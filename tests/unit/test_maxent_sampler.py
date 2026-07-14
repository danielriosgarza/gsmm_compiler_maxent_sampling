"""Coordinate hit-and-run mechanics (M5).

What is testable *here* is the machinery: feasible dispersed starts, the RNG's semantic keying, the
refresh, the singleton path, the schedule. What is **not** testable here is stationarity — a chain
that samples the wrong law still produces feasible, well-keyed, correctly-counted draws. That claim
is settled in `tests/statistical/test_uniform_targets.py`, against distributions known on paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.line_distribution import L1Objective
from gsmm_compiler.maxent_sampler import (
    SamplerError,
    dispersed_start,
    run_chain,
    run_chains,
)
from gsmm_compiler.rounding import build_transform


@pytest.fixture(scope="module")
def simplex(simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(simplex_polytope, model_id="simplex")
    return build_transform(geometry, simplex_polytope)


SHORT = SamplerConfig(n_chains=4, n_samples=200, burn_in=100, thin=1, refresh_interval=50)


class TestTheStarts:
    def test_a_dispersed_start_is_feasible(self, simplex, simplex_polytope) -> None:
        rng = np.random.default_rng(0)

        for _ in range(50):
            start, shrink = dispersed_start(simplex, simplex_polytope, rng)
            v = simplex.to_flux(start)

            assert shrink == 1.0, "a convex combination of feasible points should never need one"
            assert np.all(v >= simplex_polytope.lower_bounds - 1e-9)
            assert np.all(v <= simplex_polytope.upper_bounds + 1e-9)
            assert abs(float(v.sum()) - 1.0) < 1e-9  # the simplex's own mass balance

    def test_the_starts_are_actually_dispersed(self, simplex, simplex_polytope) -> None:
        """R̂ divides between-chain variance by within-chain variance. Chains that all start in a
        huddle around the centre make the numerator small for reasons that have nothing to do with
        convergence — which is why the Dirichlet concentration is below 1."""
        rng = np.random.default_rng(1)
        starts = np.stack(
            [dispersed_start(simplex, simplex_polytope, rng)[0] for _ in range(200)]
        )

        # Spread out relative to the support points they are drawn from, not merely nonzero.
        support_spread = simplex.support_coordinates.std(axis=0).mean()
        assert starts.std(axis=0).mean() > 0.3 * support_spread

    def test_an_infeasible_supplied_start_is_refused(self, simplex, simplex_polytope) -> None:
        far_outside = np.full(simplex.dimension, 100.0)

        with pytest.raises(SamplerError, match="not feasible"):
            run_chain(
                simplex,
                simplex_polytope,
                config=SHORT,
                model_id="simplex",
                chain_index=0,
                start=far_outside,
            )


class TestTheRngIsKeyedSemantically:
    """The property that makes a batch reproducible: a stream is named by *what it is*, not by its
    position in a `spawn()` sequence (`provenance.stream_seed`)."""

    def test_the_same_coordinates_give_the_same_chain(self, simplex, simplex_polytope) -> None:
        kwargs = dict(config=SHORT, model_id="simplex", chain_index=2)
        first = run_chain(simplex, simplex_polytope, **kwargs)
        second = run_chain(simplex, simplex_polytope, **kwargs)

        assert np.array_equal(first.coordinates, second.coordinates)

    def test_different_chains_draw_different_numbers(self, simplex, simplex_polytope) -> None:
        kwargs = dict(config=SHORT, model_id="simplex")
        first = run_chain(simplex, simplex_polytope, chain_index=0, **kwargs)
        second = run_chain(simplex, simplex_polytope, chain_index=1, **kwargs)

        assert not np.array_equal(first.coordinates, second.coordinates)

    def test_adding_a_chain_does_not_renumber_the_others(self, simplex, simplex_polytope) -> None:
        """The whole reason `stream_seed` exists. Under a flat `spawn()` sequence, running 8 chains
        instead of 4 would give chain 2 a *different* stream — and every previously published number
        would change."""
        four = run_chains(
            simplex,
            simplex_polytope,
            config=SamplerConfig(n_chains=4, n_samples=50, burn_in=10),
            model_id="simplex",
        )
        eight = run_chains(
            simplex,
            simplex_polytope,
            config=SamplerConfig(n_chains=8, n_samples=50, burn_in=10),
            model_id="simplex",
        )

        for index in range(4):
            assert np.array_equal(
                four.chains[index].coordinates, eight.chains[index].coordinates
            )

    def test_a_different_model_id_gives_a_different_stream(self, simplex, simplex_polytope) -> None:
        kwargs = dict(config=SHORT, chain_index=0)
        first = run_chain(simplex, simplex_polytope, model_id="a", **kwargs)
        second = run_chain(simplex, simplex_polytope, model_id="b", **kwargs)

        assert not np.array_equal(first.coordinates, second.coordinates)

    def test_the_spawn_key_is_recorded(self, simplex, simplex_polytope) -> None:
        """Reproducing a chain must need nothing but its manifest."""
        chain = run_chain(
            simplex, simplex_polytope, config=SHORT, model_id="simplex", chain_index=3
        )

        assert chain.diagnostics.spawn_key[-1] == 3
        assert len(chain.diagnostics.spawn_key) == 4  # model, stage, β, chain


class TestTheSchedule:
    def test_burn_in_is_discarded_and_the_count_is_exact(self, simplex, simplex_polytope) -> None:
        config = SamplerConfig(n_chains=1, n_samples=37, burn_in=13, thin=3, refresh_interval=5)
        chain = run_chain(
            simplex, simplex_polytope, config=config, model_id="simplex", chain_index=0
        )

        assert chain.coordinates.shape == (37, simplex.dimension)
        assert chain.fluxes.shape == (37, simplex.n_free)
        assert chain.diagnostics.n_sweeps == 13 + 37 * 3

    def test_the_stored_flux_is_exactly_the_stored_state(self, simplex, simplex_polytope) -> None:
        """The stored flux must be ``centre + T·y`` of the stored ``y`` **exactly**, not the
        incremental cache that happened to be in hand.

        The M5 review's finding: storing the cache leaves a reader with two quantities that are
        supposed to be the same and are not, and it makes `max_refresh_drift` the only window onto
        the discrepancy — which is not a bound, since drift can peak and partly cancel *between*
        refreshes. Now the sampler recomputes the exact flux at every stored sample, so the equality
        below is exact rather than approximate, and the drift is measured there too.
        """
        chain = run_chain(
            simplex, simplex_polytope, config=SHORT, model_id="simplex", chain_index=0
        )

        rebuilt = simplex.to_flux(chain.coordinates)
        assert np.abs(rebuilt - chain.fluxes).max() == 0.0

    def test_the_refresh_drift_is_tiny_and_measured(self, simplex, simplex_polytope) -> None:
        """The incremental ``v += t·d`` is the one place float64 perturbs the kernel. It is bounded
        by measurement, not by argument."""
        chain = run_chain(
            simplex, simplex_polytope, config=SHORT, model_id="simplex", chain_index=0
        )

        total_sweeps = SHORT.burn_in + SHORT.n_samples * SHORT.thin
        assert chain.diagnostics.n_refreshes == total_sweeps // SHORT.refresh_interval
        assert 0.0 <= chain.diagnostics.max_refresh_drift < 1e-10


class TestValidation:
    def test_a_positive_beta_needs_an_objective(self, simplex, simplex_polytope) -> None:
        """M6 supplies it. Until then the tilted path must refuse rather than silently sample flat —
        which is exactly what a missing ``J`` would do."""
        with pytest.raises(SamplerError, match="objective is required"):
            run_chain(
                simplex,
                simplex_polytope,
                config=SHORT,
                model_id="simplex",
                chain_index=0,
                beta=1.0,
            )

    def test_a_positive_beta_needs_a_positive_energy_scale(self, simplex, simplex_polytope) -> None:
        objective = L1Objective(
            biomass_index=0,
            penalized_indices=np.array([1, 2], dtype=np.intp),
            weights=np.ones(2),
            lam=0.1,
        )

        with pytest.raises(SamplerError, match="energy_scale"):
            run_chain(
                simplex,
                simplex_polytope,
                config=SHORT,
                model_id="simplex",
                chain_index=0,
                beta=1.0,
                objective=objective,
                energy_scale=0.0,
            )

    @pytest.mark.parametrize("beta", [-1.0, np.nan, np.inf])
    def test_a_nonsense_beta_is_refused(self, simplex, simplex_polytope, beta: float) -> None:
        with pytest.raises(SamplerError, match="beta"):
            run_chain(
                simplex,
                simplex_polytope,
                config=SHORT,
                model_id="simplex",
                chain_index=0,
                beta=beta,
            )


class TestTheSingletonPath:
    def test_a_zero_dimensional_polytope_yields_the_center_every_time(
        self, singleton_polytope: ReducedPolytope
    ) -> None:
        """spec §16: the feasible set is a point, so the sample *is* that point — and the sampler
        must not try to draw a coordinate out of an empty set."""
        geometry = build_geometry(singleton_polytope, model_id="singleton")
        transform = build_transform(geometry, singleton_polytope)

        result = run_chains(
            transform,
            singleton_polytope,
            config=SamplerConfig(n_chains=2, n_samples=10, burn_in=5),
            model_id="singleton",
        )

        for chain in result.chains:
            assert chain.coordinates.shape == (10, 0)
            assert np.abs(chain.fluxes - transform.center).max() == 0.0
            assert chain.diagnostics.dimension == 0


class TestNoSolverInTheInnerLoop:
    """BUILD_PLAN §1.3, and half of the M5 gate."""

    def test_sampling_performs_zero_highs_solves(self, simplex, simplex_polytope) -> None:
        before = total_solve_count()
        run_chains(simplex, simplex_polytope, config=SHORT, model_id="simplex")

        assert total_solve_count() == before

    def test_importing_the_sampler_does_not_load_highspy_or_cobra(self) -> None:
        """§1.2: a worker receives frozen arrays and a seed. It must not drag a solver or a parser
        into the process to do so — on a 14-worker Jetson that is 14 copies of each."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import gsmm_compiler.maxent_sampler; "
                "print('highspy' in sys.modules, 'cobra' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "False False"


class TestTheChainStaysInThePolytope:
    def test_every_sample_is_bound_feasible_and_mass_balanced(
        self, simplex, simplex_polytope
    ) -> None:
        result = run_chains(simplex, simplex_polytope, config=SHORT, model_id="simplex")

        for chain in result.chains:
            assert chain.diagnostics.max_bound_violation == 0.0
            assert chain.diagnostics.max_mass_balance_residual < 1e-12

    def test_a_well_rounded_polytope_produces_no_degenerate_chords(
        self, simplex, simplex_polytope
    ) -> None:
        """Not a requirement — a self-loop is exact — but a large count would mean the chain is
        pinned against a face, and that is worth noticing rather than absorbing."""
        result = run_chains(simplex, simplex_polytope, config=SHORT, model_id="simplex")

        assert all(chain.diagnostics.n_degenerate_steps == 0 for chain in result.chains)
