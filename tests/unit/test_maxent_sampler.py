"""Coordinate hit-and-run mechanics (M5).

What is testable *here* is the machinery: feasible dispersed starts, the RNG's semantic keying, the
refresh, the singleton path, the schedule. What is **not** testable here is stationarity — a chain
that samples the wrong law still produces feasible, well-keyed, correctly-counted draws. That claim
is settled in `tests/statistical/test_uniform_targets.py`, against distributions known on paper.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.diagnostics import mcse, posterior_variance
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.line_distribution import L1Objective
from gsmm_compiler.maxent_sampler import (
    SamplerError,
    dispersed_start,
    movable_reactions,
    run_chain,
    run_chains,
    run_ladder,
    trace_objective,
)
from gsmm_compiler.rounding import RoundingError, build_transform
from gsmm_compiler.sparse_objective import (
    ObjectiveError,
    ReducedObjective,
    SparseFluxObjective,
    choose_energy_scale,
    lower_objective,
    solve_sparse_objective,
)


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


# --- M6: the tilt's inputs — reduced objective, traces, the ladder --------------------------------


@pytest.fixture(scope="module")
def tilted_simplex(simplex_flux_polytope):  # type: ignore[no-untyped-def]
    """The M6 production path, once: FluxPolytope → reduce → geometry → transform → objective."""
    reduced = simplex_flux_polytope.reduce()
    objective = SparseFluxObjective.from_polytope(simplex_flux_polytope, l1_penalty=1.0)
    lowered = lower_objective(reduced, objective)

    geometry = build_geometry(reduced, model_id="m6-simplex")
    transform = build_transform(geometry, reduced)

    solution = solve_sparse_objective(reduced, objective)
    scale = choose_energy_scale(
        lowered, geometry.support_points, j_star=solution.optimum.j_star, mode=1.0
    )
    optimum = transform.to_coordinates(reduced.to_reduced(solution.optimum.v_full))

    return transform, reduced, lowered, scale, optimum


class TestTheObjectiveMustBeInReducedCoordinates:
    """The failure that produces no bad numbers, only wrong ones.

    A full-model `L1Objective` handed to the sampler indexes ``v`` (length ``n_free``) with
    full-model reaction indices. Where the index is out of range NumPy raises and the mistake is
    loud. Where it happens to be *in* range — and on any model with more free reactions than the
    biomass index, it is — the read succeeds, ``J`` rewards the wrong reaction's flux, and the chain
    tilts toward it while feasibility, mass balance, chords and R̂ all stay perfectly green. Nothing
    else in this package knows which reaction ``J`` is supposed to be about.
    """

    def test_a_biomass_index_past_the_reduced_polytope_is_refused(
        self, tilted_simplex
    ) -> None:
        transform, reduced, lowered, scale, _ = tilted_simplex
        full_model_objective = L1Objective(
            biomass_index=reduced.n_free + 5,  # a full-model index
            penalized_indices=np.array([0], dtype=np.intp),
            weights=np.array([1.0]),
            lam=1.0,
        )

        with pytest.raises(SamplerError, match="REDUCED coordinates"):
            run_chain(
                transform,
                reduced,
                config=SHORT,
                model_id="wrong-idx",
                chain_index=0,
                beta=1.0,
                objective=full_model_objective,
                energy_scale=1.0,
            )

    def test_penalized_indices_past_the_reduced_polytope_are_refused(
        self, tilted_simplex
    ) -> None:
        transform, reduced, lowered, scale, _ = tilted_simplex
        stray = L1Objective(
            biomass_index=0,
            penalized_indices=np.array([1, reduced.n_free + 2], dtype=np.intp),
            weights=np.array([1.0, 1.0]),
            lam=1.0,
        )

        with pytest.raises(SamplerError, match="REDUCED coordinates"):
            run_chain(
                transform,
                reduced,
                config=SHORT,
                model_id="wrong-pen",
                chain_index=0,
                beta=1.0,
                objective=stray,
                energy_scale=1.0,
            )

    def test_the_properly_lowered_objective_is_accepted(self, tilted_simplex) -> None:
        transform, reduced, lowered, scale, _ = tilted_simplex

        chain = run_chain(
            transform,
            reduced,
            config=SHORT,
            model_id="ok",
            chain_index=0,
            beta=1.0,
            objective=lowered.line,
            energy_scale=scale.value,
        )

        assert chain.coordinates.shape == (SHORT.n_samples, transform.dimension)


class TestTheOptimumJoinsTheStartHull:
    def test_passing_none_reproduces_m5s_draws_bit_for_bit(self, simplex, simplex_polytope) -> None:
        """M5's β=0 chains must not move because M6 added a parameter. The Dirichlet is over ``K``
        weights when no optimum is supplied, so it consumes exactly the numbers it used to."""
        rng_a = np.random.default_rng(4)
        rng_b = np.random.default_rng(4)

        first, _ = dispersed_start(simplex, simplex_polytope, rng_a)
        second, _ = dispersed_start(simplex, simplex_polytope, rng_b, optimum_coordinates=None)

        assert np.array_equal(first, second)

    def test_the_optimum_changes_the_start_distribution(self, tilted_simplex) -> None:
        """It must, or adding it bought nothing: the hull is a genuinely different set."""
        transform, reduced, _, _, optimum = tilted_simplex

        without = np.array(
            [
                dispersed_start(transform, reduced, np.random.default_rng(s))[0]
                for s in range(40)
            ]
        )
        with_optimum = np.array(
            [
                dispersed_start(
                    transform, reduced, np.random.default_rng(s), optimum_coordinates=optimum
                )[0]
                for s in range(40)
            ]
        )

        assert not np.allclose(without, with_optimum)
        # …and the starts must still be feasible, which is the only thing that was ever required.
        for start in with_optimum:
            assert reduced.contains(transform.to_flux(start), tol=1e-9)

    def test_a_misshapen_optimum_is_refused(self, tilted_simplex) -> None:
        transform, reduced, _, _, _ = tilted_simplex

        with pytest.raises(SamplerError, match="optimum_coordinates"):
            dispersed_start(
                transform,
                reduced,
                np.random.default_rng(0),
                optimum_coordinates=np.zeros(transform.dimension + 1),
            )


class TestTheObjectiveTraces:
    """Spec §24.2 — μ, C, J, the normalized log-energy, and the near-zero counts."""

    def test_the_trace_is_the_exact_objective_of_the_stored_fluxes(self, tilted_simplex) -> None:
        """Not an incremental cache, and this is what that buys: the trace is a *function of the
        stored sample*, so it cannot drift from it. Recomputing it from scratch must reproduce
        it."""
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(1.0,)),
            model_id="traces",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        rung = ladder.rungs[0]
        for chain, trace in zip(rung.result.chains, rung.traces, strict=True):
            mu, cost, j = lowered.evaluate_many(chain.fluxes)

            assert np.array_equal(trace.mu, mu)
            assert np.array_equal(trace.cost, cost)
            assert np.array_equal(trace.j, j)
            assert np.allclose(trace.j, trace.mu - lowered.l1_penalty * trace.cost)

    def test_the_log_energy_is_the_exponent_the_chain_was_actually_tilted_by(
        self, tilted_simplex
    ) -> None:
        """``(J − J*)/s_J``, with the *same* ``J*`` and ``s_J`` the draw used — not a plausible
        recomputation from different inputs, which would agree on every healthy run and diverge
        precisely when something had gone wrong."""
        transform, reduced, lowered, scale, optimum = tilted_simplex
        trace = trace_objective(
            transform.to_flux(np.zeros((3, transform.dimension))),
            lowered,
            j_star=scale.j_star,
            energy_scale=scale.value,
        )

        expected = (trace.j - scale.j_star) / scale.value
        assert np.allclose(trace.normalized_log_energy, expected)

    def test_near_zero_counts_are_per_threshold_and_over_the_free_reactions(
        self, tilted_simplex
    ) -> None:
        transform, reduced, lowered, scale, _ = tilted_simplex
        fluxes = np.array([[1.0, 1e-7, 0.0], [1e-4, 1e-4, 1e-4]])

        trace = trace_objective(
            fluxes,
            lowered,
            j_star=scale.j_star,
            energy_scale=scale.value,
            thresholds=(1e-9, 1e-6, 1e-3),
        )

        assert trace.near_zero_thresholds == (1e-9, 1e-6, 1e-3)
        assert trace.near_zero_counts.tolist() == [[1, 2, 2], [0, 0, 3]]
        assert trace.n_free == reduced.n_free

    def test_a_nonpositive_energy_scale_is_refused(self, tilted_simplex) -> None:
        _, _, lowered, _, _ = tilted_simplex

        with pytest.raises(SamplerError, match="energy_scale"):
            trace_objective(np.zeros((2, lowered.n_free)), lowered, j_star=0.0, energy_scale=0.0)


class TestTheLadder:
    def test_every_rung_gets_its_own_rng_stream(self, tilted_simplex) -> None:
        """``β_index`` names the stream, so two rungs never draw the same numbers — which they would
        if the key were built from ``β``'s *value* and a ladder repeated one."""
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.5, 0.5)),  # the same β twice, deliberately
            model_id="streams",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        first, second = ladder.rungs
        assert first.beta == second.beta
        assert first.result.chains[0].diagnostics.spawn_key != (
            second.result.chains[0].diagnostics.spawn_key
        )
        assert not np.array_equal(
            first.result.chains[0].coordinates, second.result.chains[0].coordinates
        )

    def test_the_manifest_carries_the_scale_every_rung_was_tilted_by(
        self, tilted_simplex
    ) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.0, 1.0)),
            model_id="manifest",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        manifest = ladder.manifest()

        assert manifest["betas"] == [0.0, 1.0]
        assert manifest["energy_scale"] == scale.value
        assert manifest["j_star"] == scale.j_star
        assert "monotonicity" in manifest
        assert len(manifest["rungs"]) == 2

    def test_beta_zero_needs_no_objective_but_still_gets_a_trace(self, tilted_simplex) -> None:
        """The reference rung. ``J`` never enters the β=0 draw — but ``E_0[J]`` is exactly what the
        monotonicity check compares every other rung against, so it must still be measured."""
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.0,)),
            model_id="cold",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        rung = ladder.rungs[0]
        assert rung.beta == 0.0
        assert np.all(np.isfinite(rung.j))
        assert rung.ess_j > 0.0

    def test_rung_at_finds_a_beta_and_refuses_one_that_is_absent(self, tilted_simplex) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.0, 2.0)),
            model_id="rungs",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        assert ladder.rung_at(2.0).beta_index == 1
        with pytest.raises(SamplerError, match="no rung"):
            ladder.rung_at(7.0)


class TestTheMonotonicityReport:
    """``dE_β[J]/dβ = Var_β(J)/s_J ≥ 0`` — a theorem, so a violation is noise or a bug, never
    physics.

    These tests are about the *instrument*, not the sampler: whether it measures a drop in the right
    units and compares the rungs in the right order. Whether the sampler actually satisfies the
    theorem is settled on real chains, in `tests/integration/test_m6_positive_beta.py`.
    """

    def test_a_drop_is_measured_in_pooled_standard_errors(self, tilted_simplex) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.0, 1.0, 3.0)),
            model_id="mono",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        report = ladder.monotonicity()
        means = report.mean_j
        errors = report.standard_error_j

        worst = max(
            (means[i] - means[i + 1]) / float(np.hypot(errors[i], errors[i + 1]))
            for i in range(len(means) - 1)
        )
        assert report.worst_drop_sigma == pytest.approx(worst)
        assert report.betas == (0.0, 1.0, 3.0)

    def test_the_error_bar_is_var_plus_over_ess_not_sd_over_root_n(self, tilted_simplex) -> None:
        """Two lessons in one number, and it is wrong without either.

        * ``√N`` in the denominator (the M5 lesson): with an ESS well below ``N`` it understates the
          error, and the monotonicity check would reject ladders that are perfectly consistent.
        * The **pooled sample SD** in the numerator (Codex, M6 review): the ESS is estimated against
          ``var⁺``, the overdispersed variance that counts between-chain disagreement. Pairing it
          with the pooled variance throws that conservatism away — under-reporting the error by
          ``√2`` for two chains trapped at ``±a``, which is exactly the case an error bar exists
          for.

        So the SE is ``√(var⁺/ESS)``, and both parts are pinned here.
        """
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(1.0,)),
            model_id="ess",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        rung = ladder.rungs[0]
        j = rung.j.reshape(-1)

        assert rung.ess_j < j.size, "an MCMC chain does not deliver independent draws"
        assert rung.standard_error_j > float(j.std(ddof=1)) / np.sqrt(j.size)  # not sd/√N
        assert rung.standard_error_j == pytest.approx(
            float(np.sqrt(posterior_variance(rung.j)[0] / rung.ess_j))
        )
        assert rung.standard_error_j == pytest.approx(float(mcse(rung.j)[0]))

    def test_a_non_finite_mean_or_error_is_refused_rather_than_read_as_a_pass(
        self, tilted_simplex
    ) -> None:
        """**Codex, M6 review.** ``max(-inf, nan)`` is ``-inf`` in Python — the comparison
        ``nan > -inf`` is False — so a single NaN σ would sail through the fold and be reported as
        *the most monotone ladder imaginable*. The same shape of trap as M5's
        ``np.min(x, initial=0.0)``, which reported an ESS of 0 for a sample whose every entry was
        8000: a sentinel that silently wins a comparison it was never meant to enter.

        A non-finite mean is not weak evidence of monotonicity. It is none at all, so it is refused.
        """
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(0.0, 1.0)),
            model_id="nan",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        assert max(float("-inf"), float("nan")) == float("-inf")  # the trap itself, in one line

        poisoned = replace(
            ladder,
            rungs=(
                replace(
                    ladder.rungs[0],
                    traces=(
                        replace(ladder.rungs[0].traces[0], j=np.full(SHORT.n_samples, np.nan)),
                        *ladder.rungs[0].traces[1:],
                    ),
                ),
                ladder.rungs[1],
            ),
        )

        with pytest.raises(SamplerError, match="not finite"):
            poisoned.monotonicity()

    def test_the_rungs_are_compared_in_ascending_beta_whatever_order_they_were_run_in(
        self, tilted_simplex
    ) -> None:
        """A config is free to list its ladder in any order. Comparing rungs in *config* order would
        report a spurious drop the moment someone wrote ``betas = [2.0, 0.0]``."""
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(3.0, 0.0)),  # descending, on purpose
            model_id="order",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        report = ladder.monotonicity()

        assert report.betas == (0.0, 3.0)
        assert report.mean_j[0] < report.mean_j[1]  # the tilted rung has the higher mean J

    def test_a_one_rung_ladder_is_vacuously_monotone(self, tilted_simplex) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex
        ladder = run_ladder(
            transform,
            reduced,
            config=replace(SHORT, betas=(1.0,)),
            model_id="single",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )

        assert ladder.monotonicity().is_monotone


class TestTheArtifactsMustHaveBeenBuiltFromThisPolytope:
    """**Codex, M6 review.** The failure where every diagnostic agrees, and all of them are wrong.

    Three artifacts meet in `run_ladder` — the L1 polytope, the L3 transform, the L2 objective — and
    nothing checked that they had ever met before. They are all just arrays.

    Hand it an objective lowered from a *different* model of the same size: the chain tilts by the
    reactions **that objective** names, and `ReducedObjective.evaluate_many` reports those same
    reactions as ``μ`` and ``C``. So the trace of ``J`` rises with β exactly as the theorem requires
    — because the chain really is maximizing the thing the trace is measuring. Feasibility, mass
    balance, the chords and R̂ are all perfect. Nothing else in this package knows which reaction
    ``J`` is supposed to be about, so nothing else can object.

    One string comparison per run buys it away, and it is what makes M8's cache safe to build: L2
    and
    L3 are separate artifacts on disk, and a stale key is all it takes to load two that were never
    computed against each other.
    """

    def test_an_objective_lowered_from_another_polytope_is_refused(
        self, tilted_simplex, simplex_polytope
    ) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex

        stranger = replace(lowered, polytope_key="a-key-from-some-other-model")

        with pytest.raises(SamplerError, match="not lowered from this polytope"):
            run_ladder(
                transform,
                reduced,
                config=replace(SHORT, betas=(1.0,)),
                model_id="stranger",
                objective=stranger,
                energy_scale=scale,
                optimum_coordinates=optimum,
            )

    def test_a_transform_from_another_polytope_is_refused(self, tilted_simplex) -> None:
        transform, reduced, lowered, scale, optimum = tilted_simplex

        stranger = replace(transform, polytope_key="a-key-from-some-other-model")

        with pytest.raises(SamplerError, match="not built from this polytope"):
            run_ladder(
                stranger,
                reduced,
                config=replace(SHORT, betas=(1.0,)),
                model_id="stranger-T",
                objective=lowered,
                energy_scale=scale,
                optimum_coordinates=optimum,
            )

    def test_the_properly_lowered_pair_binds(self, tilted_simplex) -> None:
        _, reduced, lowered, _, _ = tilted_simplex

        assert lowered.binds_to(reduced)
        assert lowered.polytope_key == reduced.content_key()

    def test_the_kernels_lambda_and_the_reported_lambda_cannot_disagree(
        self, tilted_simplex
    ) -> None:
        """The chain bends ``J`` by ``line.lam``; the traces report ``J = μ − l1_penalty·C``. Let
        the two drift apart and the sampler draws from one distribution while the run describes
        another — and both look entirely healthy on their own."""
        _, _, lowered, _, _ = tilted_simplex

        with pytest.raises(ObjectiveError, match="λ"):
            replace(lowered, l1_penalty=lowered.l1_penalty + 1.0)


class TestImmovableIsNotTheSameAsZero:
    """**Codex, M6 review round 2.** A zero row of ``T`` means the chain cannot move that reaction.
    It does **not** mean the reaction is at zero.

    Mass balance can pin a *free* reaction (``l < u``) at a nonzero constant, and
    `pinned_nonzero_polytope` is the minimal case: ``v0 ∈ [0, 2]`` with a row forcing ``v0 = 1``. It
    is immovable and it is nowhere near zero.

    Every one of the example model's 61 immovable reactions happens to sit at ~1e-13, so *cannot
    move* and *is at zero* are indistinguishable there — and M6's first attempt at the near-zero
    reconciliation conflated them, asserting ``all_free − movable == n_blocked`` as an identity. It
    is not one. It is a measured property of that model, and this test is how we know.
    """

    @pytest.fixture(scope="class")
    def pinned(self, pinned_nonzero_polytope):  # type: ignore[no-untyped-def]
        geometry = build_geometry(pinned_nonzero_polytope, model_id="pinned")
        transform = build_transform(geometry, pinned_nonzero_polytope)

        # Built directly rather than through `lower_objective`: what is under test here is the
        # near-zero *counting*, and the lowering has its own tests. λ = 0, so J = v1 (biomass).
        lowered = ReducedObjective(
            line=L1Objective(
                biomass_index=1,
                penalized_indices=np.empty(0, dtype=np.intp),
                weights=np.empty(0),
                lam=0.0,
            ),
            weights=np.zeros(3),
            mu_offset=0.0,
            cost_offset=0.0,
            l1_penalty=0.0,
            n_free=3,
            polytope_key=pinned_nonzero_polytope.content_key(),
        )
        return transform, pinned_nonzero_polytope, lowered

    def test_the_pinned_reaction_is_immovable(self, pinned) -> None:
        transform, reduced, _ = pinned
        movable = movable_reactions(transform)

        assert transform.dimension == 1  # only v1 = v2 can vary
        assert movable.tolist() == [1, 2]
        assert 0 not in movable

    def test_but_its_flux_is_one_not_zero(self, pinned) -> None:
        transform, reduced, _ = pinned

        assert float(transform.center[0]) == pytest.approx(1.0, abs=1e-9)

    def test_so_the_two_near_zero_counts_do_not_differ_by_n_blocked(self, pinned) -> None:
        """The claim M6 originally asserted as an identity, refuted on the polytope that can see it.

        At a threshold of 0.5 the immovable ``v0 = 1`` is *not* near zero, so it contributes nothing
        to the free-set count — and the gap between the two counts is 0, while ``n_blocked`` is 1.
        """
        transform, reduced, lowered = pinned
        fluxes = transform.to_flux(np.array([[0.0], [0.4]]))

        trace = trace_objective(
            fluxes,
            lowered,
            j_star=1.0,
            energy_scale=1.0,
            thresholds=(0.5,),
            movable=movable_reactions(transform),
        )

        assert trace.n_blocked == 1
        gap = trace.near_zero_counts_all_free - trace.near_zero_counts
        assert np.all(gap == 0), "v0 = 1 is immovable but NOT near zero, so it adds nothing"
        assert np.any(gap != trace.n_blocked), "the identity M6 first asserted is false here"


class TestEveryJoinInTheSamplerIsBound:
    """**Codex, M6 review round 6.** The sweep for the *class*, not the instance.

    Across five rounds Codex found the same bug at five different joins: two artifacts that were
    never computed against each other, silently combined. Each round I patched the join he had
    demonstrated. Round 6 asked the right question — *which joins are still unguarded?* — and found
    three more in this module's path.

    The nastiest is `rounding.build_transform`: it takes a geometry **and** a polytope, builds ``T``
    from the geometry's basis while taking the *bounds* and the `CoordinatePrecompute` from the
    polytope, and records the **geometry's** `polytope_key`. A mismatched pair therefore produces a
    hybrid that **passes `run_ladder`'s binding check** — the key it reports is the geometry's —
    while stepping against another model's bounds entirely.
    """

    def test_a_transform_built_from_another_polytope_is_refused_at_construction(
        self, simplex_polytope, coupled_box_polytope
    ) -> None:
        geometry = build_geometry(simplex_polytope, model_id="simplex")

        with pytest.raises(RoundingError, match="not built from this polytope"):
            build_transform(geometry, coupled_box_polytope)

    def test_run_chain_binds_the_transform_to_the_polytope(
        self, simplex, coupled_box_polytope
    ) -> None:
        """`run_ladder` guards this, but `run_chain` is the low-level entry point and was not."""
        with pytest.raises(SamplerError, match="not built from this polytope"):
            run_chain(
                simplex,  # built from simplex_polytope
                coupled_box_polytope,  # …but bounds-checked against a different one
                config=SHORT,
                model_id="mixed",
                chain_index=0,
            )

    def test_an_energy_scale_from_another_objective_is_refused(self, tilted_simplex) -> None:
        """``s_J`` is the range ``J`` spans over *one* objective on *one* polytope. Borrowed from
        another, every β on the ladder silently names a different selection pressure — which is the
        entire failure ``s_J`` exists to prevent."""
        transform, reduced, lowered, scale, optimum = tilted_simplex

        stranger = replace(scale, polytope_key="a-key-from-some-other-objective")

        with pytest.raises(SamplerError, match="calibrated from a different objective"):
            run_ladder(
                transform,
                reduced,
                config=replace(SHORT, betas=(1.0,)),
                model_id="mixed-scale",
                objective=lowered,
                energy_scale=stranger,
                optimum_coordinates=optimum,
            )

    def test_a_movable_set_that_does_not_index_this_flux_vector_is_refused(
        self, tilted_simplex
    ) -> None:
        transform, reduced, lowered, scale, _ = tilted_simplex

        with pytest.raises(SamplerError, match="movable"):
            trace_objective(
                np.zeros((2, lowered.n_free)),
                lowered,
                j_star=scale.j_star,
                energy_scale=scale.value,
                movable=np.array([0, lowered.n_free + 3], dtype=np.intp),
            )
