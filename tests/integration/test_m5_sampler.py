"""M5 acceptance gate — rounding and the β=0 sampler on the genome-scale model.

The gate (BUILD_PLAN M5): uniform analytic targets reproduced · transform-invariance of moments ·
positive chords at the start · ``‖S·T‖ ≈ 0`` · **zero inner-loop HiGHS solves**.

The analytic targets live in `tests/statistical/test_uniform_targets.py`, where the exact law is
known. What can only be checked *here* is that the same machinery survives 773 reactions, 894
metabolites and a 46-dimensional affine hull — and in particular that every sampled state is
feasible in the **full** model, not merely in the reduced one it was drawn in. A reduced-space
sampler internally consistent and wrong about the lift passes every test here but that one.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import GeometryConfig, SamplerConfig
from gsmm_compiler.diagnostics import convergence_report, feasibility_report
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.line_geometry import chord_on_support, feasible_chord
from gsmm_compiler.maxent_sampler import dispersed_start, run_chains
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.rounding import build_transform

pytestmark = pytest.mark.slow

DIMENSION = 46
"""Settled by M4, against an independent FVA+rank oracle. Not ``n_free − rank(S) = 55``."""


@pytest.fixture(scope="module")
def reduced(example_canonical: CanonicalModel) -> ReducedPolytope:
    return example_canonical.polytope.reduce()


@pytest.fixture(scope="module")
def geometry(reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    return build_geometry(reduced, model_id="bifido")


@pytest.fixture(scope="module")
def transform(geometry, reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    return build_transform(geometry, reduced)


@pytest.fixture(scope="module")
def samples(transform, reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    """4000 draws, not 1500, and the difference is a measurement rather than a preference.

    M5 recorded that this model mixes slowly. At 1500 draws it mixes *so* slowly that the
    diagnostics land on both sides of their own bars: across 8 seeds R̂ spans **1.089 – 1.177**
    against a 1.15 bar and min ESS spans **10.2 – 50.7** against a 20 bar, so two of eight seeds
    failed — including seed 0, this fixture's own, on **both** assertions. That is not a flaky
    test, it is a bar inside the distribution of the thing it judges: M9's *a bar a valid input
    clears only 2 times in 3 is not a tolerance, it is a coin flip*, for the third time in this
    package.

    It surfaced as a threading bug (M10.2e — the ambient BLAS thread count picked the basis, and
    R̂ with it), and that framing was wrong: the thread count was one way to toss the coin, and
    **seeds toss it just as well**. Pinning the threads only fixes which way it lands.

    R̂ → 1 as the chain grows is a theorem, so the schedule is the honest lever. Measured across 5
    seeds at 4000 draws: R̂ **1.033 – 1.059** (2.5× margin on the bar's excess) and min ESS
    **59.7 – 155.0** (3×). The bars now catch a regression that breaks mixing instead of sampling
    the noise. Costs 19.5 s against 7.4 s, once, for the whole module.
    """
    config = SamplerConfig(
        n_chains=4, n_samples=4000, burn_in=4000, thin=1, refresh_interval=250
    )
    return run_chains(transform, reduced, config=config, model_id="bifido", beta=0.0)


class TestTheTransform:
    def test_it_preconditions_all_46_directions(self, transform) -> None:
        assert transform.dimension == DIMENSION
        assert transform.transform.shape == (260, DIMENSION)
        assert transform.diagnostics.covariance_rank == DIMENSION

    def test_S_times_T_stays_on_the_mass_balance_manifold(self, transform, reduced) -> None:
        """``‖S·T‖ ≈ 0`` — the gate. Judged on a *relative* bar (M4): ``S·T_k`` sums terms of size
        ``|S|·|T_k|``, so an absolute 1e-9 charges float64's own rounding to the geometry."""
        assert transform.diagnostics.transform_mass_balance_error < 1e-9

        for k in range(transform.dimension):
            column = np.ascontiguousarray(transform.transform[:, k])
            assert float(reduced.stoichiometry.relative_residual(column).max()) < 1e-9

    def test_the_blocked_reactions_stay_exactly_zero_in_T(self, transform, geometry) -> None:
        """The M4 landmine, re-armed by the rounding multiply. 61 free reactions cannot carry flux,
        so their rows in ``B`` are *exact* zeros — and if ``L`` smeared a 1e-300 into any of them,
        `np.flatnonzero` would put that reaction back into the chord and its bound ratio (a noise
        value divided by a noise value) would produce a chord limit of order 0.03–0.5."""
        blocked = ~np.any(geometry.basis != 0.0, axis=1)

        assert int(blocked.sum()) == 61
        assert np.all(transform.transform[blocked] == 0.0)  # exactly, not approximately

        for k in range(transform.dimension):
            assert not np.any(np.isin(transform.precompute.support[k], np.flatnonzero(blocked)))

    def test_rounding_actually_conditions_this_polytope(self, transform, geometry, reduced) -> None:
        """The claim `rounding` makes about itself, on the model where it is true. The toy polytopes
        cannot show this — 2-dimensional sets are not ill-conditioned, and `diag(s)` already absorbs
        axis-aligned stretch. It takes 46 dimensions of a real metabolic network."""

        def chord_lengths(matrix: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    feasible_chord(
                        geometry.center,
                        np.ascontiguousarray(matrix[:, k]),
                        reduced.lower_bounds,
                        reduced.upper_bounds,
                    ).length
                    for k in range(matrix.shape[1])
                ]
            )

        unrounded = chord_lengths(geometry.basis * geometry.scaling[:, np.newaxis])
        rounded = chord_lengths(transform.transform)

        # The worst axis is what throttles the chain, and it is the one that improves most.
        assert unrounded.min() < 0.05
        assert rounded.min() > 0.5
        assert rounded.min() > 20.0 * unrounded.min()
        # And the axes end up comparable to each other, not spanning two orders of magnitude.
        assert unrounded.max() / unrounded.min() > 50.0
        assert rounded.max() / rounded.min() < 10.0

    def test_every_rounded_axis_admits_a_step_from_the_center(self, transform) -> None:
        """The sampler's start condition. M4 established it for ``B``'s columns; ``T``'s columns are
        different directions, so it is re-established rather than inherited."""
        assert transform.diagnostics.min_chord_at_center > 0.0


class TestTheGateNoSolverInTheInnerLoop:
    """BUILD_PLAN §1.3. The counter is process-global precisely so a sampler cannot evade it by
    building a fresh LP of its own."""

    def test_sampling_performs_zero_highs_solves(self, transform, reduced) -> None:
        config = SamplerConfig(n_chains=2, n_samples=300, burn_in=200, refresh_interval=100)

        before = total_solve_count()
        run_chains(transform, reduced, config=config, model_id="bifido", beta=0.0)

        assert total_solve_count() == before

    def test_building_the_transform_is_also_solver_free(self, geometry, reduced) -> None:
        """The geometry pays for its LPs; rounding is pure linear algebra on what it produced."""
        before = total_solve_count()
        build_transform(geometry, reduced)

        assert total_solve_count() == before


class TestEverySampleIsAFeasibleFlux:
    def test_the_reduced_samples_are_feasible(self, samples, reduced) -> None:
        report = feasibility_report(samples.fluxes, reduced)

        # Counted from the chains, not typed as a literal. This read `4 * 1500` — a copy of the
        # fixture's schedule that broke the moment M10.2e changed it, while what the assertion is
        # actually for is that the report judged **every** draw rather than a subset of them.
        assert report.n_samples == sum(chain.coordinates.shape[0] for chain in samples.chains)
        assert report.is_feasible
        assert report.n_bound_violations == 0
        assert report.max_bound_violation == 0.0
        assert report.max_mass_balance_residual < 1e-9

    def test_the_lifted_samples_are_feasible_in_the_full_773_reaction_model(
        self, samples, reduced, example_canonical: CanonicalModel
    ) -> None:
        """The one thing only this test can catch. Everything upstream lives in the 260-column
        reduced polytope; a sampler perfectly consistent *there* and wrong about ``v_full = R·v_red
        + c`` would be internally spotless and yet produce infeasible fluxes. So the check runs
        against the original model's own bounds and its own 894-row stoichiometry."""
        polytope = example_canonical.polytope
        thinned = samples.fluxes.reshape(-1, reduced.n_free)[::37]  # every 37th, ~162 samples
        full = reduced.to_full(thinned)

        assert full.shape[1] == 773

        assert np.all(full >= polytope.lower_bounds - 1e-9)
        assert np.all(full <= polytope.upper_bounds + 1e-9)

        for flux in full:
            contiguous = np.ascontiguousarray(flux)
            assert float(polytope.stoichiometry.relative_residual(contiguous).max()) < 1e-9

    def test_the_fixed_reactions_keep_their_fixed_values(
        self, samples, reduced, example_canonical: CanonicalModel
    ) -> None:
        """513 reactions are eliminated from the sampled state; every saved sample must still carry
        them at their pinned values, with the full reaction order intact (CLAUDE.md)."""
        full = reduced.to_full(samples.fluxes.reshape(-1, reduced.n_free)[:20])

        assert np.all(full[:, reduced.fixed_indices] == reduced.fixed_values)

    def test_no_flux_was_snapped_to_zero(self, samples) -> None:
        """CLAUDE.md: thresholds belong to analysis, never to chain state. A sampler that quietly
        rounded small fluxes would show a suspicious pile-up of exact zeros among the free
        reactions."""
        moving = samples.fluxes.reshape(-1, 260)
        varying = moving[:, moving.std(axis=0) > 1e-6]

        assert np.count_nonzero(varying == 0.0) == 0


class TestTransformInvarianceOfMoments:
    """The M5 gate's transform-invariance, at genome scale.

    ``range(diag(s)·B·L) = range(diag(s)·B)`` for any invertible ``L``, so ``L`` cannot move the
    target — only the speed at which the chain explores it. A ridge seven orders of magnitude larger
    gives a genuinely different ``T``; if it also gave a different *distribution*, the rounding step
    would be silently retargeting the sampler and nothing else in this suite would notice.
    """

    def test_two_ridges_sample_the_same_flux_distribution(self, geometry, reduced) -> None:
        """Compared in **units of Monte-Carlo standard error**, not in units of σ.

        The naive comparison ("the two means agree to within 0.25 σ") is not a test, it is a guess:
        with an ESS of only ~1–5% of the draws — which is what coordinate hit-and-run delivers on a
        46-dimensional metabolic polytope — the standard error of each mean is already ~0.1 σ, and
        the *maximum* over 199 reactions of a mean-zero noise term with that scale routinely reaches
        0.4 σ. Measured, it does: this exact assertion at 0.25 σ failed at 0.39 σ on a pair of runs
        that are, on the statistic below, entirely consistent with each other.

        So the difference is divided by the standard error it actually has —
        ``√(σ²/ESS₁ + σ²/ESS₂)`` — which is what makes it interpretable. Two checks, which fail
        differently:
        ``max |z|`` catches one reaction badly retargeted, and ``mean z²`` — which is ≈ 1 for
        agreeing samplers, with a standard error of ``√(2/199) ≈ 0.1`` — catches a *small* bias
        spread across all of them, which no per-reaction bar could see.
        """
        config = SamplerConfig(n_chains=4, n_samples=2000, burn_in=2000, refresh_interval=500)

        small = build_transform(geometry, reduced, config=GeometryConfig(ridge_relative=1e-8))
        large = build_transform(geometry, reduced, config=GeometryConfig(ridge_relative=1e-1))
        assert np.abs(small.transform - large.transform).max() > 1e-6, "the two T are identical"

        first = run_chains(small, reduced, config=config, model_id="ridge-small", beta=0.0)
        second = run_chains(large, reduced, config=config, model_id="ridge-large", beta=0.0)

        flux_a = first.fluxes.reshape(-1, reduced.n_free)
        flux_b = second.fluxes.reshape(-1, reduced.n_free)

        moving = flux_a.std(axis=0) > 1e-6
        assert int(moving.sum()) > 150  # the comparison must be over a real set of reactions

        ess_a = convergence_report(first.fluxes[:, :, moving]).ess
        ess_b = convergence_report(second.fluxes[:, :, moving]).ess
        assert min(ess_a.min(), ess_b.min()) > 10.0, "too few effective draws to conclude anything"

        var_a = flux_a[:, moving].var(axis=0)
        var_b = flux_b[:, moving].var(axis=0)
        standard_error = np.sqrt(var_a / ess_a + var_b / ess_b)
        z = (flux_a[:, moving].mean(axis=0) - flux_b[:, moving].mean(axis=0)) / standard_error

        assert float(np.abs(z).max()) < 5.0
        assert float(np.mean(z**2)) < 2.0

        # The spreads too: log σ has standard error ≈ 1/√(2·ESS).
        log_ratio = 0.5 * np.log(var_a / var_b)
        z_spread = log_ratio / np.sqrt(0.5 / ess_a + 0.5 / ess_b)
        assert float(np.abs(z_spread).max()) < 5.0


class TestConvergence:
    def test_the_chains_mix_and_the_diagnostics_say_so(self, samples) -> None:
        """Not a correctness claim — a *reported* one. R̂ and ESS are how a user learns that this
        model needs a long chain, and the honest number for it is recorded in DEVELOPMENT_STATUS.

        Both bars are judged at the `samples` fixture's schedule, which was chosen *so that* they
        have margin — see its docstring. Measured across 5 seeds there: R̂ ≤ 1.059 against 1.15,
        min ESS ≥ 59.7 against 20. What they now catch is a change that breaks mixing; what they
        no longer do is fail for a seed.
        """
        report = convergence_report(samples.coordinates)

        assert report.n_parameters == DIMENSION
        assert report.max_r_hat < 1.15
        assert report.min_ess > 20.0

    def test_no_chain_got_stuck(self, samples) -> None:
        for chain in samples.chains:
            assert chain.diagnostics.n_degenerate_steps == 0
            assert chain.diagnostics.start_shrink == 1.0
            assert chain.diagnostics.mean_chord_length > 0.1

    def test_the_incremental_update_drift_stays_negligible(self, samples) -> None:
        """The one place float64 perturbs the kernel (module docstring of `maxent_sampler`). Bounded
        by measurement rather than by an argument about error growth."""
        for chain in samples.chains:
            assert chain.diagnostics.max_refresh_drift < 1e-9


class TestThePrecomputeMatchesTheOracleOnRealData:
    def test_the_hot_path_chord_is_bit_for_bit_the_oracle_chord(
        self, transform, reduced, samples
    ) -> None:
        """The precompute is what M2 refused to accept from a caller, because a truncated support
        silently reintroduces the §1.6.1 bug. It is derived from ``T`` here — and held to the oracle
        on states the chain actually visited, not on synthetic ones."""
        precompute = transform.precompute
        visited = samples.fluxes.reshape(-1, reduced.n_free)[::211]

        for v in visited:
            v = np.ascontiguousarray(v)
            for k in range(transform.dimension):
                support = precompute.support[k]
                fast = chord_on_support(
                    v[support], precompute.direction[k], precompute.lower[k], precompute.upper[k]
                )
                oracle = feasible_chord(
                    v,
                    np.ascontiguousarray(transform.transform[:, k]),
                    reduced.lower_bounds,
                    reduced.upper_bounds,
                )
                assert fast.t_lo == oracle.t_lo
                assert fast.t_hi == oracle.t_hi


class TestTheStarts:
    def test_every_chain_starts_feasible_with_a_positive_chord_on_every_axis(
        self, transform, reduced
    ) -> None:
        """"Positive chords at start" — the gate, checked on the dispersed starts the chains
        actually use, rather than only at the centre."""
        rng = np.random.default_rng(0)

        for _ in range(8):
            start, shrink = dispersed_start(transform, reduced, rng)
            v = transform.to_flux(start)

            assert shrink == 1.0
            assert np.all(v >= reduced.lower_bounds - 1e-9)
            assert np.all(v <= reduced.upper_bounds + 1e-9)

            for k in range(transform.dimension):
                support = transform.precompute.support[k]
                chord = chord_on_support(
                    v[support],
                    transform.precompute.direction[k],
                    transform.precompute.lower[k],
                    transform.precompute.upper[k],
                )
                assert chord.is_samplable
                assert chord.contains(0.0)
