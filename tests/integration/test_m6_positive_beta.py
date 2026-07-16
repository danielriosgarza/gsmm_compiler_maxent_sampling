"""M6 acceptance gate — the positive-β maximum-entropy sampler on the genome-scale model.

The gate (BUILD_PLAN M6): truncated-exponential + truncated-Laplace analytic targets · **mean ``J``
nondecreasing in β** within Monte-Carlo uncertainty · large-β stress · 1-D quadrature cross-check in
the reduced coordinate.

Three of those four are settled in `tests/statistical/test_tilted_targets.py`, on polytopes whose
tilted law is known on paper — because a genome-scale model has no known law to check against, and a
test that cannot say what the right answer is cannot say the answer is wrong.

What can only be checked **here** is that the same machinery survives 773 reactions, 894 metabolites
and a 46-dimensional hull, on a *real* objective: a λ resolved against the model's own cliff
(BUILD_PLAN §1.7), an ``s_J`` set from the observed range of ``J`` over the support points, and a
piecewise ``J`` with ~199 candidate breakpoints per chord instead of the toys' one. Specifically:

* **mean ``J`` rises with β**, which is a theorem (``dE_β[J]/dβ = Var_β(J)/s_J ≥ 0``) and so a test
  of the *implementation*, not of the model;
* every tilted sample is still feasible in the **full** 773-reaction polytope — the one check a
  reduced-space sampler that is internally consistent and wrong about the lift cannot pass;
* **zero HiGHS solves** once sampling starts, still, with the objective now live in the inner loop.
  That is the invariant most at risk in M6: `build_piecewise_j` runs on every step, and if anything
  in it reached for a solver the counter would say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import ObjectiveConfig, SamplerConfig
from gsmm_compiler.diagnostics import feasibility_report
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.line_distribution import build_piecewise_j
from gsmm_compiler.line_geometry import feasible_chord
from gsmm_compiler.maxent_sampler import (
    SUGGESTED_BETA_LADDER,
    movable_reactions,
    run_ladder,
)
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.rounding import build_transform
from gsmm_compiler.sparse_objective import (
    choose_energy_scale,
    lower_objective,
    resolve_objective,
    solve_sparse_objective,
)

pytestmark = pytest.mark.slow

DIMENSION = 46
"""Settled by M4, against an independent FVA+rank oracle."""

LADDER = (0.0, 4.0, 16.0)
"""Three rungs reaching the **top** of spec §22.1's ladder, and it has to reach that far.

``s_J`` on this model comes out at 31.3, while the spread of ``J`` the β = 0 chain actually explores
is ``sd(J) = 2.6`` — a factor of **12**. The linear response ``dE_β[J]/dβ = Var_β(J)/s_J`` is
therefore only 0.22 per unit β, so ``β = 1`` moves the mean by less than one Monte-Carlo standard
error and no test at any reasonable chain length could tell it from zero. At ``β = 16`` the rise is
2.8, which clears the noise by ~6σ.

That is a fact about the *calibration*, not about the sampler — see `TestTheBetaLadderIsAWeakDial`,
which measures it rather than working around it."""

SCHEDULE = SamplerConfig(
    betas=LADDER, n_chains=4, n_samples=1200, burn_in=1200, thin=1, refresh_interval=250
)
"""Deliberately *not* the `SamplerConfig` defaults, and DEVELOPMENT_STATUS says why: this model has
a `step_scale_ratio` of 0.008 and mixes slowly, so the defaults do not reach R̂ < 1.01. The claims
below are chosen to be ones this schedule can actually support — feasibility, the solve counter, and
a monotonicity check whose error bars come from the measured ESS of ``J`` rather than from the draw
count. Nothing here asserts convergence, because at this length nothing here has it."""


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
def objective(example_canonical: CanonicalModel, reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    """The real objective: λ resolved against *this model's* cliff (BUILD_PLAN §1.7), not a raw λ.

    λ̃ = 0.5 → λ ≈ 9.4e-4 here. A raw λ of 1.0 would sit 529× past the cliff, where ``v* = 0`` and
    the ladder would tilt toward a cell that does nothing — the M3 finding, and the reason this
    fixture goes through `resolve_objective` rather than setting a number.
    """
    return resolve_objective(
        example_canonical.polytope, reduced, ObjectiveConfig(l1_penalty_scaled=0.5)
    )


@pytest.fixture(scope="module")
def ladder(transform, reduced, geometry, objective):  # type: ignore[no-untyped-def]
    lowered = lower_objective(reduced, objective.objective)
    solution = solve_sparse_objective(reduced, objective.objective)

    scale = choose_energy_scale(
        lowered,
        geometry.support_points,
        optimum=solution.optimum,
        warmup_polytope_key=geometry.polytope_key,
        mode="warmup_range",
    )
    optimum = transform.to_coordinates(reduced.to_reduced(solution.optimum.v_full))

    solves_before = total_solve_count()
    result = run_ladder(
        transform,
        reduced,
        config=SCHEDULE,
        model_id="bifido",
        objective=lowered,
        energy_scale=scale,
        optimum_coordinates=optimum,
    )
    return result, solution, solves_before


class TestTheObjectiveIsSaneBeforeAnythingIsSampled:
    def test_lambda_is_below_the_cliff_and_the_cell_still_grows(self, objective) -> None:
        """The precondition for the whole ladder. Past ``λ*`` the LP optimum is the origin, ``J*``
        is 0, and ``s_J`` would be read off a distribution concentrated on no metabolism at all —
        and *nothing in the LP can tell*: optimal status, zero residual, ``z = |v|`` exactly."""
        scale = objective.scale

        assert scale.l1_penalty_scaled == 0.5
        assert scale.l1_penalty < scale.critical_l1_penalty
        assert scale.origin_is_feasible  # this model has no forced-flux reaction

    def test_the_optimum_retains_real_growth(self, ladder) -> None:
        _, solution, _ = ladder

        assert not solution.is_sparsity_dominated
        assert solution.optimum.value.mu > 0.0
        assert 0.3 < solution.biomass_retention < 1.0

    def test_the_energy_scale_comes_from_the_observed_objective_range(self, ladder) -> None:
        """``s_J = J* − Q_{0.05}(J(W))`` over the support points (spec §22.2), so ``β`` measures
        selection against *this model's* objective spread rather than its raw magnitude."""
        result, solution, _ = ladder
        scale = result.energy_scale

        assert scale.mode == "warmup_range"
        assert not scale.fell_back
        assert scale.value > 0.0
        assert scale.j_star == pytest.approx(solution.optimum.j_star)
        assert scale.warmup_quantile_j < scale.j_star


class TestTheGateMeanJIsNondecreasingInBeta:
    """``dE_β[J]/dβ = Var_β(J)/s_J ≥ 0`` — so this is a test of the implementation, not the model.

    A sampler that inverted the sign of the tilt, weighted its segments by the wrong masses, or
    quietly flattened ``κ`` to zero would produce a curve that is flat or falls. Every one of those
    bugs leaves feasibility, mass balance, the chords and R̂ completely intact.
    """

    def test_mean_j_rises_along_the_ladder(self, ladder) -> None:
        result, _, _ = ladder
        report = result.monotonicity()

        assert report.betas == LADDER
        assert report.is_monotone, (
            f"mean J fell by {report.worst_drop_sigma:.1f}σ along {report.betas}: "
            f"means {report.mean_j}, standard errors {report.standard_error_j}"
        )

    def test_the_rise_is_real_and_not_a_rounding_error(self, ladder) -> None:
        """Monotone-within-error is satisfied by a *flat* curve too, and a flat curve is exactly
        what
        a silently-untilted sampler produces. So the ladder must also **separate**: the top rung's
        mean ``J`` has to stand clear of the bottom's by several standard errors."""
        result, _, _ = ladder
        report = result.monotonicity()

        pooled = float(np.hypot(report.standard_error_j[0], report.standard_error_j[-1]))
        rise = report.mean_j[-1] - report.mean_j[0]

        assert rise > 4.0 * pooled, f"the tilt moved E[J] by only {rise / pooled:.1f}σ"

    def test_the_chains_agree_about_E_J_so_the_rise_is_not_a_burn_in_artifact(
        self, ladder
    ) -> None:
        """**Codex, M6 review — the objection the σ-based check cannot answer by itself.**

        An ESS-based error bar assumes the draws come from the stationary law, and this schedule is
        explicitly run below convergence. So there is a rival explanation for a rising curve, with
        nothing to do with the tilt: the chains start dispersed over the support hull with ``v*``
        mixed in, and a high-β chain that merely **retained** its high-``J`` start — never having
        mixed at all — produces exactly the same picture as a chain that was pulled there by ``β``.
        No amount of ESS can tell those apart, because ESS says nothing about burn-in bias.

        R̂ can. Chains launched far apart which nonetheless *agree* about ``E[J]`` are not each
        sitting in their own initial neighbourhood. That is what this asserts, on every rung — the
        check that gives `is_monotone` its meaning rather than merely its value.
        """
        result, _, _ = ladder
        report = result.monotonicity()

        assert report.max_r_hat_j < 1.2, (
            f"R̂(J) = {report.r_hat_j} — the chains do not agree about E[J], so a rising curve may "
            "be retained initialization rather than the tilt. Lengthen the schedule."
        )

    def test_the_size_of_the_rise_matches_the_linear_response_identity(self, ladder) -> None:
        """The strongest claim available here, and the one a *sign* test cannot make.

        ``dE_β[J]/dβ = Var_β(J)/s_J`` exactly, so the rise across the ladder is
        ``∫₀^{β_max} Var_β(J)/s_J dβ`` — which, to the extent ``Var`` does not move much, is about
        ``β_max · Var₀(J)/s_J``. Measured here: ``Var₀ = 6.7``, ``s_J = 31.3``, ``β_max = 16``, so
        the
        prediction is **3.4** and the observation is **2.8** — the shortfall being ``Var`` shrinking
        as the target concentrates, exactly as it should.

        This pins the **magnitude** of ``κ = β/s_J``, which a monotonicity test cannot. Forget the
        ``s_J`` division and ``κ`` is 31× too large: the chain would slam up against ``J* = 9.5``
        instead of creeping to −9.2, and the rise would overshoot this band by an order of
        magnitude.
        Halve ``κ`` and it undershoots. Both are silent everywhere else in the suite.
        """
        result, _, _ = ladder
        cold = result.rungs[0]

        variance = float(cold.j.reshape(-1).var(ddof=1))
        predicted = max(LADDER) * variance / result.energy_scale.value
        observed = result.rungs[-1].mean_j - cold.mean_j

        assert 0.3 * predicted < observed < 2.0 * predicted, (
            f"E[J] rose by {observed:.3f}; the linear response Var₀(J)/s_J = "
            f"{variance:.3f}/{result.energy_scale.value:.3f} predicts ≈ {predicted:.3f}"
        )

    def test_the_tilt_also_raises_biomass_and_lowers_the_l1_cost(self, ladder) -> None:
        """What ``J`` rising *means*, in the two quantities a biologist would actually read.

        ``J = μ − λC``, so a higher ``J`` could in principle come from either term. Reporting them
        separately (spec §3.2) is what makes the ladder legible: here selection buys growth *and*
        pays for it with sparsity, which is the trade-off the method exists to trace.
        """
        result, _, _ = ladder
        rungs = result.rungs

        mu = [float(np.mean([trace.mu.mean() for trace in rung.traces])) for rung in rungs]
        cost = [float(np.mean([trace.cost.mean() for trace in rung.traces])) for rung in rungs]

        assert mu[-1] > mu[0], f"biomass did not rise along the ladder: {mu}"
        assert cost[-1] < cost[0], f"the L1 cost did not fall along the ladder: {cost}"


class TestEveryTiltedSampleIsAFeasibleFlux:
    """The check a reduced-space sampler that is wrong about the lift cannot pass."""

    def test_every_sample_of_every_rung_is_in_the_reduced_polytope(
        self, ladder, reduced: ReducedPolytope
    ) -> None:
        result, _, _ = ladder

        for rung in result.rungs:
            report = feasibility_report(rung.result.fluxes, reduced)

            assert report.is_feasible, f"β = {rung.beta}: {report.as_dict()}"
            assert report.n_bound_violations == 0

    def test_every_sample_lifts_to_a_feasible_full_flux_vector(
        self, ladder, reduced: ReducedPolytope, example_canonical: CanonicalModel
    ) -> None:
        """773 reactions, including the 513 the reduced state never carried. Mass balance is
        recomputed against the **full** ``S``, which the sampler has never once seen."""
        result, _, _ = ladder
        polytope = example_canonical.polytope

        for rung in result.rungs:
            fluxes = rung.result.fluxes.reshape(-1, reduced.n_free)
            sampled = fluxes[:: max(1, fluxes.shape[0] // 200)]  # 200 per rung is plenty
            full = reduced.to_full(sampled)

            assert full.shape[1] == polytope.n_reactions
            for v in full:
                assert polytope.contains(np.ascontiguousarray(v), tol=1e-7)

    def test_no_flux_was_ever_snapped_to_zero(
        self, ladder, transform, reduced: ReducedPolytope
    ) -> None:
        """Spec §3.7 / CLAUDE.md. At finite β the law is *continuous*: the L1 term promotes small
        fluxes, it does not produce exact zeros. A **movable** reaction that comes out exactly 0.0
        would mean the sampler had rounded — which moves the stationary distribution and can break
        mass balance.

        The movable set is taken from ``T``'s structural support, not from a magnitude heuristic:
        the
        61 FVA-blocked reactions genuinely *are* ~0 in every sample, and asking whether they were
        "snapped" is asking the wrong question about them.
        """
        result, _, _ = ladder
        hot = result.rung_at(LADDER[-1]).result.fluxes.reshape(-1, reduced.n_free)
        movable = movable_reactions(transform)

        exactly_zero = np.count_nonzero(hot[:, movable] == 0.0)
        assert exactly_zero == 0, f"{exactly_zero} fluxes are exactly 0.0 — something snapped"


class TestTheGateNoSolverInTheInnerLoop:
    """BUILD_PLAN §1.3, and the invariant M6 puts most at risk.

    At β = 0 the objective never enters the loop at all. At β > 0 `build_piecewise_j` runs on
    **every
    coordinate update** — 46 × 2400 sweeps × 4 chains × 3 rungs of them — and it is the first code
    in
    the hot path that knows what ``J`` is. If anything in the objective layer reached for HiGHS,
    this
    is where it would show.
    """

    def test_the_whole_ladder_performs_zero_highs_solves(self, ladder) -> None:
        result, _, solves_before = ladder

        assert total_solve_count() == solves_before
        assert len(result.rungs) == len(LADDER)


class TestThePiecewiseObjectiveOnRealChords:
    """`build_piecewise_j` against `L1Objective.evaluate_on_line` — on the genome-scale model.

    M2 held these two against each other on synthetic lines. Here the chord is a real rounded axis
    through a real 46-dimensional polytope, with ~199 penalized reactions whose breakpoints are
    spread across it — the case where a misplaced or wrongly-merged cut has somewhere to hide.
    """

    def test_the_reconstruction_matches_the_definition_on_every_rounded_axis(
        self, transform, reduced: ReducedPolytope, objective
    ) -> None:
        lowered = lower_objective(reduced, objective.objective)
        centre = transform.center

        for k in range(transform.dimension):
            direction = np.ascontiguousarray(transform.transform[:, k])
            chord = feasible_chord(
                centre, direction, reduced.lower_bounds, reduced.upper_bounds
            )
            piecewise = build_piecewise_j(centre, direction, chord, lowered.line)
            grid = np.linspace(chord.t_lo, chord.t_hi, 257)

            direct = lowered.line.evaluate_on_line(centre, direction, grid)

            assert np.allclose(piecewise.evaluate(grid), direct, rtol=1e-9, atol=1e-9)
            assert np.all(np.diff(piecewise.slopes) <= 1e-12), "J must stay concave"

    def test_the_chords_carry_real_breakpoints(self, transform, reduced, objective) -> None:
        """Otherwise the test above checks a straight line 46 times and proves nothing about the
        piecewise machinery."""
        lowered = lower_objective(reduced, objective.objective)
        centre = transform.center

        segments = [
            build_piecewise_j(
                centre,
                np.ascontiguousarray(transform.transform[:, k]),
                feasible_chord(
                    centre,
                    np.ascontiguousarray(transform.transform[:, k]),
                    reduced.lower_bounds,
                    reduced.upper_bounds,
                ),
                lowered.line,
            ).n_segments
            for k in range(transform.dimension)
        ]

        assert max(segments) > 1
        assert float(np.mean(segments)) > 1.5


class TestTheTracesAreReportable:
    def test_every_rung_reports_mu_cost_j_log_energy_and_near_zero_counts(self, ladder) -> None:
        """Spec §24.2, in full."""
        result, _, _ = ladder

        for rung in result.rungs:
            for trace in rung.traces:
                assert trace.mu.shape == (SCHEDULE.n_samples,)
                assert trace.cost.shape == (SCHEDULE.n_samples,)
                assert trace.j.shape == (SCHEDULE.n_samples,)
                assert np.all(np.isfinite(trace.normalized_log_energy))
                assert trace.near_zero_counts.shape == (
                    SCHEDULE.n_samples,
                    len(trace.near_zero_thresholds),
                )
                # More reactions are near zero at a loose threshold than at a tight one.
                counts = trace.near_zero_counts.mean(axis=0)
                assert np.all(np.diff(counts) >= 0.0)

    def test_the_near_zero_count_excludes_the_reactions_that_could_never_move(
        self, ladder, transform, reduced: ReducedPolytope
    ) -> None:
        """The count is over the **199 movable** reactions, not the 260 free ones — and if it were
        not, it would be a constant.

        61 of this model's free reactions are FVA-blocked (BUILD_PLAN §1.4.1): mass balance pins
        them
        at zero whatever the file's bounds say, so their flux is the centre's residual noise
        (~1e-13)
        in every sample of every chain. Counted among the near-zero reactions they supply **61 at
        every threshold and every β** — and, measured, they supply *all of it*: the 199 reactions
        that
        can move contribute exactly 0 below 1e-3.

        So a free-set count reads 61.0 at β = 0 and 61.0 at β = 16. It looks like a sparsity signal,
        it is pure geometry, and it would have gone straight into a cross-model activity table.
        """
        result, _, _ = ladder
        movable = movable_reactions(transform)
        blocked = np.setdiff1d(np.arange(reduced.n_free), movable)

        assert movable.size == 199
        assert blocked.size == 61

        for rung in result.rungs:
            for trace in rung.traces:
                assert trace.n_movable == 199
                assert trace.n_free == 260
                # The tight thresholds see nothing among the movable reactions…
                assert trace.near_zero_counts[:, 0].max() == 0
                # …which is why the blocked ones would have dominated: all 61 sit below it.
                fluxes = rung.result.chains[0].fluxes
                assert np.abs(fluxes[:, blocked]).max() < 1e-9

    def test_both_counts_are_reported_and_they_reconcile(
        self, ladder, transform, reduced: ReducedPolytope
    ) -> None:
        """**Codex, M6 review.** Excluding the structural zeros from the *selection* statistic is
        right; deleting them from the record is not.

        That 61 reactions cannot carry flux under this medium is a real biological fact about the
        model, and a run should be able to state it. What it must not do is let that constant 61
        masquerade as a *response to β*. So both counts are kept — over the movable reactions, and
        over every free reaction — with `n_blocked` to reconcile them.

        The gap between them is a **per-threshold constant** — the immovable reactions never move,
        so they contribute the same count to every sample. It is *not* ``n_blocked`` in general:
        immovable means "pinned at its own value", and that value need not be zero (see
        `tests/unit/test_maxent_sampler.py::TestImmovableIsNotTheSameAsZero`). On *this* model all
        61 happen to sit at ~1e-13, so here the gap **is** 61 at every threshold — asserted as a
        measurement of this model, next to the measurement that licenses it, rather than as an
        identity. (An earlier version asserted it as an identity. Codex, M6 review round 2.)
        """
        result, _, _ = ladder

        for rung in result.rungs:
            for trace in rung.traces:
                assert trace.n_blocked == 61
                assert trace.n_movable + trace.n_blocked == trace.n_free

                gap = trace.near_zero_counts_all_free - trace.near_zero_counts

                # The gap is the same in every sample: immovable reactions do not move.
                assert np.all(gap == gap[0]), "immovable reactions contribute a constant"

                # …and on this model it is all 61, *because* they all sit below every threshold —
                # which is the premise, measured here rather than assumed.
                blocked_flux = np.abs(rung.result.chains[0].fluxes).max(axis=0)
                movable = movable_reactions(transform)
                blocked = np.setdiff1d(np.arange(reduced.n_free), movable)
                assert blocked_flux[blocked].max() < min(trace.near_zero_thresholds)
                assert np.all(gap == 61)

    def test_the_conventional_thresholds_cannot_see_this_models_flux_scale(
        self, ladder, transform, reduced: ReducedPolytope
    ) -> None:
        """Why five thresholds and not the usual three (spec §3.7: the threshold is *declared*).

        The median ``|v_r|`` over the movable reactions is ~53 and the maximum is 1000. A threshold
        of
        1e-6 — what an FBA paper reaches for — sits eight orders of magnitude below that and reports
        zero, at every β. That is a true statement and a useless one, and reading it as "this cell
        is
        dense" would be a mistake about units, not about biology.
        """
        result, _, _ = ladder
        movable = movable_reactions(transform)
        fluxes = result.rungs[0].result.fluxes.reshape(-1, reduced.n_free)[:, movable]

        assert np.median(np.abs(fluxes)) > 1.0, "if the fluxes were O(1e-3) this test is obsolete"

        counts = result.rungs[0].traces[0].near_zero_counts.mean(axis=0)
        thresholds = result.rungs[0].traces[0].near_zero_thresholds

        # Not *bit-exactly* zero — the law is continuous, so it can put a hair of mass anywhere, and
        # a run does occasionally catch one movable reaction under 1e-3. The claim is that the tight
        # thresholds see nothing *worth reporting*: 0.002 reactions out of 199, against 7 at 1.0.
        assert counts[thresholds.index(1e-3)] < 0.1
        assert counts[-1] > 1.0, "the widest declared threshold must at least see something"

    def test_the_l1_cost_is_where_the_sparsity_signal_actually_shows(self, ladder) -> None:
        """``C(v) = Σ w_r|v_r|`` is what the L1 term penalizes, and it is the quantity that moves.

        The near-zero *count* does not budge on this model, and that is an honest result rather than
        a
        missing feature: at ``s_J = 31`` the tilt is weak (`TestTheBetaLadderIsAWeakDial`), and a
        continuous law does not pile mass onto a measure-zero hyperplane in any case (spec §3.7).
        What
        selection actually does here is shrink the *magnitude* of the flux vector — measurably, and
        by
        1.9e3 units across the ladder.
        """
        result, _, _ = ladder
        cost = [
            float(np.mean([trace.cost.mean() for trace in rung.traces])) for rung in result.rungs
        ]

        assert cost[-1] < cost[0], f"the L1 cost did not fall along the ladder: {cost}"
        assert cost[0] - cost[-1] > 500.0, f"…and it must fall by a real amount: {cost}"

    def test_the_log_energy_is_at_most_a_solver_tolerance_above_zero(self, ladder) -> None:
        """``J*`` is an LP optimum computed to a tolerance, **not** a strict numeric upper bound on
        ``J`` (BUILD_PLAN §1.6 delta 4) — so ``(J − J*)/s_J`` may be a hair positive, and nothing
        may
        assume otherwise. It may not be *substantially* positive: that would mean the chain had
        found
        flux the LP said was unreachable."""
        result, _, _ = ladder

        for rung in result.rungs:
            for trace in rung.traces:
                assert trace.normalized_log_energy.max() < 1e-6


class TestTheBetaLadderIsAWeakDialOnThisModel:
    """The M6 finding, measured rather than worked around — and an argument M10 can act on.

    ``s_J = J* − Q_{0.05}(J(W))`` is spec §22.2's formula, and the ``W`` available at this stage is
    M4's **support points**: the vertices the geometry's support LPs returned. Those are *extreme*
    points, where the L1 cost is enormous — ``J`` runs down to −28 there. The chain, meanwhile,
    lives
    in the interior, where ``J ≈ −12`` with a standard deviation of **2.6**.

    So ``s_J`` is calibrated to a range **12× wider** than the one the sampler explores, and since
    ``dE_β[J]/dβ = Var_β(J)/s_J``, every β on the spec's ladder is divided by that same factor of
    12.
    β = 16 — the *top* rung — moves ``E[J]`` by 2.8 out of the 21.5 that separates it from ``J*``.
    The ladder works; it is a fine-tuning knob rather than a switch.

    Spec §22.2 already names the remedy in passing: define ``s_J`` "from the objective values of
    support **or pilot** points". A β = 0 pilot's own ``J`` spread is 2.6, so the identical ladder
    would tilt ~12× harder. That is **M10** (pilot-based ``s_J``, already deferred there alongside
    pilot rounding), and it is a change to the *calibration of β*, not to the target at any given
    ``β`` — the distribution this milestone validates is untouched either way.

    These assertions are loose on purpose. They exist to make the finding **fail loudly if it stops
    being true** — if a future ``s_J`` lands near the chain's own spread, the ladder becomes a
    switch
    and the schedule in `SCHEDULE` needs revisiting.
    """

    def test_s_j_is_far_wider_than_the_spread_of_j_the_chain_explores(self, ladder) -> None:
        result, _, _ = ladder
        spread = float(result.rungs[0].j.reshape(-1).std(ddof=1))

        ratio = result.energy_scale.value / spread
        assert ratio > 5.0, (
            f"s_J = {result.energy_scale.value:.1f} is only {ratio:.1f}× the chain's own sd(J) = "
            f"{spread:.2f}. The ladder is no longer a weak dial — revisit SCHEDULE and M10."
        )

    def test_even_the_top_rung_stays_far_below_the_lp_optimum(self, ladder) -> None:
        """The concrete consequence. If this ever fails, β has become a strong dial and the ladder
        should be re-centred — a good problem, but one that changes what a run means."""
        result, solution, _ = ladder
        hot = result.rungs[-1]

        assert hot.mean_j < solution.optimum.j_star
        travelled = (hot.mean_j - result.rungs[0].mean_j) / (
            solution.optimum.j_star - result.rungs[0].mean_j
        )
        assert travelled < 0.5, f"β = {max(LADDER)} closed {travelled:.0%} of the gap to J*"


class TestTheSuggestedLadderIsAvailableButNotImposed:
    def test_the_spec_ladder_is_offered_verbatim(self) -> None:
        assert SUGGESTED_BETA_LADDER == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

    def test_it_is_not_the_default(self) -> None:
        """Spec §22.1: "no universal ladder should be hard-coded as scientifically correct". The β
        at
        which a strain's flux distribution concentrates is a property of *its* polytope."""
        assert SamplerConfig().betas == (0.0,)
