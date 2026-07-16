"""Reweighted-L1 (M7): the weight update, the loop, and the invariants that make it safe.

The gate (BUILD_PLAN M7) is four claims:
  * deterministic weights for a fixed input,
  * the active set converges within tolerance,
  * the weights are **frozen before sampling** and never updated from chain state,
  * the sampler still reproduces its analytic targets under the reweighted ``J``.

The first three live here. The fourth is a statistical test (`test_tilted_targets` already samples
against a `lower_objective`, and reweighting produces exactly such an objective — verified here to
be a well-formed one that binds).

Two of these tests exist to catch failures that produce *no error*: the median normalization
silently rescaling the target (it must not — λ absorbs it), and a too-tight clip silently turning
reweighting into plain L1 (it does — and the run must say so).
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.flux_polytope import FluxPolytope
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.reweighting import (
    ReweightingError,
    ReweightingNotConvergedError,
    _relative_weight_change,
    reweight_objective,
    update_weights,
)
from gsmm_compiler.sparse_objective import lower_objective, solve_sparse_objective

# --- a toy where reweighting has a lever ---------------------------------------------------------
#
# Two parallel routes from a fixed source SRC (pinned at 3) to a biomass sink BIO. The DIRECT route
# is one reaction; the SPLIT route is two. Both deliver the same flux, so plain L1 with unit weights
# already prefers DIRECT (cost 1 vs 2 per unit) — but a small flux can leak down SPLIT. Reweighting
# is what drives that leak to zero: once SPLIT carries a little, its weight climbs and it is shut.

REACTIONS = ("SRC", "DIRECT", "SPLIT1", "SPLIT2", "BIO")
METABOLITES = ("A", "B", "C")
#          SRC  DIRECT SPLIT1 SPLIT2  BIO
STOICH = [
    [1.0, -1.0, -1.0, 0.0, 0.0],  # A: SRC feeds DIRECT and SPLIT1
    [0.0, 1.0, 0.0, 1.0, -1.0],  # B: DIRECT and SPLIT2 feed BIO
    [0.0, 0.0, 1.0, -1.0, 0.0],  # C: SPLIT1 → SPLIT2
]


def _polytope() -> FluxPolytope:
    return FluxPolytope(
        reaction_ids=REACTIONS,
        metabolite_ids=METABOLITES,
        stoichiometry=NativeCSC.from_dense(np.asarray(STOICH, dtype=np.float64)),
        lower_bounds=np.asarray([3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        upper_bounds=np.asarray([3.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64),
        biomass_index=4,
    )


@pytest.fixture
def polytope() -> FluxPolytope:
    return _polytope()


@pytest.fixture
def config() -> ObjectiveConfig:
    # A forced source (SRC = 3) means the origin is infeasible, so λ̃ ≥ 1 is *allowed* — but we keep
    # a real interior pressure so the loop has something to do.
    return ObjectiveConfig(l1_penalty_scaled=0.5, reweighting_enabled=True)


# --- step 2–4: the weight update in isolation ----------------------------------------------------


class TestTheWeightUpdate:
    def test_the_formula_is_w_base_over_abs_v_plus_epsilon(self) -> None:
        base = np.array([1.0, 1.0, 1.0, 0.0])
        mask = np.array([True, True, True, False])
        v = np.array([0.0, 1.0, 3.0, 5.0])

        # ε huge and clip wide open, so we read the raw formula before normalization bites.
        updated, _, _ = update_weights(base, v, mask, epsilon=1.0, clip=(1e-30, 1e30))

        raw = np.array([1 / 1.0, 1 / 2.0, 1 / 4.0, 0.0])
        assert updated == pytest.approx(raw / np.median(raw[raw > 0]))

    def test_the_median_of_the_positive_weights_is_one(self) -> None:
        base = np.ones(5)
        mask = np.array([True, True, True, True, True])
        v = np.array([0.0, 0.5, 1.0, 2.0, 4.0])

        updated, _, _ = update_weights(base, v, mask, epsilon=1e-6, clip=(1e-6, 1e6))

        assert float(np.median(updated[updated > 0])) == pytest.approx(1.0)

    def test_weights_are_exactly_zero_off_the_penalty_set(self) -> None:
        base = np.array([1.0, 1.0, 0.0])
        mask = np.array([True, True, False])
        v = np.array([2.0, 3.0, 9.0])

        updated, _, _ = update_weights(base, v, mask, epsilon=1e-6, clip=(1e-6, 1e6))

        assert updated[2] == 0.0

    def test_the_clip_bounds_the_raw_update_and_the_counts_report_it(self) -> None:
        base = np.ones(3)
        mask = np.array([True, True, True])
        # |v| = 0 → raw 1/ε = 1e6, way over the ceiling; |v| = 1000 → raw ~1e-3, under the floor.
        v = np.array([0.0, 1.0, 1000.0])

        _, at_low, at_high = update_weights(base, v, mask, epsilon=1e-6, clip=(1e-2, 1e2))

        assert at_high == 1  # the |v| = 0 reaction
        assert at_low == 1  # the |v| = 1000 reaction

    def test_a_reaction_at_zero_flux_gets_the_largest_weight(self) -> None:
        """The whole mechanism: an off reaction is the most expensive to switch back on."""
        base = np.ones(3)
        mask = np.array([True, True, True])
        v = np.array([0.0, 1.0, 2.0])

        updated, _, _ = update_weights(base, v, mask, epsilon=1e-6, clip=(1e-6, 1e6))

        assert updated[0] == updated.max()

    def test_snapping_is_never_applied_the_exact_flux_is_used(self) -> None:
        """CLAUDE.md: never snap a small flux to zero. A 1e-12 flux and a 0 flux must give
        *different* weights, or the update has quietly thresholded."""
        base = np.ones(2)
        mask = np.array([True, True])

        wide = {"epsilon": 1e-16, "clip": (1e-30, 1e30)}
        at_zero, _, _ = update_weights(base, np.array([0.0, 1.0]), mask, **wide)
        at_tiny, _, _ = update_weights(base, np.array([1e-12, 1.0]), mask, **wide)

        assert at_zero[0] != at_tiny[0]

    def test_an_all_zero_weight_vector_is_refused(self) -> None:
        with pytest.raises(ReweightingError, match="nothing to reweight"):
            update_weights(
                np.zeros(3), np.ones(3), np.zeros(3, dtype=bool), epsilon=1e-6, clip=(1e-6, 1e6)
            )


class TestTheConvergenceMetricSeesSmallCoordinates:
    """The blindness Codex found: a *flux*-based stop test, normalized by the whole vector, misses
    the small sparsity-critical coordinate whose weight is still moving. The *weight*-based metric
    is per-reaction relative, so it does not. These tests fail on the old flux criterion."""

    def test_a_large_flux_hides_a_small_coordinate_from_the_flux_metric(self) -> None:
        # The exact counterexample: one flux at 1e3, a sparsity-critical flux moving 1e-3 → 2e-3.
        prev = np.array([1e3, 1.0, 1e-3])
        v = np.array([1e3, 1.0, 2e-3])

        denom = max(1.0, float(np.max(np.abs(prev))))
        flux_metric = float(np.max(np.abs(v - prev))) / denom
        assert flux_metric < 1e-5  # the flux metric says "converged" — and is wrong

        # But the weight the map produces on that coordinate has HALVED.
        base = np.ones(3)
        mask = np.array([True, True, True])
        w_prev, _, _ = update_weights(base, prev, mask, epsilon=1e-6, clip=(1e-6, 1e6))
        w_v, _, _ = update_weights(base, v, mask, epsilon=1e-6, clip=(1e-6, 1e6))
        assert _relative_weight_change(w_v, w_prev, mask) > 0.3  # the weight metric sees it

    def test_the_weight_metric_is_per_reaction_relative(self) -> None:
        mask = np.array([True, True])
        # Tiny absolute change on a small weight is a LARGE relative change — and must register.
        old = np.array([1000.0, 0.001])
        new = np.array([1000.0, 0.002])
        assert _relative_weight_change(new, old, mask) == pytest.approx(0.5)

    def test_identical_weights_are_a_fixed_point(self) -> None:
        mask = np.array([True, True, True])
        w = np.array([1.0, 0.01, 5.0])
        assert _relative_weight_change(w, w, mask) == 0.0


# --- the loop ------------------------------------------------------------------------------------


class TestTheLoop:
    def test_it_converges_and_freezes(self, polytope, config) -> None:
        report = reweight_objective(polytope, config=config)

        assert report.converged
        assert report.n_iterations >= 1
        # The frozen weights are physically read-only — "frozen" is an invariant, not a convention.
        with pytest.raises(ValueError, match="read-only|write"):
            report.objective.weights[0] = 999.0

    def test_it_is_deterministic(self, polytope, config) -> None:
        """Gate: deterministic weights for a fixed solver and seed (no seed here; it is pure)."""
        a = reweight_objective(polytope, config=config)
        b = reweight_objective(_polytope(), config=config)

        assert np.array_equal(a.objective.weights, b.objective.weights)
        assert a.objective.l1_penalty == b.objective.l1_penalty
        assert a.objective.content_key() == b.objective.content_key()

    def test_the_active_set_converges(self, polytope, config) -> None:
        """Gate: the active set settles. By the last step its symmetric difference is empty."""
        report = reweight_objective(polytope, config=config)

        assert report.history[-1].active_set_changes == 0

    def test_convergence_is_on_the_weights_not_the_fluxes(self, polytope, config) -> None:
        """The frozen artifact is the *weights*, so convergence must mean the weights are a fixed
        point of ``w ↦ F(solve(w))`` — a flux-based test is blind on small sparsity-critical
        coordinates (Codex, M7 review). The stop is keyed on `max_weight_change`, not
        `max_flux_change`."""
        report = reweight_objective(polytope, config=config)

        assert report.history[-1].max_weight_change <= config.reweighting_solution_tol

    def test_the_final_solution_belongs_to_the_frozen_objective(self, polytope, config) -> None:
        """The join `choose_energy_scale` refuses to make on trust: the L2 optimum must be the one
        solved against the frozen weights, not a rebuild that might differ in its last ulp."""
        report = reweight_objective(polytope, config=config)

        assert report.solution.optimum.objective_key == report.objective.content_key()

    def test_the_frozen_objective_is_a_fixed_point(self, polytope, config) -> None:
        """Reweighting the frozen objective's own optimum reproduces its own weights — that is what
        being a fixed point *means*, and it is the property that makes 'frozen' meaningful."""
        report = reweight_objective(polytope, config=config)

        v = report.solution.optimum.v_full
        again, _, _ = update_weights(
            report.history[0].weights,  # w^base
            v,
            report.objective.penalty_mask,
            epsilon=config.reweighting_epsilon,
            clip=(config.weight_clip_min, config.weight_clip_max),
        )
        assert again == pytest.approx(np.asarray(report.objective.weights))

    def test_the_frozen_objective_lowers_and_binds(self, polytope, config) -> None:
        """Gate (the fourth claim's precondition): the reweighted objective is a well-formed input
        to the sampler — it lowers onto the polytope and the lowered form binds to it."""
        reduced = polytope.reduce()
        report = reweight_objective(polytope, reduced, config)

        lowered = lower_objective(reduced, report.objective)
        assert lowered.binds_to(reduced)
        assert lowered.objective_key == report.objective.content_key()

    def test_an_unconverged_loop_raises_rather_than_shipping_an_iterate(self, polytope) -> None:
        """A weight vector that is not a fixed point is a target chosen by the iteration budget.
        The loop refuses to freeze one unless the caller explicitly accepts it."""
        stingy = ObjectiveConfig(
            l1_penalty_scaled=0.5,
            reweighting_enabled=True,
            reweighting_max_iterations=1,  # cannot possibly measure convergence (needs a prior)
        )
        with pytest.raises(ReweightingNotConvergedError, match="did not converge"):
            reweight_objective(polytope, config=stingy)

        accepted = reweight_objective(polytope, config=stingy, allow_unconverged=True)
        assert not accepted.converged


# --- the two silent-failure guards ---------------------------------------------------------------


class TestNormalizationCannotMoveTheTarget:
    """λ = λ̃·λ*(w) makes any rescaling of w a no-op: w → c·w sends λ* → λ*/c, so λw is invariant.

    So the median normalization — which multiplies every weight by an arbitrary 1/median —
    cannot move the sampled distribution. This is what makes step 4 a *conditioning* step and not a
    *modelling* one, and it is the reason re-resolving λ (not freezing it) is the right policy:
    under a frozen λ the normalization would rescale the pressure by that median every iteration.

    Tested at the objective/LP layer directly, **not** through the loop — the clip step is applied
    to the raw (unnormalized) update and is genuinely *not* scale-invariant, so pushing a scale
    through the base weights would confound the two effects. The claim is specifically about the
    normalization: scaling the weights the LP is *solved with*, then re-resolving λ, is a no-op.
    """

    def test_scaling_the_weights_and_re_resolving_lambda_leaves_the_optimum_fixed(
        self, polytope
    ) -> None:
        from gsmm_compiler.sparse_objective import (
            SparseFluxObjective,
            resolve_objective,
        )

        reduced = polytope.reduce()
        config = ObjectiveConfig(l1_penalty_scaled=0.5)
        seed = SparseFluxObjective.from_polytope(polytope, l1_penalty=0.0)

        # A generic, non-uniform weight vector — the kind an iteration actually produces.
        w = np.where(seed.penalty_mask, np.array([2.0, 0.5, 3.0, 0.25, 0.0]), 0.0)

        a = resolve_objective(polytope, reduced, config, weights=w)
        b = resolve_objective(polytope, reduced, config, weights=7.31 * w)

        # λ and w each move; λ* moves the other way; λ·w is invariant to the last ulp the LP allows.
        assert a.scale.l1_penalty != pytest.approx(b.scale.l1_penalty)
        cost_a = a.objective.l1_penalty * np.asarray(a.objective.weights)
        cost_b = b.objective.l1_penalty * np.asarray(b.objective.weights)
        assert cost_a == pytest.approx(cost_b, rel=1e-9, abs=1e-12)

        # And so the optimum is the same point.
        opt_a = solve_sparse_objective(reduced, a.objective).optimum.v_full
        opt_b = solve_sparse_objective(reduced, b.objective).optimum.v_full
        assert opt_a == pytest.approx(opt_b, rel=1e-6, abs=1e-9)
