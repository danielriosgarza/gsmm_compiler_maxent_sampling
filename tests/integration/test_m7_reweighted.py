"""M7 acceptance gate — reweighted-L1 with frozen weights, on the genome-scale model.

The gate (BUILD_PLAN M7): deterministic weights for a fixed seed · the active set converges · the
weights are **frozen before MCMC** and never updated from chain state · the sampler still reproduces
analytic targets under the reweighted ``J``.

The analytic-target claim is a statistical test (`test_tilted_targets`, which already samples
against a `lower_objective` — and a reweighted objective *is* one). What can only be checked
**here**, on a real model with a real cliff, is the rest:

* reweighting actually **sheds** reactions plain L1 leaves on (measured: 134 → 131), and a too-tight
  clip **silently** stops shedding — the run must report that, because nothing else can;
* the frozen objective flows through the entire sampler with **zero HiGHS solves after the freeze**,
  the invariant the whole milestone is built to protect: a weight that moved mid-chain would have to
  be recomputed, and recomputing it means a solve;
* ``λ*`` moves by five-plus orders of magnitude across the reweighting loop, which is *why* λ is
  re-resolved rather than frozen (the M3 decision, settled by M7);
* the whole run is deterministic — same weights, same λ, same content key, twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import ObjectiveConfig, SamplerConfig
from gsmm_compiler.diagnostics import feasibility_report
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.maxent_sampler import run_ladder
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.reweighting import reweight_objective
from gsmm_compiler.rounding import build_transform
from gsmm_compiler.sparse_objective import (
    choose_energy_scale,
    lower_objective,
)

pytestmark = pytest.mark.slow

BASE_ACTIVE = 134
"""Active reactions at the base-weight L1 optimum (λ̃ = 0.25), measured. Reweighting sheds 3."""

SCHEDULE = SamplerConfig(
    betas=(0.0, 4.0), n_chains=2, n_samples=600, burn_in=600, thin=1, refresh_interval=250
)
"""Short — this gate is about the frozen weights and the solve counter, not convergence. Feasibility
and 'no solves after freeze' hold at any length; nothing here asserts R̂."""


@pytest.fixture(scope="module")
def reduced(example_canonical: CanonicalModel) -> ReducedPolytope:
    return example_canonical.polytope.reduce()


@pytest.fixture(scope="module")
def config() -> ObjectiveConfig:
    return ObjectiveConfig(l1_penalty_scaled=0.25, reweighting_enabled=True)


@pytest.fixture(scope="module")
def report(example_canonical, reduced, config):  # type: ignore[no-untyped-def]
    return reweight_objective(example_canonical.polytope, reduced, config)


class TestTheLoopOnARealModel:
    def test_it_converges(self, report) -> None:
        assert report.converged
        assert report.history[-1].active_set_changes == 0

    def test_it_sheds_reactions_plain_l1_leaves_on(self, report) -> None:
        """Reweighting's whole purpose: a stronger zero pressure than plain L1. Measured 134→131."""
        assert report.n_active_base == BASE_ACTIVE
        assert report.n_shed >= 1
        assert report.n_active_final < report.n_active_base

    def test_lambda_star_moves_by_orders_of_magnitude_which_is_why_lambda_is_re_resolved(
        self, report
    ) -> None:
        """The M3 decision, settled by measurement. ``λ*`` is a function of ``w``: across the
        reweighting loop it moves from ~1.9e-3 to ~4e2 (at the default [1e-3, 1e3] clip; with a
        wider clip it reaches ~2e5) because ``C_w`` changes *units* — a sum of fluxes becomes very
        nearly a count of active reactions. A λ frozen at the base value would leave the effective
        selection pressure λ/λ*(w) five-plus orders of magnitude below the requested one,
        annihilating the sparsity pressure — and the resulting sub-dual-tolerance costs make M3's
        ``z == |v|`` gate fail outright (reproduced in the module docstring's measurement)."""
        first = report.history[0].critical_l1_penalty
        last = report.history[-1].critical_l1_penalty

        assert last / first > 1e4  # five orders here; the point is "orders", not a precise ratio
        # λ̃ is held: the effective pressure λ/λ*(w) stays at the configured 0.25 throughout.
        for step in report.history:
            assert step.l1_penalty / step.critical_l1_penalty == pytest.approx(0.25, rel=1e-6)

    def test_it_is_deterministic(self, example_canonical, reduced, config, report) -> None:
        again = reweight_objective(example_canonical.polytope, reduced, config)

        assert np.array_equal(report.objective.weights, again.objective.weights)
        assert report.objective.l1_penalty == again.objective.l1_penalty
        assert report.objective.content_key() == again.objective.content_key()

    def test_the_frozen_weights_are_physically_read_only(self, report) -> None:
        with pytest.raises(ValueError, match="read-only|write"):
            report.objective.weights[0] = 1.0


class TestATooTightClipSilentlyBecomesPlainL1:
    """The signature bug of this codebase, at the reweighting layer — and here it is *measured*, on
    the model where the clip actually flips the answer, so the no-op test can fail for the right
    reason (the M4/M6 lesson: a test that cannot fail proves nothing).

    Clip [1e-3, 1e3]: sheds 3. Clip [1e-2, 1e2]: sheds 0 — the ceiling merges 'nearly off' (|v| ~
    1e-3) with 'off', and reweighted-L1 becomes plain L1 with no error anywhere.
    """

    def test_the_default_clip_sheds(self, report) -> None:
        assert report.n_shed >= 1  # [1e-3, 1e3] by default
        assert not report.support_unchanged

    def test_a_ceiling_below_the_nearly_off_band_leaves_the_support_unchanged(
        self, example_canonical, reduced
    ) -> None:
        tight = ObjectiveConfig(
            l1_penalty_scaled=0.25,
            reweighting_enabled=True,
            weight_clip_min=1e-2,
            weight_clip_max=1e2,
        )
        degenerate = reweight_objective(example_canonical.polytope, reduced, tight)

        assert degenerate.converged  # it runs and converges — that is the trap
        # `support_unchanged` is the honest signal (a straight swap gives net n_shed = 0 too).
        assert degenerate.support_unchanged
        assert degenerate.n_turned_off == 0
        assert degenerate.n_turned_on == 0


class TestTheFrozenObjectiveFlowsThroughTheSampler:
    """The frozen weights, sampled — and no solver touched once the chain starts."""

    @pytest.fixture(scope="class")
    def sampled(self, example_canonical, reduced, report):  # type: ignore[no-untyped-def]
        geometry = build_geometry(reduced, model_id="bifido-rw")
        transform = build_transform(geometry, reduced)

        lowered = lower_objective(reduced, report.objective)
        # s_J is built from the FROZEN objective's own optimum — the join choose_energy_scale now
        # refuses to make on trust. report.solution IS that optimum (same weights, same solve).
        scale = choose_energy_scale(
            lowered,
            geometry.support_points,
            optimum=report.solution.optimum,
            warmup_polytope_key=geometry.polytope_key,
            mode="warmup_range",
        )
        optimum = transform.to_coordinates(
            reduced.to_reduced(report.solution.optimum.v_full)
        )

        solves_before = total_solve_count()
        result = run_ladder(
            transform,
            reduced,
            config=SCHEDULE,
            model_id="bifido-rw",
            objective=lowered,
            energy_scale=scale,
            optimum_coordinates=optimum,
        )
        return result, solves_before, transform

    def test_zero_highs_solves_after_sampling_starts(self, sampled) -> None:
        """The invariant the milestone exists to protect. A weight that moved mid-chain would have
        to be recomputed, and recomputing it is a solve. The counter says none happened."""
        _, solves_before, _ = sampled
        assert total_solve_count() == solves_before

    def test_the_energy_scale_binds_to_the_frozen_objective(self, sampled, report) -> None:
        result, _, _ = sampled
        assert result.energy_scale.objective_key == report.objective.content_key()

    def test_every_tilted_sample_is_feasible_in_the_full_polytope(
        self, sampled, reduced, example_canonical
    ) -> None:
        """The one check a reduced-space sampler that is internally consistent but wrong about the
        lift cannot pass — checked on the β > 0 rung, under the reweighted objective. The lift
        restores all 773 reactions, including the 513 the reduced state never carried, and mass
        balance is recomputed against the **full** ``S`` the sampler has never seen."""
        result, _, _ = sampled
        polytope = example_canonical.polytope
        rung = result.rungs[-1]  # β = 4

        report_ = feasibility_report(rung.result.fluxes, reduced)
        assert report_.is_feasible
        assert report_.n_bound_violations == 0

        fluxes = rung.result.fluxes.reshape(-1, reduced.n_free)
        sampled_rows = fluxes[:: max(1, fluxes.shape[0] // 200)]
        for v in reduced.to_full(sampled_rows):
            assert polytope.contains(np.ascontiguousarray(v), tol=1e-7)

    def test_mean_j_rises_with_beta_under_the_reweighted_objective(self, sampled) -> None:
        """The tilt still works after reweighting: E[J] is nondecreasing in β (the linear-response
        theorem holds for the reweighted J too — it is still concave in v)."""
        result, _, _ = sampled
        means = [
            float(np.mean([np.mean(trace.j) for trace in rung.traces]))
            for rung in result.rungs
        ]
        assert means[-1] >= means[0]


class TestReweightingCannotSeeAChain:
    """The stationarity guarantee, made structural. A weight that moved from MCMC state would
    retarget every one-dimensional conditional and destroy the invariance argument (BUILD_PLAN
    §1.6.2). Enforced by construction: the two modules cannot import each other."""

    def test_reweighting_takes_no_sampler_input(self) -> None:
        import inspect

        from gsmm_compiler import reweighting

        source = inspect.getsource(reweighting)
        # The reweighter names no sampler symbol — it cannot receive a chain, a flux trace, or a
        # coordinate. Every v it uses is an LP optimum it solved itself.
        assert "maxent_sampler" not in source
        assert "run_chain" not in source
        assert "run_ladder" not in source
