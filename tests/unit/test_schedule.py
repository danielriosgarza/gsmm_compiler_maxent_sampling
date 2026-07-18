"""M11.5 — the dimension-scaled sampling schedule.

Three groups: (1) `resolve_schedule`'s arithmetic, with lightweight pilot/transform stand-ins so a
known autocorrelation structure can be built and the sizing formula checked exactly; (2) the
`_sampling_convergence` verification logic (two separate booleans, nonfinite → fail); (3) the KEYING
regression through the real `batch.prepare_model` — that the *resolved* schedule (not the raw) is
what reaches `sample_recipe_key`. The last is the one the spec calls THE TRAP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from gsmm_compiler.config import Config, ConfigError, SamplerConfig
from gsmm_compiler.diagnostics import SAMPLING_RHAT_BAR, _sampling_convergence
from gsmm_compiler.schedule import ScheduleError, resolve_schedule

# ---- stand-ins: resolve_schedule reads only these attributes ----------------------------------


@dataclass
class _FakeRecipe:
    transform_key: str
    n_chains: int
    n_draws: int
    thin: int = 1

    def content_key(self) -> str:
        return f"pilot::{self.transform_key}::{self.n_chains}x{self.n_draws}"


@dataclass
class _FakePilot:
    fluxes: np.ndarray
    recipe: _FakeRecipe


@dataclass
class _FakeTransform:
    transform: np.ndarray  # (n_free, d); movable_reactions reads the nonzero rows
    key: str

    def content_key(self) -> str:
        return self.key


def _pilot(fluxes: np.ndarray, *, key: str = "T", thin: int = 1) -> _FakePilot:
    nc, nd, n_free = fluxes.shape
    return _FakePilot(
        fluxes=np.ascontiguousarray(fluxes, dtype=np.float64),
        recipe=_FakeRecipe(transform_key=key, n_chains=nc, n_draws=nd, thin=thin),
    )


def _transform(n_free: int, *, key: str = "T") -> _FakeTransform:
    # Every row nonzero ⇒ every reduced flux is "movable", so the mask is the whole column set and
    # the test controls exactly which coordinates the resolver sees.
    return _FakeTransform(transform=np.ones((n_free, 2), dtype=np.float64), key=key)


def _iid_fluxes(nc: int, nd: int, n_free: int, *, seed: int) -> np.ndarray:
    """White-noise coordinates: τ_int ≈ 1, so ESS ≈ nc·nd. Deterministic given the seed."""
    return np.random.default_rng(seed).standard_normal((nc, nd, n_free))


def _ar1_fluxes(nc: int, nd: int, n_free: int, *, phi: float, seed: int) -> np.ndarray:
    """AR(1) coordinates: φ near 1 ⇒ long autocorrelation ⇒ small ESS ⇒ large τ."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((nc, nd, n_free))
    x = np.empty_like(eps)
    x[:, 0, :] = eps[:, 0, :]
    for t in range(1, nd):
        x[:, t, :] = phi * x[:, t - 1, :] + eps[:, t, :]
    return x


def _pilot_ess_sampler(**overrides: object) -> SamplerConfig:
    base = dict(
        betas=(0.0,),
        n_chains=4,
        n_samples=100,
        burn_in=10,
        energy_scale="pilot_sd",
        schedule_mode="pilot_ess",
        target_ess=400,
    )
    base.update(overrides)
    return SamplerConfig(**base)  # type: ignore[arg-type]


class TestResolveScheduleArithmetic:
    def test_fixed_mode_is_the_identity(self) -> None:
        sampler = SamplerConfig(n_samples=1234)  # schedule_mode defaults to "fixed"
        resolved, res = resolve_schedule(sampler, transform=None, scale_pilot=None)  # type: ignore[arg-type]
        assert resolved is sampler  # the *same object* — nothing downstream can tell M11.5 ran
        assert res.mode == "fixed"
        assert res.resolved_n_samples == 1234 == res.requested_n_samples

    def test_pilot_ess_sizes_to_its_own_measured_tau(self) -> None:
        # Self-consistency: the resolved n must equal ceil(target · τ_q / n_chains) for the τ_q the
        # resolver itself recorded — the formula and the record cannot disagree.
        fluxes = _iid_fluxes(4, 400, 30, seed=0)
        sampler = _pilot_ess_sampler(n_chains=4, target_ess=400, n_samples=1)
        resolved, res = resolve_schedule(sampler, _transform(30), _pilot(fluxes))
        assert res.mode == "pilot_ess"
        assert res.n_movable == 30
        expected = math.ceil(400 * res.quantile_tau_int / 4)  # thin=1 ⇒ τ_int(sweeps) == τ_draws
        assert resolved.n_samples == res.resolved_n_samples == expected
        assert not res.cap_hit

    def test_resolution_is_deterministic(self) -> None:
        fluxes = _ar1_fluxes(4, 400, 20, phi=0.7, seed=3)
        sampler = _pilot_ess_sampler()
        a, ra = resolve_schedule(sampler, _transform(20), _pilot(fluxes))
        b, rb = resolve_schedule(sampler, _transform(20), _pilot(fluxes))
        assert a.n_samples == b.n_samples
        assert ra.quantile_tau_int == rb.quantile_tau_int
        assert ra.as_dict() == rb.as_dict()

    def test_n_samples_scales_linearly_with_target_ess(self) -> None:
        fluxes = _ar1_fluxes(4, 400, 20, phi=0.6, seed=7)
        pilot, transform = _pilot(fluxes), _transform(20)
        _, r1 = resolve_schedule(_pilot_ess_sampler(target_ess=200, n_samples=1), transform, pilot)
        _, r4 = resolve_schedule(_pilot_ess_sampler(target_ess=800, n_samples=1), transform, pilot)
        assert r1.quantile_tau_int == r4.quantile_tau_int  # τ depends on the pilot, not target
        assert r4.uncapped_n_samples is not None and r1.uncapped_n_samples is not None
        # 4× the target ⇒ 4× the sweeps (up to the ceil rounding of one sample).
        assert abs(r4.uncapped_n_samples - 4 * r1.uncapped_n_samples) <= 4

    def test_worse_mixing_needs_more_sweeps(self) -> None:
        # The d-scaling mechanism: a slower-mixing pilot (higher τ) resolves to a longer schedule.
        good = _pilot(_iid_fluxes(4, 400, 20, seed=1))
        bad = _pilot(_ar1_fluxes(4, 400, 20, phi=0.85, seed=1))
        sampler = _pilot_ess_sampler(n_samples=1)
        _, r_good = resolve_schedule(sampler, _transform(20), good)
        _, r_bad = resolve_schedule(sampler, _transform(20), bad)
        assert r_bad.quantile_tau_int > r_good.quantile_tau_int
        assert r_bad.resolved_n_samples > r_good.resolved_n_samples

    def test_constant_coordinates_resolve_to_the_cap(self) -> None:
        # τ = ∞ on a constant coordinate; if the protected quantile lands on one, no finite budget
        # reaches the target, so the resolver goes to the cap and flags it (ceil(inf) must not
        # raise).
        fluxes = _iid_fluxes(4, 400, 20, seed=2)
        fluxes[:, :, :5] = 7.0  # 25% of coords constant ⇒ p90 τ is +∞
        sampler = _pilot_ess_sampler(max_schedule_sweeps=50_000)
        resolved, res = resolve_schedule(sampler, _transform(20), _pilot(fluxes))
        assert res.cap_hit is True
        assert res.uncapped_n_samples is None
        assert resolved.n_samples == res.resolved_n_samples == 50_000
        assert not np.isfinite(res.quantile_tau_int)

    def test_resolved_never_exceeds_the_cap(self) -> None:
        fluxes = _ar1_fluxes(4, 400, 20, phi=0.8, seed=5)
        sampler = _pilot_ess_sampler(target_ess=100_000, n_samples=10, max_schedule_sweeps=500)
        resolved, res = resolve_schedule(sampler, _transform(20), _pilot(fluxes))
        assert res.uncapped_n_samples is not None and res.uncapped_n_samples > 500
        assert resolved.n_samples == 500 and res.cap_hit is True

    def test_cap_is_a_sweep_budget_under_thinning(self) -> None:
        # The cap is in SWEEPS, not retained draws (Codex, M11.5 review): with thin=5 a resolved
        # draw count of N costs 5N sweeps, so a 1000-sweep cap must clamp the draws to 1000//5 = 200
        # — enforcing it in draw units would let the run spend 5× its declared sweep budget.
        fluxes = _ar1_fluxes(4, 400, 20, phi=0.85, seed=11)
        pilot = _pilot(fluxes, thin=5)
        sampler = _pilot_ess_sampler(
            target_ess=100_000, n_chains=4, n_samples=10, thin=5, max_schedule_sweeps=1000,
        )
        resolved, res = resolve_schedule(sampler, _transform(20), pilot)
        assert res.cap_hit is True
        assert resolved.n_samples == 200  # 1000 sweeps // thin 5, not 1000 draws
        assert res.resolved_n_samples * sampler.thin <= sampler.max_schedule_sweeps

    def test_resolved_never_falls_below_the_requested_floor(self) -> None:
        fluxes = _iid_fluxes(4, 400, 20, seed=9)
        sampler = _pilot_ess_sampler(target_ess=1, n_samples=3000)  # tiny target ⇒ uncapped < floor
        resolved, res = resolve_schedule(sampler, _transform(20), _pilot(fluxes))
        assert res.uncapped_n_samples is not None and res.uncapped_n_samples < 3000
        assert resolved.n_samples == 3000 and not res.cap_hit

    def test_a_pilot_from_another_frame_is_refused(self) -> None:
        fluxes = _iid_fluxes(4, 400, 20, seed=0)
        pilot = _pilot(fluxes, key="frame-A")
        with pytest.raises(ScheduleError, match="different transform"):
            resolve_schedule(_pilot_ess_sampler(), _transform(20, key="frame-B"), pilot)

    def test_pilot_ess_without_a_pilot_is_refused(self) -> None:
        with pytest.raises(ScheduleError, match="scale pilot"):
            resolve_schedule(_pilot_ess_sampler(), _transform(20), None)


class TestConfigValidation:
    def test_pilot_ess_requires_target_ess(self) -> None:
        with pytest.raises(ConfigError, match="target_ess"):
            SamplerConfig(schedule_mode="pilot_ess", energy_scale="pilot_sd")

    def test_pilot_ess_requires_pilot_sd(self) -> None:
        with pytest.raises(ConfigError, match="pilot_sd"):
            SamplerConfig(schedule_mode="pilot_ess", target_ess=400)  # warmup_range default

    def test_cap_must_be_at_least_n_samples(self) -> None:
        with pytest.raises(ConfigError, match="max_schedule_sweeps"):
            SamplerConfig(
                schedule_mode="pilot_ess", target_ess=400, energy_scale="pilot_sd",
                n_samples=2000, max_schedule_sweeps=1000,
            )

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
    def test_quantile_must_be_in_the_open_unit_interval(self, q: float) -> None:
        with pytest.raises(ConfigError, match="schedule_ess_quantile"):
            SamplerConfig(schedule_ess_quantile=q)

    def test_bad_mode_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="schedule_mode"):
            SamplerConfig(schedule_mode="adaptive")

    def test_fixed_mode_with_no_target_is_fine(self) -> None:
        SamplerConfig()  # the default — must construct


class TestSamplingConvergence:
    def test_a_well_mixed_rung_passes_and_meets_a_modest_target(self) -> None:
        samples = _iid_fluxes(4, 500, 8, seed=4)  # τ≈1 ⇒ ESS≈2000
        out = _sampling_convergence(samples, target_ess=100, bar=SAMPLING_RHAT_BAR)
        assert out["convergence_diagnostic_passed"] is True
        assert out["ess_target_met"] is True and out["target_verified"] is True
        assert out["flux_worst_rhat"] <= SAMPLING_RHAT_BAR

    def test_trapped_chains_fail_convergence_even_at_a_low_target(self) -> None:
        # Each chain constant at a different value: within-chain variance 0, between-chain > 0 ⇒
        # split-R̂ is +∞. A nonfinite R̂ must be a FAILURE, never a silent pass (Codex, M11.5).
        samples = np.zeros((4, 200, 6))
        for c in range(4):
            samples[c] = float(c)  # frozen apart
        out = _sampling_convergence(samples, target_ess=1, bar=SAMPLING_RHAT_BAR)
        assert not np.isfinite(out["flux_worst_rhat"])
        assert out["convergence_diagnostic_passed"] is False
        assert out["target_verified"] is False

    def test_no_target_leaves_ess_met_unknown(self) -> None:
        samples = _iid_fluxes(4, 500, 8, seed=6)
        out = _sampling_convergence(samples, target_ess=None, bar=SAMPLING_RHAT_BAR)
        assert out["ess_target_met"] is None and out["target_verified"] is None
        assert out["convergence_diagnostic_passed"] is True  # still reported


class TestKeyingTrap:
    """THE TRAP: the resolved schedule — not the raw config — must reach `sample_recipe_key`."""

    @staticmethod
    def _prepare(toy_path: Path, target_ess: int):
        from gsmm_compiler.batch import ModelSpec, prepare_model, sample_recipe_key

        config = Config(
            sampler=SamplerConfig(
                betas=(0.0,), n_chains=2, n_samples=5, burn_in=10,
                energy_scale="pilot_sd", pilot_chains=2, pilot_burn_in=20, pilot_samples=60,
                schedule_mode="pilot_ess", target_ess=target_ess,
            )
        )
        plan = prepare_model(ModelSpec(model_path=str(toy_path), model_id="toy"), config)
        key = sample_recipe_key(plan, beta_index=0, chain_index=0)
        return config, plan, key

    def test_the_resolved_n_samples_is_what_reaches_the_key(self, toy_path: Path) -> None:
        from gsmm_compiler.batch import sample_recipe_key

        config, plan, key = self._prepare(toy_path, target_ess=500)
        # The schedule actually resolved to something other than the requested floor of 5.
        assert plan.sampler.n_samples != config.sampler.n_samples
        assert plan.sampler.n_samples == plan.reports["schedule"]["resolved_n_samples"]
        # Non-vacuity: rebuild the key from the RAW (unresolved) sampler. If prepare_model had left
        # the raw config in the plan, this would equal `key` and the test could not catch the trap.
        raw_plan = replace(plan, sampler=config.sampler)
        raw_key = sample_recipe_key(raw_plan, beta_index=0, chain_index=0)
        assert raw_key != key

    def test_different_targets_miss_and_the_same_target_hits(self, toy_path: Path) -> None:
        _, _, key_500a = self._prepare(toy_path, target_ess=500)
        _, _, key_500b = self._prepare(toy_path, target_ess=500)
        _, _, key_900 = self._prepare(toy_path, target_ess=900)
        assert key_500a == key_500b       # same config → same experiment → cache HIT
        assert key_500a != key_900        # different target → different draws → cache MISS

    def test_fixed_mode_key_ignores_the_new_schedule_fields(self, toy_path: Path) -> None:
        # A fixed-mode run's sample key must not move when a schedule-only field changes — proof the
        # new config fields do not leak into the sample artifact identity (byte-compatibility).
        from gsmm_compiler.batch import ModelSpec, prepare_model, sample_recipe_key

        def key_for(max_sweeps: int) -> str:
            config = Config(sampler=SamplerConfig(
                betas=(0.0,), n_chains=1, n_samples=10, burn_in=10, max_schedule_sweeps=max_sweeps,
            ))
            plan = prepare_model(ModelSpec(model_path=str(toy_path), model_id="toy"), config)
            assert plan.sampler is config.sampler  # fixed mode is the identity
            return sample_recipe_key(plan, beta_index=0, chain_index=0)

        assert key_for(200_000) == key_for(50_000)
