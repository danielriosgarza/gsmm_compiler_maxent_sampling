"""The dimension-scaled sampling schedule (M11.5, spec §M11.5 / `benchmarks/M11_5_SCHEDULE_TAU.md`).

The flat 2000-sweep default was tuned on one d=46 anaerobe. A MEASURE-FIRST sweep of the integrated
autocorrelation time τ across 9 strains (d=34…145) × β∈{0,1,8,16} showed mixing efficiency degrades
super-linearly with d (p90-τ ∝ d^1.63) *and* with β (up to 27×), with ±1.5–2× strain-to-strain
scatter at fixed d. A single fitted d-power rule is a guess — the exponent is not even a constant
(1.2/1.6/2.2 by which quantile you protect). So instead of guessing, **measure this strain's own τ
from the β=0 scale pilot** (which already ran) and size production to a declared target ESS.

`resolve_schedule` is the whole module: a pure, deterministic function from
``(SamplerConfig, final transform, scale pilot)`` to the *effective* config whose ``n_samples`` the
run actually uses. Two properties are load-bearing:

* **It is the identity in ``schedule_mode="fixed"``** (default), so every pre-M11.5 run and every
  sample artifact key is byte-unchanged.
* **The resolved integer is what flows into the content key.** `batch.prepare_model` calls this and
  puts the result into `batch.ModelPlan.sampler` — the object `batch.sample_recipe_key` hashes and
  the object the workers step under. Resolve it anywhere the key cannot see and two runs of
  one config draw different numbers and collide in the cache (the M10.2b hazard, one layer out). The
  `energy_scale_value` field already follows this pattern for `s_J`: the *mode* is config, the
  *resolved number* is keyed.

**What it does NOT do.** It does not touch ``burn_in`` — autocorrelation is the wrong instrument
for burn-in (which is about reaching stationarity, not mixing speed), so ``burn_in`` keeps its own
policy (Codex, M11.5 review). It makes **no β correction**: the pilot is β=0 and the β-inflation is
erratic and unpredictable from d (the largest inflation was the smallest model), so the resolved
schedule is a **β=0 prediction**. `diagnostics.run_diagnostics` reports the *achieved* per-rung
flux-ESS separately and never claims the target was met where it was not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from gsmm_compiler.config import SamplerConfig

if TYPE_CHECKING:  # runtime import avoided; these modules do not import `schedule`, so it is safe
    from gsmm_compiler.calibration import ScalePilot
    from gsmm_compiler.rounding import RoundedTransform

SCHEDULE_IMPL_VERSION = 1
"""Bump when `resolve_schedule`'s arithmetic changes. Recorded in the manifest, **not** in a key:
the resolved integer ``n_samples`` is already a component of `batch.sample_recipe_key`, and two
policies that produce the same integer produce the same sample bytes (Codex, M11.5 review). A key
component here would only assert the same fact twice — the `movable` exclusion argument, again."""


class ScheduleError(ValueError):
    """The schedule could not be resolved — a missing pilot, or a pilot from another frame."""


@dataclass(frozen=True)
class ScheduleResolution:
    """How `resolve_schedule` sized a run — recorded in the manifest beside the resolved config.

    Recomputed from the keyed pilot on every run, never read back from a store (M10.2b: evidence you
    recompute is evidence; evidence you store and read back is a claim). Its job is *visibility
    beside elimination* (M10.2e): the resolved schedule is opaque without the τ it came from, so a
    reader can see the base config, the pilot it derived from, the uncapped requirement, and the cap
    decision.
    """

    mode: str
    requested_n_samples: int
    resolved_n_samples: int
    target_ess: int | None = None
    ess_quantile: float | None = None
    pilot_content_key: str | None = None
    n_movable: int | None = None
    quantile_tau_int: float | None = None
    """τ_int at the protected quantile, in **sweeps** (frame- and chain-count-independent). ``inf``
    if more than ``1 − ess_quantile`` of movable coordinates were constant over the pilot."""
    pilot_ess_at_quantile: float | None = None
    """The pilot's own ESS at the protected coordinate — the number the sizing formula inverts."""
    uncapped_n_samples: int | None = None
    max_schedule_sweeps: int | None = None
    cap_hit: bool = False
    schedule_impl_version: int = SCHEDULE_IMPL_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_n_samples": self.requested_n_samples,
            "resolved_n_samples": self.resolved_n_samples,
            "target_ess": self.target_ess,
            "ess_quantile": self.ess_quantile,
            "pilot_content_key": self.pilot_content_key,
            "n_movable": self.n_movable,
            "quantile_tau_int": self.quantile_tau_int,
            "pilot_ess_at_quantile": self.pilot_ess_at_quantile,
            "uncapped_n_samples": self.uncapped_n_samples,
            "max_schedule_sweeps": self.max_schedule_sweeps,
            "cap_hit": self.cap_hit,
            "schedule_impl_version": self.schedule_impl_version,
        }


def resolve_schedule(
    sampler: SamplerConfig,
    transform: RoundedTransform,
    scale_pilot: ScalePilot | None,
) -> tuple[SamplerConfig, ScheduleResolution]:
    """Return the effective ``SamplerConfig`` and a record of how it was sized.

    ``"fixed"`` mode is the identity — the same object back, so nothing downstream can tell M11.5
    ran. ``"pilot_ess"`` mode sizes ``n_samples`` from the scale pilot's flux autocorrelation; see
    the module docstring for why the pilot rather than a fitted rule, and why no β correction.
    """
    requested = int(sampler.n_samples)
    if sampler.schedule_mode == "fixed":
        return sampler, ScheduleResolution(
            mode="fixed", requested_n_samples=requested, resolved_n_samples=requested
        )

    # ---- pilot_ess ----------------------------------------------------------------------------
    if scale_pilot is None:
        # Config validation forbids this pairing, so reaching here means a caller wired the pilot
        # out — refuse rather than fall back to `fixed` (which would silently ignore the target).
        raise ScheduleError(
            'schedule_mode="pilot_ess" needs the β=0 scale pilot (energy_scale="pilot_sd"), but '
            "none was provided"
        )
    if sampler.target_ess is None:  # pragma: no cover - config validation guarantees this
        raise ScheduleError('schedule_mode="pilot_ess" needs target_ess set')

    # Bind the movable mask to the frame that produced these fluxes. `movable_reactions(transform)`
    # indexes the pilot's reduced fluxes, so a transform from another frame would select the wrong
    # columns — same shape, wrong reactions, every downstream shape-check green. The pilot records
    # the transform it ran under; a mismatch is a wiring bug, and this is where it dies.
    if transform.content_key() != scale_pilot.recipe.transform_key:
        raise ScheduleError(
            "the scale pilot ran under a different transform than the one handed to the resolver "
            f"({scale_pilot.recipe.transform_key[:16]}… vs {transform.content_key()[:16]}…); its "
            "flux columns index that frame's movable reactions, not this one's"
        )

    from gsmm_compiler.diagnostics import effective_sample_size
    from gsmm_compiler.maxent_sampler import movable_reactions

    movable = movable_reactions(transform)
    fluxes = np.asarray(scale_pilot.fluxes)[:, :, movable]  # (n_chains, n_draws, n_movable)
    ess = effective_sample_size(fluxes)  # (n_movable,) — pooled across chains

    n_chains_pilot = int(scale_pilot.recipe.n_chains)
    n_draws_pilot = int(scale_pilot.recipe.n_draws)
    # τ_int = n_chains·n_draws / ESS, in pilot **retained-draw** units. A constant coordinate has
    # ESS = 0 (diagnostics returns exactly 0); its τ is +∞, RETAINED so the quantile counts it —
    # dropping the worst-mixing coordinates before a high quantile of τ would defeat the protection
    # (Codex, M11.5 review).
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_pilot_draws = np.where(ess > 0.0, (n_chains_pilot * n_draws_pilot) / ess, np.inf)
        # When the protected quantile lands between two +∞ coordinates, numpy's linear
        # interpolation computes ∞ − ∞ = nan; both nan and ∞ are non-finite and route to the cap
        # below, so the `invalid` warning is suppressed rather than the (correct) result changed.
        quantile_tau_draws = float(
            np.percentile(tau_pilot_draws, 100.0 * sampler.schedule_ess_quantile)
        )

    # Convert pilot-retained-draw units to production-retained-draw units when thinning differs:
    # τ in sweeps is invariant, and one retained draw is `thin` sweeps, so
    # τ_prod_draws = τ_pilot_draws · (pilot_thin / prod_thin) (Codex, M11.5 review).
    pilot_thin = int(scale_pilot.recipe.thin)
    prod_thin = int(sampler.thin)
    quantile_tau_prod = quantile_tau_draws * (pilot_thin / prod_thin)
    # τ_int in **sweeps** for the manifest — the frame/chain/thin-independent physical number.
    quantile_tau_sweeps = (
        quantile_tau_draws * pilot_thin if np.isfinite(quantile_tau_draws) else float("inf")
    )
    pilot_ess_at_quantile = (
        (n_chains_pilot * n_draws_pilot) / quantile_tau_draws
        if quantile_tau_draws > 0.0 and np.isfinite(quantile_tau_draws)
        else 0.0
    )

    # The cap is in **sweeps**; the schedule it bounds is in retained draws, one draw being `thin`
    # sweeps — so convert with floor division, which is what stops a `thin > 1` run from exceeding
    # the declared sweep budget (Codex, M11.5 review). Config validation guarantees
    # max_schedule_sweeps ≥ n_samples·thin, so `cap_draws ≥ requested` and [requested, cap_draws] is
    # non-empty.
    cap_sweeps = int(sampler.max_schedule_sweeps)
    cap_draws = cap_sweeps // prod_thin
    if not np.isfinite(quantile_tau_prod):
        # More than (1 − q) of the movable coordinates were constant over the pilot: no finite
        # budget reaches the target on them. `math.ceil(inf)` raises, so resolve straight to the cap
        # and record the miss (Codex, M11.5 review implementation guard).
        uncapped: int | None = None
        resolved = cap_draws
        cap_hit = True
    else:
        # ESS_prod = n_chains · n_prod / τ_prod = target  ⇒  n_prod = target · τ_prod / n_chains.
        uncapped = math.ceil(sampler.target_ess * quantile_tau_prod / int(sampler.n_chains))
        resolved = min(cap_draws, max(requested, uncapped))
        cap_hit = uncapped > cap_draws

    resolution = ScheduleResolution(
        mode="pilot_ess",
        requested_n_samples=requested,
        resolved_n_samples=int(resolved),
        target_ess=int(sampler.target_ess),
        ess_quantile=float(sampler.schedule_ess_quantile),
        pilot_content_key=scale_pilot.recipe.content_key(),
        n_movable=int(movable.size),
        quantile_tau_int=quantile_tau_sweeps,
        pilot_ess_at_quantile=float(pilot_ess_at_quantile),
        uncapped_n_samples=None if uncapped is None else int(uncapped),
        max_schedule_sweeps=cap_sweeps,
        cap_hit=cap_hit,
    )
    return replace(sampler, n_samples=int(resolved)), resolution
