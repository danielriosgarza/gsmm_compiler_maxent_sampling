"""The pilot DAG: bootstrap geometry → β=0 pilot → {final ``T``, ``s_J``}  (M10, spec §17.4/§22.2).

Two of v1's recorded findings converge on one structural change, and it is the same change:

- **M5**: this model mixes slowly (`step_scale_ratio = 0.008`). Re-rounding from a β=0 pilot's own
  covariance instead of M4's support-LP vertices improves ``cond(C_q)`` 1.54e4 → 5.11e3 and **ESS by
  ~2.5×** (spec §17.4).
- **M6**: the β axis is uncalibrated. ``s_J`` is read off the support *vertices*, where the L1
  cost is enormous, while the chain lives in the interior — so ``s_J = 32.5`` calibrates β against
  a range 13× wider than the one actually explored (spec §22.2, BUILD_PLAN §2.1).

Both want the same object: **a frozen β=0 pilot chain**. This module runs it, and hands its
covariance to `rounding.reround_transform` and its ``J`` spread to
`sparse_objective.pilot_energy_scale`.

## The pipeline, and why it is sequential

```
1. geometry pilot at β=0 under T₀          (OBJECTIVE-INDEPENDENT)
2. freeze its covariance → build T₁
3. INDEPENDENT scale pilot at β=0 under T₁  (better mixing → better ESS for σ̂₀)
4. freeze σ̂₀ → production chains on independent streams
```

The two pilots are **separate streams on purpose**. One shared pilot would be perfectly valid — the
transform cannot move the stationary law and both artifacts are frozen before production — but it
would make pilot-seed sensitivity **unattributable**: geometry quality and the selected target
would move together, so a run that looked odd could not be diagnosed into "the rounding got
unlucky" versus "the scale got unlucky". Separating them separates *random efficiency calibration*
from *random target calibration*. (Codex, M10 review round 2.)

The scale pilot runs under ``T₁`` deliberately, and the compounding worry is answered by noticing
what can and cannot propagate: a poor ``T₀`` **cannot deform the neutral target** — only the
efficiency of estimating σ̂₀ from it. So the errors do not compound as target deformation; they
compound as imprecision, which is measured and reported.

## What this module may and may not do

`calibration` imports `maxent_sampler`; **the sampler must never import `calibration`**. That is not
style, it is the invariant: an adaptive ``T`` or a ``s_J`` that could be re-derived mid-chain makes
the transition kernel depend on the chain's own history, and the samples are then not from ``π_β``
at
all (spec §17.4, §18.3). The dependency runs one way so a production chain *structurally cannot*
re-calibrate itself — the same guard shape M7 uses to keep reweighting out of the sampler, pinned by
`tests/unit/test_calibration_cannot_be_imported_by_sampler`.

## What the DAG guarantees, stated precisely

Freezing ``T₁`` and ``σ̂₀`` before production gives a **time-homogeneous kernel with a fixed
conditional invariant law**. It does *not* give stationarity from iteration zero — burn-in provides
convergence, not stationarity, unless the initial state is drawn from the law. And conditional on
the
pilot artifact the invariant target is ``π_{β/σ̂₀}``, **not** the ideal ``π_{β/σ₀}``; marginalising
over pilot randomness gives a *mixture* of calibrated targets. That is **calibration uncertainty,
not
an MCMC invariance failure** — a real thing, reported in `PilotScaleReport`, and different in kind
from a bug. (Codex, M10 review round 3; recorded rather than blurred.)

The transform's own invariance is a theorem (`range(diag(s)·B·L)` is ``L``-invariant, §1.6.1), but
range-invariance alone is *not* the clean condition: ``T₁`` must be a nonsingular affine coordinate
change **on the affine hull**. The real risks are implementation ones — rank loss, feasibility
tolerance, state carry-over, residual adaptation — which is what `rounding`'s SVD rank check and
this
module's freezing exist to close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.affine_geometry import ReducedGeometry
from gsmm_compiler.config import GeometryConfig, SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.logging_utils import get_logger
from gsmm_compiler.maxent_sampler import run_chains
from gsmm_compiler.provenance import content_key
from gsmm_compiler.rounding import RoundedTransform, reround_transform
from gsmm_compiler.sparse_objective import (
    EnergyScale,
    LPOptimum,
    ReducedObjective,
    pilot_energy_scale,
)

_log = get_logger(__name__)

VALUE_DTYPE = np.float64

CALIBRATION_IMPL_VERSION = 1
"""Bump when the pilot's arithmetic or schedule semantics change — invalidates cached pilots."""

GEOMETRY_PILOT_STAGE = "geometry_pilot"
SCALE_PILOT_STAGE = "scale_pilot"
"""The two `provenance.stream_seed` stages. Distinct names are what make the pilots independent:
the RNG is keyed on ``(model_id, stage, β_index, chain_index)``, so two stages that share a name
share their draws — which is exactly the coupling this DAG separates."""


class CalibrationError(ValueError):
    """A pilot could not be run, or its artifacts do not bind to each other."""


@dataclass(frozen=True)
class NeutralPilot:
    """A frozen β=0 pilot chain — **objective-independent**, and that is load-bearing.

    The β=0 law does not see ``J``: the target is uniform on the polytope and the kernel draws
    uniformly on each chord. So **one neutral pilot serves every objective on a polytope**, which is
    not a micro-optimisation — M7 puts a base *and* a reweighted objective on one polytope, and they
    must be calibrated against the *same* neutral ensemble or their β axes are not comparable with
    each other, let alone across strains.

    Hence the split: this artifact carries **no objective key**, and the derived `EnergyScale` does.
    (Codex, M10 review round 2.)
    """

    coordinates: NDArray[np.float64]
    """``(n_chains, n_draws, d)`` — the chains' rounded ``y`` under `transform_key`'s transform."""

    fluxes: NDArray[np.float64]
    """``(n_chains, n_draws, n_free)`` — reduced fluxes. `sparse_objective.pilot_energy_scale`
    evaluates ``J`` from these; `rounding.reround_transform` uses `coordinates` instead."""

    model_id: str

    stage: str
    """`GEOMETRY_PILOT_STAGE` or `SCALE_PILOT_STAGE` — also the RNG stream's name."""

    polytope_key: str

    transform_key: str
    """`RoundedTransform.content_key` of the transform this pilot **ran under**.

    A pilot's coordinates are only meaningful in the frame that produced them. `reround_transform`
    maps them back with ``q = L₀·y``, which is the right change of coordinates for *that* ``L₀`` and
    a silent corruption for any other — same shape, different geometry, every check green.
    """

    n_chains: int
    n_draws: int
    burn_in: int
    thin: int

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[2])

    def pooled_coordinates(self) -> NDArray[np.float64]:
        """``(n_chains·n_draws, d)`` — the draws as one point set, for a covariance."""
        return self.coordinates.reshape(-1, self.dimension)

    def content_key(self) -> str:
        """Everything that can change this pilot's bytes (BUILD_PLAN §1.1).

        Not merely polytope+stream: the **input transform** and the **schedule** change the draws
        too, and a key that omits them lets a pilot run under one ``T`` be reused under another.
        (Codex, M10 review round 3.)
        """
        return content_key(
            model_id=self.model_id,
            stage=self.stage,
            polytope_key=self.polytope_key,
            transform_key=self.transform_key,
            n_chains=self.n_chains,
            n_draws=self.n_draws,
            burn_in=self.burn_in,
            thin=self.thin,
            calibration_impl_version=CALIBRATION_IMPL_VERSION,
            numpy_version=np.__version__,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "pilot_content_key": self.content_key(),
            "pilot_stage": self.stage,
            "pilot_model_id": self.model_id,
            "pilot_polytope_key": self.polytope_key,
            "pilot_transform_key": self.transform_key,
            "pilot_n_chains": self.n_chains,
            "pilot_n_draws": self.n_draws,
            "pilot_burn_in": self.burn_in,
            "pilot_thin": self.thin,
            "calibration_impl_version": CALIBRATION_IMPL_VERSION,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """What the DAG froze, plus the evidence for it."""

    transform: RoundedTransform
    """``T₁`` — re-rounded from the geometry pilot, or ``T₀`` unchanged if re-rounding was off."""

    energy_scale: EnergyScale
    """``s_J``, in whichever mode the config asked for."""

    geometry_pilot: NeutralPilot | None
    scale_pilot: NeutralPilot | None

    bootstrap_condition_number: float
    """``cond(C_q)`` of ``T₀`` — the support-vertex rounding, kept so the improvement is *shown*
    rather than asserted."""

    def manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bootstrap_condition_number": self.bootstrap_condition_number,
            "final_condition_number": self.transform.diagnostics.condition_number,
            "rerounded": self.geometry_pilot is not None,
            **self.energy_scale.manifest(),
        }
        if self.geometry_pilot is not None:
            out["geometry_pilot"] = self.geometry_pilot.manifest()
        if self.scale_pilot is not None:
            out["scale_pilot"] = self.scale_pilot.manifest()
        return out


def run_neutral_pilot(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig, model_id: str, stage: str,
    optimum_coordinates: NDArray[np.float64] | None = None,
) -> NeutralPilot:
    """Run a β=0 pilot under ``transform`` and freeze it.

    ``stage`` names the RNG stream, so `GEOMETRY_PILOT_STAGE` and `SCALE_PILOT_STAGE` draw
    independent numbers on the same model — which is the whole point of running two.

    No ``objective`` argument, and that is deliberate rather than an omission: at β=0 the target is
    flat and ``J`` provably never enters the draw, so an objective here could only be decoration
    that
    a later reader would mistake for a dependency.
    """
    if stage not in (GEOMETRY_PILOT_STAGE, SCALE_PILOT_STAGE):
        raise CalibrationError(
            f"unknown pilot stage {stage!r}; expected {GEOMETRY_PILOT_STAGE!r} or "
            f"{SCALE_PILOT_STAGE!r} — the stage names the RNG stream, so a typo would silently "
            "give two pilots the same draws"
        )
    if transform.polytope_key != reduced.content_key():
        raise CalibrationError(
            "the transform was not built from this polytope; the pilot would step along one "
            "model's directions and be bounds-checked against another's"
        )

    result = run_chains(
        transform, reduced,
        config=config,
        model_id=model_id,
        beta=0.0,
        beta_index=0,
        objective=None,
        optimum_coordinates=optimum_coordinates,
        stage=stage,
    )

    coordinates = np.ascontiguousarray(
        np.stack([chain.coordinates for chain in result.chains]), dtype=VALUE_DTYPE
    )
    fluxes = np.ascontiguousarray(
        np.stack([chain.fluxes for chain in result.chains]), dtype=VALUE_DTYPE
    )
    coordinates.flags.writeable = False
    fluxes.flags.writeable = False

    return NeutralPilot(
        coordinates=coordinates,
        fluxes=fluxes,
        model_id=model_id,
        stage=stage,
        polytope_key=reduced.content_key(),
        transform_key=transform.content_key(),
        n_chains=int(coordinates.shape[0]),
        n_draws=int(coordinates.shape[1]),
        burn_in=int(config.burn_in),
        thin=int(config.thin),
    )


def calibrate(
    geometry: ReducedGeometry,
    reduced: ReducedPolytope,
    bootstrap: RoundedTransform,
    objective: ReducedObjective,
    *,
    model_id: str,
    optimum: LPOptimum,
    sampler: SamplerConfig,
    geometry_config: GeometryConfig | None = None,
    optimum_coordinates: NDArray[np.float64] | None = None,
) -> CalibrationResult:
    """Run the pilot DAG and freeze ``T₁`` and ``s_J`` before any production chain exists.

    Both stages are opt-in via `SamplerConfig`: ``pilot_reround`` re-rounds the transform, and
    ``energy_scale = "pilot_sd"`` calibrates β against the neutral ensemble's own spread. They are
    independent switches — either alone is coherent — but they share the machinery, so this runs
    whichever the config asked for and no pilot the config did not.

    Returns ``T₀`` and the support-vertex scale unchanged when neither is enabled, so a caller can
    route every run through here without a branch and without changing v1's numbers.
    """
    geometry_config = geometry_config or GeometryConfig()

    if bootstrap.polytope_key != reduced.content_key():
        raise CalibrationError("the bootstrap transform was not built from this polytope")
    if objective.polytope_key != reduced.content_key():
        raise CalibrationError(
            "the objective was not lowered from this polytope — the M6 'two artifacts that never "
            "met' join (BUILD_PLAN §1.6.3)"
        )

    wants_reround = sampler.pilot_reround
    wants_pilot_scale = sampler.energy_scale == "pilot_sd"
    bootstrap_condition = bootstrap.diagnostics.condition_number

    transform = bootstrap
    geometry_pilot: NeutralPilot | None = None
    if wants_reround:
        pilot_config = _pilot_schedule(sampler)
        _log.info(
            "geometry pilot: %d chains × (%d burn-in + %d) sweeps under T₀ (cond %.3g)",
            pilot_config.n_chains, pilot_config.burn_in, pilot_config.n_samples,
            bootstrap_condition,
        )
        geometry_pilot = run_neutral_pilot(
            bootstrap, reduced,
            config=pilot_config,
            model_id=model_id,
            stage=GEOMETRY_PILOT_STAGE,
            optimum_coordinates=optimum_coordinates,
        )
        transform = reround_transform(
            geometry, reduced, bootstrap,
            pilot_coordinates=geometry_pilot.pooled_coordinates(),
            config=geometry_config,
        )
        _log.info(
            "re-rounded: cond(C_q) %.3g → %.3g (%.2f×), step_scale_ratio %.3g → %.3g",
            bootstrap_condition,
            transform.diagnostics.condition_number,
            bootstrap_condition / transform.diagnostics.condition_number,
            bootstrap.diagnostics.step_scale_ratio,
            transform.diagnostics.step_scale_ratio,
        )

    scale_pilot: NeutralPilot | None = None
    if wants_pilot_scale:
        pilot_config = _pilot_schedule(sampler)
        scale_pilot = run_neutral_pilot(
            transform, reduced,
            config=pilot_config,
            model_id=model_id,
            stage=SCALE_PILOT_STAGE,
            optimum_coordinates=_recoordinate(bootstrap, transform, optimum_coordinates),
        )
        energy_scale = pilot_energy_scale(
            objective, scale_pilot.fluxes,
            optimum=optimum,
            pilot_polytope_key=scale_pilot.polytope_key,
        )
        assert energy_scale.pilot is not None
        _log.info(
            "s_J = σ̂₀ = %.4g (±%.1f%%), E₀[J] = %.4g, Δ₀ = %.4g, G = %.2f, R̂(J) = %.3f",
            energy_scale.value,
            100.0 * energy_scale.pilot.relative_se,
            energy_scale.pilot.mean_j, energy_scale.pilot.gap,
            energy_scale.pilot.headroom,
            energy_scale.pilot.r_hat_j,
        )
    else:
        from gsmm_compiler.sparse_objective import choose_energy_scale

        energy_scale = choose_energy_scale(
            objective, geometry.support_points,
            optimum=optimum,
            warmup_polytope_key=reduced.content_key(),
            mode=sampler.energy_scale,
            quantile=sampler.energy_scale_quantile,
            fallback=sampler.energy_scale_fallback,
        )

    return CalibrationResult(
        transform=transform,
        energy_scale=energy_scale,
        geometry_pilot=geometry_pilot,
        scale_pilot=scale_pilot,
        bootstrap_condition_number=bootstrap_condition,
    )


def _pilot_schedule(sampler: SamplerConfig) -> SamplerConfig:
    """The pilot's own schedule — β=0 only, and its own lengths.

    A pilot is not the production run and must not inherit its ladder: `run_chains` is called at
    β=0, so a `betas` tuple here would be silently ignored, which is worse than being absent.
    """
    from dataclasses import replace

    return replace(
        sampler,
        betas=(0.0,),
        n_chains=sampler.pilot_chains,
        burn_in=sampler.pilot_burn_in,
        n_samples=sampler.pilot_samples,
    )


def _recoordinate(
    bootstrap: RoundedTransform,
    transform: RoundedTransform,
    optimum_coordinates: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Re-express a start hint in the new transform's frame, if the transform changed.

    ``optimum_coordinates`` is a *start hint* and nothing else (BUILD_PLAN §1.6.5) — it enters only
    the initial state, never the kernel, the objective or ``s_J``, so a wrong one cannot change the
    invariant target, only seed a poorer start. But it is expressed in ``T₀``'s coordinates, and
    handing those to a chain stepping in ``T₁``'s frame would point it at an arbitrary interior
    point. Cheaper and clearer to lift through the flux space, which both frames share.
    """
    if optimum_coordinates is None or transform is bootstrap:
        return optimum_coordinates
    flux = bootstrap.to_flux(np.asarray(optimum_coordinates, dtype=VALUE_DTYPE))
    return np.ascontiguousarray(transform.to_coordinates(flux), dtype=VALUE_DTYPE)
