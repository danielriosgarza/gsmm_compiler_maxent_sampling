"""Configuration: TOML load/resolve, CLI overrides, resolved-config echo.

Two properties matter more than convenience here:

* **Unknown keys are errors.** A typo'd ``sampler.n_chian`` that is silently ignored means the run
  quietly used a default the user did not intend — and the numbers still come out looking plausible.
  Every section rejects keys it does not define.
* **The resolved config is echoed, not the file.** What lands in the run manifest is the config
  *after* defaults and CLI overrides are folded in: the thing that actually parameterized the run.

Stdlib only (``tomllib`` ships with 3.11), so a worker can read a config without importing cobra.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """The configuration is malformed, has unknown keys, or violates a stated constraint."""


DEFAULT_NEAR_ZERO_THRESHOLDS: tuple[float, ...] = (1e-9, 1e-6, 1e-3, 1e-1, 1.0)
"""Declared thresholds for *analysing* near-zero fluxes (spec §3.7). Never used by the chain.

Five, spanning nine orders of magnitude, because **one absolute threshold cannot serve two models**
and the conventional FBA values cannot serve this one. Measured on the example model at β = 0: the
median ``|v_r|`` over the 199 reactions that can move is **53**, the 1st percentile is 0.18, and the
maximum is 1000. So ``1e-9``, ``1e-6`` and ``1e-3`` — the thresholds an FBA paper would reach for —
each report a count of **exactly zero**, at every β. They are not wrong; they are five to eleven
orders of magnitude below the fluxes they are asked about, and the honest reading of a zero count is
"your threshold cannot see this model", not "this cell is dense".

At ``1e-1`` the count is 0.2 reactions and at ``1.0`` it is 7.3, so the ladder brackets the scale
rather than sitting under it. The tight three are kept because they are the ones the literature uses
and a run must be able to report them.

⚠️ **An absolute threshold is not comparable across strains** — it is the λ problem again (§1.7): two
organisms whose fluxes differ by 100× would get activity counts that say more about their units than
their biology. The cross-model activity tables are **M8**'s (§1.1), and this decision is recorded
there as open: they will need a *relative* scale (a quantile of the movable flux, most likely), and
this constant is what a per-model run reports in the meantime.
"""


@dataclass(frozen=True)
class ModelConfig:
    """Which model to sample, and what counts as its biomass reaction."""

    path: str = ""
    biomass_id: str | None = None
    """``None`` means "use the model's own objective", which must then name exactly one reaction."""


@dataclass(frozen=True)
class ObjectiveConfig:
    """The sparse objective ``J(v) = v_biomass − λ Σ w_r |v_r|`` (spec §3)."""

    l1_penalty_scaled: float = 0.5
    """``λ̃``, **dimensionless** — the raw λ is ``λ̃ · λ*``, resolved per model (BUILD_PLAN §1.7).

    λ is not a model-independent knob: it compares a biomass flux against a sum of hundreds of
    absolute fluxes, and the exchange rate differs per organism. Above ``λ* = max_v μ(v)/C(v)`` the
    LP optimum is *the origin* — the cell's best move is to stop growing. On the example model
    ``λ* = 1.9e-3``, so a raw λ of 1.0 (our old default) sits 529× past the cliff and the spec's
    suggested 0.01 sits 5.3× past it. Both would have sampled a distribution concentrated on no
    metabolism at all.

    So the config takes ``λ̃`` instead: 0 is plain FBA, and → 1 is the most sparsity pressure the
    model can carry while still growing. The same ``λ̃`` then means the same *selection pressure* in
    every strain of a batch — which is what makes the cross-model comparison (§1.1) mean anything.
    Both ``λ̃`` and the resolved raw λ are written to the manifest; nothing is scaled in secret.
    """
    exclude_biomass_from_penalty: bool = True
    """Penalizing biomass would have the objective fight its own reward term."""
    near_zero_thresholds: tuple[float, ...] = DEFAULT_NEAR_ZERO_THRESHOLDS
    """Declared thresholds for counting near-zero fluxes in the objective traces (spec §3.7, §24.2).

    **Analysis only.** At finite β the sampled law is continuous, so *no* reaction is exactly zero
    unless the polytope itself pins it there — the L1 term promotes small fluxes, it does not
    produce zeros. Reporting "how many reactions are off" therefore requires a threshold to be
    *declared* — and several of them, because the answer depends on which one you pick, and a single
    number would hide that dependence.

    The sampler never sees these. Snapping a chain's flux to zero would alter the stationary
    distribution and can break mass balance (spec §3.7, CLAUDE.md) — the thresholds apply to the
    stored samples, downstream, and to nothing else.
    """
    reweighting_enabled: bool = False
    """M7. Weights are frozen before sampling begins — never updated from chain state."""
    reweighting_epsilon: float = 1e-6
    reweighting_max_iterations: int = 10
    weight_clip_min: float = 1e-3
    weight_clip_max: float = 1e3

    def __post_init__(self) -> None:
        if self.l1_penalty_scaled < 0.0:
            raise ConfigError(
                f"objective.l1_penalty_scaled must be >= 0, got {self.l1_penalty_scaled}"
            )
        if not self.near_zero_thresholds:
            raise ConfigError("objective.near_zero_thresholds must list at least one threshold")
        if any(
            not isfinite(threshold) or threshold <= 0.0
            for threshold in self.near_zero_thresholds
        ):
            raise ConfigError(
                "objective.near_zero_thresholds must all be finite and > 0, got "
                f"{list(self.near_zero_thresholds)}"
            )
        if not 0.0 < self.weight_clip_min < self.weight_clip_max:
            raise ConfigError(
                "objective weight clips must satisfy 0 < min < max, got "
                f"{self.weight_clip_min} / {self.weight_clip_max}"
            )
        if self.reweighting_epsilon <= 0.0:
            raise ConfigError("objective.reweighting_epsilon must be > 0")
        if self.reweighting_max_iterations < 1:
            raise ConfigError("objective.reweighting_max_iterations must be >= 1")


@dataclass(frozen=True)
class GeometryConfig:
    """Affine basis discovery and the span certificate (M4)."""

    feasibility_tol: float = 1e-9
    span_tol: float = 1e-9
    """Width tolerance: an LP width below this is a *flat* direction, not a sampling dimension."""
    rank_tol: float = 1e-9
    """Residual-norm tolerance for admitting a direction into the basis (spec §15.3 step 10).

    Not tighter than `span_tol`, and deliberately not tighter than what the LP can deliver: HiGHS is
    *allowed* to return a point that misses feasibility by its own primal tolerance (1e-9), and a
    rank tolerance below the noise that licenses would call a perfectly good geometry inconclusive.
    `affine_geometry` checks that coupling explicitly rather than leaving it to luck.
    """
    dual_tol: float = 1e-7
    """Dual infeasibility above which a support LP's *width* is not to be believed.

    A width is a difference of two optima, so a solve that stopped short of optimality reports it
    too
    **small** — and a real dimension gets certified flat. Primal feasibility says nothing about
    this.
    `affine_geometry` sup-normalizes each support objective precisely so that this absolute bar is
    meaningful, and a probe that breaches it is inconclusive.
    """
    scale_floor: float = 1.0
    """Floor on the reaction scale ``s_i = u_i − l_i`` (spec §15.2).

    The scaled coordinates divide by ``s_i``, so they also divide the solver's feasibility error by
    ``s_i``: an unfloored range of 1e-12 would amplify a 1e-12 LP residual into a scaled coordinate
    of order 1. The floor bounds that amplification.

    It is not a free parameter. ``__post_init__`` requires ``scale_floor ≥ blocked_tol / span_tol``,
    which keeps the two resolutions the geometry uses from contradicting each other: a reaction
    blocked for spanning less than `blocked_tol` in **flux** has a scaled axis width of at most
    ``blocked_tol / scale_floor``, and unless that is ≤ `span_tol` the certificate sweep will turn
    around and report the very axis the blocked projection just removed — a direction it then cannot
    append, because the projection zeroed it. With the defaults the bound is exactly 1.0.
    """
    stall_probes: int = 24
    """Consecutive random probes that must find nothing before discovery hands off to the sweep."""
    blocked_tol: float = 1e-9
    """FVA range below which a *free* reaction is judged unable to carry flux at all.

    Such a reaction is an exact zero of the direction space, and `affine_geometry` projects it out
    of the basis — see that module on why a noise-valued basis row there corrupts the chord. The
    example model has 61 of them among its 260 free reactions. This is not a knob to tune: the
    widest blocked range there is 8e-10 and the narrowest moving one is 0.30, and geometry refuses
    to proceed when that separation is not wide enough to make the split unambiguous. The ranges are
    *upper* bounds from weak duality, never primal readings — a lower bound would be the wrong end
    entirely, since an LP that stopped short would report zero for a reaction that is wide open.
    """
    seed: int = 0
    max_geometry_memory_gb: float = 2.0
    exhaustive_span_certificate: bool = True
    """False downgrades this to a randomized partial check (§1.4), noted in the manifest."""
    max_span_probes: int | None = None
    """Cap on certificate probes. Any cap forces ``span_certificate_exhaustive = false``."""

    # ---- rounding (M5, spec §17) ---------------------------------------------------------------

    ridge_relative: float = 1e-6
    """Ridge ``ε`` on the support covariance, as a fraction of ``trace(C_q)/d`` (spec §17.2).

    Relative, never absolute: coordinate variances span orders of magnitude between models, and an
    absolute ridge would be a rounding error on one and the dominant term on another.

    **It cannot change the sampled distribution.** ``C_q + εI`` is positive definite for any
    ``ε > 0``, so ``L`` is invertible and ``range(T) = range(diag(s)·B)`` regardless — see
    `rounding`. What it sets is the *step scale along the thinnest measured direction*, which is
    ``√ε`` against a mean of 1. Too small and a near-flat direction gets a needle-thin axis the
    chain crawls along; too large and every genuinely thin direction is over-inflated, so the
    chord truncates the proposal and the step is wasted.

    Measured on the example model: the ridge at 1e-6 moves ``cond(C_ε)`` from 1.5659e4 to 1.5615e4
    — 0.3%. It is numerically inert there while still providing a floor, which is what a ridge is
    supposed to be.
    """
    ridge_growth: float = 10.0
    """Geometric escalation factor when Cholesky fails (spec §17.2)."""
    max_ridge_escalations: int = 12
    """Escalations before the covariance is called corrupt rather than merely ill-conditioned."""
    covariance_rank_tol: float = 1e-12
    """Eigenvalue cutoff for the rank of ``C_q``, relative to ``trace(C_q)/d``.

    Diagnostic only. The ridge makes Cholesky succeed either way, and a rank-deficient covariance
    is a *mixing* defect rather than a correctness one (`rounding` explains why). It is reported so
    that a chain crawling along a ridge-held direction is legible in the manifest, not invisible.
    """

    def __post_init__(self) -> None:
        if self.feasibility_tol <= 0.0 or self.span_tol <= 0.0 or self.rank_tol <= 0.0:
            raise ConfigError("geometry tolerances must be > 0")
        if self.blocked_tol <= 0.0 or self.dual_tol <= 0.0:
            raise ConfigError("geometry.blocked_tol and geometry.dual_tol must be > 0")
        if self.scale_floor <= 0.0:
            raise ConfigError("geometry.scale_floor must be > 0")
        required_floor = self.blocked_tol / self.span_tol
        if self.scale_floor < required_floor:
            raise ConfigError(
                f"geometry.scale_floor ({self.scale_floor:g}) must be >= blocked_tol/span_tol "
                f"({required_floor:g}), or the two resolutions contradict each other: a reaction "
                "blocked below blocked_tol would still show a scaled width above span_tol, and the "
                "span sweep would report a direction the blocked projection has already removed"
            )
        if self.rank_tol > self.span_tol:
            # width = pᵀ·Δx with ‖p‖ = 1 and p ⊥ B, so width ≤ ‖residual‖ always. A rank tolerance
            # above the width tolerance would therefore reject residuals the width test just
            # accepted — the two tests would disagree about a direction the LP proved is real.
            raise ConfigError(
                f"geometry.rank_tol ({self.rank_tol}) must be <= geometry.span_tol "
                f"({self.span_tol}): a width always lower-bounds its residual norm"
            )
        if self.stall_probes < 1:
            raise ConfigError("geometry.stall_probes must be >= 1")
        if self.max_geometry_memory_gb <= 0.0:
            raise ConfigError("geometry.max_geometry_memory_gb must be > 0")
        if self.max_span_probes is not None and self.max_span_probes < 1:
            raise ConfigError("geometry.max_span_probes must be >= 1 when set")
        if self.ridge_relative <= 0.0:
            # Zero is not "no ridge". It is a Cholesky that fails on a rank-deficient covariance
            # and then escalates from zero — which multiplies back to zero, forever.
            raise ConfigError("geometry.ridge_relative must be > 0")
        if self.ridge_growth <= 1.0:
            raise ConfigError("geometry.ridge_growth must be > 1 (it escalates the ridge)")
        if self.max_ridge_escalations < 1:
            raise ConfigError("geometry.max_ridge_escalations must be >= 1")
        if self.covariance_rank_tol <= 0.0:
            raise ConfigError("geometry.covariance_rank_tol must be > 0")


@dataclass(frozen=True)
class SamplerConfig:
    """The β-ladder and the MCMC schedule (M5/M6).

    **Every count here is in *sweeps*, and one sweep is ``d`` random-scan coordinate updates.** The
    scan is still random — each update draws its own coordinate — but counting single updates would
    make ``burn_in = 1000`` mean 21 passes over a 46-dimensional model and 1000 passes over a
    1-dimensional one. A schedule has to mean the same thing across a batch, or the cross-model
    comparison this package exists for is comparing chains of different lengths.
    """

    betas: tuple[float, ...] = (0.0,)
    """The ladder. ``maxent_sampler.SUGGESTED_BETA_LADDER`` holds spec §22.1's starting suggestion —
    which that section is careful *not* to call scientifically correct, and neither is this default.
    """
    n_chains: int = 4
    n_samples: int = 1000
    burn_in: int = 1000
    thin: int = 1
    refresh_interval: int = 1000
    """Sweeps between exact rebuilds of ``v`` from ``y`` — bounds incremental drift (§1.3)."""
    energy_scale: str | float = "warmup_range"
    """``s_J`` (spec §3.6). ``"warmup_range"`` sets it from the observed objective range of the
    support points; a positive number declares it, putting ``β`` in reciprocal raw-objective units.

    Only ``warmup_range`` makes ``β`` comparable across strains — see
    `sparse_objective.EnergyScale`.
    """
    energy_scale_quantile: float = 0.05
    """``q`` in ``s_J = J* − Q_q(J(W))`` (spec §22.2). Ignored when `energy_scale` is a number."""
    energy_scale_fallback: float | None = None
    """The scale to use if the warm-up range turns out unresolvable — ``None`` means **raise**.

    Spec §22.2 says to fall back on a "**declared** positive scale", and a library default is not a
    declaration. A degenerate range means every support vertex has essentially the LP-optimal
    objective, i.e. ``J`` barely varies over this polytope and *no* β would mean much. Silently
    substituting ``s_J = 1`` there would rescale every rung of this strain's ladder — so its β = 2
    would name a different selection pressure from every other strain's β = 2, which is the one
    thing
    ``s_J`` exists to prevent (BUILD_PLAN §1.1's cross-model comparison). It would arrive as a
    warning in a log nobody reads.

    So the default is to stop. A caller who wants to proceed anyway says so here, and the manifest
    records `energy_scale_fell_back = true` next to the number that was used.
    """
    seed: int = 0
    """Base entropy. Each chain's stream is keyed on ``(model_id, stage, β_index, chain_index)``
    on top of it (`provenance.stream_seed`) — never on a position in a `spawn()` sequence."""

    def __post_init__(self) -> None:
        if not self.betas:
            raise ConfigError("sampler.betas must list at least one β")
        if any(beta < 0.0 for beta in self.betas):
            raise ConfigError(f"sampler.betas must all be >= 0, got {list(self.betas)}")
        if any(not isfinite(beta) for beta in self.betas):
            raise ConfigError(f"sampler.betas must all be finite, got {list(self.betas)}")
        for name, value, minimum in (
            ("n_chains", self.n_chains, 1),
            ("n_samples", self.n_samples, 1),
            ("burn_in", self.burn_in, 0),
            ("thin", self.thin, 1),
            ("refresh_interval", self.refresh_interval, 1),
        ):
            if value < minimum:
                raise ConfigError(f"sampler.{name} must be >= {minimum}, got {value}")

        if isinstance(self.energy_scale, str):
            if self.energy_scale != "warmup_range":
                raise ConfigError(
                    "sampler.energy_scale must be \"warmup_range\" or a positive number, got "
                    f"{self.energy_scale!r}"
                )
        elif not isfinite(self.energy_scale) or self.energy_scale <= 0.0:
            # A zero or negative s_J does not merely rescale β — it flips or annihilates the tilt,
            # so the chain would sample exp(−βJ/|s_J|) or a flat law while reporting the β it was
            # asked for. `line_distribution` refuses it too; refusing it *here* means the run
            # dies at config time rather than several thousand sweeps in.
            raise ConfigError(
                f"sampler.energy_scale must be finite and > 0, got {self.energy_scale}"
            )
        if not 0.0 < self.energy_scale_quantile < 1.0:
            raise ConfigError(
                "sampler.energy_scale_quantile must lie strictly in (0, 1), got "
                f"{self.energy_scale_quantile}"
            )
        if self.energy_scale_fallback is not None and (
            not isfinite(self.energy_scale_fallback) or self.energy_scale_fallback <= 0.0
        ):
            raise ConfigError(
                "sampler.energy_scale_fallback must be finite and > 0 when set (or None to refuse "
                f"a degenerate warm-up range), got {self.energy_scale_fallback}"
            )


@dataclass(frozen=True)
class OutputConfig:
    """Where results land and how densely they are stored (§1.3)."""

    directory: str = "results"
    store_mode: str = "full_flux"
    """``full_flux`` keeps full-length vectors; ``reduced`` keeps y-states plus summaries."""
    store_flux_dtype: str = "float64"
    """Storage width only — computation stays float64 regardless (CLAUDE.md)."""

    def __post_init__(self) -> None:
        if self.store_mode not in {"full_flux", "reduced"}:
            raise ConfigError(
                f"output.store_mode must be 'full_flux' or 'reduced', got {self.store_mode!r}"
            )
        if self.store_flux_dtype not in {"float64", "float32"}:
            raise ConfigError(
                "output.store_flux_dtype must be 'float64' or 'float32', "
                f"got {self.store_flux_dtype!r}"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-pool sizing. One global pool spans all ``(model, β, chain)`` units (§1.2)."""

    n_workers: int = 4
    solver_threads: int = 1

    def __post_init__(self) -> None:
        if self.n_workers < 1:
            raise ConfigError("runtime.n_workers must be >= 1")
        if self.solver_threads != 1:
            raise ConfigError(
                "runtime.solver_threads must be 1: multi-threaded HiGHS makes the geometry "
                "nondeterministic (BUILD_PLAN §1.2)"
            )


@dataclass(frozen=True)
class Config:
    """The resolved configuration for one run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def as_dict(self) -> dict[str, Any]:
        """The resolved config, ready for the run manifest."""
        return asdict(self)

    def echo(self) -> str:
        """Render the resolved config as TOML, for logging and for the run manifest.

        TOML has no null, so an unset optional key is emitted as a **comment** rather than as an
        empty string. Writing ``biomass_id = ""`` would read back as a literal empty reaction ID —
        a value, not an absence. Commenting it out means the echo still reloads to the same config,
        which is the property the manifest depends on. (`as_dict` keeps the true ``None`` for JSON.)
        """
        lines: list[str] = []
        for section, values in self.as_dict().items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                if value is None:
                    lines.append(f"# {key} = (unset — uses the default)")
                else:
                    lines.append(f"{key} = {_to_toml(value)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# ---- loading -----------------------------------------------------------------------------------

_SECTION_TYPES: dict[str, type] = {
    "model": ModelConfig,
    "objective": ObjectiveConfig,
    "geometry": GeometryConfig,
    "sampler": SamplerConfig,
    "output": OutputConfig,
    "runtime": RuntimeConfig,
}


def _to_toml(value: Any) -> str:
    if value is None:
        raise ValueError("None has no TOML representation; `Config.echo` comments the key out")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_to_toml(item) for item in value) + "]"
    return repr(value)


def _build(cls: type[T], data: dict[str, Any], section: str) -> T:
    """Instantiate a config section from a mapping, rejecting keys it does not define."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {unknown}. Known keys: {sorted(known)}")

    # ``betas`` is the one tuple-typed field, and TOML hands us a list.
    coerced = {
        key: tuple(value) if isinstance(value, list) else value for key, value in data.items()
    }
    try:
        return cls(**coerced)
    except TypeError as error:
        raise ConfigError(f"invalid value in [{section}]: {error}") from error


def from_dict(data: dict[str, Any]) -> Config:
    """Build a `Config` from nested mappings, rejecting unknown sections and keys."""
    unknown = sorted(set(data) - set(_SECTION_TYPES))
    if unknown:
        raise ConfigError(f"unknown config section(s): {unknown}. Known: {sorted(_SECTION_TYPES)}")

    kwargs: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        if name not in data:
            continue
        values = data[name]
        if not isinstance(values, dict):
            raise ConfigError(f"section [{name}] must be a table, got {type(values).__name__}")
        kwargs[name] = _build(section_type, values, name)

    return Config(**kwargs)


def _parse_override_value(raw: str) -> Any:
    """Type an override's value the way TOML would, falling back to a bare string.

    So ``sampler.n_chains=8`` yields an int, ``sampler.betas=[0,1]`` a list, and
    ``model.path=x.json`` a string — without the caller having to quote anything.
    """
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Fold ``section.key=value`` CLI overrides into a config mapping."""
    merged: dict[str, Any] = {
        section: dict(values) if isinstance(values, dict) else values
        for section, values in data.items()
    }

    for override in overrides:
        dotted, separator, raw = override.partition("=")
        section, dot, key = dotted.strip().partition(".")
        if not separator or not section or not dot or not key:
            raise ConfigError(f"override {override!r} is not of the form section.key=value")
        if not isinstance(merged.get(section, {}), dict):
            raise ConfigError(f"cannot override into non-table section [{section}]")
        merged.setdefault(section, {})[key] = _parse_override_value(raw.strip())

    return merged


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a TOML config (or start from defaults), then apply CLI overrides.

    The result is fully resolved: what `Config.echo` prints is what the run used.
    """
    data: dict[str, Any] = {}
    if path is not None:
        source = Path(path)
        if not source.is_file():
            raise ConfigError(f"config file not found: {source}")
        data = tomllib.loads(source.read_text())

    if overrides:
        data = apply_overrides(data, overrides)

    return from_dict(data)
