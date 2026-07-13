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
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """The configuration is malformed, has unknown keys, or violates a stated constraint."""


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
    seed: int = 0
    max_geometry_memory_gb: float = 2.0
    exhaustive_span_certificate: bool = True
    """False downgrades this to a randomized partial check (§1.4), noted in the manifest."""
    max_span_probes: int | None = None
    """Cap on certificate probes. Any cap forces ``span_certificate_exhaustive = false``."""

    def __post_init__(self) -> None:
        if self.feasibility_tol <= 0.0 or self.span_tol <= 0.0:
            raise ConfigError("geometry tolerances must be > 0")
        if self.max_geometry_memory_gb <= 0.0:
            raise ConfigError("geometry.max_geometry_memory_gb must be > 0")
        if self.max_span_probes is not None and self.max_span_probes < 1:
            raise ConfigError("geometry.max_span_probes must be >= 1 when set")


@dataclass(frozen=True)
class SamplerConfig:
    """The β-ladder and the MCMC schedule (M5/M6)."""

    betas: tuple[float, ...] = (0.0,)
    n_chains: int = 4
    n_samples: int = 1000
    burn_in: int = 1000
    thin: int = 1
    refresh_interval: int = 1000
    """Steps between exact rebuilds of ``v`` from ``y`` — bounds incremental-update drift (§1.3)."""

    def __post_init__(self) -> None:
        if not self.betas:
            raise ConfigError("sampler.betas must list at least one β")
        if any(beta < 0.0 for beta in self.betas):
            raise ConfigError(f"sampler.betas must all be >= 0, got {list(self.betas)}")
        for name, value, minimum in (
            ("n_chains", self.n_chains, 1),
            ("n_samples", self.n_samples, 1),
            ("burn_in", self.burn_in, 0),
            ("thin", self.thin, 1),
            ("refresh_interval", self.refresh_interval, 1),
        ):
            if value < minimum:
                raise ConfigError(f"sampler.{name} must be >= {minimum}, got {value}")


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
