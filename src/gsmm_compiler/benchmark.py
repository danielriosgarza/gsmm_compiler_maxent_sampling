"""The M9 benchmark suite: where the wall-clock actually goes, stage by stage.

**New module, not in spec §6** — like `provenance`, it earned its place by being needed. The M9 gate
asks for a *report*, and a report that only ever runs inside a test cannot be regenerated on a new
machine or a new strain, which is precisely when the numbers matter.

Three measurement decisions are load-bearing, and each exists because the naive alternative
reports a number that looks fine and means something else.

**1. The sweep rate is a slope, not a ratio.** `maxent_sampler.run_chain` pays a fixed cost — the
dispersed start (which shrinks until feasible), the sample buffers, the closing feasibility pass —
plus a per-sweep cost. Dividing total sweeps by total seconds blends the two, so the reported rate
would *rise* with the schedule it was measured at and flatter every long run. We time the same chain
at ``S`` and ``2S`` sweeps and take the slope; the intercept cancels exactly and is reported on its
own, because a batch pays it once per ``(model, β, chain)`` unit and that count is large.

**2. Every stage reports its raw repeats, and the headline is the median.** A mean over three
runs on a Jetson is one thermal event away from fiction. The median is reported, the spread is
reported next to it, and the raw seconds go in the JSON so a reader can see a bimodal stage rather
than infer a
clean one.

**3. The LP stages are split cold from warm.** `build_flux_lp` (which is where `passModel` happens)
and the first solve are *construction*; every later solve on the same `HighsLinearProgram` is a warm
start off the retained basis. M3 established that warm starts are what make the sequential geometry
phase affordable — this quantifies it rather than restating it.

The report is JSON (`BenchmarkReport.as_dict`) plus a human table (`format_report`). It records the
platform and library versions alongside the timings, because a benchmark without them is a number
without units.

Implemented in **M9** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler import __version__

T = TypeVar("T")

DEFAULT_SWEEPS = 200
"""Base sweep count for the sampling slope. The chain is also run at ``2 ×`` this."""

DEFAULT_WARM_LPS = 50
"""Random objectives re-solved on one warm `HighsLinearProgram`, matching M3's warm-start probe."""

DEFAULT_KERNEL_DRAWS = 2000
"""`sample_line` calls timed per kernel stage. Large enough that the loop overhead is amortized."""


class BenchmarkError(RuntimeError):
    """The benchmark could not be run as specified."""


@dataclass(frozen=True)
class StageTiming:
    """One pipeline stage's wall-clock, with the raw repeats kept rather than collapsed.

    ``n_units`` and ``unit`` turn seconds into a rate that survives a change of schedule: a stage
    that does 50 LPs reports seconds *per LP*, so re-running it with 100 does not look 2× slower.
    A stage with no natural unit leaves ``n_units`` at 1 and reports seconds.
    """

    name: str
    seconds: tuple[float, ...]
    n_units: int = 1
    unit: str = "call"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.seconds:
            raise BenchmarkError(f"stage {self.name!r} has no timings")
        if self.n_units < 1:
            raise BenchmarkError(f"stage {self.name!r} has n_units={self.n_units}")

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)

    @property
    def fastest(self) -> float:
        return min(self.seconds)

    @property
    def slowest(self) -> float:
        return max(self.seconds)

    @property
    def spread(self) -> float:
        """``(slowest − fastest) / median`` — the relative spread across repeats.

        Reported next to every median so a reader can tell a stable stage from one that happened to
        be measured through a thermal throttle. There is no threshold here; it is information.
        """
        median = self.median
        return (self.slowest - self.fastest) / median if median > 0.0 else 0.0

    @property
    def seconds_per_unit(self) -> float:
        return self.median / self.n_units

    @property
    def units_per_second(self) -> float:
        per_unit = self.seconds_per_unit
        return 1.0 / per_unit if per_unit > 0.0 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "median_seconds": self.median,
            "fastest_seconds": self.fastest,
            "slowest_seconds": self.slowest,
            "spread": self.spread,
            "n_repeats": len(self.seconds),
            "n_units": self.n_units,
            "unit": self.unit,
            "seconds_per_unit": self.seconds_per_unit,
            "units_per_second": self.units_per_second,
            "raw_seconds": list(self.seconds),
            "note": self.note,
        }


@dataclass(frozen=True)
class SweepRate:
    """The marginal cost of one sweep, separated from the fixed cost of starting a chain.

    Measured from two chains of ``n_sweeps_low`` and ``n_sweeps_high`` total sweeps. The slope is
    the per-sweep cost with the intercept cancelled; the intercept is what a `run_chain` call costs
    before it walks anywhere. Both matter and they are not interchangeable: a batch of 46 strains ×
    8 β × 4 chains pays the intercept 1472 times.

    ``seconds_per_sweep`` can come out non-positive if the two timings invert under noise. That is
    reported honestly (`is_resolved` false) rather than clamped, because a clamped slope would be a
    fabricated measurement, and the caller's remedy is more sweeps, not a floor.
    """

    beta: float
    dimension: int
    n_sweeps_low: int
    n_sweeps_high: int
    seconds_low: float
    seconds_high: float

    @property
    def seconds_per_sweep(self) -> float:
        return (self.seconds_high - self.seconds_low) / (self.n_sweeps_high - self.n_sweeps_low)

    @property
    def fixed_seconds(self) -> float:
        """The intercept: what a chain costs before the first sweep."""
        return self.seconds_low - self.seconds_per_sweep * self.n_sweeps_low

    @property
    def is_resolved(self) -> bool:
        """False when noise inverted the two points, so the slope means nothing."""
        return self.seconds_per_sweep > 0.0

    @property
    def sweeps_per_second(self) -> float:
        per_sweep = self.seconds_per_sweep
        return 1.0 / per_sweep if per_sweep > 0.0 else float("nan")

    @property
    def updates_per_second(self) -> float:
        """Single coordinate updates per second. One sweep is ``d`` of them."""
        return self.sweeps_per_second * self.dimension

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "dimension": self.dimension,
            "n_sweeps_low": self.n_sweeps_low,
            "n_sweeps_high": self.n_sweeps_high,
            "seconds_low": self.seconds_low,
            "seconds_high": self.seconds_high,
            "seconds_per_sweep": self.seconds_per_sweep,
            "sweeps_per_second": self.sweeps_per_second,
            "updates_per_second": self.updates_per_second,
            "fixed_seconds": self.fixed_seconds,
            "is_resolved": self.is_resolved,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Every stage's timing plus the model size and platform that produced them."""

    model_id: str
    n_reactions: int
    n_metabolites: int
    n_free: int
    dimension: int
    stages: tuple[StageTiming, ...]
    sweep_rates: tuple[SweepRate, ...] = ()
    platform_info: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> StageTiming:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"no stage named {name!r}; have {[s.name for s in self.stages]}")

    def rate_at(self, beta: float) -> SweepRate:
        for rate in self.sweep_rates:
            if rate.beta == beta:
                return rate
        raise KeyError(f"no sweep rate at β={beta}; have {[r.beta for r in self.sweep_rates]}")

    @property
    def total_seconds(self) -> float:
        """Summed medians. A budget for one model's stages, not a wall-clock of the whole run."""
        return sum(stage.median for stage in self.stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_reactions": self.n_reactions,
            "n_metabolites": self.n_metabolites,
            "n_free": self.n_free,
            "dimension": self.dimension,
            "total_seconds": self.total_seconds,
            "stages": [stage.as_dict() for stage in self.stages],
            "sweep_rates": [rate.as_dict() for rate in self.sweep_rates],
            "platform": self.platform_info,
            "notes": self.notes,
        }


def _version(distribution: str) -> str:
    """A distribution's version from the installed metadata.

    Not ``module.__version__``: `highspy` does not define one, so reading the attribute reports
    ``"unknown"`` for the single most performance-relevant dependency in the package. The
    metadata is authoritative and is what `uv.lock` pins.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover - the venv always has these
        return "unknown"


def platform_info() -> dict[str, Any]:
    """Platform + library versions. A timing without these is a number without units."""
    return {
        "gsmm_compiler": __version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "numpy": _version("numpy"),
        "cobra": _version("cobra"),
        "highspy": _version("highspy"),
        "blas_threads_env": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def time_repeats(fn: Callable[[], T], repeats: int) -> tuple[list[float], T]:
    """Run ``fn`` ``repeats`` times, returning every elapsed time and the last result.

    `time.perf_counter` rather than `time.process_time`: we are measuring what a user waits for,
    and the LP stages spend real time inside HiGHS.

    ``repeats >= 1`` is checked up front, which is what makes the last result well-defined. It is
    *not* asserted to be non-``None`` afterwards: several stages here are timed for their effect and
    return nothing, and a guard that cannot tell "no result" from "a result that is None" would
    reject them.
    """
    if repeats < 1:
        raise BenchmarkError(f"repeats must be >= 1, got {repeats}")
    seconds: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - start)
    return seconds, cast("T", result)


def benchmark_pipeline(
    model_path: str | Path,
    *,
    biomass_id: str | None = None,
    config: Any = None,
    repeats: int = 1,
    cheap_repeats: int = 3,
    sweeps: int = DEFAULT_SWEEPS,
    n_warm_lps: int = DEFAULT_WARM_LPS,
    kernel_draws: int = DEFAULT_KERNEL_DRAWS,
    seed: int = 0,
) -> BenchmarkReport:
    """Time every stage of the pipeline on one model, end to end.

    ``repeats`` applies to the expensive stages (geometry, the sparse LP, the chains); the cheap
    ones
    (parse, CSC, reduce, rounding) get ``cheap_repeats``, since their variance is what needs
    resolving and their cost does not.

    Stages, in the order BUILD_PLAN M9 names them: parse → CSC → reduce → passModel → first LP →
    warm-start LPs → sparse LP → geometry → rounding → β=0 sweeps → β>0 sweeps → breakpoint kernel →
    output. Nothing here is cached: the point is the cold cost.
    """
    from gsmm_compiler.affine_geometry import build_geometry
    from gsmm_compiler.config import Config, ObjectiveConfig
    from gsmm_compiler.model_input import build_canonical_model, load_model
    from gsmm_compiler.rounding import build_transform
    from gsmm_compiler.sparse_objective import (
        build_flux_lp,
        choose_energy_scale,
        lower_objective,
        resolve_objective,
        solve_sparse_objective,
    )

    resolved_config: Any = Config() if config is None else config
    path = Path(model_path)
    if not path.exists():
        raise BenchmarkError(f"model file not found: {path}")

    stages: list[StageTiming] = []

    # ── L0: parse → CSC → reduce ────────────────────────────────────────────────────────────────
    parse_seconds, cobra_model = time_repeats(lambda: load_model(path), cheap_repeats)
    stages.append(
        StageTiming("parse", tuple(parse_seconds), note="cobra JSON load; no CSC assembly yet")
    )

    csc_seconds, canonical = time_repeats(
        lambda: build_canonical_model(cobra_model, source_path=path, biomass_id=biomass_id),
        cheap_repeats,
    )
    stages.append(
        StageTiming(
            "csc_assembly",
            tuple(csc_seconds),
            note="validation + native int32/float64 CSC + L0 content key",
        )
    )

    reduce_seconds, reduced = time_repeats(canonical.polytope.reduce, cheap_repeats)
    stages.append(
        StageTiming("reduce", tuple(reduce_seconds), note="eliminate l==u; build the affine RHS")
    )

    # ── L2: LPs. Construction (passModel) and the cold first solve are separated from the warm
    # re-solves, because they are different costs with different scaling and M3's warm-start result
    # is about the difference. ───────────────────────────────────────────────────────────────────
    pass_model_seconds, _ = time_repeats(lambda: build_flux_lp(reduced), cheap_repeats)
    stages.append(
        StageTiming(
            "pass_model",
            tuple(pass_model_seconds),
            note="HighsLp construction + passModel; no solve",
        )
    )

    def _cold_solve() -> Any:
        program = build_flux_lp(reduced)
        return program.solve()

    first_lp_seconds, _ = time_repeats(_cold_solve, cheap_repeats)
    stages.append(
        StageTiming(
            "first_lp",
            tuple(first_lp_seconds),
            note="build + cold solve, no basis to start from",
        )
    )

    warm_program = build_flux_lp(reduced)
    warm_program.solve()  # the cold one, excluded from the timing below
    rng = np.random.default_rng(seed)
    warm_costs = [rng.standard_normal(warm_program.n_cols) for _ in range(n_warm_lps)]

    def _warm_solves() -> None:
        for costs in warm_costs:
            warm_program.maximize(costs)

    warm_seconds, _ = time_repeats(_warm_solves, repeats)
    stages.append(
        StageTiming(
            "warm_start_lps",
            tuple(warm_seconds),
            n_units=n_warm_lps,
            unit="LP",
            note=f"{n_warm_lps} random objectives re-solved off the retained basis",
        )
    )

    objective_config = getattr(resolved_config, "objective", ObjectiveConfig())
    resolved = resolve_objective(canonical.polytope, reduced, objective_config)

    sparse_seconds, solution = time_repeats(
        lambda: solve_sparse_objective(reduced, resolved.objective), repeats
    )
    stages.append(
        StageTiming(
            "sparse_lp",
            tuple(sparse_seconds),
            note="the (v,z) sparse-objective LP + the biomass-only diagnostic LP",
        )
    )

    # ── L3: geometry + rounding. The expensive stage, and the only one M8 caches. ───────────────
    geometry_config = getattr(resolved_config, "geometry", None)
    geometry_seconds, geometry = time_repeats(
        lambda: build_geometry(reduced, model_id=canonical.model_id, config=geometry_config),
        repeats,
    )
    stages.append(
        StageTiming(
            "geometry",
            tuple(geometry_seconds),
            note="FVA + support LPs + orthonormal basis + span certificate (cold; cache bypassed)",
        )
    )

    rounding_seconds, transform = time_repeats(
        lambda: build_transform(geometry, reduced, config=geometry_config), cheap_repeats
    )
    stages.append(
        StageTiming(
            "rounding",
            tuple(rounding_seconds),
            note="support covariance → ridge → Cholesky → T=diag(s)·B·L + precompute",
        )
    )

    # ── Sampling. Rates are slopes (see the module docstring), so each β is two chains. ─────────
    lowered = lower_objective(reduced, resolved.objective)
    scale = choose_energy_scale(
        lowered,
        geometry.support_points,
        optimum=solution.optimum,
        warmup_polytope_key=reduced.content_key(),
        mode="warmup_range",
    )
    optimum_coordinates = transform.to_coordinates(reduced.to_reduced(solution.optimum.v_full))

    sweep_rates: list[SweepRate] = []
    for beta_index, beta in ((0, 0.0), (1, 4.0)):
        rate, low_stage, high_stage = _measure_sweep_rate(
            transform=transform,
            reduced=reduced,
            model_id=canonical.model_id,
            beta=beta,
            beta_index=beta_index,
            objective=lowered.line if beta > 0.0 else None,
            energy_scale=scale.value,
            sweeps=sweeps,
            optimum_coordinates=optimum_coordinates,
            repeats=repeats,
        )
        sweep_rates.append(rate)
        stages.extend((low_stage, high_stage))

    # ── The line kernel itself, isolated from the walk: uniform draw vs the piecewise-exp one. ──
    stages.extend(
        _benchmark_kernel(
            transform=transform,
            reduced=reduced,
            objective=lowered.line,
            energy_scale=scale.value,
            draws=kernel_draws,
            seed=seed,
        )
    )

    # ── Output: both storage modes, on a real chain. ────────────────────────────────────────────
    stages.extend(
        _benchmark_output(
            transform=transform,
            reduced=reduced,
            objective=lowered,
            model_id=canonical.model_id,
            energy_scale=scale.value,
            repeats=cheap_repeats,
        )
    )

    return BenchmarkReport(
        model_id=canonical.model_id,
        n_reactions=len(canonical.polytope.reaction_ids),
        n_metabolites=len(canonical.polytope.metabolite_ids),
        n_free=reduced.n_free,
        dimension=geometry.dimension,
        stages=tuple(stages),
        sweep_rates=tuple(sweep_rates),
        platform_info=platform_info(),
        notes={
            "l1_penalty_scaled": resolved.scale.l1_penalty_scaled,
            "l1_penalty": resolved.scale.l1_penalty,
            "critical_l1_penalty": resolved.scale.critical_l1_penalty,
            "energy_scale": scale.value,
            "step_scale_ratio": transform.diagnostics.step_scale_ratio,
            "condition_number": transform.diagnostics.condition_number,
            "n_blocked": geometry.manifest()["n_blocked"],
            "sweep_definition": "one sweep = d coordinate updates",
            "cache": "bypassed; every stage is a cold cost",
        },
    )


def _measure_sweep_rate(
    *,
    transform: Any,
    reduced: Any,
    model_id: str,
    beta: float,
    beta_index: int,
    objective: Any,
    energy_scale: float,
    sweeps: int,
    optimum_coordinates: NDArray[np.float64] | None,
    repeats: int,
) -> tuple[SweepRate, StageTiming, StageTiming]:
    """Time one chain at ``2·sweeps`` and ``4·sweeps`` total sweeps; return the slope + both stages.

    The two schedules keep the *same* 50/50 split of storing to non-storing sweeps as a production
    ``burn_in == n_samples`` run, so the slope is the cost of a production sweep and not of a
    cheaper one that never writes a sample.
    """
    from gsmm_compiler.config import SamplerConfig
    from gsmm_compiler.maxent_sampler import run_chain

    def _chain(n: int) -> Callable[[], Any]:
        config = SamplerConfig(
            n_chains=1,
            n_samples=n,
            burn_in=n,
            thin=1,
            refresh_interval=max(n // 4, 1),
        )

        def run() -> Any:
            return run_chain(
                transform,
                reduced,
                config=config,
                model_id=model_id,
                chain_index=0,
                beta=beta,
                beta_index=beta_index,
                objective=objective,
                energy_scale=energy_scale,
                optimum_coordinates=optimum_coordinates,
            )

        return run

    label = f"beta{beta:g}".replace(".", "p")
    low_seconds, _ = time_repeats(_chain(sweeps), repeats)
    high_seconds, _ = time_repeats(_chain(2 * sweeps), repeats)

    low = StageTiming(
        f"sample_{label}_low",
        tuple(low_seconds),
        n_units=2 * sweeps,
        unit="sweep",
        note=f"β={beta:g}, {2 * sweeps} sweeps incl. fixed chain cost",
    )
    high = StageTiming(
        f"sample_{label}_high",
        tuple(high_seconds),
        n_units=4 * sweeps,
        unit="sweep",
        note=f"β={beta:g}, {4 * sweeps} sweeps incl. fixed chain cost",
    )
    rate = SweepRate(
        beta=beta,
        dimension=transform.dimension,
        n_sweeps_low=2 * sweeps,
        n_sweeps_high=4 * sweeps,
        seconds_low=low.median,
        seconds_high=high.median,
    )
    return rate, low, high


def _benchmark_kernel(
    *,
    transform: Any,
    reduced: Any,
    objective: Any,
    energy_scale: float,
    draws: int,
    seed: int,
) -> list[StageTiming]:
    """Time `sample_line` alone — the β=0 uniform path against the β>0 piecewise-exponential one.

    Isolating the kernel from the walk answers the question the chain-level rates cannot: how much
    of a β>0 sweep is the tilt, and how much is the chord and the incremental update it shares with
    β=0. Both paths run on chords taken from the real transform at the real centre, so the
    breakpoint counts are the ones production sees, not a synthetic best case.
    """
    from gsmm_compiler.line_distribution import build_piecewise_j, sample_line

    precompute = transform.precompute
    d = transform.dimension
    v = np.ascontiguousarray(transform.to_flux(np.zeros(d, dtype=np.float64)), dtype=np.float64)
    columns = tuple(transform.transform[:, k] for k in range(d))

    from gsmm_compiler.line_geometry import chord_on_support

    chords = [
        chord_on_support(
            v[precompute.support[k]], precompute.direction[k], precompute.lower[k],
            precompute.upper[k],
        )
        for k in range(d)
    ]
    samplable = [k for k in range(d) if chords[k].is_samplable]
    if not samplable:
        raise BenchmarkError("no samplable chord at the centre; the geometry is degenerate")

    def _draw(beta: float, obj: Any) -> Callable[[], None]:
        def run() -> None:
            rng = np.random.default_rng(seed)
            for i in range(draws):
                k = samplable[i % len(samplable)]
                sample_line(v, columns[k], chords[k], obj, beta, energy_scale, rng)

        return run

    def _build_only() -> None:
        for i in range(draws):
            k = samplable[i % len(samplable)]
            build_piecewise_j(v, columns[k], chords[k], objective)

    uniform_seconds, _ = time_repeats(_draw(0.0, None), 3)
    tilted_seconds, _ = time_repeats(_draw(4.0, objective), 3)
    breakpoint_seconds, _ = time_repeats(_build_only, 3)

    # `knots` spans the chord end to end, so the interior bends — the ones the sort orders and the
    # ones that cost anything — are `knots.size - 2`.
    n_breaks = [
        build_piecewise_j(v, columns[k], chords[k], objective).knots.size - 2 for k in samplable
    ]

    return [
        StageTiming(
            "kernel_uniform",
            tuple(uniform_seconds),
            n_units=draws,
            unit="draw",
            note="sample_line at β=0: chord → uniform draw, no piecewise machinery",
        ),
        StageTiming(
            "kernel_tilted",
            tuple(tilted_seconds),
            n_units=draws,
            unit="draw",
            note="sample_line at β=4: breakpoints → piecewise J → segment masses → inverse CDF",
        ),
        StageTiming(
            "kernel_breakpoints",
            tuple(breakpoint_seconds),
            n_units=draws,
            unit="build",
            note=(
                f"build_piecewise_j alone (the sort); median {int(np.median(n_breaks))} "
                f"breakpoints/chord, max {max(n_breaks)}"
            ),
        ),
    ]


def _benchmark_output(
    *,
    transform: Any,
    reduced: Any,
    objective: Any,
    model_id: str,
    energy_scale: float,
    repeats: int,
) -> list[StageTiming]:
    """Time `store_chain` in each storage mode on one real chain.

    ``reduced`` mode writes ``(n_samples, d)``; ``full_flux`` lifts to ``(n_samples, n_full)``. On a
    genome-scale model that is 46 columns against 773 — a 17× ratio in the array, which this stage
    turns from an arithmetic claim into a measured one.
    """
    import tempfile

    from gsmm_compiler.config import SamplerConfig
    from gsmm_compiler.maxent_sampler import run_chain, trace_objective
    from gsmm_compiler.output import SampleStorage, store_chain

    config = SamplerConfig(n_chains=1, n_samples=200, burn_in=50, thin=1, refresh_interval=25)
    chain = run_chain(
        transform,
        reduced,
        config=config,
        model_id=model_id,
        chain_index=0,
        beta=0.0,
        beta_index=0,
    )
    trace = trace_objective(
        chain.fluxes, objective, j_star=0.0, energy_scale=energy_scale
    )

    modes = (
        SampleStorage(mode="full_flux", flux_dtype="float64"),
        SampleStorage(mode="full_flux", flux_dtype="float32"),
        SampleStorage(mode="reduced", flux_dtype="float64"),
    )
    stages: list[StageTiming] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, storage in enumerate(modes):
            def _store(storage: Any = storage, index: int = index) -> None:
                directory = root / f"store_{index}_{time.perf_counter_ns()}"
                store_chain(
                    directory,
                    chain=chain,
                    trace=trace,
                    reduced=reduced,
                    model_id=model_id,
                    beta=0.0,
                    beta_index=0,
                    chain_index=0,
                    storage=storage,
                )

            seconds, _ = time_repeats(_store, repeats)
            suffix = storage.mode if storage.mode == "reduced" else f"{storage.flux_dtype}"
            stages.append(
                StageTiming(
                    f"output_{suffix}",
                    tuple(seconds),
                    n_units=config.n_samples,
                    unit="sample",
                    note=f"store_chain, mode={storage.mode} dtype={storage.flux_dtype}, atomic",
                )
            )
    return stages


def format_report(report: BenchmarkReport) -> str:
    """A human table. The JSON is authoritative; this is what a reader actually looks at."""
    lines = [
        f"benchmark — {report.model_id}",
        f"  {report.n_reactions} reactions · {report.n_metabolites} metabolites · "
        f"{report.n_free} free · d = {report.dimension}",
        f"  {report.platform_info.get('system', '?')} {report.platform_info.get('machine', '?')} · "
        f"python {report.platform_info.get('python', '?')} · "
        f"numpy {report.platform_info.get('numpy', '?')} · "
        f"highspy {report.platform_info.get('highspy', '?')}",
        "",
        f"{'stage':<24} {'median':>10} {'spread':>8} {'per unit':>12}  unit",
        "─" * 78,
    ]
    for stage in report.stages:
        per_unit = (
            f"{stage.seconds_per_unit * 1e6:.1f} µs"
            if stage.seconds_per_unit < 1e-3
            else f"{stage.seconds_per_unit * 1e3:.2f} ms"
        )
        lines.append(
            f"{stage.name:<24} {stage.median:>9.4f}s {stage.spread:>7.1%} {per_unit:>12}  "
            f"{stage.unit}"
        )
    lines.append("─" * 78)
    lines.append(f"{'sum of medians':<24} {report.total_seconds:>9.4f}s")

    if report.sweep_rates:
        lines.extend(["", "sweep rates (slope of two schedules; the chain intercept is removed)"])
        for rate in report.sweep_rates:
            if not rate.is_resolved:
                lines.append(f"  β={rate.beta:<5g} UNRESOLVED — noise inverted the two timings")
                continue
            lines.append(
                f"  β={rate.beta:<5g} {rate.sweeps_per_second:>8.1f} sweeps/s · "
                f"{rate.updates_per_second:>10.0f} updates/s · "
                f"fixed {rate.fixed_seconds * 1e3:.1f} ms/chain"
            )
        if len(report.sweep_rates) == 2 and all(r.is_resolved for r in report.sweep_rates):
            base, tilted = report.sweep_rates
            ratio = tilted.seconds_per_sweep / base.seconds_per_sweep
            lines.append(f"  β>0 costs {ratio:.2f}× a β=0 sweep")
    return "\n".join(lines) + "\n"


def write_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write the report JSON atomically, alongside the human table."""
    from gsmm_compiler.output import write_json

    destination = Path(path)
    write_json(destination, report.as_dict())
    return destination


def benchmark_worker_sweep(
    specs: Sequence[Any],
    config: Any,
    *,
    worker_counts: Sequence[int],
    output_root: str | Path,
    batch_name: str = "worker_sweep",
) -> list[dict[str, Any]]:
    """Run the same batch at each worker count and report ESS per wall-second.

    **Wall-clock alone is the wrong axis and ESS alone is the wrong axis.** A batch that finishes
    fast having mixed badly is not faster at anything a user wants; ESS is the currency the sampler
    actually produces, and wall-seconds is what it costs. So the ratio is what ranks a worker count.

    Every run writes to its own batch directory and passes ``cache_dir=None``, so no run is handed
    another's geometry: the comparison is between identical amounts of work. M8 established that the
    fluxes are byte-identical across worker counts — so ESS is expected to be *constant* here, and
    the ratio is really wall-clock in disguise. That is the point: if ESS moves at all, the
    determinism guarantee has broken, and this sweep would be the thing that saw it.

    **A failed strain is fatal here, and that is a deliberate departure from `run_batch`.** M8 makes
    a failure recorded-but-not-fatal, which is right for a batch of 46 organisms where one bad model
    must not sink the run. It is wrong for a *measurement*: this function inherited that tolerance
    and, when every strain failed, produced a full table of zeros — ``total_ess_j = 0.0`` beside a
    wall-clock that still looked like a plausible speedup curve. Nothing in the table said "nothing
    ran". A benchmark's whole job is to be believed, so a sweep that did not sample refuses to
    report instead.
    """
    from gsmm_compiler.batch import run_batch
    from gsmm_compiler.diagnostics import convergence_report
    from gsmm_compiler.output import RunLayout, load_traces

    results: list[dict[str, Any]] = []
    for n_workers in worker_counts:
        root = Path(output_root) / f"workers_{n_workers}"
        start = time.perf_counter()
        batch = run_batch(
            list(specs),
            config,
            batch_name=batch_name,
            output_root=root,
            cache_dir=None,
            n_workers=n_workers,
        )
        elapsed = time.perf_counter() - start

        failed = [outcome for outcome in batch.outcomes if outcome.status != "complete"]
        if failed:
            detail = "; ".join(
                f"{outcome.model_id}: {(outcome.error or '').splitlines()[0]}" for outcome in failed
            )
            raise BenchmarkError(
                f"the worker sweep at n_workers={n_workers} had {len(failed)} of "
                f"{len(batch.outcomes)} strains fail, so there is no throughput to measure — "
                f"{detail}"
            )

        # ESS is taken on **J**, not on the 46 coordinates. J is the scalar the study reports and
        # the ladder is judged on, and summing ESS over coordinates would let a worker count look
        # productive by mixing well in directions nobody asked about.
        layout = RunLayout(root=root, batch=batch_name)
        total_ess = 0.0
        n_units = 0
        for outcome in batch.outcomes:
            for beta_index in range(len(config.sampler.betas)):
                draws = []
                for chain_index in range(config.sampler.n_chains):
                    directory = layout.chain_dir(outcome.model_id, beta_index, chain_index)
                    draws.append(load_traces(directory)["j"])
                    n_units += 1
                report = convergence_report(np.stack(draws)[:, :, None])
                total_ess += float(np.sum(report.ess))

        if total_ess <= 0.0 or n_units == 0:
            raise BenchmarkError(
                f"the worker sweep at n_workers={n_workers} completed {n_units} units for a total "
                f"ESS(J) of {total_ess} — there is nothing to divide by wall-clock"
            )

        results.append(
            {
                "n_workers": n_workers,
                "wall_seconds": elapsed,
                "total_ess_j": total_ess,
                "ess_per_wall_second": total_ess / elapsed,
                "n_units": n_units,
                "n_models": len(batch.outcomes),
            }
        )
    return results
