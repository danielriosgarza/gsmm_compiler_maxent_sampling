"""The M9 benchmark harness + the `reduced` storage-mode validation.

The harness's own measurement logic is tested here on the toy — the *numbers* are a property of the
machine and nothing can assert them, but the arithmetic that turns seconds into a rate is ours and
is wrong in ways a report would not reveal. The slope, the intercept, the median, the vacuous-stage
guard: each is checked against a synthetic timing whose right answer is known.

The `reduced` storage mode gets its end-to-end validation here rather than in `tests/integration`
because M9 is where it was scoped, and because what needed proving turned out to be a performance
claim as much as a correctness one: it is worth having only if it is smaller, and worth *trusting*
only if what comes back is the same flux.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gsmm_compiler.benchmark import (
    BenchmarkError,
    BenchmarkReport,
    StageTiming,
    SweepRate,
    benchmark_pipeline,
    format_report,
    platform_info,
    time_repeats,
    write_report,
)

# ── The timing primitives ───────────────────────────────────────────────────────────────────────


def test_time_repeats_runs_exactly_n_times_and_returns_the_last_result() -> None:
    calls: list[int] = []

    def fn() -> int:
        calls.append(len(calls))
        return len(calls)

    seconds, result = time_repeats(fn, 4)
    assert len(seconds) == 4
    assert len(calls) == 4
    assert result == 4
    assert all(second >= 0.0 for second in seconds)


def test_time_repeats_accepts_a_stage_that_returns_none() -> None:
    """The regression the toy smoke-test caught: several stages are timed for effect only.

    An earlier version asserted the result was not ``None`` as a stand-in for "the loop ran", which
    cannot distinguish *no result* from *a result that is None* and rejected every void stage.
    ``repeats >= 1`` is what makes the result well-defined, and it is checked up front.
    """
    seconds, result = time_repeats(lambda: None, 2)
    assert len(seconds) == 2
    assert result is None


def test_time_repeats_refuses_zero_repeats() -> None:
    with pytest.raises(BenchmarkError, match="repeats must be >= 1"):
        time_repeats(lambda: None, 0)


# ── StageTiming: the seconds → rate arithmetic ──────────────────────────────────────────────────


def test_stage_timing_reports_the_median_not_the_mean() -> None:
    """A mean over three runs on a thermally-throttling Jetson is one event away from fiction."""
    stage = StageTiming("s", (1.0, 2.0, 100.0))
    assert stage.median == 2.0
    assert stage.fastest == 1.0
    assert stage.slowest == 100.0


def test_stage_timing_spread_exposes_an_unstable_stage() -> None:
    stable = StageTiming("stable", (1.0, 1.01, 1.02))
    noisy = StageTiming("noisy", (1.0, 2.0, 9.0))
    assert stable.spread < 0.05
    assert noisy.spread > 3.0


def test_stage_timing_normalizes_by_unit_so_a_longer_run_is_not_slower() -> None:
    """50 LPs in 1 s and 100 LPs in 2 s are the same speed; the per-unit rate must say so."""
    fifty = StageTiming("lp", (1.0,), n_units=50, unit="LP")
    hundred = StageTiming("lp", (2.0,), n_units=100, unit="LP")
    assert fifty.seconds_per_unit == pytest.approx(hundred.seconds_per_unit)
    assert fifty.units_per_second == pytest.approx(50.0)


def test_stage_timing_refuses_a_stage_with_no_timings() -> None:
    with pytest.raises(BenchmarkError, match="no timings"):
        StageTiming("empty", ())


# ── SweepRate: the slope that removes the chain's fixed cost ────────────────────────────────────


def test_sweep_rate_recovers_a_known_slope_and_intercept() -> None:
    """The whole point of two schedules: a synthetic 2 ms/sweep + 500 ms fixed must come back exact.

    Constructed so the naive ``total/elapsed`` answer is visibly wrong — at 100 sweeps it would
    report 143 sweeps/s against a true 500, because the 500 ms intercept is 71% of that run.
    """
    seconds_per_sweep, fixed = 0.002, 0.5
    rate = SweepRate(
        beta=0.0,
        dimension=46,
        n_sweeps_low=100,
        n_sweeps_high=200,
        seconds_low=fixed + seconds_per_sweep * 100,
        seconds_high=fixed + seconds_per_sweep * 200,
    )
    assert rate.seconds_per_sweep == pytest.approx(seconds_per_sweep)
    assert rate.fixed_seconds == pytest.approx(fixed)
    assert rate.sweeps_per_second == pytest.approx(500.0)
    assert rate.updates_per_second == pytest.approx(500.0 * 46)
    assert rate.is_resolved

    naive = 100 / rate.seconds_low
    assert naive < 150.0, "the naive ratio should be badly wrong here, or this test proves nothing"


def test_sweep_rate_reports_an_inverted_measurement_rather_than_clamping_it() -> None:
    """Noise can invert the two points. A clamped slope would be a fabricated measurement.

    The honest report is ``is_resolved = False`` and a NaN rate; the caller's remedy is more sweeps,
    not a floor. Same principle as M4's refusal to certify flatness from a primal width.
    """
    rate = SweepRate(
        beta=0.0, dimension=46, n_sweeps_low=100, n_sweeps_high=200,
        seconds_low=1.0, seconds_high=0.9,
    )
    assert not rate.is_resolved
    assert rate.seconds_per_sweep < 0.0
    assert np.isnan(rate.sweeps_per_second)


# ── The report ──────────────────────────────────────────────────────────────────────────────────


def test_platform_info_resolves_highspy_from_metadata_not_a_module_attribute() -> None:
    """highspy defines no ``__version__``; reading the attribute reports "unknown" for the single
    most performance-relevant dependency in the package. The installed metadata is authoritative.
    """
    info = platform_info()
    assert info["highspy"] != "unknown"
    assert info["highspy"][0].isdigit()
    assert info["cobra"] != "unknown"
    assert info["numpy"] != "unknown"


def test_report_round_trips_through_json_and_formats(tmp_path: Path) -> None:
    report = BenchmarkReport(
        model_id="m",
        n_reactions=7,
        n_metabolites=3,
        n_free=5,
        dimension=2,
        stages=(StageTiming("parse", (0.1, 0.2)), StageTiming("geometry", (1.0,))),
        sweep_rates=(
            SweepRate(
                beta=0.0, dimension=2, n_sweeps_low=10, n_sweeps_high=20,
                seconds_low=0.2, seconds_high=0.3,
            ),
        ),
        platform_info=platform_info(),
    )
    assert report.stage("geometry").median == 1.0
    assert report.rate_at(0.0).is_resolved
    assert report.total_seconds == pytest.approx(0.15 + 1.0)

    destination = write_report(report, tmp_path / "b.json")
    from gsmm_compiler.output import read_json

    loaded = read_json(destination)
    assert loaded["model_id"] == "m"
    assert loaded["stages"][0]["raw_seconds"] == [0.1, 0.2]

    text = format_report(report)
    assert "parse" in text and "geometry" in text and "sweeps/s" in text


def test_report_raises_on_an_unknown_stage() -> None:
    report = BenchmarkReport("m", 1, 1, 1, 1, stages=(StageTiming("parse", (0.1,)),))
    with pytest.raises(KeyError, match="no stage named"):
        report.stage("nope")


# ── The harness end to end ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_pipeline_covers_every_stage_the_gate_names(toy_path: Path) -> None:
    """Every stage BUILD_PLAN M9 lists is present, timed, and positive.

    Run on the toy so it is affordable in the suite; the genome-scale report is produced by the CLI
    and checked into ``benchmarks/``. What this asserts is that the harness *drives* the whole
    pipeline — a stage silently missing from the report is exactly the failure a report cannot show.
    """
    report = benchmark_pipeline(
        toy_path, repeats=1, cheap_repeats=2, sweeps=20, n_warm_lps=5, kernel_draws=50
    )

    expected = {
        "parse",
        "csc_assembly",
        "reduce",
        "pass_model",
        "first_lp",
        "warm_start_lps",
        "sparse_lp",
        "geometry",
        "rounding",
        "sample_beta0_low",
        "sample_beta0_high",
        "sample_beta4_low",
        "sample_beta4_high",
        "kernel_uniform",
        "kernel_tilted",
        "kernel_breakpoints",
        "output_float64",
        "output_float32",
        "output_reduced",
    }
    assert {stage.name for stage in report.stages} == expected
    assert all(stage.median > 0.0 for stage in report.stages)
    assert report.dimension == 2
    assert {rate.beta for rate in report.sweep_rates} == {0.0, 4.0}
    assert report.total_seconds > 0.0


def test_benchmark_pipeline_refuses_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="model file not found"):
        benchmark_pipeline(tmp_path / "nope.json")


# ── The worker sweep ────────────────────────────────────────────────────────────────────────────


def test_worker_sweep_refuses_to_report_a_batch_whose_strains_failed(
    toy_path: Path, tmp_path: Path
) -> None:
    """A sweep over a batch that sampled nothing must raise, not report a table of zeros.

    This is the regression that motivated the guard, and it is this codebase's signature bug in the
    benchmark's own clothes. M8 makes a failed strain **recorded, not fatal** — correct for 46
    organisms where one bad model must not sink the run. `benchmark_worker_sweep` inherited that
    tolerance and, with every strain failing, produced ``total_ess_j = 0.0`` for all five worker
    counts *beside a wall-clock that still looked like a plausible speedup curve*. Nothing in the
    output said "nothing ran": a reader would have concluded the Jetson does not scale.

    A batch runner should survive a bad strain. A **measurement** of throughput must not.
    """
    from gsmm_compiler.batch import ModelSpec
    from gsmm_compiler.benchmark import benchmark_worker_sweep
    from gsmm_compiler.config import Config, SamplerConfig

    config = Config(sampler=SamplerConfig(betas=(0.0,), n_chains=1, n_samples=5, burn_in=5,
                                          refresh_interval=5))
    specs = [
        ModelSpec(model_path=str(toy_path), model_id="good"),
        ModelSpec(model_path=str(tmp_path / "does_not_exist.json"), model_id="bad"),
    ]

    with pytest.raises(BenchmarkError, match="strains fail"):
        benchmark_worker_sweep(
            specs, config, worker_counts=[1], output_root=tmp_path / "sweep"
        )


# ── The `reduced` storage mode, validated at genome scale ───────────────────────────────────────


@pytest.mark.slow
def test_reduced_storage_round_trips_to_the_same_flux_and_is_smaller(
    example_canonical: Any, tmp_path: Path
) -> None:
    """A ``reduced`` unit reconstructs to ~1e-13 — **not** bit-identically — and is smaller.

    Both halves matter and neither implies the other.

    **The size claim** is the mode's entire reason to exist — 46 coordinates against 773 fluxes —
    and an arithmetic ratio is not a measurement of what lands on disk (`.npy` headers, the manifest
    and six trace arrays ride along in both modes, so the realised saving is strictly less than 17×
    and worth knowing).

    **The fidelity claim is where M9 found something.** The obvious bar is `array_equal`: both paths
    evaluate ``centre + T·y``, so why would they differ? They differ because `_walk` stores
    ``to_flux(y)`` for a **1-D** ``y`` — a matrix-vector product — while `load_chain` lifts the
    whole
    ``(n_samples, d)`` block at once, a matrix-matrix product. Different BLAS kernels accumulate in
    different orders and round differently: measured on the example model, 8642 of 26000 entries
    differ, by up to 4096 ULP (max **1.1e-13** relative).

    That is not a defect and the tolerance is not a concession. Neither rounding is more correct —
    they are two float64 evaluations of the same exact quantity — and 1.1e-13 is ~100× *below* the
    refresh drift (6e-12) the sampler already measures and reports, and four orders below the 1e-9
    feasibility tolerance. What would be a defect is a *wrong* lift, and a loose tolerance alone
    could not tell the two apart. So the test localizes it: the **per-row** lift is asserted
    bit-identical, which pins the discrepancy to the batching and rules out a mis-rebuilt transform
    — the M6 "two artifacts that never met" failure — without appealing to a tolerance at all.

    The honest consequence, recorded rather than buried: **byte-identity holds within a storage
    mode, not across them.** M8's serial-vs-pool guarantee is untouched (same mode, same path).
    """
    from gsmm_compiler.affine_geometry import build_geometry
    from gsmm_compiler.config import SamplerConfig
    from gsmm_compiler.maxent_sampler import run_chain, trace_objective
    from gsmm_compiler.output import SampleStorage, load_chain, store_chain
    from gsmm_compiler.rounding import build_transform
    from gsmm_compiler.sparse_objective import lower_objective, resolve_objective

    reduced = example_canonical.polytope.reduce()
    geometry = build_geometry(reduced, model_id=example_canonical.model_id)
    transform = build_transform(geometry, reduced)
    resolved = resolve_objective(example_canonical.polytope, reduced)
    lowered = lower_objective(reduced, resolved.objective)

    chain = run_chain(
        transform,
        reduced,
        config=SamplerConfig(n_chains=1, n_samples=100, burn_in=50, refresh_interval=25),
        model_id=example_canonical.model_id,
        chain_index=0,
    )
    trace = trace_objective(chain.fluxes, lowered, j_star=0.0, energy_scale=1.0)

    def store(mode: str, dtype: str, where: Path) -> None:
        store_chain(
            where,
            chain=chain,
            trace=trace,
            reduced=reduced,
            model_id=example_canonical.model_id,
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage(mode=mode, flux_dtype=dtype),
        )

    full_dir, reduced_dir = tmp_path / "full", tmp_path / "red"
    store("full_flux", "float64", full_dir)
    store("reduced", "float64", reduced_dir)

    full = load_chain(full_dir)
    lifted = load_chain(reduced_dir, reduced=reduced, transform=transform)

    assert lifted.fluxes.shape == full.fluxes.shape == (100, len(reduced.reaction_ids))

    # The lift is correct to float64 rounding of centre + T·y ...
    scale = np.maximum(np.abs(full.fluxes), 1.0)
    relative = np.max(np.abs(lifted.fluxes - full.fluxes) / scale)
    assert relative < 1e-11, (
        f"a reduced-mode unit reconstructed fluxes {relative:.2e} away from the full-flux path — "
        "far past the ~1e-13 that a different BLAS accumulation order can explain"
    )

    # ... and the residual is *entirely* the batched matrix-matrix lift. Applying the same transform
    # one row at a time — as `_walk` does when it stores the exact flux — is bit-identical. This is
    # what separates "two roundings of the same quantity" from "the wrong transform", which no
    # tolerance on its own could distinguish.
    per_row = np.stack([transform.to_flux(y) for y in chain.coordinates])
    assert np.array_equal(np.asarray(reduced.to_full(per_row)), full.fluxes), (
        "the per-row lift does not reproduce the stored flux bit-for-bit, so the difference above "
        "is not merely the batched BLAS path — the transform or the polytope is wrong"
    )

    def total_bytes(directory: Path) -> int:
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    saving = total_bytes(full_dir) / total_bytes(reduced_dir)
    assert saving > 1.5, f"reduced mode saved only {saving:.2f}×; it has no reason to exist"

    # The 46-vs-773 array ratio is 16.8×; the realised saving is lower because the manifest and the
    # six trace arrays are written identically in both modes. Report it rather than assume it.
    array_ratio = len(reduced.reaction_ids) / transform.dimension
    assert saving < array_ratio, (
        f"reduced mode claims a {saving:.2f}× saving, above the {array_ratio:.1f}× the arrays "
        "alone can give — the accounting is wrong"
    )


@pytest.mark.slow
def test_reduced_storage_refuses_reconstruction_against_the_wrong_polytope(
    toy_canonical: Any, example_canonical: Any, tmp_path: Path
) -> None:
    """The read boundary is keyed: a reduced unit will not lift through a foreign geometry.

    This is the M6 invariant applied where M8 put it — a stored ``y`` is meaningless without the
    exact ``T`` that produced it, and lifting it through another model's transform would yield
    confident, feasible-looking fluxes for the wrong organism.
    """
    from gsmm_compiler.affine_geometry import build_geometry
    from gsmm_compiler.config import SamplerConfig
    from gsmm_compiler.maxent_sampler import run_chain, trace_objective
    from gsmm_compiler.output import OutputError, SampleStorage, load_chain, store_chain
    from gsmm_compiler.rounding import build_transform
    from gsmm_compiler.sparse_objective import lower_objective, resolve_objective

    reduced = toy_canonical.polytope.reduce()
    transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)
    resolved = resolve_objective(toy_canonical.polytope, reduced)
    chain = run_chain(
        transform,
        reduced,
        config=SamplerConfig(n_chains=1, n_samples=20, burn_in=10, refresh_interval=5),
        model_id="toy",
        chain_index=0,
    )
    trace = trace_objective(
        chain.fluxes, lower_objective(reduced, resolved.objective), j_star=0.0, energy_scale=1.0
    )
    store_chain(
        tmp_path / "unit",
        chain=chain,
        trace=trace,
        reduced=reduced,
        model_id="toy",
        beta=0.0,
        beta_index=0,
        chain_index=0,
        storage=SampleStorage(mode="reduced"),
    )

    other = example_canonical.polytope.reduce()
    with pytest.raises(OutputError, match="polytope mismatch"):
        load_chain(tmp_path / "unit", reduced=other, transform=transform)


def test_load_traces_needs_no_geometry_in_either_storage_mode(
    toy_canonical: Any, tmp_path: Path
) -> None:
    """A trace is a stored scalar per sample; reading one must not require the transform.

    `load_chain` legitimately demands the geometry for a reduced unit — it reconstructs fluxes. A
    trace has no such dependency, and `diagnostics` was already hand-rolling its own reader to get
    around the coupling. `output.load_traces` is that reader, and this pins that it works in *both*
    modes — the reduced one being the case `load_chain` cannot serve.
    """
    from gsmm_compiler.affine_geometry import build_geometry
    from gsmm_compiler.config import SamplerConfig
    from gsmm_compiler.maxent_sampler import run_chain, trace_objective
    from gsmm_compiler.output import SampleStorage, load_traces, store_chain
    from gsmm_compiler.rounding import build_transform
    from gsmm_compiler.sparse_objective import lower_objective, resolve_objective

    reduced = toy_canonical.polytope.reduce()
    transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)
    resolved = resolve_objective(toy_canonical.polytope, reduced)
    chain = run_chain(
        transform,
        reduced,
        config=SamplerConfig(n_chains=1, n_samples=20, burn_in=10, refresh_interval=5),
        model_id="toy",
        chain_index=0,
    )
    lowered = lower_objective(reduced, resolved.objective)
    trace = trace_objective(chain.fluxes, lowered, j_star=0.0, energy_scale=1.0)

    for index, mode in enumerate(("full_flux", "reduced")):
        directory = tmp_path / mode
        store_chain(
            directory,
            chain=chain,
            trace=trace,
            reduced=reduced,
            model_id="toy",
            beta=0.0,
            beta_index=index,
            chain_index=0,
            storage=SampleStorage(mode=mode),
        )
        traces = load_traces(directory)
        assert set(traces) >= {"mu", "cost", "j"}
        assert np.array_equal(traces["j"], trace.j)
