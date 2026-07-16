"""M8 acceptance gate — cache, restart, and batch orchestration on the toy network.

The gate (BUILD_PLAN M8): **kill-and-resume resumes only the missing ``(model, chain)`` units**; a
**partial batch still yields valid cross-model tables**; **concurrent writers are safe**; a
**corrupted artifact is rejected** on load; and **same-env traces are deterministic**. Everything
runs on the toy network, which is small enough that a real ladder finishes in well under a second.

Concurrent-writer safety is exercised where its primitive lives — the atomic-``mkdir`` writer claim
in `tests/unit/test_cache.py`, with eight threads proving the compute runs exactly once. Here focus
is the batch layer on top: that restart is per-unit, that a failed strain does not sink the run, and
that the pool and the in-process path produce byte-identical draws.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from gsmm_compiler.batch import (
    ModelSpec,
    _jobs_for,
    prepare_model,
    run_batch,
    run_sample_unit,
)
from gsmm_compiler.cache import ArtifactCache, CacheError
from gsmm_compiler.config import Config, OutputConfig, SamplerConfig
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.output import OutputError, RunLayout, load_chain


def _config(betas: tuple[float, ...] = (0.0, 2.0), n_chains: int = 2) -> Config:
    return Config(
        sampler=SamplerConfig(
            betas=betas, n_chains=n_chains, n_samples=40, burn_in=40, refresh_interval=20
        ),
        output=OutputConfig(),
    )


@pytest.fixture
def toy_spec(toy_path: Path) -> ModelSpec:
    return ModelSpec(model_path=str(toy_path), model_id="toy")


class TestSingleModelEndToEnd:
    def test_every_artifact_is_written_and_every_unit_completes(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        config = _config()
        result = run_batch(
            [toy_spec], config, batch_name="b", output_root=tmp_path / "r", cache_dir=tmp_path / "c"
        )
        assert [o.status for o in result.outcomes] == ["complete"]

        layout = RunLayout(root=tmp_path / "r", batch="b")
        assert layout.model_manifest_path("toy").is_file()
        assert layout.model_complete_marker("toy").is_file()
        assert (layout.diagnostics_dir("toy") / "diagnostics.json").is_file()
        for beta_index in range(len(config.sampler.betas)):
            for chain_index in range(config.sampler.n_chains):
                assert layout.is_chain_complete("toy", beta_index, chain_index)

    def test_every_stored_sample_is_feasible_in_the_full_polytope(
        self, tmp_path: Path, toy_spec: ModelSpec, toy_canonical
    ) -> None:  # type: ignore[no-untyped-def]
        run_batch([toy_spec], _config(), batch_name="b", output_root=tmp_path / "r")
        layout = RunLayout(root=tmp_path / "r", batch="b")

        polytope = toy_canonical.polytope
        for sample in load_chain(layout.chain_dir("toy", 1, 0)).fluxes:
            assert polytope.contains(sample)  # full 7-reaction bounds *and* mass balance

    def test_no_hiGHS_solve_happens_once_sampling_starts(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        plan = prepare_model(toy_spec, _config())  # all the LP work happens here
        layout = RunLayout(root=tmp_path / "r", batch="b")

        before = total_solve_count()
        for job in _jobs_for(plan, layout):
            run_sample_unit(job)
        assert total_solve_count() == before  # the inner loop never touched a solver


class TestDeterminism:
    def test_the_process_pool_and_in_process_paths_agree_bit_for_bit(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        config = _config()
        run_batch([toy_spec], config, batch_name="serial", output_root=tmp_path / "s", n_workers=1)
        run_batch([toy_spec], config, batch_name="pool", output_root=tmp_path / "p", n_workers=2)

        serial = RunLayout(root=tmp_path / "s", batch="serial")
        pool = RunLayout(root=tmp_path / "p", batch="pool")
        for beta_index in range(len(config.sampler.betas)):
            for chain_index in range(config.sampler.n_chains):
                a = load_chain(serial.chain_dir("toy", beta_index, chain_index)).fluxes
                b = load_chain(pool.chain_dir("toy", beta_index, chain_index)).fluxes
                np.testing.assert_array_equal(a, b)


class TestKillAndResume:
    def test_resume_recomputes_only_the_missing_unit(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        config = _config()
        run_batch(
            [toy_spec], config, batch_name="b", output_root=tmp_path / "r", cache_dir=tmp_path / "c"
        )
        layout = RunLayout(root=tmp_path / "r", batch="b")

        kept = layout.chain_dir("toy", 0, 0) / "flux.npy"
        kept_mtime = kept.stat().st_mtime_ns
        victim_dir = layout.chain_dir("toy", 1, 1)
        victim_before = np.load(victim_dir / "flux.npy")

        shutil.rmtree(victim_dir)  # simulate a kill that lost exactly one unit
        assert not layout.is_chain_complete("toy", 1, 1)

        run_batch(
            [toy_spec], config, batch_name="b", output_root=tmp_path / "r", cache_dir=tmp_path / "c"
        )

        # The missing unit came back, reproduced bit-for-bit from its semantic RNG key ...
        assert layout.is_chain_complete("toy", 1, 1)
        np.testing.assert_array_equal(np.load(victim_dir / "flux.npy"), victim_before)
        # ... and the unit that survived was never rewritten.
        assert kept.stat().st_mtime_ns == kept_mtime


class TestPartialBatch:
    def test_a_failed_strain_does_not_sink_the_batch(self, tmp_path: Path, toy_path: Path) -> None:
        specs = [
            ModelSpec(model_path=str(toy_path), model_id="good"),
            ModelSpec(model_path="/does/not/exist.json", model_id="broken"),
        ]
        result = run_batch(specs, _config(betas=(0.0,)), batch_name="b", output_root=tmp_path / "r")

        status = {o.model_id: o.status for o in result.outcomes}
        assert status == {"good": "complete", "broken": "failed"}

        # The cross-model β-summary still covers the strain that finished.
        from gsmm_compiler.output import read_json

        summary = read_json(result.batch_dir / "cross_model" / "beta_summary.json")
        assert [row["model_id"] for row in summary] == ["good"]

    def test_a_broken_strain_records_its_error(self, tmp_path: Path, toy_path: Path) -> None:
        specs = [ModelSpec(model_path="/nope.json", model_id="broken")]
        result = run_batch(specs, _config(betas=(0.0,)), batch_name="b", output_root=tmp_path / "r")
        (broken,) = result.outcomes
        assert broken.status == "failed"
        assert broken.error is not None and "FileNotFoundError" in broken.error


class TestCorruptedArtifactRejection:
    def test_a_corrupted_sample_is_rejected_on_load(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        run_batch([toy_spec], _config(betas=(0.0,)), batch_name="b", output_root=tmp_path / "r")
        layout = RunLayout(root=tmp_path / "r", batch="b")
        flux_path = layout.chain_dir("toy", 0, 0) / "flux.npy"

        tampered = np.load(flux_path)
        tampered[0, 0] += 1.0
        np.save(flux_path, tampered)  # same shape/dtype, different bytes

        with pytest.raises(OutputError, match="content hash mismatch"):
            load_chain(layout.chain_dir("toy", 0, 0))

    def test_a_corrupted_geometry_cache_entry_is_rejected(
        self, tmp_path: Path, toy_spec: ModelSpec
    ) -> None:
        from gsmm_compiler.batch import geometry_cache_key

        config = _config(betas=(0.0,))
        cache = ArtifactCache(tmp_path / "cache")
        plan = prepare_model(toy_spec, config, cache=cache)

        key = geometry_cache_key(plan.reduced, config, model_id="toy")
        transform_npy = cache.artifact_dir("L3", key) / "transform.npy"
        tampered = np.load(transform_npy)
        tampered.flat[0] += 1.0
        np.save(transform_npy, tampered)

        with pytest.raises((CacheError, OutputError), match="content hash mismatch"):
            prepare_model(toy_spec, config, cache=cache)  # a fresh run must not trust it
