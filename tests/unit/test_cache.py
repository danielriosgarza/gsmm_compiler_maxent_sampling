"""The content-addressed artifact cache: store/load, validation, and writer-claim locking (M8)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from gsmm_compiler.cache import ArtifactCache, CacheError
from gsmm_compiler.output import COMPLETE_MARKER


def _bundle() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    return (
        {"matrix": np.arange(6, dtype=np.float64).reshape(2, 3), "vec": np.array([1, 2, 3])},
        {"note": "hello", "count": 3},
    )


class TestStoreLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        arrays, meta = _bundle()
        cache.store("L3", "abc", arrays=arrays, meta=meta)

        assert cache.is_cached("L3", "abc")
        loaded = cache.load("L3", "abc")
        np.testing.assert_array_equal(loaded.arrays["matrix"], arrays["matrix"])
        np.testing.assert_array_equal(loaded.arrays["vec"], arrays["vec"])
        assert loaded.meta == meta

    def test_absent_artifact_is_not_cached(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        assert not cache.is_cached("L3", "nope")
        with pytest.raises(CacheError, match="no complete artifact"):
            cache.load("L3", "nope")

    def test_a_directory_without_a_marker_is_not_cached(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        cache.artifact_dir("L3", "half").mkdir(parents=True)  # exists but never COMPLETE
        assert not cache.is_cached("L3", "half")

    def test_corrupted_array_is_rejected_on_load(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        arrays, meta = _bundle()
        directory = cache.store("L3", "abc", arrays=arrays, meta=meta)

        tampered = np.load(directory / "matrix.npy")
        tampered[0, 0] += 1.0
        np.save(directory / "matrix.npy", tampered)  # same shape/dtype, different content

        with pytest.raises(Exception, match="content hash mismatch"):
            cache.load("L3", "abc")

    @pytest.mark.parametrize("bad", ["..", "a/b", ""])
    def test_unsafe_keys_are_refused(self, tmp_path: Path, bad: str) -> None:
        cache = ArtifactCache(tmp_path)
        with pytest.raises(CacheError, match="safe path component"):
            cache.artifact_dir("L3", bad)


class TestGetOrCompute:
    def test_computes_on_a_miss_then_reads_the_cache(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        calls = {"n": 0}

        def compute() -> tuple[dict[str, np.ndarray], dict[str, object]]:
            calls["n"] += 1
            return {"x": np.array([1.0, 2.0])}, {"tag": "v1"}

        first = cache.get_or_compute("L3", "k", compute)
        second = cache.get_or_compute("L3", "k", compute)

        assert calls["n"] == 1  # the second call is a pure cache hit
        np.testing.assert_array_equal(first.arrays["x"], np.array([1.0, 2.0]))
        assert second.meta == {"tag": "v1"}

    def test_concurrent_callers_compute_exactly_once(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        lock = threading.Lock()
        calls = {"n": 0}

        def compute() -> tuple[dict[str, np.ndarray], dict[str, object]]:
            with lock:
                calls["n"] += 1
            time.sleep(0.05)  # widen the window so racing threads actually collide on the claim
            return {"x": np.array([3.0, 1.0, 4.0])}, {}

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [pool.submit(cache.get_or_compute, "L3", "shared", compute) for _ in range(8)]
            arrays = [r.result().arrays["x"] for r in results]

        assert calls["n"] == 1  # the writer-claim let exactly one thread do the work
        for got in arrays:
            np.testing.assert_array_equal(got, np.array([3.0, 1.0, 4.0]))

    def test_a_dead_writers_stale_claim_is_stolen(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        # Simulate a worker that claimed the key and then died without publishing.
        claim = cache.artifact_dir("L3", "k").with_name("k.claim")
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.mkdir()

        computed = cache.get_or_compute(
            "L3", "k", lambda: ({"x": np.zeros(2)}, {}), stale_after_seconds=-1.0
        )
        np.testing.assert_array_equal(computed.arrays["x"], np.zeros(2))
        assert cache.is_cached("L3", "k")

    def test_a_live_claim_that_never_publishes_times_out(self, tmp_path: Path) -> None:
        cache = ArtifactCache(tmp_path)
        claim = cache.artifact_dir("L3", "k").with_name("k.claim")
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.mkdir()  # held forever, and not stale — a genuinely stuck writer

        with pytest.raises(CacheError, match="timed out"):
            cache.get_or_compute(
                "L3",
                "k",
                lambda: ({"x": np.zeros(2)}, {}),
                timeout_seconds=0.1,
                poll_seconds=0.01,
                stale_after_seconds=1e9,
            )


def test_store_publishes_atomically_with_the_marker_last(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    arrays, meta = _bundle()
    directory = cache.store("L3", "abc", arrays=arrays, meta=meta)

    assert (directory / COMPLETE_MARKER).is_file()
    assert not directory.with_name("abc.staging").exists()
