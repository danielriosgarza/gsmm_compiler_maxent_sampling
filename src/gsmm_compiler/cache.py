"""The content-addressed artifact cache (BUILD_PLAN §1.1, M8).

A generic store for the four-layer DAG — parsed model (L0), reduced polytope (L1), objective + LP
optimum + ``s_J`` (L2), geometry (L3) — plus any run's per-``(β, chain)`` samples. Each artifact is
an immutable directory named by the hash of everything upstream that can change its bytes, so a
refactor that changes array semantics *misses* the cache rather than loading stale bytes.

This module is **deliberately generic and domain-free**: it stores ``{name: ndarray}`` bundles under
caller-supplied key strings and knows nothing about polytopes or transforms. That keeps it
importable by a solver-free worker (no cobra, no HiGHS), and keeps the *key derivation* — which
layer depends on which upstream hash — with the artifacts (e.g. `rounding.RoundedTransform`).

Three invariants, each earning a slice of the M8 gate:

* **Lookup by an inputs key, validate by a content hash.** The key you look an artifact up by is a
  function of its *inputs* (so it is computable before the artifact is built); every stored array
  additionally carries a sha256 that `load` recomputes and refuses on mismatch. That split is the
  same one the L0 fix drew — a lookup key is a cheap proxy, a content hash is the authority — and it
  is what makes "corrupted-artifact rejected" hold.

* **One writer per key.** `get_or_compute` claims a key with an atomic ``mkdir``; whoever wins
  computes and publishes, everyone else waits for the ``COMPLETE`` marker and loads. Two jobs asked
  for the same geometry never both compute it ("concurrent-writer safe").

* **All-or-nothing publication.** Artifacts are staged whole and swapped in with the marker written
  last (`output.staged_directory`), so a killed writer leaves *nothing* to read — never a torn
  artifact a later run would trust.

Implemented in **M8** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.output import (
    OutputError,
    is_complete,
    load_array,
    read_json,
    save_array,
    staged_directory,
    write_json,
)

CACHE_SCHEMA_VERSION = 1
"""Bump when the on-disk artifact envelope (``artifact.json`` layout) changes."""

CLAIM_SUFFIX = ".claim"
ARTIFACT_MANIFEST = "artifact.json"


class CacheError(RuntimeError):
    """A cached artifact is missing, malformed, or a claim could not be resolved."""


def _safe_key(key: str) -> str:
    """A cache key becomes a directory name, so it must be a single, tame path segment."""
    if not key or key in {".", ".."} or "/" in key or "\\" in key or "\x00" in key:
        raise CacheError(f"cache key {key!r} is not a safe path component")
    return key


@dataclass(frozen=True)
class Artifact:
    """One cache entry read back: its arrays (validated) and its JSON metadata."""

    arrays: dict[str, NDArray[Any]]
    meta: dict[str, Any]


@dataclass(frozen=True)
class ArtifactCache:
    """A content-addressed store rooted at one directory, one subdirectory per layer."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def layer_dir(self, layer: str) -> Path:
        return self.root / _safe_key(layer)

    def artifact_dir(self, layer: str, key: str) -> Path:
        return self.layer_dir(layer) / _safe_key(key)

    def is_cached(self, layer: str, key: str) -> bool:
        """True once the artifact's ``COMPLETE`` marker exists — nothing partial counts."""
        return is_complete(self.artifact_dir(layer, key))

    # ---- store / load ------------------------------------------------------------------------

    def store(
        self,
        layer: str,
        key: str,
        *,
        arrays: Mapping[str, NDArray[Any]],
        meta: Mapping[str, Any],
    ) -> Path:
        """Write an artifact atomically: each array as a hashed ``.npy``, plus an envelope JSON."""
        target = self.artifact_dir(layer, key)
        with staged_directory(target) as staging:
            refs: dict[str, dict[str, Any]] = {}
            for name, value in arrays.items():
                array = np.ascontiguousarray(value)
                filename = f"{_safe_key(name)}.npy"
                digest = save_array(staging / filename, array)
                refs[name] = {
                    "file": filename,
                    "sha256": digest,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
            write_json(
                staging / ARTIFACT_MANIFEST,
                {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "layer": layer,
                    "key": key,
                    "arrays": refs,
                    "meta": dict(meta),
                },
            )
        return target

    def load(self, layer: str, key: str) -> Artifact:
        """Load an artifact, validating every array's shape, dtype and content hash."""
        directory = self.artifact_dir(layer, key)
        if not is_complete(directory):
            raise CacheError(f"no complete artifact at {layer}/{key}")
        manifest = read_json(directory / ARTIFACT_MANIFEST)
        arrays: dict[str, NDArray[Any]] = {}
        for name, ref in manifest["arrays"].items():
            arrays[name] = load_array(
                directory / ref["file"],
                sha256=ref["sha256"],
                dtype=ref["dtype"],
                shape=tuple(ref["shape"]),
            )
        return Artifact(arrays=arrays, meta=manifest["meta"])

    # ---- claim-and-compute -------------------------------------------------------------------

    @contextmanager
    def _claim(self, layer: str, key: str) -> Iterator[bool]:
        """Try to become the sole writer of a key via an atomic ``mkdir``.

        Yields ``True`` to the winner (who must compute + publish) and ``False`` to everyone else.
        Only the winner clears the claim on the way out, so a loser can never delete the claim of
        the worker that is actually computing.
        """
        claim = self.artifact_dir(layer, key).with_name(_safe_key(key) + CLAIM_SUFFIX)
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(claim)
        except FileExistsError:
            yield False
            return
        try:
            yield True
        finally:
            shutil.rmtree(claim, ignore_errors=True)

    def get_or_compute(
        self,
        layer: str,
        key: str,
        compute: Callable[[], tuple[Mapping[str, NDArray[Any]], Mapping[str, Any]]],
        *,
        poll_seconds: float = 0.02,
        timeout_seconds: float = 600.0,
        stale_after_seconds: float = 300.0,
    ) -> Artifact:
        """Return the cached artifact, computing it exactly once across concurrent callers.

        The winner of the ``mkdir`` claim runs ``compute`` (which returns ``(arrays, meta)``) and
        publishes it; losers wait for the ``COMPLETE`` marker and load. A claim that outlives
        ``stale_after_seconds`` with no artifact — a writer that died mid-compute — is stolen, so
        the work is not wedged forever.
        """
        if self.is_cached(layer, key):
            return self.load(layer, key)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() <= deadline:
            with self._claim(layer, key) as owned:
                if owned:
                    # A concurrent winner may have finished between our is_cached check and the
                    # claim — trust the artifact over recomputing it.
                    if self.is_cached(layer, key):
                        return self.load(layer, key)
                    arrays, meta = compute()
                    self.store(layer, key, arrays=arrays, meta=meta)
                    return self.load(layer, key)

            # Someone else holds the claim. If they have since published, load it; otherwise wait a
            # beat, steal the claim if it has gone stale (a dead writer), and try to claim again.
            if self.is_cached(layer, key):
                return self.load(layer, key)
            self._steal_if_stale(layer, key, stale_after_seconds)
            time.sleep(poll_seconds)

        if self.is_cached(layer, key):
            return self.load(layer, key)
        raise CacheError(
            f"timed out after {timeout_seconds:g}s waiting for {layer}/{key}; the writing worker "
            "appears to have died without publishing"
        )

    def _steal_if_stale(self, layer: str, key: str, stale_after: float) -> None:
        """Remove a claim that outlived a plausible compute, so a dead writer cannot wedge it."""
        claim = self.artifact_dir(layer, key).with_name(_safe_key(key) + CLAIM_SUFFIX)
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            return
        if age > stale_after and not self.is_cached(layer, key):
            shutil.rmtree(claim, ignore_errors=True)


def bundle_refs_ok(cache: ArtifactCache, layer: str, key: str) -> bool:
    """Cheap integrity probe used by tests/tools: does the artifact load and validate?"""
    try:
        cache.load(layer, key)
    except (CacheError, OutputError):
        return False
    return True
