"""Run-directory layout, crash-safe writes, and configurable sample storage (M8, BUILD_PLAN §1.3).

This is the persistence layer the batch runner and the cache sit on top of. It answers three
questions and nothing else:

* **Where do results go?** `RunLayout` fixes the ``results/<batch>/<model_id>/…`` tree (§1.1), with
  a ``cross_model/`` sibling for the aggregated tables. One place owns the paths, so a resume
  computes the *same* path a crash left half-written.

* **How are they written so a kill never corrupts them?** Every file lands via a temp file +
  ``fsync`` + atomic ``os.replace``; every *directory* of artifacts is staged whole and swapped into
  place with a ``COMPLETE`` marker written last, so a reader sees an artifact that is either absent
  or finished — never a half-written one. This is what makes "resume only the missing
  ``(model, chain)`` units" (the M8 gate) safe: presence-of-``COMPLETE`` is the sole truth for done.

* **How densely?** `SampleStorage` implements the three modes of §1.3 — ``full_flux`` at float64
  (default) or float32, and ``reduced`` (store the ``d``-dim rounded coordinates, reconstruct the
  full flux on demand from the retained geometry). Objective traces are stored in **every** mode.

**No cobra, no HiGHS, no solver** — a sampling worker imports this module, so it stays in the
frozen-array world (BUILD_PLAN §1.2). Every array carries a stored sha256; `load_array` recomputes
it and refuses a mismatch, which is the M8 "corrupted-artifact rejected" gate.

Implemented in **M8** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.provenance import hash_array

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids importing the sampler at module load
    from gsmm_compiler.flux_polytope import ReducedPolytope
    from gsmm_compiler.maxent_sampler import ChainResult, ObjectiveTrace
    from gsmm_compiler.rounding import RoundedTransform

OUTPUT_IMPL_VERSION = 1
"""Bump when a change here alters the **bytes a unit stores** — the arrays written, their names, the
casts applied, or the manifest's fields. Feeds `batch.sample_recipe_key` (§1.1: "provenance in every
key: parser + code + artifact-schema versions").

It exists because that rule was not applied to samples. The recipe key carried
`SAMPLER_IMPL_VERSION`, which by its own docstring covers **the transition kernel** — while
`store_chain` decides everything about the stored artifact and carried no version at all. So an
output-only change kept the recipe key identical and left stale units looking resumable, which is a
false hit on the one key that guards a user's results tree. (Codex, M10.2 review round 4.)
"""

COMPLETE_MARKER = "COMPLETE"
"""The file whose *presence* means an artifact directory is finished. Written last, atomically."""

STAGING_SUFFIX = ".staging"
"""A partly-written directory is built here, then atomically renamed onto its final name."""


class OutputError(RuntimeError):
    """A stored artifact is missing, malformed, or fails its integrity check."""


# ---- crash-safe primitives ---------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    """Flush a file's bytes to disk. Strict: a durability failure on data is a real error."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_directory(path: Path) -> None:
    """Flush a directory entry so a rename into it survives a power loss.

    Best-effort: some filesystems reject ``fsync`` on a directory with ``EINVAL``. That costs
    *durability across power loss*, never *correctness* — the atomic ``os.replace`` still makes the
    swap all-or-nothing to any reader on a running kernel, which is what the kill-and-resume gate
    exercises. So an unsupported directory fsync is swallowed; anything else is raised.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:  # pragma: no cover - filesystem-dependent
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EACCES}:
            raise
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a reader sees either the old bytes or all the new ones.

    Temp file in the *same directory* (so the rename cannot cross filesystems), ``fsync`` the data,
    then ``os.replace`` — atomic on POSIX — and ``fsync`` the parent directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def write_json(path: Path, obj: Any) -> None:
    """Atomically write ``obj`` as sorted, indented JSON (numpy scalars/arrays coerced).

    ``allow_nan=True`` because manifests legitimately carry non-finite *diagnostics* — a
    ``blocked_separation`` of ``inf`` is the honest reading when a model has no blocked reactions,
    and a condition number can overflow. That is a report, not sampled data: array files get a
    finite bar from `load_array`, where a NaN flux would be a bug. Python's own ``json`` reads the
    ``Infinity``/``NaN`` tokens back exactly, which is what the cache round-trip relies on.
    """
    text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default, allow_nan=True)
    atomic_write_bytes(Path(path), (text + "\n").encode())


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise OutputError(f"expected JSON at {path}, found nothing")
    return json.loads(path.read_text())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def save_array(path: Path, array: NDArray[Any]) -> str:
    """Atomically save ``array`` as an uncompressed, memory-mappable ``.npy`` and return its sha256.

    Uncompressed on purpose (§1.1): a compressed ``.npz`` cannot be zero-copy ``mmap``-ed, and the
    flux matrix is the one artifact big enough to care. The returned hash is of the **contiguous**
    bytes actually written, so it matches what `load_array` recomputes.
    """
    path = Path(path)
    contiguous = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, contiguous, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)
    return hash_array(contiguous)


def load_array(
    path: Path,
    *,
    sha256: str | None = None,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
    finite: bool = True,
) -> NDArray[Any]:
    """Load a ``.npy`` and **validate it** — shape, dtype, finiteness, and content hash (§1.1).

    A silent bit-flip on disk is exactly the failure the cache design guards against, so this is not
    optional belt-and-suspenders: it is the check the "corrupted-artifact rejected" gate relies on.
    """
    path = Path(path)
    if not path.is_file():
        raise OutputError(f"expected an array at {path}, found nothing")
    array: NDArray[Any] = np.load(path, allow_pickle=False)

    if dtype is not None and array.dtype.str != dtype:
        raise OutputError(f"{path}: dtype is {array.dtype.str!r}, expected {dtype!r}")
    if shape is not None and array.shape != tuple(shape):
        raise OutputError(f"{path}: shape is {array.shape}, expected {tuple(shape)}")
    if finite and np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise OutputError(f"{path}: array holds NaN or infinity")
    if sha256 is not None and hash_array(array) != sha256:
        raise OutputError(
            f"{path}: content hash mismatch — the file is corrupt or was written by another version"
        )
    return array


def _array_ref(filename: str, array: NDArray[Any], sha256: str) -> dict[str, Any]:
    """The manifest record for one stored array: enough to validate it on the way back in."""
    return {
        "file": filename,
        "sha256": sha256,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


# ---- atomic directory publication --------------------------------------------------------------


def is_complete(directory: Path) -> bool:
    """True once the directory's ``COMPLETE`` marker exists — the only "is it done?" test.

    A directory can exist without being complete (a crash between the swap and the marker, or a
    half-built staging dir). Presence of the marker, and nothing else, decides.
    """
    return (Path(directory) / COMPLETE_MARKER).is_file()


@contextmanager
def staged_directory(final_dir: Path) -> Iterator[Path]:
    """Build an artifact directory off to the side, then swap it into place atomically.

    Yields a **staging** directory to write into. On clean exit the ``COMPLETE`` marker is written
    last, everything is fsync'd, and the staging dir is renamed onto ``final_dir`` — so a reader
    only ever sees ``final_dir`` once it is whole and marked. A raised exception leaves the staging
    dir behind and ``final_dir`` untouched, so the unit reads as *not done* and a resume recomputes
    it.

    The staging name is **deterministic** per final directory (``<name>.staging``), not random: two
    different units have different final names and so never collide, while a retry of the *same*
    unit reclaims its own staging dir. That keeps sibling chains writing into a shared ``beta_NNN/``
    parent from ever deleting each other's in-progress work — the trap a prefix sweep would spring.
    """
    final_dir = Path(final_dir)
    staging = final_dir.with_name(final_dir.name + STAGING_SUFFIX)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    yield staging

    # Commit. The marker goes in last: if a crash interrupts anything above, there is no marker and
    # the whole staging dir is discarded on resume.
    _fsync_file(_touch(staging / COMPLETE_MARKER))
    fsync_directory(staging)

    # Swap. A stale `final_dir` (a partial from an earlier crash, never one with COMPLETE that the
    # caller should have skipped) is cleared first: renaming a directory onto a non-empty one fails.
    # Single-writer-per-unit (the batch runner assigns each unit to one worker) makes the tiny
    # rmtree→replace window unobservable.
    shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(staging, final_dir)
    fsync_directory(final_dir.parent)


def _touch(path: Path) -> Path:
    with open(path, "wb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return path


# ---- run-directory layout ----------------------------------------------------------------------


def _safe_component(name: str, *, what: str) -> str:
    """A single path segment, refusing anything that could climb out of the run tree."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise OutputError(f"{what} {name!r} is not a safe path component")
    return name


@dataclass(frozen=True)
class RunLayout:
    """The ``results/<batch>/…`` tree (§1.1). One owner of every path a run reads or writes."""

    root: Path
    batch: str

    def __post_init__(self) -> None:
        _safe_component(self.batch, what="batch name")
        object.__setattr__(self, "root", Path(self.root))

    @property
    def batch_dir(self) -> Path:
        return self.root / self.batch

    def model_dir(self, model_id: str) -> Path:
        return self.batch_dir / _safe_component(model_id, what="model_id")

    def geometry_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "geometry"

    def samples_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "samples"

    def chain_dir(self, model_id: str, beta_index: int, chain_index: int) -> Path:
        return (
            self.samples_dir(model_id)
            / f"beta_{int(beta_index):03d}"
            / f"chain_{int(chain_index):03d}"
        )

    def diagnostics_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "diagnostics"

    def cross_model_dir(self) -> Path:
        return self.batch_dir / "cross_model"

    def model_manifest_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "run_manifest.json"

    def model_complete_marker(self, model_id: str) -> Path:
        return self.model_dir(model_id) / COMPLETE_MARKER

    def is_chain_complete(self, model_id: str, beta_index: int, chain_index: int) -> bool:
        return is_complete(self.chain_dir(model_id, beta_index, chain_index))


# ---- configurable sample storage (§1.3) --------------------------------------------------------

_TRACE_ARRAYS = (
    "mu",
    "cost",
    "j",
    "normalized_log_energy",
    "near_zero_counts",
    "near_zero_counts_all_free",
)


@dataclass(frozen=True)
class SampleStorage:
    """The §1.3 storage choice: which vectors to keep, and at what width.

    ``full_flux`` writes the lifted 773-length flux vectors (best fidelity; float64 default, float32
    to halve the disk). ``reduced`` writes only the ``d``-dimensional rounded coordinates and leans
    on the retained geometry to reconstruct the flux on demand (smallest on disk, for big batches).
    Either way the objective traces are stored — they are what the cross-model comparison reads.
    """

    mode: str = "full_flux"
    flux_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.mode not in {"full_flux", "reduced"}:
            raise OutputError(f"store_mode must be 'full_flux' or 'reduced', got {self.mode!r}")
        if self.flux_dtype not in {"float64", "float32"}:
            raise OutputError(
                f"store_flux_dtype must be 'float64' or 'float32', got {self.flux_dtype!r}"
            )

    @classmethod
    def from_config(cls, output_config: Any) -> SampleStorage:
        return cls(mode=output_config.store_mode, flux_dtype=output_config.store_flux_dtype)


def store_chain(
    chain_dir: Path,
    *,
    chain: ChainResult,
    trace: ObjectiveTrace,
    reduced: ReducedPolytope,
    model_id: str,
    beta: float,
    beta_index: int,
    chain_index: int,
    storage: SampleStorage,
    recipe_key: str = "",
) -> dict[str, Any]:
    """Persist one ``(model, β, chain)`` unit atomically, returning its manifest.

    In ``full_flux`` mode the reduced draws are lifted to full model vectors *here* and stored at
    the configured width; in ``reduced`` mode the rounded coordinates ``y`` are stored and the lift
    is deferred to `load_chain` (which then needs the geometry). The traces go in regardless.

    The manifest carries the polytope's L1 ``content_key``: a sample loaded for aggregation can then
    be checked against the geometry it is reconstructed with, so the M6/M8 "two artifacts that never
    met" failure cannot slip in at read time either.

    ``recipe_key`` (`batch.sample_recipe_key`, M10.2) is the rest of that identity: the polytope key
    says *which model*, and this says **which experiment** — the transform, the objective, ``s_J``,
    the schedule, the seed and the storage mode that produced these bytes. It is what lets restart
    tell "this unit is already done" from "this unit is a different run's". Defaulted for the
    direct-library caller who has no batch plan; `batch` always passes it.
    """
    to_store: dict[str, NDArray[Any]] = {}
    if storage.mode == "full_flux":
        full = reduced.to_full(np.asarray(chain.fluxes))
        to_store["flux"] = np.ascontiguousarray(full, dtype=storage.flux_dtype)
    else:
        to_store["coordinates"] = np.ascontiguousarray(chain.coordinates)

    to_store["trace_mu"] = trace.mu
    to_store["trace_cost"] = trace.cost
    to_store["trace_j"] = trace.j
    to_store["trace_normalized_log_energy"] = trace.normalized_log_energy
    to_store["trace_near_zero_counts"] = trace.near_zero_counts
    to_store["trace_near_zero_counts_all_free"] = trace.near_zero_counts_all_free

    with staged_directory(Path(chain_dir)) as staging:
        arrays: dict[str, dict[str, Any]] = {}
        for name, array in to_store.items():
            filename = f"{name}.npy"
            digest = save_array(staging / filename, array)
            arrays[name] = _array_ref(filename, array, digest)

        manifest = {
            "kind": "chain_samples",
            "model_id": model_id,
            "beta": float(beta),
            "beta_index": int(beta_index),
            "chain_index": int(chain_index),
            "store_mode": storage.mode,
            "store_flux_dtype": storage.flux_dtype,
            "polytope_key": reduced.content_key(),
            "recipe_key": recipe_key,
            "n_samples": int(np.asarray(chain.fluxes).shape[0]),
            "n_free": int(reduced.n_free),
            "n_full": int(reduced.n_full),
            "dimension": int(np.asarray(chain.coordinates).shape[1]),
            "arrays": arrays,
            "diagnostics": chain.diagnostics.as_dict(),
            "trace_summary": trace.as_dict(),
        }
        write_json(staging / "manifest.json", manifest)

    return manifest


@dataclass(frozen=True)
class LoadedChain:
    """One chain read back and validated: full-length fluxes plus the objective traces."""

    fluxes: NDArray[np.float64]
    """``(n_samples, n_full)`` — full-model flux, reconstructed if the unit was stored reduced."""
    traces: dict[str, NDArray[Any]]
    manifest: dict[str, Any]


def load_chain(
    chain_dir: Path,
    *,
    reduced: ReducedPolytope | None = None,
    transform: RoundedTransform | None = None,
) -> LoadedChain:
    """Read a stored unit, validate every array, and return full-length fluxes + traces.

    A ``reduced``-mode unit stores only coordinates, so reconstructing the flux needs the geometry:
    pass the same ``reduced`` polytope and ``transform`` the chain was sampled with (their
    ``content_key``/``polytope_key`` are cross-checked against the manifest, so a mismatch raises
    rather than silently reconstructing against the wrong model).
    """
    chain_dir = Path(chain_dir)
    if not is_complete(chain_dir):
        raise OutputError(f"{chain_dir} has no {COMPLETE_MARKER} marker — not a finished unit")
    manifest = read_json(chain_dir / "manifest.json")
    refs = manifest["arrays"]

    def _load(name: str) -> NDArray[Any]:
        ref = refs[name]
        return load_array(
            chain_dir / ref["file"],
            sha256=ref["sha256"],
            dtype=ref["dtype"],
            shape=tuple(ref["shape"]),
        )

    traces = {name: _load(f"trace_{name}") for name in _TRACE_ARRAYS}

    if manifest["store_mode"] == "full_flux":
        fluxes = _load("flux").astype(np.float64, copy=False)
    else:
        if reduced is None or transform is None:
            raise OutputError(
                "a 'reduced'-mode unit needs the geometry to reconstruct its fluxes: pass both "
                "`reduced` and `transform`"
            )
        _check_polytope(manifest, reduced)
        coordinates = _load("coordinates")
        v_reduced = transform.to_flux(coordinates)
        fluxes = np.asarray(reduced.to_full(v_reduced), dtype=np.float64)

    return LoadedChain(fluxes=fluxes, traces=traces, manifest=manifest)


def load_traces(chain_dir: Path) -> dict[str, NDArray[Any]]:
    """Read a stored unit's objective traces, validated, **without needing the geometry**.

    `load_chain` reconstructs full-length fluxes, and a ``reduced``-mode unit cannot do that without
    the transform it was sampled with. A trace has no such dependency — it is a stored scalar per
    sample, written identically in every storage mode — so requiring the geometry to read one would
    be a coupling the data does not have. Callers that want ``J`` and nothing else (diagnostics,
    the M9 worker sweep) use this and stay independent of the storage mode.
    """
    chain_dir = Path(chain_dir)
    if not is_complete(chain_dir):
        raise OutputError(f"{chain_dir} has no {COMPLETE_MARKER} marker — not a finished unit")
    manifest = read_json(chain_dir / "manifest.json")
    refs = manifest["arrays"]
    return {
        name: load_array(
            chain_dir / refs[f"trace_{name}"]["file"],
            sha256=refs[f"trace_{name}"]["sha256"],
            dtype=refs[f"trace_{name}"]["dtype"],
            shape=tuple(refs[f"trace_{name}"]["shape"]),
        )
        for name in _TRACE_ARRAYS
    }


def _check_polytope(manifest: dict[str, Any], reduced: ReducedPolytope) -> None:
    stored = manifest.get("polytope_key")
    live = reduced.content_key()
    if stored != live:
        raise OutputError(
            "polytope mismatch reconstructing a stored sample: it was written against polytope "
            f"{str(stored)[:16]}… but is being lifted with {live[:16]}…. Reconstructing against a "
            "wrong geometry would silently produce fluxes for a different model."
        )
