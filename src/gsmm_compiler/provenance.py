"""Content-addressed hashing and provenance keys for the cache DAG (BUILD_PLAN §1.1).

A module beyond the spec's §6 list: the spec assumes hashing happens inside the cache layer, but M1
needs L0/L1 keys long before M8's cache exists, and the numerical core must be able to hash arrays
without importing cobra.

The rule the keys must satisfy: **a refactor that changes array semantics must miss the cache, not
silently load stale bytes.** So every key folds in dtype *and byte order*, shape, the schema version
of the artifact, and the versions of the code that produced it.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PARSER_SCHEMA_VERSION = 1
"""Bump when `model_input` changes what it extracts — invalidates L0 and everything below."""

IR_SCHEMA_VERSION = 1
"""Bump when the canonical/reduced IR layout changes — invalidates L1 and everything below."""

_CHUNK_BYTES = 1 << 20


def hash_file(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: NDArray[Any]) -> str:
    """sha256 of an array's *semantics*, not just its bytes.

    ``dtype.str`` carries the byte order (``'<f8'``), so the same numbers stored big-endian hash
    differently — what we want from a key guarding cached ``.npy`` files. C-contiguity is forced
    first, so a transposed view and its copy agree.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(repr(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonicalize(value: Any) -> Any:
    """Render a key component as something ``json.dumps`` can order deterministically."""
    if isinstance(value, np.ndarray):
        return {"__array__": hash_array(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot hash key component of type {type(value).__name__}")


def content_key(**components: Any) -> str:
    """A stable sha256 over named components — arrays included, by their own hashes.

    Keyword order does not matter (components are sorted), so a caller cannot change a key by
    reordering its arguments.
    """
    payload = json.dumps(
        {name: _canonicalize(value) for name, value in sorted(components.items())},
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """The environment an artifact was produced in — written into every run manifest (spec §7)."""

    python_version: str
    numpy_version: str
    cobra_version: str | None
    highspy_version: str | None
    gsmm_compiler_version: str
    byte_order: str

    @classmethod
    def capture(cls) -> Provenance:
        """Record the current environment. cobra/highspy are optional — a worker has neither."""
        from gsmm_compiler import __version__

        return cls(
            python_version=".".join(str(n) for n in sys.version_info[:3]),
            numpy_version=np.__version__,
            cobra_version=_installed_version("cobra"),
            highspy_version=_installed_version("highspy"),
            gsmm_compiler_version=__version__,
            byte_order=sys.byteorder,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _installed_version(distribution: str) -> str | None:
    """Version of an installed distribution, without importing it. ``None`` if absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None
