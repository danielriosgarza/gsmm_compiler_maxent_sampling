"""Content hashing and provenance keys (BUILD_PLAN §1.1).

The property under test is the one the cache's correctness rests on: **anything that can change the
bytes of a downstream artifact must change its key.** A key that collides across a semantic change
means a stale artifact loads silently — the worst failure mode in the whole design, because the run
still produces numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsmm_compiler.provenance import (
    Provenance,
    content_key,
    hash_array,
    hash_file,
)


class TestHashArray:
    def test_is_stable_across_calls(self) -> None:
        array = np.arange(10, dtype=np.float64)
        assert hash_array(array) == hash_array(array.copy())

    def test_distinguishes_values(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.000000001])
        assert hash_array(a) != hash_array(b)

    def test_distinguishes_dtype_at_identical_values(self) -> None:
        """float32 and float64 arrays of "the same" numbers are different artifacts."""
        values = [1.0, 2.0, 3.0]
        assert hash_array(np.array(values, dtype=np.float64)) != hash_array(
            np.array(values, dtype=np.float32)
        )

    def test_distinguishes_byte_order(self) -> None:
        """dtype.str carries endianness, so a big-endian .npy cannot masquerade as a little-endian
        one — the exact hazard §1.1 calls out."""
        little = np.array([1.0, 2.0], dtype="<f8")
        big = np.array([1.0, 2.0], dtype=">f8")
        assert np.array_equal(little, big)  # numerically identical...
        assert hash_array(little) != hash_array(big)  # ...but not interchangeable on disk

    def test_distinguishes_shape_at_identical_bytes(self) -> None:
        flat = np.arange(6, dtype=np.float64)
        assert hash_array(flat) != hash_array(flat.reshape(2, 3))

    def test_a_transposed_view_hashes_as_its_own_copy(self) -> None:
        """Contiguity is a memory layout, not a semantic difference."""
        matrix = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert hash_array(matrix.T) == hash_array(np.ascontiguousarray(matrix.T))


class TestContentKey:
    def test_keyword_order_does_not_matter(self) -> None:
        assert content_key(a=1, b=2) == content_key(b=2, a=1)

    def test_any_component_change_changes_the_key(self) -> None:
        base = content_key(schema=1, ids=["r1", "r2"], values=np.array([1.0]))
        assert base != content_key(schema=2, ids=["r1", "r2"], values=np.array([1.0]))
        assert base != content_key(schema=1, ids=["r1", "r3"], values=np.array([1.0]))
        assert base != content_key(schema=1, ids=["r1", "r2"], values=np.array([2.0]))

    def test_reordering_a_list_changes_the_key(self) -> None:
        """Reaction order is semantic: it decides which column of S a flux belongs to."""
        assert content_key(ids=["a", "b"]) != content_key(ids=["b", "a"])

    def test_arrays_are_keyed_by_content_not_identity(self) -> None:
        assert content_key(v=np.arange(5.0)) == content_key(v=np.arange(5.0))

    def test_none_is_hashable(self) -> None:
        assert content_key(x=None) != content_key(x="")

    def test_unhashable_component_is_rejected_loudly(self) -> None:
        with pytest.raises(TypeError, match="cannot hash"):
            content_key(x={1, 2, 3})

    def test_nan_is_rejected_rather_than_silently_keyed(self) -> None:
        with pytest.raises(ValueError):
            content_key(x=float("nan"))


class TestHashFile:
    def test_matches_a_known_sha256(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.txt"
        target.write_bytes(b"abc")
        # sha256("abc"), the standard test vector.
        assert hash_file(target) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_differs_for_different_content(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        assert hash_file(first) != hash_file(second)


class TestProvenance:
    def test_captures_the_environment(self) -> None:
        captured = Provenance.capture()

        assert captured.python_version.startswith("3.11")
        assert captured.numpy_version
        assert captured.byte_order in {"little", "big"}
        assert captured.gsmm_compiler_version

    def test_records_cobra_and_highspy_without_importing_them(self) -> None:
        captured = Provenance.capture()
        assert captured.cobra_version is not None
        assert captured.highspy_version is not None

    def test_is_json_serializable(self) -> None:
        import json

        json.dumps(Provenance.capture().as_dict())
