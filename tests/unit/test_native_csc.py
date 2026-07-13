"""`NativeCSC` — the hand-checked CSC the M1 gate asks for, plus the invariants it must enforce."""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.native_csc import (
    INDEX_DTYPE,
    VALUE_DTYPE,
    InvalidCSCError,
    NativeCSC,
)

# The matrix used throughout, written out so the expected arrays below are checkable by eye:
#
#         c0    c1    c2   c3          starts  = [0, 2, 3, 3, 5]   (c2 is structurally empty)
#   r0 [ 1.0   0.0   0.0  -1.0 ]       indices = [0, 2, 1, 0, 2]
#   r1 [ 0.0   4.0   0.0   0.0 ]       values  = [1.0, -2.0, 4.0, -1.0, 3.0]
#   r2 [-2.0   0.0   0.0   3.0 ]
DENSE = np.array(
    [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 4.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 3.0],
    ],
    dtype=VALUE_DTYPE,
)
EXPECTED_STARTS = np.array([0, 2, 3, 3, 5], dtype=INDEX_DTYPE)
EXPECTED_INDICES = np.array([0, 2, 1, 0, 2], dtype=INDEX_DTYPE)
EXPECTED_VALUES = np.array([1.0, -2.0, 4.0, -1.0, 3.0], dtype=VALUE_DTYPE)


@pytest.fixture
def matrix() -> NativeCSC:
    return NativeCSC.from_dense(DENSE)


def test_csc_arrays_match_the_hand_computed_answer(matrix: NativeCSC) -> None:
    """The gate check: the compressed arrays are exactly what you get doing it on paper."""
    assert matrix.shape == (3, 4)
    assert matrix.nnz == 5
    np.testing.assert_array_equal(matrix.starts, EXPECTED_STARTS)
    np.testing.assert_array_equal(matrix.indices, EXPECTED_INDICES)
    np.testing.assert_array_equal(matrix.values, EXPECTED_VALUES)


def test_dtypes_are_the_widths_highs_expects(matrix: NativeCSC) -> None:
    assert matrix.starts.dtype == np.int32
    assert matrix.indices.dtype == np.int32
    assert matrix.values.dtype == np.float64


def test_to_dense_round_trips(matrix: NativeCSC) -> None:
    np.testing.assert_array_equal(matrix.to_dense(), DENSE)


def test_from_columns_sorts_rows_within_a_column() -> None:
    """Callers may hand us a column in any order; the stored form is canonical."""
    scrambled = NativeCSC.from_columns(3, [{2: -2.0, 0: 1.0}, {1: 4.0}, {}, {2: 3.0, 0: -1.0}])
    np.testing.assert_array_equal(scrambled.indices, EXPECTED_INDICES)
    np.testing.assert_array_equal(scrambled.values, EXPECTED_VALUES)


def test_from_columns_keeps_explicit_zero_coefficients() -> None:
    """A curator's explicit 0 is a modelling statement — dropping it would collide two hashes."""
    with_zero = NativeCSC.from_columns(2, [{0: 0.0, 1: 5.0}])
    assert with_zero.nnz == 2
    np.testing.assert_array_equal(with_zero.values, np.array([0.0, 5.0]))


class TestProducts:
    """`matvec`/`rmatvec` against dense reference products, including the empty-column case."""

    def test_matvec(self, matrix: NativeCSC) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=VALUE_DTYPE)
        np.testing.assert_allclose(matrix.matvec(x), DENSE @ x)

    def test_rmatvec(self, matrix: NativeCSC) -> None:
        y = np.array([1.0, -2.0, 0.5], dtype=VALUE_DTYPE)
        np.testing.assert_allclose(matrix.rmatvec(y), DENSE.T @ y)

    def test_rmatvec_yields_zero_for_a_structurally_empty_column(self, matrix: NativeCSC) -> None:
        """The bug `np.add.reduceat` would have introduced: column 2 has no entries, so its
        transpose product must be 0 — not a stray re-read of a neighbouring value."""
        y = np.array([7.0, 7.0, 7.0], dtype=VALUE_DTYPE)
        assert matrix.rmatvec(y)[2] == 0.0

    def test_products_are_float64(self, matrix: NativeCSC) -> None:
        assert matrix.matvec(np.ones(4)).dtype == VALUE_DTYPE
        assert matrix.rmatvec(np.ones(3)).dtype == VALUE_DTYPE

    @pytest.mark.parametrize("bad_shape", [3, 5])
    def test_matvec_rejects_a_mismatched_vector(self, matrix: NativeCSC, bad_shape: int) -> None:
        with pytest.raises(ValueError, match="expected"):
            matrix.matvec(np.ones(bad_shape))

    def test_random_products_agree_with_dense(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(11))
        dense = rng.standard_normal((17, 23))
        dense[rng.random((17, 23)) < 0.7] = 0.0  # genuinely sparse, with empty rows and columns

        sparse = NativeCSC.from_dense(dense)
        x, y = rng.standard_normal(23), rng.standard_normal(17)

        np.testing.assert_allclose(sparse.matvec(x), dense @ x, atol=1e-12)
        np.testing.assert_allclose(sparse.rmatvec(y), dense.T @ y, atol=1e-12)


class TestSelectColumns:
    def test_selects_and_reorders(self, matrix: NativeCSC) -> None:
        selected = matrix.select_columns([3, 0])
        np.testing.assert_array_equal(selected.to_dense(), DENSE[:, [3, 0]])

    def test_selecting_an_empty_column_keeps_it_empty(self, matrix: NativeCSC) -> None:
        selected = matrix.select_columns([2, 1])
        np.testing.assert_array_equal(selected.starts, np.array([0, 0, 1], dtype=INDEX_DTYPE))
        np.testing.assert_array_equal(selected.to_dense(), DENSE[:, [2, 1]])

    def test_selecting_nothing_yields_an_empty_matrix(self, matrix: NativeCSC) -> None:
        selected = matrix.select_columns([])
        assert selected.shape == (3, 0)
        assert selected.nnz == 0

    def test_duplicate_selection_duplicates_the_column(self, matrix: NativeCSC) -> None:
        selected = matrix.select_columns([1, 1])
        np.testing.assert_array_equal(selected.to_dense(), DENSE[:, [1, 1]])

    def test_rejects_out_of_range(self, matrix: NativeCSC) -> None:
        with pytest.raises(ValueError, match="out of range"):
            matrix.select_columns([4])


class TestValidation:
    """Malformed CSC must be rejected at construction, not discovered inside HiGHS."""

    def test_rejects_wrong_starts_length(self) -> None:
        with pytest.raises(InvalidCSCError, match="starts has length"):
            NativeCSC(
                n_rows=2,
                n_cols=2,
                starts=np.array([0, 1], dtype=INDEX_DTYPE),
                indices=np.array([0], dtype=INDEX_DTYPE),
                values=np.array([1.0]),
            )

    def test_rejects_trailing_start_that_disagrees_with_nnz(self) -> None:
        with pytest.raises(InvalidCSCError, match="expected nnz"):
            NativeCSC(
                n_rows=2,
                n_cols=1,
                starts=np.array([0, 5], dtype=INDEX_DTYPE),
                indices=np.array([0], dtype=INDEX_DTYPE),
                values=np.array([1.0]),
            )

    def test_rejects_decreasing_starts(self) -> None:
        # starts[-1] still equals nnz, so this isolates the monotonicity check itself.
        with pytest.raises(InvalidCSCError, match="nondecreasing"):
            NativeCSC(
                n_rows=2,
                n_cols=3,
                starts=np.array([0, 2, 1, 2], dtype=INDEX_DTYPE),
                indices=np.array([0, 1], dtype=INDEX_DTYPE),
                values=np.array([1.0, 2.0]),
            )

    def test_rejects_row_index_beyond_the_matrix(self) -> None:
        with pytest.raises(InvalidCSCError, match="out of range"):
            NativeCSC(
                n_rows=2,
                n_cols=1,
                starts=np.array([0, 1], dtype=INDEX_DTYPE),
                indices=np.array([9], dtype=INDEX_DTYPE),
                values=np.array([1.0]),
            )

    def test_rejects_nan_values(self) -> None:
        with pytest.raises(InvalidCSCError, match="NaN or inf"):
            NativeCSC(
                n_rows=1,
                n_cols=1,
                starts=np.array([0, 1], dtype=INDEX_DTYPE),
                indices=np.array([0], dtype=INDEX_DTYPE),
                values=np.array([np.nan]),
            )

    def test_rejects_duplicate_entries_within_a_column(self) -> None:
        """HiGHS would silently *sum* these, changing the model behind our back."""
        with pytest.raises(InvalidCSCError, match="strictly increasing"):
            NativeCSC(
                n_rows=3,
                n_cols=1,
                starts=np.array([0, 2], dtype=INDEX_DTYPE),
                indices=np.array([1, 1], dtype=INDEX_DTYPE),
                values=np.array([1.0, 2.0]),
            )

    def test_rejects_unsorted_rows_within_a_column(self) -> None:
        with pytest.raises(InvalidCSCError, match="strictly increasing"):
            NativeCSC(
                n_rows=3,
                n_cols=1,
                starts=np.array([0, 2], dtype=INDEX_DTYPE),
                indices=np.array([2, 0], dtype=INDEX_DTYPE),
                values=np.array([1.0, 2.0]),
            )

    def test_accepts_a_descending_row_index_across_a_column_boundary(self) -> None:
        """Rows must ascend *within* a column; the join between columns may of course drop."""
        matrix = NativeCSC(
            n_rows=3,
            n_cols=2,
            starts=np.array([0, 2, 3], dtype=INDEX_DTYPE),
            indices=np.array([0, 2, 1], dtype=INDEX_DTYPE),  # 2 -> 1 crosses a column boundary
            values=np.array([1.0, 2.0, 3.0]),
        )
        assert matrix.nnz == 3

    def test_rejects_int64_indices(self) -> None:
        """int64 would be narrowed by highspy on every passModel — store the native width."""
        with pytest.raises(InvalidCSCError, match="dtype int64"):
            NativeCSC(
                n_rows=1,
                n_cols=1,
                starts=np.array([0, 1], dtype=np.int64),
                indices=np.array([0], dtype=np.int64),
                values=np.array([1.0]),
            )

    def test_rejects_float32_values(self) -> None:
        with pytest.raises(InvalidCSCError, match="dtype float32"):
            NativeCSC(
                n_rows=1,
                n_cols=1,
                starts=np.array([0, 1], dtype=INDEX_DTYPE),
                indices=np.array([0], dtype=INDEX_DTYPE),
                values=np.array([1.0], dtype=np.float32),
            )

    def test_rejects_a_row_reference_outside_the_matrix_at_build_time(self) -> None:
        with pytest.raises(InvalidCSCError, match="outside"):
            NativeCSC.from_columns(2, [{5: 1.0}])
