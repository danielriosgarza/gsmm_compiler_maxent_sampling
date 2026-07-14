"""Native CSC sparse matrix (`NativeCSC`) built column-wise from NumPy arrays.

No ``scipy.sparse`` — these arrays are handed straight to ``highspy.Highs.passModel``.

**Index width (delta from spec §6).** The spec sketches ``starts``/``indices`` as ``int64``. HiGHS
is built here with a 32-bit ``HighsInt`` (``highspy.kHighsIInf == 2**31 - 1``), so int64 arrays are
accepted only by being narrowed on the way in — a silent per-call conversion. We store **int32**
so ``passModel`` takes the arrays as they are, and guard the 2**31 limit at construction.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

INDEX_DTYPE = np.int32
"""Matches HiGHS's ``HighsInt``. See the module docstring."""

VALUE_DTYPE = np.float64
"""float64 everywhere in computation (CLAUDE.md conventions)."""

MASS_BALANCE_SCALE_FLOOR = 1.0
"""Floor on the denominator of `NativeCSC.relative_residual`, in flux units.

Below one flux unit a row's cancellation scale is not a scale, it is noise — and dividing by it is
the M4 error, not a defence against it. See `NativeCSC.relative_residual` for the measured case that
forced this: a metabolite whose only two free reactions are both FVA-blocked reports an *unfloored*
relative residual of exactly 1.0, from an absolute residual of 3.4e-14.

One flux unit is also `GeometryConfig.scale_floor`, and for the same reason: it is the magnitude
below which this model's numbers stop carrying information.
"""

_HIGHS_INT_MAX = 2**31 - 1


class InvalidCSCError(ValueError):
    """A CSC triple violates the compressed-column invariants."""


@dataclass(frozen=True)
class NativeCSC:
    """A compressed-sparse-column matrix in exactly the layout HiGHS consumes.

    ``starts`` has length ``n_cols + 1``; column ``j`` owns entries
    ``indices[starts[j]:starts[j+1]]`` with the matching ``values``. Row indices within a column are
    kept sorted and duplicate-free, which HiGHS does not require but which makes the arrays
    canonical — two structurally equal matrices then hash identically (L1 cache keys, §1.1).
    """

    n_rows: int
    n_cols: int
    starts: NDArray[np.int32]
    indices: NDArray[np.int32]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.validate()

    # ---- invariants ---------------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``InvalidCSCError`` unless every compressed-column invariant holds."""
        if self.n_rows < 0 or self.n_cols < 0:
            raise InvalidCSCError(f"negative shape ({self.n_rows}, {self.n_cols})")

        for name, array, dtype in (
            ("starts", self.starts, INDEX_DTYPE),
            ("indices", self.indices, INDEX_DTYPE),
            ("values", self.values, VALUE_DTYPE),
        ):
            if array.dtype != dtype:
                raise InvalidCSCError(f"{name} has dtype {array.dtype}, expected {np.dtype(dtype)}")
            if array.ndim != 1:
                raise InvalidCSCError(f"{name} must be 1-D, got {array.ndim}-D")

        if self.starts.size != self.n_cols + 1:
            raise InvalidCSCError(
                f"starts has length {self.starts.size}, expected n_cols + 1 = {self.n_cols + 1}"
            )
        if self.indices.size != self.values.size:
            raise InvalidCSCError(
                f"indices ({self.indices.size}) and values ({self.values.size}) differ in length"
            )
        if self.starts[0] != 0:
            raise InvalidCSCError(f"starts[0] must be 0, got {self.starts[0]}")
        if self.starts[-1] != self.values.size:
            raise InvalidCSCError(
                f"starts[-1] is {self.starts[-1]}, expected nnz = {self.values.size}"
            )
        if np.any(np.diff(self.starts) < 0):
            raise InvalidCSCError("starts must be nondecreasing")

        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= self.n_rows):
            raise InvalidCSCError(
                f"row indices out of range [0, {self.n_rows}): "
                f"[{self.indices.min()}, {self.indices.max()}]"
            )
        if not np.all(np.isfinite(self.values)):
            raise InvalidCSCError("values contain NaN or inf")

        # Canonical form: strictly increasing row indices inside each column. This catches an
        # unsorted column *and* a duplicated (row, col) entry — the latter would otherwise be
        # summed implicitly by HiGHS, silently changing the model.
        if self.indices.size > 1:
            nnz = self.indices.size
            ascending = np.diff(self.indices) > 0  # ascending[i] compares entries i and i+1
            # A column boundary at entry j exempts the pair (j−1, j) from ascending: crossing into a
            # new column, the row index is *allowed* to descend. Only j strictly inside (0, nnz)
            # names such a pair. `j == nnz` is a boundary past the last entry — what a **trailing
            # empty column** produces — and indexing `ascending[nnz−1]` for it walks off the end of
            # an array of length nnz−1. A reaction appearing in no metabolite is unusual but legal,
            # and it made this raise IndexError instead of validating.
            joins = self.starts[1:-1]
            boundaries = joins[(joins > 0) & (joins < nnz)]
            ascending[boundaries - 1] = True
            if not np.all(ascending):
                raise InvalidCSCError(
                    "row indices within a column must be strictly increasing "
                    "(unsorted or duplicate entries)"
                )

        if max(self.n_rows, self.n_cols, self.values.size) > _HIGHS_INT_MAX:
            raise InvalidCSCError(
                f"matrix exceeds the 32-bit HighsInt limit ({_HIGHS_INT_MAX}): "
                f"shape=({self.n_rows}, {self.n_cols}), nnz={self.values.size}"
            )

    # ---- structure ----------------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @cached_property
    def _column_of_entry(self) -> NDArray[np.intp]:
        """Column index of every stored entry — the companion of ``starts`` for scatter ops."""
        return np.repeat(np.arange(self.n_cols, dtype=np.intp), np.diff(self.starts))

    # ---- products -----------------------------------------------------------------------------

    def relative_residual(
        self,
        x: NDArray[np.float64],
        rhs: NDArray[np.float64] | None = None,
        *,
        scale_floor: float = MASS_BALANCE_SCALE_FLOOR,
    ) -> NDArray[np.float64]:
        """``|A·x − b|`` per row, divided by that row's cancellation scale — **with a floor**.

        The division is the M4 lesson: ``A·x`` sums terms of size ``|A|·|x|``, so evaluating it in
        float64 already costs ``~eps`` of that before any solver or sampler error enters. An
        absolute bar charges that arithmetic to whoever produced ``x``.

        **The floor is the same lesson applied to the instrument itself.** A relative bar exists to
        forgive *cancellation*: when a sum of large terms collapses to near zero, the result cannot
        be demanded small in absolute terms. Where there is no cancellation, there is nothing to
        forgive — and dividing by the row's own scale then divides by noise, which is the error M4
        catalogued rather than a defence against it.

        Measured on the example model: metabolite ``cpd02375_c0`` is touched by exactly two free
        reactions, and **both are FVA-blocked**, so their fluxes are the centre's residual noise
        (−3.4e-14 and exactly 0). Its residual is ``|S·v| = 3.4e-14`` and its cancellation scale is
        ``|S|·|v| = 3.4e-14`` — *the same number*, because there is one nonzero term and nothing
        cancels. Unfloored, the row reports a relative residual of exactly **1.0** and drowns out
        every real row. With the floor it reports 3.4e-14, which is the truth. 24 rows of this model
        touch a single free reaction, and for every one of them the unfloored ratio is identically 1
        whenever the flux is not exactly zero.

        So the denominator is ``max(|A|·|x|, scale_floor)``: a residual is judged against its row's
        own term magnitudes, but never against a scale below one flux unit, where relative and
        absolute coincide anyway.
        """
        residual = np.abs(self.matvec(x) if rhs is None else self.matvec(x) - rhs)
        denominator = np.maximum(self.cancellation_scale(x), scale_floor)
        return np.asarray(residual / denominator, dtype=VALUE_DTYPE)

    def cancellation_scale(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``|S|·|x|`` — the magnitude of the terms that ``matvec(x)`` cancels, row by row.

        A steady-state residual is a *difference of large numbers*: on the example model
        ``S·v`` sums terms of size ~1e5, so evaluating it in float64 costs ~1e-11 of
        rounding before any solver error is involved, and a polytope with 1e10 bounds
        costs ~1e-6. Callers that want to know whether a residual is a **defect** rather
        than arithmetic need this scale to divide by.

        Note it cannot be had from ``matvec(np.abs(x))``: that still applies the *signed* entries of
        ``S``, so the very cancellation being measured happens all over again.
        """
        vector = np.asarray(x, dtype=VALUE_DTYPE)
        if vector.shape != (self.n_cols,):
            raise ValueError(f"x has shape {vector.shape}, expected ({self.n_cols},)")
        contributions = np.abs(self.values) * np.abs(vector[self._column_of_entry])
        return np.bincount(self.indices, weights=contributions, minlength=self.n_rows).astype(
            VALUE_DTYPE, copy=False
        )

    def cancellation_scale_transpose(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """``|A|ᵀ·|y|`` — the column-wise companion of `cancellation_scale`."""
        vector = np.asarray(y, dtype=VALUE_DTYPE)
        if vector.shape != (self.n_rows,):
            raise ValueError(f"y has shape {vector.shape}, expected ({self.n_rows},)")
        contributions = np.abs(self.values) * np.abs(vector[self.indices])
        return np.bincount(
            self._column_of_entry, weights=contributions, minlength=self.n_cols
        ).astype(VALUE_DTYPE, copy=False)

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``A @ x`` (length ``n_rows``).

        ``bincount`` rather than ``np.add.at``: identical scatter-add semantics, but it runs in C
        instead of taking the slow unbuffered-ufunc path.
        """
        if x.shape != (self.n_cols,):
            raise ValueError(f"x has shape {x.shape}, expected ({self.n_cols},)")
        contributions = self.values * x[self._column_of_entry]
        return np.bincount(self.indices, weights=contributions, minlength=self.n_rows).astype(
            VALUE_DTYPE, copy=False
        )

    def rmatvec(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``Aᵀ @ y`` (length ``n_cols``).

        Scattering by ``_column_of_entry`` rather than ``np.add.reduceat`` over ``starts``:
        ``reduceat`` mishandles empty columns (it re-reads ``values[starts[j]]`` instead of yielding
        0), and structurally empty reaction columns do occur.
        """
        if y.shape != (self.n_rows,):
            raise ValueError(f"y has shape {y.shape}, expected ({self.n_rows},)")
        contributions = self.values * y[self.indices]
        return np.bincount(
            self._column_of_entry, weights=contributions, minlength=self.n_cols
        ).astype(VALUE_DTYPE, copy=False)

    # ---- views --------------------------------------------------------------------------------

    def select_columns(self, columns: NDArray[np.int_] | Sequence[int]) -> NativeCSC:
        """Return the submatrix holding ``columns``, in the given order, rows unchanged.

        This is how the reduced polytope IR drops fixed reactions (`flux_polytope`).
        """
        selected = np.asarray(columns, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("columns must be 1-D")
        if selected.size and (selected.min() < 0 or selected.max() >= self.n_cols):
            raise ValueError(f"column indices out of range [0, {self.n_cols})")

        counts = np.diff(self.starts)[selected].astype(np.intp)
        starts = np.zeros(selected.size + 1, dtype=INDEX_DTYPE)
        np.cumsum(counts, out=starts[1:])

        # Flat positions of the gathered entries: each selected column's block, laid end to end.
        nnz = int(counts.sum())
        offsets_within_column = np.arange(nnz, dtype=np.intp) - np.repeat(
            starts[:-1].astype(np.intp), counts
        )
        entry_positions = np.repeat(self.starts[selected].astype(np.intp), counts) + (
            offsets_within_column
        )

        return NativeCSC(
            n_rows=self.n_rows,
            n_cols=int(selected.size),
            starts=starts,
            indices=self.indices[entry_positions].copy(),
            values=self.values[entry_positions].copy(),
        )

    def to_dense(self) -> NDArray[np.float64]:
        """Return the dense matrix. Tests and small models only — never the numerical path."""
        dense = np.zeros((self.n_rows, self.n_cols), dtype=VALUE_DTYPE)
        dense[self.indices, self._column_of_entry] = self.values
        return dense

    # ---- construction -------------------------------------------------------------------------

    @classmethod
    def from_columns(cls, n_rows: int, columns: Sequence[dict[int, float]]) -> NativeCSC:
        """Build from one ``{row: coefficient}`` mapping per column.

        Rows are sorted here, so callers need not. Explicit zero coefficients are **kept** as
        structural entries: a coefficient a curator wrote as 0 is a modelling statement, and
        dropping it would collapse the CSC content hash of two models that genuinely differ.
        """
        starts = np.zeros(len(columns) + 1, dtype=INDEX_DTYPE)
        indices: list[int] = []
        values: list[float] = []

        for j, column in enumerate(columns):
            for row in sorted(column):
                if not 0 <= row < n_rows:
                    raise InvalidCSCError(f"column {j} references row {row}, outside [0, {n_rows})")
                indices.append(row)
                values.append(column[row])
            starts[j + 1] = len(indices)

        return cls(
            n_rows=n_rows,
            n_cols=len(columns),
            starts=starts,
            indices=np.asarray(indices, dtype=INDEX_DTYPE),
            values=np.asarray(values, dtype=VALUE_DTYPE),
        )

    @classmethod
    def from_dense(cls, dense: NDArray[np.float64]) -> NativeCSC:
        """Build from a dense matrix, keeping only structurally nonzero entries (tests)."""
        matrix = np.asarray(dense, dtype=VALUE_DTYPE)
        if matrix.ndim != 2:
            raise ValueError("dense matrix must be 2-D")
        n_rows, n_cols = matrix.shape
        return cls.from_columns(
            n_rows,
            [
                {int(row): float(matrix[row, col]) for row in np.flatnonzero(matrix[:, col])}
                for col in range(n_cols)
            ],
        )
