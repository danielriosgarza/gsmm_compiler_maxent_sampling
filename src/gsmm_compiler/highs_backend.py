"""`HighsLinearProgram` — the one place in the package that talks to ``highspy``.

Everything the scientific modules need from a solver goes through this adapter: build a `HighsLp`
from native CSC arrays, `passModel`, change objective coefficients for warm-started re-solves, solve
with status checks, and extract the solution. Raw HiGHS calls are not scattered through the
scientific code (spec §11).

Three properties this module exists to guarantee:

* **No solution is ever returned unchecked.** `solve` raises unless HiGHS reports
  ``kOptimal``; an infeasible or unbounded LP cannot be mistaken for a feasible answer that happens
  to have odd numbers in it.
* **Solutions are extracted in one shot.** M0 established that highspy attribute reads return Python
  ``list``\\ s, not NumPy views — the pybind layer copies on every access. So a solution is
  converted with a single `np.asarray`, never indexed element-wise (the M9 assertion). We also keep
  our own float64 arrays rather than reading the model back out of highspy.
* **Solves are counted, and can be forbidden.** The MCMC inner loop must never call a solver
  (BUILD_PLAN §1.3). `total_solve_count` counts every solve in the *process*, so M5's integration
  test can assert the count is unchanged across sampling even if the sampler were to construct a
  fresh LP; `freeze` turns that assertion into an outright prohibition.

**The objective offset carries the fixed-reaction constant.** Eliminating the ``l == u`` reactions
(§1.5) removes their contribution to ``J`` from the LP's variables, but not from ``J`` itself.
`offset` is added to HiGHS's reported objective with its sign preserved under ``kMaximize`` —
probed, and pinned by ``test_highs_applies_the_objective_offset_under_maximize`` — so the solver
reports the *full* objective and the M3 gate check "solver objective == directly recomputed J" is
one complete equation rather than one that quietly omits a constant.

Implemented in **M3** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.native_csc import INDEX_DTYPE, VALUE_DTYPE, NativeCSC

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

BACKEND_IMPL_VERSION: Final = 1
"""Bumped when a change here can alter the bytes of a solve. Feeds the L2/L3 cache keys (§1.1)."""

DEFAULT_PRIMAL_FEASIBILITY_TOL: Final = 1e-9
"""What we *demand* of a returned solution. HiGHS's tolerance is set to match (see `_OPTIONS`)."""


class HighsBackendError(RuntimeError):
    """HiGHS rejected the model, or returned something we will not trust."""


class LPNotOptimalError(HighsBackendError):
    """The LP terminated at a non-optimal model status (infeasible, unbounded, limit reached…)."""

    def __init__(self, model_status: str, name: str) -> None:
        super().__init__(
            f"LP {name!r} terminated with model status {model_status}, expected kOptimal"
        )
        self.model_status = model_status


class SolverFrozenError(HighsBackendError):
    """A solve was attempted on a program frozen for the MCMC phase (BUILD_PLAN §1.3)."""


# ---- process-global solve counter ---------------------------------------------------------------
#
# Per-instance counts would let a sampler evade the "no solver in the inner loop" assertion just by
# building a new HighsLinearProgram. Counting per process closes that door.

_TOTAL_SOLVES = 0


def total_solve_count() -> int:
    """Number of HiGHS solves executed in this process, across every `HighsLinearProgram`."""
    return _TOTAL_SOLVES


def reset_solve_count() -> None:
    """Zero the process-global counter. Tests only — production code reads deltas, not absolutes."""
    global _TOTAL_SOLVES
    _TOTAL_SOLVES = 0


def _record_solve() -> None:
    global _TOTAL_SOLVES
    _TOTAL_SOLVES += 1


# ---- solutions ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LPSolution:
    """A checked LP solution. Only `HighsLinearProgram.solve` constructs one, and only after HiGHS
    has reported ``kOptimal`` — so holding one of these *is* the guarantee that the LP solved."""

    model_status: str
    run_status: str
    objective_value: float
    """HiGHS's objective, **including** the offset (see the module docstring)."""
    primal: NDArray[np.float64]
    """Column values, length ``n_cols``. Extracted in one conversion."""
    row_activity: NDArray[np.float64]
    """Row values, length ``n_rows`` — the solver's own view of ``A x``."""
    max_primal_infeasibility: float
    simplex_iterations: int
    elapsed_seconds: float


# ---- the adapter --------------------------------------------------------------------------------


class HighsLinearProgram:
    """A HiGHS model built once from native arrays, then re-solved under changing objectives.

    The matrix and the bounds are fixed at construction; only the objective coefficients, the sense,
    and the basis may change afterwards. That is exactly the surface the geometry phase needs (§11
    "Solver reuse"), and refusing to expose more means no scientific module can quietly mutate a
    constraint out from under a cached artifact.
    """

    def __init__(
        self,
        matrix: NativeCSC,
        col_lower: NDArray[np.float64],
        col_upper: NDArray[np.float64],
        row_lower: NDArray[np.float64],
        row_upper: NDArray[np.float64],
        col_cost: NDArray[np.float64] | None = None,
        *,
        maximize: bool = True,
        offset: float = 0.0,
        threads: int = 1,
        name: str = "lp",
    ) -> None:
        import highspy

        self._highs_module = highspy
        self.name = name
        self._n_rows = matrix.n_rows
        self._n_cols = matrix.n_cols
        self._solve_count = 0
        self._frozen = False

        cost = (
            np.zeros(self._n_cols, dtype=VALUE_DTYPE)
            if col_cost is None
            else np.asarray(col_cost, dtype=VALUE_DTYPE).copy()
        )
        self._cost = _check_vector(cost, self._n_cols, "col_cost", allow_infinite=False)
        self._col_lower = _check_vector(col_lower, self._n_cols, "col_lower")
        self._col_upper = _check_vector(col_upper, self._n_cols, "col_upper")
        self._row_lower = _check_vector(row_lower, self._n_rows, "row_lower")
        self._row_upper = _check_vector(row_upper, self._n_rows, "row_upper")

        for kind, lower, upper in (
            ("column", self._col_lower, self._col_upper),
            ("row", self._row_lower, self._row_upper),
        ):
            crossed = np.flatnonzero(lower > upper)
            if crossed.size:
                raise HighsBackendError(
                    f"{kind} bounds are crossed (lower > upper) at indices "
                    f"{crossed[:5].tolist()} — the LP is empty by construction"
                )

        if not np.isfinite(offset):
            raise HighsBackendError(f"objective offset must be finite, got {offset}")
        self._offset = float(offset)
        self._matrix = matrix

        # highspy is a pybind extension: its constructor carries no annotations, so mypy --strict
        # sees an untyped call. The import is deliberately *inside* the constructor, not at module
        # scope, so that an MCMC worker can import this module's callers without loading a solver.
        self._highs = highspy.Highs()  # type: ignore[no-untyped-call]
        for option, value in _OPTIONS.items():
            self._highs.setOptionValue(option, value)
        self._highs.setOptionValue("threads", int(threads))

        lp = highspy.HighsLp()
        lp.num_col_ = self._n_cols
        lp.num_row_ = self._n_rows
        lp.col_cost_ = self._cost
        lp.col_lower_ = self._col_lower
        lp.col_upper_ = self._col_upper
        lp.row_lower_ = self._row_lower
        lp.row_upper_ = self._row_upper
        lp.offset_ = self._offset
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = matrix.starts
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.values
        lp.sense_ = highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize

        status = self._highs.passModel(lp)
        if status == highspy.HighsStatus.kError:
            raise HighsBackendError(f"HiGHS rejected the model {name!r} (passModel gave kError)")

    # ---- structure ------------------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def offset(self) -> float:
        """The constant added to the reported objective — the fixed reactions' contribution to J."""
        return self._offset

    @property
    def objective(self) -> NDArray[np.float64]:
        """The current cost vector (a copy — the caller cannot mutate the model through it)."""
        return self._cost.copy()

    @property
    def solve_count(self) -> int:
        """Solves executed on *this* program. See `total_solve_count` for the process-wide count."""
        return self._solve_count

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # ---- objective ------------------------------------------------------------------------------

    def set_objective(self, costs: NDArray[np.float64]) -> None:
        """Replace every objective coefficient, keeping the basis for a warm-started re-solve."""
        cost = _check_vector(
            np.asarray(costs, dtype=VALUE_DTYPE), self._n_cols, "costs", allow_infinite=False
        )
        columns = np.arange(self._n_cols, dtype=INDEX_DTYPE)
        status = self._highs.changeColsCost(self._n_cols, columns, cost)
        if status == self._highs_module.HighsStatus.kError:
            raise HighsBackendError(f"HiGHS rejected the objective change on {self.name!r}")
        self._cost = cost.copy()

    def set_maximize(self) -> None:
        self._set_sense(maximize=True)

    def set_minimize(self) -> None:
        self._set_sense(maximize=False)

    def _set_sense(self, *, maximize: bool) -> None:
        sense = (
            self._highs_module.ObjSense.kMaximize
            if maximize
            else self._highs_module.ObjSense.kMinimize
        )
        if self._highs.changeObjectiveSense(sense) == self._highs_module.HighsStatus.kError:
            raise HighsBackendError(f"HiGHS rejected the sense change on {self.name!r}")

    # ---- basis ----------------------------------------------------------------------------------

    def get_basis(self) -> Any:
        """The current simplex basis, for explicit reuse across programs (§11 "Solver reuse")."""
        return self._highs.getBasis()

    def set_basis(self, basis: Any) -> None:
        if self._highs.setBasis(basis) == self._highs_module.HighsStatus.kError:
            raise HighsBackendError(f"HiGHS rejected the basis supplied to {self.name!r}")

    # ---- solving --------------------------------------------------------------------------------

    def freeze(self) -> None:
        """Forbid all further solves on this program.

        Called once production sampling starts. The solve *counter* lets a test detect a solver call
        in the inner loop after the fact; freezing makes it impossible in the first place, which is
        the stronger guarantee and costs one boolean.
        """
        self._frozen = True

    def solve(self) -> LPSolution:
        """Solve, and return the solution only if HiGHS reports ``kOptimal``.

        Raises `LPNotOptimalError` on any other model status — an infeasible or unbounded LP must
        not reach the scientific code wearing the clothes of an answer.
        """
        if self._frozen:
            raise SolverFrozenError(
                f"{self.name!r} is frozen: no HiGHS solve may run once sampling has begun "
                "(BUILD_PLAN §1.3)"
            )

        highspy = self._highs_module
        started = time.perf_counter()
        run_status = self._highs.run()
        elapsed = time.perf_counter() - started

        self._solve_count += 1
        _record_solve()

        if run_status == highspy.HighsStatus.kError:
            raise HighsBackendError(f"HiGHS failed to run {self.name!r} (run returned kError)")

        model_status = self._highs.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise LPNotOptimalError(str(model_status), self.name)

        solution = self._highs.getSolution()
        info = self._highs.getInfo()

        # One conversion each, per the M0 finding: these attributes are Python lists.
        primal = np.asarray(solution.col_value, dtype=VALUE_DTYPE)
        row_activity = np.asarray(solution.row_value, dtype=VALUE_DTYPE)
        if primal.shape != (self._n_cols,):
            raise HighsBackendError(
                f"HiGHS returned {primal.size} column values, expected {self._n_cols}"
            )

        return LPSolution(
            model_status=str(model_status),
            run_status=str(run_status),
            objective_value=float(info.objective_function_value),
            primal=primal,
            row_activity=row_activity,
            max_primal_infeasibility=float(info.max_primal_infeasibility),
            simplex_iterations=int(info.simplex_iteration_count),
            elapsed_seconds=elapsed,
        )

    def maximize(self, costs: NDArray[np.float64]) -> LPSolution:
        """Set an objective and solve — the geometry phase's whole interaction with the solver."""
        self.set_objective(costs)
        self.set_maximize()
        return self.solve()


# ---- helpers ------------------------------------------------------------------------------------

_OPTIONS: Final[dict[str, Any]] = {
    "output_flag": False,
    "solver": "simplex",
    # Simplex, not the default choice: the geometry phase re-solves hundreds of LPs that differ only
    # in their objective, and only simplex can warm-start from the previous basis (§11).
    "primal_feasibility_tolerance": DEFAULT_PRIMAL_FEASIBILITY_TOL,
    "dual_feasibility_tolerance": DEFAULT_PRIMAL_FEASIBILITY_TOL,
}


def _check_vector(
    values: NDArray[np.float64], length: int, name: str, *, allow_infinite: bool = True
) -> NDArray[np.float64]:
    """Validate a bound/cost vector and return it as a contiguous float64 array.

    Infinities are legitimate in *row* bounds (that is how a one-sided inequality is written) but
    never in a cost, and NaN is never legitimate anywhere: HiGHS would take a NaN bound without
    complaint and return a solution shaped like an answer.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=VALUE_DTYPE))
    if array.shape != (length,):
        raise HighsBackendError(f"{name} has shape {array.shape}, expected ({length},)")
    if np.any(np.isnan(array)):
        raise HighsBackendError(f"{name} contains NaN")
    if not allow_infinite and not np.all(np.isfinite(array)):
        raise HighsBackendError(f"{name} must be finite")
    return array
