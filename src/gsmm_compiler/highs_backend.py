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

BACKEND_IMPL_VERSION: Final = 2
"""Bumped when a change here can alter the bytes of a solve. Feeds the L2/L3 cache keys (§1.1).

2 (M11.0): the sentence above was **false for the pre-build lookup key** from v1 until now, which is
why `solver_identity` exists. This bump is also the first live exercise of that fix: it must reach
warm L3 caches as a *miss*, and before M11.0 it would have reached them as a hit whose bundle then
failed its content-key check — an error where §1.1 requires a rebuild.
"""

DEFAULT_PRIMAL_FEASIBILITY_TOL: Final = 1e-9
"""What we *demand* of a returned solution. HiGHS's tolerance is set to match (see `_OPTIONS`)."""


def solver_identity() -> dict[str, Any]:
    """Everything about the solver that can move a solve's bytes, knowable **before** the solve.

    This exists because of the asymmetry between the two keys an artifact has. `content_key` is
    computed *from* a built artifact, so it can hash the artifact's own bytes; the **lookup** key
    must decide whether to build at all, so it may only hash *inputs* — §1.1: "hashing the artifact
    to decide whether to build the artifact is circular." `ReducedGeometry.content_key` therefore
    folded in `BACKEND_IMPL_VERSION` while `batch.geometry_cache_key` did not, and the constant's
    own docstring claimed both.

    **The HiGHS version is the half that was silent.** `BACKEND_IMPL_VERSION` is ours and a
    reviewer can bump it; the solver's version is not, and nothing bumps on a ``uv sync``. It was
    absent from the content key *and* the lookup key, so upgrading highspy served a cache warmed by
    the previous solver under an unchanged name. The geometry's support points are LP outputs — a
    different HiGHS can move them.

    Read via `importlib.metadata`, never ``highspy.__version__``: a warm run must be able to key the
    cache **without importing the solver**, which is the same property M10.2d bought for cobra at
    L0, and which `HighsLinearProgram`'s constructor-scoped import exists to protect.
    """
    from gsmm_compiler.provenance import _installed_version

    return {
        "backend_impl_version": BACKEND_IMPL_VERSION,
        "highs_version": _installed_version("highspy"),
    }


class HighsBackendError(RuntimeError):
    """HiGHS rejected the model, or returned something we will not trust."""


class LPNotOptimalError(HighsBackendError):
    """The LP terminated at a model status the caller does not accept.

    ``solve`` accepts only ``kOptimal``. The dual-witness path (`solve_dual_witness`) accepts a
    caller-declared whitelist, so the message names *what was expected* rather than hardcoding
    ``kOptimal`` — a witness that legitimately accepts ``kUnknown`` must not report that a status
    it accepted was "expected kOptimal". Callers still discriminate on ``model_status`` (e.g.
    `sparse_objective.critical_l1_penalty` on unbounded ``J*``, `affine_geometry`'s ``kUnknown``
    escalation), which is why this stays one exception type carrying the raw status string.
    """

    def __init__(
        self, model_status: str, name: str, *, accepted: frozenset[str] | None = None
    ) -> None:
        expected = "kOptimal" if accepted is None else f"one of {sorted(accepted)}"
        super().__init__(
            f"LP {name!r} terminated with model status {model_status}, expected {expected}"
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
    reduced_costs: NDArray[np.float64]
    """Column duals, length ``n_cols`` — HiGHS's own reduced costs."""
    row_duals: NDArray[np.float64]
    """Row duals, length ``n_rows``. The raw material of a *rigorous* optimality bound.

    A caller measuring a quantity as a **difference of two optima** — a support width, say — must
    know how much the solver may have left on the table, and no primal quantity can tell it. What
    can is **weak duality**: for the box-constrained ``max cᵀv s.t. Av = b, l ≤ v ≤ u`` and *any*
    ``y`` whatsoever,

        ``max cᵀv  ≤  bᵀy + Σⱼ max(dⱼlⱼ, dⱼuⱼ)``,   ``d = c − Aᵀy``

    — an upper bound that assumes nothing about ``y`` being optimal, or even good. That
    is the point:
    a bound built instead from complementary slackness would inherit every assumption about how the
    solver chose to stop, including that its returned point is exactly row-feasible,
    which it is not.
    """
    max_primal_infeasibility: float
    max_dual_infeasibility: float
    """The optimality side of the story, and the one it is tempting to omit.

    ``kOptimal`` means "optimal to the configured tolerances", not "optimal". A returned point
    can be
    perfectly primal-feasible and still leave an improving reduced cost on the table, if that
    cost is
    below the dual tolerance — and a caller that measures a *width* by optimizing in two opposite
    directions would then read that width as **too small**, silently. Primal infeasibility cannot
    see
    this; only this number can. `affine_geometry` gates its span certificate on it.
    """
    simplex_iterations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class LPDualWitness:
    """Row multipliers from a solve that may **not** be optimal — the raw material of a weak-duality
    bound, and nothing more.

    Deliberately not an `LPSolution`. Holding an `LPSolution` *proves* HiGHS reported ``kOptimal``
    (that is the whole point of `solve` raising otherwise); holding one of these proves only that
    the solver ran and returned a status the caller had **declared acceptable** for a dual-only use.
    It carries no ``primal``, no ``objective_value`` — because a caller that reads those would be
    trusting a point the solver did not certify. The one legitimate consumer is a bound of the form
    ``max cᵀv ≤ bᵀy + Σⱼ max(dⱼlⱼ, dⱼuⱼ)`` (see `LPSolution.row_duals`), which holds for **any**
    finite ``y`` — no optimality, dual feasibility, or primal feasibility required. So a
    ``kUnknown`` solve's duals still give an upper bound; only *tightness* is at the solver's mercy.

    `solve_dual_witness` validates ``row_duals.shape == (n_rows,)`` before constructing one: a model
    status alone does not guarantee HiGHS populated the dual vector (``HighsSolution::clear`` leaves
    it empty, and some ``kUnknown`` paths discard duals), so an empty or wrong-length vector must
    fail closed rather than be read as "no binding constraints". (Codex, M11.3 review, verified
    against the HiGHS 1.15.1 source.)
    """

    model_status: str
    run_status: str
    row_duals: NDArray[np.float64]
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

    def _run_solver(self) -> tuple[Any, Any, float]:
        """Run HiGHS once and return ``(model_status, run_status, elapsed)`` — no solution read.

        The shared prefix of `solve` and `solve_dual_witness`: the frozen-check, the timed ``run``,
        the process solve counter, and the ``kError`` guard. Neither the ``kOptimal`` gate nor
        ``getSolution`` lives here, because that is exactly where the two paths diverge — `solve`
        accepts only ``kOptimal`` and reads a full `LPSolution`; the witness path accepts a declared
        whitelist and reads only the row duals. Extracting this keeps `solve`'s observable behaviour
        byte-identical (it still raises before ``getSolution`` on a non-optimal status).
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

        return self._highs.getModelStatus(), run_status, elapsed

    def solve(self) -> LPSolution:
        """Solve, and return the solution only if HiGHS reports ``kOptimal``.

        Raises `LPNotOptimalError` on any other model status — an infeasible or unbounded LP must
        not reach the scientific code wearing the clothes of an answer. **This gate is load-bearing
        elsewhere** and must not be loosened: `sparse_objective.critical_l1_penalty` reads an
        unbounded status *through* this exception to detect a zero-cost growth path, and
        `affine_geometry` escalates a ``kUnknown`` FVA solve by catching it. A caller that needs the
        duals of a non-optimal solve uses `solve_dual_witness`, which does not touch this method.
        """
        highspy = self._highs_module
        model_status, run_status, elapsed = self._run_solver()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise LPNotOptimalError(str(model_status), self.name)

        solution = self._highs.getSolution()
        info = self._highs.getInfo()

        # One conversion each, per the M0 finding: these attributes are Python lists.
        primal = np.asarray(solution.col_value, dtype=VALUE_DTYPE)
        row_activity = np.asarray(solution.row_value, dtype=VALUE_DTYPE)
        reduced_costs = np.asarray(solution.col_dual, dtype=VALUE_DTYPE)
        row_duals = np.asarray(solution.row_dual, dtype=VALUE_DTYPE)
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
            reduced_costs=reduced_costs,
            row_duals=row_duals,
            max_primal_infeasibility=float(info.max_primal_infeasibility),
            max_dual_infeasibility=float(info.max_dual_infeasibility),
            simplex_iterations=int(info.simplex_iteration_count),
            elapsed_seconds=elapsed,
        )

    def maximize(self, costs: NDArray[np.float64]) -> LPSolution:
        """Set an objective and solve — the geometry phase's whole interaction with the solver."""
        self.set_objective(costs)
        self.set_maximize()
        return self.solve()

    def solve_dual_witness(self, *, accept: frozenset[str]) -> LPDualWitness:
        """Solve, and return the **row duals** whenever HiGHS reports a status in ``accept``.

        For a caller whose only use of the solution is a weak-duality bound (`LPDualWitness`), the
        ``kOptimal`` gate in `solve` is too strong: the bound holds for *any* finite duals, so a
        ``kUnknown`` solve — a warm-started instance that could not *certify* optimality though its
        duals stay sound — is usable. This reads them without loosening `solve` (which must go on
        refusing for every other caller, `sparse_objective.critical_l1_penalty` among them).

        ``accept`` is a whitelist of `HighsModelStatus` **member names** (e.g. ``"kOptimal"``,
        ``"kUnknown"``), and it is the caller's assertion that every status in it is one whose duals
        are meaningful *for its LP*. It is matched by exact name — not the substring test used
        loosely elsewhere, which would let ``"Unbounded"`` also accept ``kUnboundedOrInfeasible`` —
        and the tokens are validated against the live enum, so a typo fails loudly here rather than
        silently refusing every solve. A status outside ``accept`` raises `LPNotOptimalError` before
        `getSolution` is ever called, exactly as `solve` refuses before reading.

        No dual-**quality** signal is consulted (``max_dual_infeasibility`` and the like): gating a
        dual-based bound on a quality reading is the anti-pattern the whole M11 family traces to.
        The only guards are structural — the status whitelist, and that the returned dual vector has
        the right length (a model status does not guarantee HiGHS populated it).
        """
        highspy = self._highs_module
        valid = set(highspy.HighsModelStatus.__members__)
        unknown_tokens = accept - valid
        if unknown_tokens:
            raise HighsBackendError(
                f"solve_dual_witness accept={sorted(accept)} names statuses that are not HiGHS "
                f"model statuses: {sorted(unknown_tokens)} — a typo would silently refuse every LP"
            )

        model_status, run_status, elapsed = self._run_solver()
        if model_status.name not in accept:
            raise LPNotOptimalError(str(model_status), self.name, accepted=accept)

        solution = self._highs.getSolution()
        row_duals = np.asarray(solution.row_dual, dtype=VALUE_DTYPE)
        if row_duals.shape != (self._n_rows,):
            raise HighsBackendError(
                f"HiGHS returned {row_duals.size} row duals at status {model_status!s}, expected "
                f"{self._n_rows}: an accepted status does not guarantee a populated dual vector, "
                "so no bound can be built from this one"
            )
        return LPDualWitness(
            model_status=str(model_status),
            run_status=str(run_status),
            row_duals=row_duals,
            elapsed_seconds=elapsed,
        )

    def maximize_dual_witness(
        self, costs: NDArray[np.float64], *, accept: frozenset[str]
    ) -> LPDualWitness:
        """Set an objective and solve for its dual witness — the witness twin of `maximize`."""
        self.set_objective(costs)
        self.set_maximize()
        return self.solve_dual_witness(accept=accept)


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
