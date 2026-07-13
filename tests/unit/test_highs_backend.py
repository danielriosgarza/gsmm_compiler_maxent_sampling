"""`HighsLinearProgram`: the adapter must never hand back a solution it has not checked.

The tests that matter here are the negative ones. A solver that returns `kInfeasible` alongside a
plausible-looking vector of numbers is the failure mode this class exists to prevent — an infeasible
LP whose "solution" flows into geometry would produce a basis for a polytope that does not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.highs_backend import (
    HighsBackendError,
    HighsLinearProgram,
    LPNotOptimalError,
    SolverFrozenError,
    reset_solve_count,
    total_solve_count,
)
from gsmm_compiler.native_csc import NativeCSC


def _program(
    matrix: list[list[float]],
    col_lower: list[float],
    col_upper: list[float],
    row_lower: list[float],
    row_upper: list[float],
    cost: list[float] | None = None,
    **kwargs: object,
) -> HighsLinearProgram:
    dense = np.asarray(matrix, dtype=np.float64)
    return HighsLinearProgram(
        matrix=NativeCSC.from_dense(dense),
        col_lower=np.asarray(col_lower, dtype=np.float64),
        col_upper=np.asarray(col_upper, dtype=np.float64),
        row_lower=np.asarray(row_lower, dtype=np.float64),
        row_upper=np.asarray(row_upper, dtype=np.float64),
        col_cost=None if cost is None else np.asarray(cost, dtype=np.float64),
        **kwargs,  # type: ignore[arg-type]
    )


class TestSolving:
    def test_solves_a_hand_computable_lp(self) -> None:
        """max x + y  s.t.  x + y = 3,  0 ≤ x ≤ 2,  0 ≤ y ≤ 2  →  objective 3."""
        program = _program(
            [[1.0, 1.0]], [0.0, 0.0], [2.0, 2.0], [3.0], [3.0], [1.0, 1.0], maximize=True
        )
        solution = program.solve()

        assert solution.objective_value == pytest.approx(3.0)
        assert solution.primal.sum() == pytest.approx(3.0)
        assert solution.max_primal_infeasibility < 1e-9

    def test_the_primal_vector_is_a_numpy_array_not_a_python_list(self) -> None:
        """The M0 finding, enforced. highspy hands back lists; the adapter converts in one shot, so
        no caller is ever tempted to index a list element-wise (the M9 assertion)."""
        program = _program([[1.0]], [0.0], [1.0], [1.0], [1.0], [1.0])
        solution = program.solve()

        assert isinstance(solution.primal, np.ndarray)
        assert solution.primal.dtype == np.float64
        assert isinstance(solution.row_activity, np.ndarray)

    def test_maximize_and_minimize_disagree(self) -> None:
        program = _program([[1.0]], [-2.0], [5.0], [-np.inf], [np.inf], [1.0])

        program.set_maximize()
        assert program.solve().objective_value == pytest.approx(5.0)
        program.set_minimize()
        assert program.solve().objective_value == pytest.approx(-2.0)

    def test_an_infeasible_lp_raises_rather_than_returning_numbers(self) -> None:
        """x ≥ 1 and x ≤ 2 with the row forcing x = 5: no feasible point exists."""
        program = _program([[1.0]], [1.0], [2.0], [5.0], [5.0], [1.0])

        with pytest.raises(LPNotOptimalError) as raised:
            program.solve()
        assert "Infeasible" in raised.value.model_status

    def test_an_unbounded_lp_raises(self) -> None:
        program = _program([[1.0]], [0.0], [np.inf], [-np.inf], [np.inf], [1.0], maximize=True)

        with pytest.raises(LPNotOptimalError) as raised:
            program.solve()
        assert "Unbounded" in raised.value.model_status

    def test_crossed_bounds_are_rejected_at_construction(self) -> None:
        with pytest.raises(HighsBackendError, match="crossed"):
            _program([[1.0]], [3.0], [1.0], [0.0], [0.0], [1.0])

    def test_nan_bounds_are_rejected(self) -> None:
        """HiGHS would accept a NaN bound and return a solution shaped like an answer."""
        with pytest.raises(HighsBackendError, match="NaN"):
            _program([[1.0]], [np.nan], [1.0], [0.0], [0.0], [1.0])

    def test_an_infinite_cost_is_rejected_but_an_infinite_row_bound_is_not(self) -> None:
        """±inf is how a one-sided row is written; in a cost it is nonsense."""
        with pytest.raises(HighsBackendError, match="finite"):
            _program([[1.0]], [0.0], [1.0], [-np.inf], [0.0], [np.inf])

        program = _program([[1.0]], [0.0], [1.0], [-np.inf], [np.inf], [1.0])
        assert program.solve().objective_value == pytest.approx(1.0)


class TestObjectiveReuse:
    """§11 "Solver reuse": build once, then change only the objective."""

    def test_changing_the_objective_changes_the_optimum(self) -> None:
        # max c·(x, y) over the segment x + y = 1, both in [0, 1]: the optimum is the larger cost.
        program = _program([[1.0, 1.0]], [0.0, 0.0], [1.0, 1.0], [1.0], [1.0], [1.0, 0.0])

        first = program.maximize(np.array([1.0, 0.0]))
        second = program.maximize(np.array([0.0, 3.0]))

        assert first.primal == pytest.approx([1.0, 0.0])
        assert second.primal == pytest.approx([0.0, 1.0])
        assert second.objective_value == pytest.approx(3.0)

    def test_the_objective_property_does_not_alias_the_model(self) -> None:
        program = _program([[1.0]], [0.0], [1.0], [0.0], [1.0], [1.0])

        program.objective[0] = 999.0  # a copy — must not reach HiGHS
        assert program.objective[0] == pytest.approx(1.0)

    def test_a_basis_round_trips(self) -> None:
        program = _program([[1.0, 1.0]], [0.0, 0.0], [1.0, 1.0], [1.0], [1.0], [1.0, 0.0])
        program.solve()

        basis = program.get_basis()
        program.set_basis(basis)

        assert program.solve().objective_value == pytest.approx(1.0)

    def test_warm_started_resolves_are_cheap(self) -> None:
        """The premise of the geometry phase: a re-solve under a new objective starts from the last
        basis, so it costs a handful of pivots rather than a fresh factorization.

        Asserted as a *bound*, not an exact count — the claim is "warm starts work", and pinning the
        pivot count would only pin this HiGHS build's tie-breaking.
        """
        rng = np.random.default_rng(np.random.SeedSequence(3))
        n = 20
        # A simplex-like polytope: sum(x) = 1, x ≥ 0. Every random objective has a vertex optimum.
        program = _program(
            [[1.0] * n], [0.0] * n, [1.0] * n, [1.0], [1.0], [1.0] + [0.0] * (n - 1)
        )
        program.solve()

        iterations = [
            program.maximize(rng.standard_normal(n)).simplex_iterations for _ in range(20)
        ]
        assert max(iterations) <= n, f"warm starts are not working: {iterations}"


class TestSolveCounter:
    """BUILD_PLAN §1.3: no HiGHS solve may run inside the MCMC loop."""

    def test_the_counter_is_process_global(self) -> None:
        """Per-instance counting would let a sampler evade the assertion by building a fresh LP."""
        reset_solve_count()
        first = _program([[1.0]], [0.0], [1.0], [0.0], [1.0], [1.0])
        first.solve()
        first.solve()

        second = _program([[1.0]], [0.0], [1.0], [0.0], [1.0], [1.0])
        second.solve()

        assert first.solve_count == 2
        assert second.solve_count == 1
        assert total_solve_count() == 3

    def test_a_frozen_program_refuses_to_solve(self) -> None:
        """The counter detects an inner-loop solve after the fact; freezing prevents it."""
        program = _program([[1.0]], [0.0], [1.0], [0.0], [1.0], [1.0])
        program.solve()
        program.freeze()

        assert program.is_frozen
        with pytest.raises(SolverFrozenError, match="sampling"):
            program.solve()

    def test_a_refused_solve_is_not_counted(self) -> None:
        program = _program([[1.0]], [0.0], [1.0], [0.0], [1.0], [1.0])
        program.freeze()
        before = total_solve_count()

        with pytest.raises(SolverFrozenError):
            program.solve()

        assert total_solve_count() == before
        assert program.solve_count == 0


def test_highs_applies_the_objective_offset_under_maximize() -> None:
    """Pins the fact `sparse_objective` relies on to report a *complete* J*.

    The eliminated fixed reactions contribute a constant to J that no LP variable can express. It
    rides in as ``lp.offset_``, and this asserts HiGHS adds it with its sign intact under
    ``kMaximize`` — i.e. that HiGHS does not negate the offset along with the costs when it converts
    a maximization internally. If a future HiGHS changed that, every reported J* on a model with a
    nonzero fixed flux would be silently wrong by a constant, and *only* this test would say so.
    """
    program = _program(
        [[1.0]], [0.0], [2.0], [-np.inf], [np.inf], [1.0], maximize=True, offset=10.0
    )

    solution = program.solve()

    assert solution.primal[0] == pytest.approx(2.0)  # the true maximum of x
    assert solution.objective_value == pytest.approx(12.0)  # ...plus the offset, not minus it
