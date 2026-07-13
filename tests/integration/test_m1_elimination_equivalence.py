"""M1 gate: eliminating the ``l == u`` reactions must not change the problem.

The unit tests show that reduced *points* map to feasible full points. That is necessary but not
sufficient: it would still hold if the reduction had quietly shrunk the polytope. The gate is
**equivalence**, so this checks the two things a shrunken polytope could not fake —

* the LP optimum is identical whether solved over the full model or the reduced one, and the reduced
  optimizer reconstructs to the full optimizer's flux vector;
* every vertex the full LP can reach, under many random objectives, is reproduced by the reduced LP.

The LP here is assembled inline from the CSC arrays. M3 builds the real adapter; this is the gate's
own instrument, deliberately independent of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.native_csc import INDEX_DTYPE, NativeCSC


def _solve(
    stoichiometry: NativeCSC,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
    cost: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Maximize ``cost·v`` over ``{S v = rhs, lower ≤ v ≤ upper}``. Returns (objective, argmax)."""
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_ = stoichiometry.n_cols
    lp.num_row_ = stoichiometry.n_rows
    lp.col_cost_ = cost
    lp.col_lower_ = lower
    lp.col_upper_ = upper
    lp.row_lower_ = rhs
    lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = stoichiometry.starts
    lp.a_matrix_.index_ = stoichiometry.indices
    lp.a_matrix_.value_ = stoichiometry.values
    lp.sense_ = highspy.ObjSense.kMaximize

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("threads", 1)
    highs.passModel(lp)
    highs.run()

    assert highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return (
        float(highs.getInfo().objective_function_value),
        np.asarray(highs.getSolution().col_value, dtype=np.float64),
    )


def _solve_full(polytope: FluxPolytope, cost: NDArray[np.float64]) -> tuple[float, NDArray]:
    return _solve(
        polytope.stoichiometry,
        polytope.lower_bounds,
        polytope.upper_bounds,
        np.zeros(polytope.n_metabolites),  # S v = 0
        cost,
    )


def _solve_reduced(
    reduced: ReducedPolytope, cost_full: NDArray[np.float64]
) -> tuple[float, NDArray]:
    """Solve over the reduced polytope, folding the fixed reactions' constant into the objective.

    The fixed fluxes still contribute ``cost·c`` to the objective even though they are no longer
    variables. Dropping that constant is the objective-lowering bug BUILD_PLAN §1.5 warns about: the
    *argmax* would survive it, but the reported optimum would silently differ.
    """
    objective, v_reduced = _solve(
        reduced.stoichiometry,
        reduced.lower_bounds,
        reduced.upper_bounds,
        reduced.rhs,
        cost_full[reduced.free_indices],
    )
    constant = float(cost_full @ reduced.offset)
    return objective + constant, v_reduced


def test_highs_index_width_is_what_native_csc_stores() -> None:
    """The M1 int-width deliverable, pinned rather than remembered.

    ``kHighsIInf == 2**31 - 1`` says HiGHS was built with a 32-bit ``HighsInt``. highspy *accepts*
    int64 arrays — pybind casts them — but each ``passModel`` then narrows a fresh copy. Storing the
    native width keeps the hand-off zero-copy, and makes the 2**31 nnz ceiling an explicit
    construction-time check rather than an overflow discovered inside the solver.

    If a future highspy ships a 64-bit build, this test fails and `native_csc.INDEX_DTYPE` should
    follow it.
    """
    import highspy

    assert highspy.kHighsIInf == 2**31 - 1
    assert np.dtype(INDEX_DTYPE) == np.dtype(np.int32)


class TestToyNetwork:
    """The toy model's FIX reaction is pinned at 2.0, so its reduced RHS is genuinely nonzero."""

    @pytest.fixture
    def polytope(self, toy_canonical: CanonicalModel) -> FluxPolytope:
        return toy_canonical.polytope

    @pytest.fixture
    def reduced(self, polytope: FluxPolytope) -> ReducedPolytope:
        return polytope.reduce()

    def test_biomass_optimum_is_identical(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        cost = np.zeros(polytope.n_reactions)
        cost[polytope.biomass_index] = 1.0

        full_objective, full_v = _solve_full(polytope, cost)
        reduced_objective, reduced_v = _solve_reduced(reduced, cost)

        # A supplies at most 10; 2 of it is forced through FIX; so BIO = 10.
        assert full_objective == pytest.approx(10.0)
        assert reduced_objective == pytest.approx(full_objective, abs=1e-9)
        assert polytope.contains(reduced.to_full(reduced_v))
        np.testing.assert_allclose(reduced.to_full(reduced_v), full_v, atol=1e-9)

    def test_optima_agree_under_many_random_objectives(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """Random costs probe the polytope's vertices in every direction. If the reduction had lost
        (or gained) any part of the feasible set, some direction would expose it."""
        rng = np.random.default_rng(np.random.SeedSequence(2026))

        for _ in range(200):
            cost = rng.standard_normal(polytope.n_reactions)

            full_objective, _ = _solve_full(polytope, cost)
            reduced_objective, reduced_v = _solve_reduced(reduced, cost)

            assert reduced_objective == pytest.approx(full_objective, abs=1e-7)
            assert polytope.contains(reduced.to_full(reduced_v), tol=1e-7)

    def test_forgetting_the_fixed_flux_constant_would_change_the_reported_optimum(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """Guards the guard: prove the objective constant is not vacuously zero on this model, so
        `_solve_reduced` is really testing the lowering rather than adding 0."""
        cost = np.zeros(polytope.n_reactions)
        cost[polytope.reaction_ids.index("FIX")] = 1.0  # FIX is fixed at 2.0

        assert float(cost @ reduced.offset) == pytest.approx(2.0)

        full_objective, _ = _solve_full(polytope, cost)
        reduced_objective, _ = _solve_reduced(reduced, cost)
        assert full_objective == pytest.approx(2.0)
        assert reduced_objective == pytest.approx(full_objective)


class TestExampleModel:
    """The genome-scale case: 513 of 773 reactions eliminated, all of them at zero."""

    @pytest.fixture
    def polytope(self, example_canonical: CanonicalModel) -> FluxPolytope:
        return example_canonical.polytope

    @pytest.fixture
    def reduced(self, polytope: FluxPolytope) -> ReducedPolytope:
        return polytope.reduce()

    def test_reduction_matches_the_documented_geometry(self, reduced: ReducedPolytope) -> None:
        assert reduced.n_free == 260
        assert reduced.n_fixed == 513
        assert reduced.n_full == 773

    def test_the_rhs_is_zero_here_because_every_fixed_reaction_is_blocked(
        self, reduced: ReducedPolytope
    ) -> None:
        """Why the toy network has to exist: this model cannot exercise the affine RHS at all."""
        np.testing.assert_array_equal(reduced.fixed_values, np.zeros(513))
        np.testing.assert_allclose(reduced.rhs, 0.0, atol=0.0)

    def test_fba_optimum_survives_the_reduction(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        cost = np.zeros(polytope.n_reactions)
        cost[polytope.biomass_index] = 1.0

        full_objective, _ = _solve_full(polytope, cost)
        reduced_objective, reduced_v = _solve_reduced(reduced, cost)

        assert full_objective > 0.0
        assert reduced_objective == pytest.approx(full_objective, rel=1e-9)
        assert polytope.contains(reduced.to_full(reduced_v), tol=1e-7)

    def test_optima_agree_under_random_objectives(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(99))

        for _ in range(10):
            cost = rng.standard_normal(polytope.n_reactions)

            full_objective, _ = _solve_full(polytope, cost)
            reduced_objective, reduced_v = _solve_reduced(reduced, cost)

            assert reduced_objective == pytest.approx(full_objective, rel=1e-6, abs=1e-6)
            assert polytope.contains(reduced.to_full(reduced_v), tol=1e-6)

    def test_the_reduced_lp_is_a_smaller_problem(
        self, polytope: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """The point of the exercise: two thirds of the columns are gone."""
        assert reduced.stoichiometry.n_cols == 260
        assert reduced.stoichiometry.nnz < polytope.stoichiometry.nnz
