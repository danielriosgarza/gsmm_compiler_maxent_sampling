"""The sparse objective and its ``(v, z)`` LP, against a toy whose optimum is computable on paper.

The toy is a metabolic fork — one metabolite reachable by a short route or a long one:

    EX_A: → A      DIRECT: A → B                   BIO: B →
                   SLOW1:  A → C,  SLOW2: C → B

Both routes carry the same flux to biomass, but the long one lights up three penalized reactions
instead of two. So with unit weights the L1 term prefers ``DIRECT``, and every optimum below is a
line of arithmetic rather than a number recorded from a previous run:

    route t through DIRECT →  C = 2t,  J = t(1 − 2λ)
    route t through SLOW   →  C = 3t,  J = t(1 − 3λ)
    the origin             →  C = 0,   J = 0        ← always feasible, always available

which makes ``λ* = 1/2`` the exact point where growth stops being worth its cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.highs_backend import LPNotOptimalError
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.sparse_objective import (
    LPCheckError,
    ObjectiveError,
    SparseFluxObjective,
    SparseObjectiveLP,
    _assemble_expanded_csc,
    biomass_maximum,
    build_flux_lp,
    build_sparse_objective_lp,
    solve_sparse_objective,
)

REACTIONS = ("EX_A", "DIRECT", "SLOW1", "SLOW2", "BIO")
METABOLITES = ("A", "B", "C")
BIOMASS = 4

#        EX_A DIRECT SLOW1 SLOW2  BIO
FORK = [
    [1.0, -1.0, -1.0, 0.0, 0.0],  # A
    [0.0, 1.0, 0.0, 1.0, -1.0],  # B
    [0.0, 0.0, 1.0, -1.0, 0.0],  # C
]


def _fork(
    lower: list[float] | None = None, upper: list[float] | None = None
) -> FluxPolytope:
    return FluxPolytope(
        reaction_ids=REACTIONS,
        metabolite_ids=METABOLITES,
        stoichiometry=NativeCSC.from_dense(np.asarray(FORK, dtype=np.float64)),
        lower_bounds=np.asarray(lower if lower else [0.0] * 5, dtype=np.float64),
        upper_bounds=np.asarray(upper if upper else [10.0] * 5, dtype=np.float64),
        biomass_index=BIOMASS,
    )


def _objective(polytope: FluxPolytope, l1_penalty: float, **kwargs: object) -> SparseFluxObjective:
    return SparseFluxObjective.from_polytope(
        polytope, ObjectiveConfig(l1_penalty=l1_penalty), **kwargs  # type: ignore[arg-type]
    )


@pytest.fixture
def fork() -> FluxPolytope:
    return _fork()


@pytest.fixture
def reduced(fork: FluxPolytope) -> ReducedPolytope:
    return fork.reduce()


class TestObjectiveDefinition:
    def test_biomass_is_excluded_from_the_penalty_by_default(self, fork: FluxPolytope) -> None:
        """Penalizing biomass would have J fight its own reward term."""
        objective = _objective(fork, 1.0)

        assert not objective.penalty_mask[BIOMASS]
        assert objective.weights[BIOMASS] == 0.0
        assert objective.penalty_indices.tolist() == [0, 1, 2, 3]

    def test_exchanges_stay_in_the_penalty_set(self, fork: FluxPolytope) -> None:
        """Spec §3.2: excluding them would change the meaning of the cost — import would be free."""
        assert _objective(fork, 1.0).penalty_mask[REACTIONS.index("EX_A")]

    def test_j_is_evaluated_from_the_flux_vector_alone(self, fork: FluxPolytope) -> None:
        """μ = 10, C = 10 + 10 = 20, J = 10 − 0.25·20 = 5."""
        objective = _objective(fork, 0.25)
        v = np.array([10.0, 10.0, 0.0, 0.0, 10.0])

        value = objective.evaluate(v)

        assert value.mu == pytest.approx(10.0)
        assert value.cost == pytest.approx(20.0)
        assert value.total == pytest.approx(5.0)

    def test_components_are_reported_separately(self, fork: FluxPolytope) -> None:
        """Spec §3.2 requires μ and C to survive alongside J, not be collapsed into it."""
        value = _objective(fork, 0.5).evaluate(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert value.total == pytest.approx(value.mu - 0.5 * value.cost)

    def test_weights_outside_the_penalty_set_are_rejected(self, fork: FluxPolytope) -> None:
        weights = np.ones(5)  # includes biomass, which the default mask excludes
        with pytest.raises(ObjectiveError, match="zero outside the penalty set"):
            SparseFluxObjective(
                reaction_ids=REACTIONS,
                biomass_index=BIOMASS,
                l1_penalty=1.0,
                penalty_mask=_objective(fork, 1.0).penalty_mask,
                weights=weights,
            )

    def test_custom_weights_are_zeroed_outside_the_mask_rather_than_silently_kept(
        self, fork: FluxPolytope
    ) -> None:
        objective = _objective(fork, 1.0, weights=np.full(5, 3.0))

        assert objective.weights.tolist() == [3.0, 3.0, 3.0, 3.0, 0.0]

    def test_a_negative_penalty_is_rejected(self, fork: FluxPolytope) -> None:
        """`ObjectiveConfig` already rejects λ < 0 on the config path, so this guards the *other*
        path: an objective built in memory, which is how M7's reweighting will construct them."""
        with pytest.raises(ObjectiveError, match="λ"):
            SparseFluxObjective(
                reaction_ids=REACTIONS,
                biomass_index=BIOMASS,
                l1_penalty=-1.0,
                penalty_mask=_objective(fork, 1.0).penalty_mask,
                weights=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
            )

    def test_an_unknown_penalty_reaction_is_rejected(self, fork: FluxPolytope) -> None:
        with pytest.raises(ObjectiveError, match="not in the model"):
            SparseFluxObjective.from_polytope(fork, penalty_ids=("NOPE",))

    def test_the_manifest_records_the_exact_penalty_set_and_weights(
        self, fork: FluxPolytope
    ) -> None:
        """Spec §3.2 — and §"Hiding the penalty set or weights" is listed as a way to go wrong."""
        manifest = _objective(fork, 0.25).manifest()

        assert manifest["penalty_reaction_ids"] == ["EX_A", "DIRECT", "SLOW1", "SLOW2"]
        assert manifest["weights"] == [1.0, 1.0, 1.0, 1.0]
        assert manifest["l1_penalty"] == 0.25
        assert manifest["biomass_id"] == "BIO"

    def test_the_content_key_moves_with_the_weights(self, fork: FluxPolytope) -> None:
        """L2 keys must miss the cache when the objective changes, or M7 would reload a stale J*."""
        objective = _objective(fork, 0.25)
        reweighted = objective.with_weights(np.array([1.0, 2.0, 1.0, 1.0, 0.0]))

        assert objective.content_key() != reweighted.content_key()
        assert objective.content_key() == _objective(fork, 0.25).content_key()

    def test_with_weights_does_not_mutate_the_original(self, fork: FluxPolytope) -> None:
        """M7's loop must never retarget an objective a chain is already sampling against."""
        objective = _objective(fork, 0.25)
        objective.with_weights(np.array([9.0, 9.0, 9.0, 9.0, 0.0]))

        assert objective.weights.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


class TestExpandedMatrix:
    """§12's direct CSC assembly, checked against an independently written dense matrix."""

    def test_the_expanded_matrix_is_what_section_12_describes(self) -> None:
        """Hand-check: one penalized reaction (column 1), so p = 1, and the layout is

            rows:  [ A B C | DIRECT − z ≤ 0 | −DIRECT − z ≤ 0 ]
            cols:  [ EX_A DIRECT SLOW1 SLOW2 BIO | z_0 ]
        """
        stoichiometry = NativeCSC.from_dense(np.asarray(FORK, dtype=np.float64))
        expanded = _assemble_expanded_csc(stoichiometry, np.array([1], dtype=np.intp))

        expected = np.zeros((5, 6))
        expected[:3, :5] = np.asarray(FORK)
        expected[3, 1] = 1.0  # +v_DIRECT
        expected[3, 5] = -1.0  # −z_0
        expected[4, 1] = -1.0  # −v_DIRECT
        expected[4, 5] = -1.0  # −z_0

        np.testing.assert_array_equal(expanded.to_dense(), expected)

    def test_the_assembly_matches_a_dense_reference_on_every_penalty_subset(self) -> None:
        """The index arithmetic is the part that could be subtly wrong, so vary what it indexes:
        every subset of penalized columns, including none and all."""
        stoichiometry = NativeCSC.from_dense(np.asarray(FORK, dtype=np.float64))
        m, n = stoichiometry.shape

        for subset in range(1 << n):
            penalized = np.array([j for j in range(n) if subset >> j & 1], dtype=np.intp)
            p = penalized.size

            expected = np.zeros((m + 2 * p, n + p))
            expected[:m, :n] = np.asarray(FORK)
            for k, j in enumerate(penalized):
                expected[m + 2 * k, j] = 1.0
                expected[m + 2 * k, n + k] = -1.0
                expected[m + 2 * k + 1, j] = -1.0
                expected[m + 2 * k + 1, n + k] = -1.0

            built = _assemble_expanded_csc(stoichiometry, penalized)
            np.testing.assert_array_equal(built.to_dense(), expected, err_msg=f"subset {subset:b}")

    def test_the_assembly_keeps_the_csc_canonical(self) -> None:
        """`NativeCSC` validates strictly-increasing row indices per column, so a mis-ordered
        append would raise here rather than reaching HiGHS as a silently summed duplicate."""
        stoichiometry = NativeCSC.from_dense(np.asarray(FORK, dtype=np.float64))
        expanded = _assemble_expanded_csc(stoichiometry, np.array([0, 1, 2, 3], dtype=np.intp))

        expanded.validate()  # would raise on unsorted or duplicated entries
        assert expanded.shape == (3 + 8, 5 + 4)


class TestAnalyticOptimum:
    """Every expected value below is the arithmetic in the module docstring, not a recorded run."""

    def test_the_sparse_optimum_takes_the_short_route(self, reduced: ReducedPolytope) -> None:
        """λ = 0.25 < λ*: grow at full tilt, through DIRECT.  J = 10(1 − 2·0.25) = 5."""
        objective = _objective(_fork(), 0.25)
        optimum = build_sparse_objective_lp(reduced, objective).solve()

        assert optimum.j_star == pytest.approx(5.0)
        assert optimum.value.mu == pytest.approx(10.0)
        assert optimum.value.cost == pytest.approx(20.0)
        np.testing.assert_allclose(optimum.v_full, [10.0, 10.0, 0.0, 0.0, 10.0], atol=1e-9)

    def test_the_long_route_is_left_unused_because_it_costs_one_more_reaction(
        self, reduced: ReducedPolytope
    ) -> None:
        """The L1 term's entire job, isolated: both routes reach the same biomass, and the sparser
        one wins. If SLOW carried flux, the penalty would not be doing anything."""
        objective = _objective(_fork(), 0.25)
        optimum = build_sparse_objective_lp(reduced, objective).solve()

        assert optimum.v_full[REACTIONS.index("SLOW1")] == pytest.approx(0.0, abs=1e-9)
        assert optimum.v_full[REACTIONS.index("SLOW2")] == pytest.approx(0.0, abs=1e-9)

    def test_z_equals_absolute_v_at_the_optimum(self, reduced: ReducedPolytope) -> None:
        """Spec §3.3: each z carries a negative cost, so the optimum drives it onto |v|."""
        lp = build_sparse_objective_lp(reduced, _objective(_fork(), 0.25))
        optimum = lp.solve()

        expected = np.abs(optimum.v_full[[0, 1, 2, 3]])  # the four penalized reactions
        np.testing.assert_allclose(optimum.z, expected, atol=1e-9)
        assert optimum.max_z_deviation < 1e-9

    def test_the_solver_objective_equals_the_directly_recomputed_j(
        self, reduced: ReducedPolytope
    ) -> None:
        """The M3 gate, in one line. `solve` would have raised on a mismatch; assert it anyway, so
        the gate criterion is visible as a test rather than buried in a check."""
        optimum = build_sparse_objective_lp(reduced, _objective(_fork(), 0.25)).solve()

        assert optimum.solver_objective == pytest.approx(optimum.value.total, rel=1e-9)

    @pytest.mark.parametrize("l1_penalty", [0.0, 0.1, 0.25, 0.4, 0.49])
    def test_j_star_follows_the_analytic_formula_below_the_cliff(
        self, reduced: ReducedPolytope, l1_penalty: float
    ) -> None:
        """J*(λ) = 10(1 − 2λ) as long as growing beats the origin."""
        optimum = build_sparse_objective_lp(reduced, _objective(_fork(), l1_penalty)).solve()

        assert optimum.j_star == pytest.approx(10.0 * (1.0 - 2.0 * l1_penalty))

    @pytest.mark.parametrize("l1_penalty", [0.51, 1.0, 10.0])
    def test_above_the_cliff_the_optimum_is_the_origin(
        self, reduced: ReducedPolytope, l1_penalty: float
    ) -> None:
        """λ > λ* = 1/2: every unit of biomass costs more L1 than it earns, so the LP stops growing.

        The LP is not wrong here — the origin genuinely maximizes J. **J is wrong.** See
        `SparseObjectiveSolution.is_sparsity_dominated`, and the M3 findings in DEVELOPMENT_STATUS.
        """
        optimum = build_sparse_objective_lp(reduced, _objective(_fork(), l1_penalty)).solve()

        assert optimum.j_star == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(optimum.v_full, np.zeros(5), atol=1e-9)

    def test_j_star_is_nonincreasing_in_lambda(self, reduced: ReducedPolytope) -> None:
        """J*(λ) = max_v (μ − λC) is a maximum of lines with slopes −C ≤ 0: concave, nonincreasing.
        A J* that rose with λ would mean the LP was not finding the optimum."""
        j_stars = [
            build_sparse_objective_lp(reduced, _objective(_fork(), lam)).solve().j_star
            for lam in np.linspace(0.0, 1.0, 11)
        ]

        assert np.all(np.diff(j_stars) <= 1e-9), j_stars

    def test_lambda_zero_builds_no_auxiliary_columns_at_all(
        self, reduced: ReducedPolytope
    ) -> None:
        """With λ = 0 nothing pushes z down onto |v|, so a z column would float anywhere in
        [|v|, z_max] and fail its own check. There is nothing to linearize, so we do not."""
        lp = build_sparse_objective_lp(reduced, _objective(_fork(), 0.0))

        assert lp.n_z_columns == 0
        assert lp.program.n_cols == reduced.n_free
        assert lp.solve().j_star == pytest.approx(10.0)  # pure FBA

    def test_the_biomass_only_diagnostic_reports_the_growth_that_was_available(
        self, reduced: ReducedPolytope
    ) -> None:
        assert biomass_maximum(reduced, _objective(_fork(), 0.25)) == pytest.approx(10.0)


class TestWeightsSteerTheSolution:
    def test_a_heavy_weight_on_the_short_route_pushes_flux_onto_the_long_one(self) -> None:
        """The sharpest test that weights reach the LP: hold λ fixed at 0.1 and change *only* the
        weight of DIRECT, and the cell reroutes its metabolism.

            unit weights, DIRECT: C = 10 + 10 = 20  → J = 10 − 0.1·20 = 8   ← wins
            w_DIRECT = 10,  DIRECT: C = 10 + 100 = 110 → J = 10 − 11 = −1
            w_DIRECT = 10,  SLOW:   C = 10 + 10 + 10 = 30  → J = 10 − 3 = 7  ← wins
        """
        polytope = _fork()
        reduced = polytope.reduce()

        cheap = build_sparse_objective_lp(reduced, _objective(polytope, 0.1)).solve()
        expensive = build_sparse_objective_lp(
            reduced,
            _objective(polytope, 0.1, weights=np.array([1.0, 10.0, 1.0, 1.0, 0.0])),
        ).solve()

        assert cheap.j_star == pytest.approx(8.0)
        np.testing.assert_allclose(cheap.v_full, [10.0, 10.0, 0.0, 0.0, 10.0], atol=1e-9)

        assert expensive.j_star == pytest.approx(7.0)
        np.testing.assert_allclose(expensive.v_full, [10.0, 0.0, 10.0, 10.0, 10.0], atol=1e-9)


class TestDegeneratePolytopes:
    def test_a_fixed_reaction_forces_a_unique_point_and_a_nonzero_objective_offset(self) -> None:
        """The affine case (§1.5): EX_A pinned at 5 forces DIRECT = BIO = 5 through mass balance,
        even though no *free* reaction has l == u. The eliminated EX_A still costs L1, and that
        cost — which no LP variable can express — must reach J via the objective offset.

            J = μ − λC = 5 − 0.25·(5 + 5) = 2.5,  of which J(c) = −0.25·5 = −1.25 is the constant.
        """
        polytope = _fork(lower=[5.0, 0.0, 0.0, 0.0, 0.0], upper=[5.0, 10.0, 0.0, 10.0, 10.0])
        reduced = polytope.reduce()
        objective = _objective(polytope, 0.25)
        lp = build_sparse_objective_lp(reduced, objective)

        # EX_A (pinned at 5) and SLOW1 (closed at 0) are both l == u, so both are eliminated.
        # Only EX_A carries flux, so only EX_A contributes to the constant: J(c) = −0.25·5.
        assert reduced.n_fixed == 2
        assert [REACTIONS[i] for i in reduced.free_indices] == ["DIRECT", "SLOW2", "BIO"]
        assert lp.program.offset == pytest.approx(-1.25)

        optimum = lp.solve()
        assert optimum.j_star == pytest.approx(2.5)
        np.testing.assert_allclose(optimum.v_full, [5.0, 5.0, 0.0, 0.0, 5.0], atol=1e-9)
        assert polytope.contains(optimum.v_full)

    def test_dropping_the_offset_would_have_reported_the_wrong_optimum(self) -> None:
        """Guards the guard: prove the offset is not vacuously zero on the model above, so the
        previous test really is exercising the constant rather than adding nothing."""
        polytope = _fork(lower=[5.0, 0.0, 0.0, 0.0, 0.0], upper=[5.0, 10.0, 0.0, 10.0, 10.0])
        offset = _objective(polytope, 0.25).evaluate(polytope.reduce().offset).total

        assert offset == pytest.approx(-1.25)
        assert offset != 0.0

    def test_an_infeasible_polytope_raises_instead_of_returning_a_vector(self) -> None:
        """A metabolite with a forced inflow and no outlet: mass balance cannot hold."""
        polytope = _fork(
            lower=[1.0, 0.0, 0.0, 0.0, 0.0],  # EX_A must carry ≥ 1...
            upper=[10.0, 0.0, 0.0, 0.0, 10.0],  # ...but nothing can consume A
        )

        with pytest.raises(LPNotOptimalError):
            build_sparse_objective_lp(polytope.reduce(), _objective(polytope, 0.25)).solve()

    def test_a_singleton_polytope_has_no_lp_to_build(self) -> None:
        """Every reaction fixed: there are no variables. Callers must take M4's dim-0 path rather
        than hand HiGHS a model with zero columns."""
        polytope = _fork(lower=[0.0] * 5, upper=[0.0] * 5)
        reduced = polytope.reduce()

        assert reduced.is_singleton
        for build in (
            lambda: build_sparse_objective_lp(reduced, _objective(polytope, 0.25)),
            lambda: build_flux_lp(reduced),
        ):
            with pytest.raises(ObjectiveError, match="single point"):
                build()

    def test_a_biomass_that_is_itself_fixed_still_reports_its_maximum(self) -> None:
        """μ_max with biomass eliminated is its fixed value — there is no LP to solve for it."""
        polytope = _fork(lower=[0.0, 0.0, 0.0, 0.0, 3.0], upper=[10.0, 10.0, 10.0, 10.0, 3.0])
        reduced = polytope.reduce()

        assert reduced.biomass_index is None
        assert biomass_maximum(reduced, _objective(polytope, 0.25)) == pytest.approx(3.0)

    def test_tied_optima_still_give_the_right_objective(self) -> None:
        """Two identical short routes: the LP may pick either, but J* is a property of the polytope
        and must not depend on which vertex the simplex happened to land on."""
        reactions = ("EX_A", "DIRECT", "TWIN", "BIO")
        stoichiometry = np.array(
            [
                [1.0, -1.0, -1.0, 0.0],  # A
                [0.0, 1.0, 1.0, -1.0],  # B
            ]
        )
        polytope = FluxPolytope(
            reaction_ids=reactions,
            metabolite_ids=("A", "B"),
            stoichiometry=NativeCSC.from_dense(stoichiometry),
            lower_bounds=np.zeros(4),
            upper_bounds=np.full(4, 10.0),
            biomass_index=3,
        )
        optimum = build_sparse_objective_lp(
            polytope.reduce(), _objective(polytope, 0.25)
        ).solve()

        # Whichever route it takes: μ = 10, C = 10 (EX_A) + 10 (one route) = 20, J = 5.
        assert optimum.j_star == pytest.approx(5.0)
        assert optimum.v_full[1] + optimum.v_full[2] == pytest.approx(10.0)


class TestSparsityDomination:
    def test_the_diagnostic_flags_an_objective_that_has_abandoned_growth(self) -> None:
        """The failure this bundle exists to catch. From inside the LP, a collapsed optimum looks
        perfectly healthy: optimal status, zero mass-balance residual, z == |v| exactly. Only μ_max
        standing next to μ(v*) reveals that the objective threw away all 10 units of growth."""
        polytope = _fork()
        solution = solve_sparse_objective(polytope.reduce(), _objective(polytope, 5.0))

        assert solution.optimum.j_star == pytest.approx(0.0, abs=1e-9)
        assert solution.biomass_maximum == pytest.approx(10.0)
        assert solution.biomass_retention == pytest.approx(0.0)
        assert solution.is_sparsity_dominated

    def test_a_sane_lambda_is_not_flagged(self) -> None:
        polytope = _fork()
        solution = solve_sparse_objective(polytope.reduce(), _objective(polytope, 0.25))

        assert solution.biomass_retention == pytest.approx(1.0)
        assert not solution.is_sparsity_dominated

    def test_the_diagnostics_block_carries_both_numbers(self) -> None:
        polytope = _fork()
        solution = solve_sparse_objective(polytope.reduce(), _objective(polytope, 0.25))
        diagnostics = solution.diagnostics()

        assert diagnostics["biomass_maximum"] == pytest.approx(10.0)
        assert diagnostics["mu_at_optimum"] == pytest.approx(10.0)
        assert diagnostics["sparsity_dominated"] is False


class TestLPChecks:
    def test_a_disagreement_between_the_solver_and_direct_j_is_fatal(
        self, reduced: ReducedPolytope
    ) -> None:
        """§12 check 8, provoked: hand the checker an objective that is *not* the one HiGHS
        optimized. λ = 0.25 makes the LP report J = 5 at the DIRECT vertex; recomputing that same
        vertex under λ = 0.5 gives J = 10 − 0.5·20 = 0. The two must not be allowed to disagree.

        This is the check that catches a mis-lowered objective — the class of bug that shifts J* by
        a constant and would otherwise pass every other test in this file, because the *flux* is
        perfectly feasible and the *z* are perfectly bound. Only the recomputation notices.
        """
        lp = build_sparse_objective_lp(reduced, _objective(_fork(), 0.25))
        mismatched = SparseObjectiveLP(
            program=lp.program,
            reduced=lp.reduced,
            objective=_objective(_fork(), 0.5),  # a different J than the LP is maximizing
            z_columns=lp.z_columns,
        )

        with pytest.raises(LPCheckError, match="recomputed"):
            mismatched.solve()

    def test_a_z_column_with_no_cost_would_float_free_and_is_caught(
        self, reduced: ReducedPolytope
    ) -> None:
        """Why `effective_costs` gates z columns on λw > 0 rather than on membership of R_p.

        Strip the objective and the rows ``z ≥ ±v`` still hold — but nothing pushes z *down* onto
        |v| any more, so it settles at its upper bound and the linearization silently stops meaning
        `z = |v|`. The check fires. A z column we never build cannot do this.
        """
        lp = build_sparse_objective_lp(reduced, _objective(_fork(), 0.25))
        lp.program.set_objective(np.zeros(lp.program.n_cols))

        with pytest.raises(LPCheckError, match="linearization did not close"):
            lp.solve()

    def test_z_is_never_part_of_the_flux_vector(self, reduced: ReducedPolytope) -> None:
        """Spec §3.4: the auxiliaries are an LP device. If they leaked into v, every flux would gain
        artificial volume and the sampled distribution would not be the one we claim."""
        lp = build_sparse_objective_lp(reduced, _objective(_fork(), 0.25))
        optimum = lp.solve()

        assert optimum.v_full.shape == (len(REACTIONS),)
        assert optimum.z.shape == (lp.n_z_columns,)
        assert lp.program.n_cols == reduced.n_free + lp.n_z_columns
