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

import dataclasses

import numpy as np
import pytest

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.highs_backend import LPNotOptimalError
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.sparse_objective import (
    DegenerateEnergyScaleError,
    IncompatibleObjectiveError,
    LPCheckError,
    ObjectiveError,
    ReducedObjective,
    SparseFluxObjective,
    SparseObjectiveLP,
    _assemble_expanded_csc,
    biomass_maximum,
    build_flux_lp,
    build_sparse_objective_lp,
    check_compatible,
    choose_energy_scale,
    critical_l1_penalty,
    energy_scale_resolution,
    lower_objective,
    origin_is_feasible,
    resolve_objective,
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
    """A *raw*-λ objective. `resolve_objective` is what turns the config's λ̃ into a raw λ;
    these tests pin the mathematics of J, so they set λ directly."""
    return SparseFluxObjective.from_polytope(
        polytope, l1_penalty=l1_penalty, **kwargs  # type: ignore[arg-type]
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


class TestTheCriticalPenalty:
    """``λ* = max_v μ(v)/C(v)`` — the cliff, and the unit λ is expressed in (BUILD_PLAN §1.7).

    The fork toy pins it analytically. Both routes reach biomass ``t``; the short one costs ``2t``
    and the long one ``3t``, so ``max μ/C = t/2t = 1/2`` **exactly** — the same ``λ* = 1/2`` this
    module's docstring derives from the other direction, as the λ where ``J = t(1 − 2λ)`` turns
    negative and the origin takes over.
    """

    def test_lambda_star_is_exactly_one_half_on_the_fork(self, reduced: ReducedPolytope) -> None:
        """From one LP (Charnes–Cooper), not a bisection — so it is exact, not merely bracketed."""
        critical = critical_l1_penalty(reduced, _objective(_fork(), 0.0))

        assert critical == pytest.approx(0.5, rel=1e-12)

    def test_lambda_star_does_not_depend_on_lambda(self, reduced: ReducedPolytope) -> None:
        """``max μ/C`` is a property of the polytope, the penalty set and the *weights*. If it moved
        with λ, `resolve_objective` would be solving a fixed-point problem instead of a division."""
        criticals = [
            critical_l1_penalty(reduced, _objective(_fork(), lam)) for lam in (0.0, 0.25, 7.0)
        ]

        assert criticals == pytest.approx([0.5, 0.5, 0.5], rel=1e-12)

    def test_lambda_star_does_move_with_the_weights(self, reduced: ReducedPolytope) -> None:
        """Doubling every weight doubles ``C`` and so halves ``λ*``. This is why M7's reweighting
        cannot be allowed to silently drift λ: it changes the very scale λ is measured in."""
        doubled = _objective(_fork(), 0.0, weights=np.array([2.0, 2.0, 2.0, 2.0, 0.0]))

        assert critical_l1_penalty(reduced, doubled) == pytest.approx(0.25, rel=1e-12)

    def test_lambda_star_is_exactly_where_growth_dies(self, reduced: ReducedPolytope) -> None:
        """The property that makes λ* *the* right unit: it is not merely near the cliff, it is the
        cliff. A hair below it the LP still grows; a hair above, the optimum is the origin."""
        polytope = _fork()
        critical = critical_l1_penalty(reduced, _objective(polytope, 0.0))

        below = build_sparse_objective_lp(reduced, _objective(polytope, 0.999 * critical)).solve()
        above = build_sparse_objective_lp(reduced, _objective(polytope, 1.001 * critical)).solve()

        assert below.value.mu > 0.0
        assert below.j_star > 0.0
        assert above.value.mu == pytest.approx(0.0, abs=1e-9)
        assert above.j_star == pytest.approx(0.0, abs=1e-9)

    def test_an_unpenalized_growth_path_has_no_cliff_at_all(self, reduced: ReducedPolytope) -> None:
        """With an empty penalty set, ``C ≡ 0``: growth is free, no λ can ever suppress it, and the
        Charnes–Cooper LP is genuinely unbounded. Reported as ``inf``, not as a solver error."""
        free_lunch = SparseFluxObjective.from_polytope(_fork(), penalty_ids=())

        assert critical_l1_penalty(reduced, free_lunch) == float("inf")


class TestScaleReferencedLambda:
    """`resolve_objective`: the config's dimensionless λ̃ becomes the raw λ that ``J`` uses."""

    def test_lambda_tilde_is_scaled_by_the_model_s_own_cliff(self, fork: FluxPolytope) -> None:
        resolved = resolve_objective(fork, fork.reduce(), ObjectiveConfig(l1_penalty_scaled=0.5))

        assert resolved.scale.critical_l1_penalty == pytest.approx(0.5)
        assert resolved.scale.l1_penalty == pytest.approx(0.25)  # λ̃ · λ* = 0.5 · 0.5
        assert resolved.objective.l1_penalty == pytest.approx(0.25)

    def test_lambda_tilde_zero_is_plain_fba(self, fork: FluxPolytope) -> None:
        resolved = resolve_objective(fork, fork.reduce(), ObjectiveConfig(l1_penalty_scaled=0.0))
        optimum = build_sparse_objective_lp(fork.reduce(), resolved.objective).solve()

        assert resolved.objective.l1_penalty == 0.0
        assert optimum.value.mu == pytest.approx(10.0)  # the full μ_max

    def test_growth_always_survives_below_one(self, fork: FluxPolytope) -> None:
        """The guarantee λ̃ buys: every λ̃ < 1 leaves the cell growing, on any model. Under a raw λ
        you get no such promise — 1.0 is harmless here, ruinous on the genome-scale model."""
        reduced = fork.reduce()

        for scaled in (0.1, 0.5, 0.9, 0.99):
            resolved = resolve_objective(
                fork, reduced, ObjectiveConfig(l1_penalty_scaled=scaled)
            )
            solution = solve_sparse_objective(reduced, resolved.objective)

            assert not solution.is_sparsity_dominated, f"λ̃ = {scaled} collapsed"
            assert solution.optimum.value.mu > 0.0

    def test_sparsity_pressure_rises_monotonically_with_lambda_tilde(
        self, fork: FluxPolytope
    ) -> None:
        """λ̃ has to be a *dial*, not just a safe number: turning it up must cost biomass."""
        reduced = fork.reduce()
        costs = [
            solve_sparse_objective(
                reduced,
                resolve_objective(
                    fork, reduced, ObjectiveConfig(l1_penalty_scaled=scaled)
                ).objective,
            ).optimum.value.cost
            for scaled in (0.0, 0.5, 0.9)
        ]

        assert np.all(np.diff(costs) <= 1e-9), costs

    def test_lambda_tilde_at_or_past_one_is_refused_when_the_origin_is_feasible(
        self, fork: FluxPolytope
    ) -> None:
        """Refused rather than warned: λ̃ ≥ 1 is a *guaranteed* collapse, and there is nothing to
        sample on the far side of it."""
        with pytest.raises(ObjectiveError, match="sparsity cliff"):
            resolve_objective(fork, fork.reduce(), ObjectiveConfig(l1_penalty_scaled=1.0))

    def test_lambda_tilde_past_one_is_allowed_when_flux_is_forced(self) -> None:
        """A cell that *must* spend ATP cannot answer a large λ by shutting down — the origin is not
        available to it. So there is no cliff, and λ̃ ≥ 1 is a legitimate (very sparse) request."""
        forced = _fork(lower=[5.0, 0.0, 0.0, 0.0, 0.0], upper=[5.0, 10.0, 10.0, 10.0, 10.0])

        assert not origin_is_feasible(forced)

        resolved = resolve_objective(
            forced, forced.reduce(), ObjectiveConfig(l1_penalty_scaled=3.0)
        )
        solution = solve_sparse_objective(forced.reduce(), resolved.objective)

        assert not resolved.scale.origin_is_feasible
        assert solution.optimum.value.mu > 0.0  # still alive: EX_A = 5 has to go somewhere

    def test_the_manifest_records_both_lambdas_and_the_cliff(self, fork: FluxPolytope) -> None:
        """Spec §3.6: "No hidden scaling is permitted." A reader must be able to recover the raw λ
        the mathematics actually used, *and* the λ* it was measured against."""
        manifest = resolve_objective(
            fork, fork.reduce(), ObjectiveConfig(l1_penalty_scaled=0.5)
        ).manifest()
        scale = manifest["scale"]
        assert isinstance(scale, dict)

        assert scale["l1_penalty_scaled"] == pytest.approx(0.5)
        assert scale["critical_l1_penalty"] == pytest.approx(0.5)
        assert scale["l1_penalty"] == pytest.approx(0.25)
        assert scale["origin_is_feasible"] is True
        assert manifest["l1_penalty"] == pytest.approx(0.25)  # the raw λ, on the objective itself

    def test_a_free_lunch_model_is_refused_rather_than_scaled_by_infinity(
        self, fork: FluxPolytope
    ) -> None:
        with pytest.raises(ObjectiveError, match="unpenalized"):
            resolve_objective(
                fork, fork.reduce(), ObjectiveConfig(l1_penalty_scaled=0.5), penalty_ids=()
            )


class TestOriginFeasibility:
    def test_the_origin_is_feasible_when_nothing_is_forced(self, fork: FluxPolytope) -> None:
        assert origin_is_feasible(fork)

    def test_a_forced_flux_puts_the_origin_out_of_reach(self) -> None:
        """The ATP-maintenance case: ``l > 0`` on any reaction and ``v = 0`` leaves the polytope."""
        assert not origin_is_feasible(_fork(lower=[1.0, 0.0, 0.0, 0.0, 0.0]))

    def test_a_negative_forced_flux_counts_too(self) -> None:
        """``u < 0`` forces flux just as firmly as ``l > 0``, in the other direction."""
        assert not origin_is_feasible(
            _fork(lower=[-10.0, 0.0, 0.0, 0.0, 0.0], upper=[-1.0, 10.0, 10.0, 10.0, 10.0])
        )


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


# --- M6: lowering J onto the reduced polytope, and the energy scale s_J ---------------------------


class TestLoweringOntoTheReducedPolytope:
    """`lower_objective` — and the one equation the whole of M6 rests on.

    The sampler tilts by an objective indexed in **reduced** coordinates, while ``J*``, the LP
    optimum it is compared against, was computed from a **full** 5-reaction flux vector. Those two
    agree only if the re-indexing, the fixed-flux constant and the penalty set are all right. It is
    the same shape of check as M3's gate ("solver objective == directly recomputed J") and it is
    load-bearing for the same reason: nothing downstream can see it fail.
    """

    def test_the_reduced_objective_equals_the_full_objective_on_the_lift(
        self, fork: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        objective = _objective(fork, 0.25)
        lowered = lower_objective(reduced, objective)
        rng = np.random.default_rng(0)

        for _ in range(200):
            v_reduced = rng.uniform(0.0, 10.0, size=reduced.n_free)
            expected = objective.evaluate(reduced.to_full(v_reduced))
            actual = lowered.evaluate(v_reduced)

            assert actual.mu == pytest.approx(expected.mu)
            assert actual.cost == pytest.approx(expected.cost)
            assert actual.total == pytest.approx(expected.total)

    def test_the_fixed_reactions_l1_cost_is_carried_as_a_constant(self) -> None:
        """``EX_A`` pinned at 2.0 still costs L1 — it is just no longer a variable.

        The whole point of `cost_offset`. Drop it and every reported ``J`` sits ``λ·2`` away from
        the
        ``J*`` it is compared with, so ``s_J = J* − Q(J(W))`` is wrong by nothing (both shift) while
        ``(J − J*)/s_J`` is wrong by ``λ·2/s_J`` on every single sample.
        """
        polytope = _fork(lower=[2.0, 0.0, 0.0, 0.0, 0.0], upper=[2.0, 10.0, 10.0, 10.0, 10.0])
        reduced = polytope.reduce()
        objective = _objective(polytope, 0.25)

        lowered = lower_objective(reduced, objective)

        assert reduced.n_free == 4  # EX_A is gone
        assert lowered.cost_offset == pytest.approx(2.0)  # w = 1, |c| = 2
        assert lowered.mu_offset == 0.0  # biomass is still free
        assert lowered.j_offset == pytest.approx(-0.25 * 2.0)

        v_reduced = np.array([3.0, 1.0, 1.0, 3.0])
        assert lowered.evaluate(v_reduced).total == pytest.approx(
            objective.evaluate(reduced.to_full(v_reduced)).total
        )

    def test_a_fixed_biomass_becomes_a_constant_and_the_line_objective_says_so(self) -> None:
        """``BIO`` pinned at 3.0: the reduced polytope has no biomass column, so ``μ`` is a
        constant.

        `L1Objective.biomass_index` must then be ``None``. Pointing it at any other reduced index
        would tilt the chain by that reaction's flux while every check in the package stayed green.
        """
        polytope = _fork(lower=[0.0, 0.0, 0.0, 0.0, 3.0], upper=[10.0, 10.0, 10.0, 10.0, 3.0])
        reduced = polytope.reduce()
        objective = _objective(polytope, 0.25)

        lowered = lower_objective(reduced, objective)

        assert reduced.biomass_index is None
        assert lowered.line.biomass_index is None
        assert lowered.mu_offset == pytest.approx(3.0)
        assert lowered.cost_offset == 0.0  # BIO is excluded from the penalty set

        v_reduced = np.array([4.0, 2.0, 1.0, 1.0])
        assert lowered.evaluate(v_reduced).mu == pytest.approx(3.0)
        assert lowered.evaluate(v_reduced).total == pytest.approx(
            objective.evaluate(reduced.to_full(v_reduced)).total
        )

    def test_lambda_zero_bends_nothing_but_still_reports_the_cost(
        self, fork: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """The one case where the two penalty sets genuinely differ, and why there are two.

        At ``λ = 0`` no reaction has ``λw > 0``, so nothing bends ``J`` and `line` has no
        breakpoints — correct, and worth the saved work. But ``C(v) = Σ w_r|v_r|`` is defined
        *without* λ and is still a number every run must report. One shared index set would report
        ``C = 0`` for a cell that is plainly metabolizing.
        """
        lowered = lower_objective(reduced, _objective(fork, 0.0))

        assert lowered.line.penalized_indices.size == 0  # nothing bends J
        assert np.count_nonzero(lowered.weights) == 4  # …but four reactions still cost

        v_reduced = np.full(reduced.n_free, 2.0)
        value = lowered.evaluate(v_reduced)

        assert value.cost == pytest.approx(8.0)  # 4 penalized reactions × |2|
        assert value.total == pytest.approx(value.mu)  # J = μ − 0·C

    def test_the_bending_set_is_exactly_the_lps_z_columns(
        self, fork: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """The LP that finds ``J*`` and the kernel that samples around it must agree on which
        reactions are penalized. They do so by construction — both test ``λw > 0`` — and this pins
        it, because a divergence would put the chain's peak somewhere other than at ``v*``."""
        objective = _objective(fork, 0.25)

        lowered = lower_objective(reduced, objective)
        lp = build_sparse_objective_lp(reduced, objective)

        assert lowered.line.penalized_indices.tolist() == lp.z_columns.tolist()

    def test_a_batch_evaluates_to_the_same_thing_as_one_at_a_time(
        self, fork: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        lowered = lower_objective(reduced, _objective(fork, 0.25))
        rng = np.random.default_rng(3)
        batch = rng.uniform(-5.0, 10.0, size=(64, reduced.n_free))

        mu, cost, total = lowered.evaluate_many(batch)

        for i, v in enumerate(batch):
            one = lowered.evaluate(v)
            assert mu[i] == pytest.approx(one.mu)
            assert cost[i] == pytest.approx(one.cost)
            assert total[i] == pytest.approx(one.total)

    def test_a_wrongly_shaped_batch_is_refused(
        self, fork: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        lowered = lower_objective(reduced, _objective(fork, 0.25))

        with pytest.raises(ObjectiveError, match="expected"):
            lowered.evaluate_many(np.zeros((4, reduced.n_free + 1)))


class TestTheEnergyScale:
    """``s_J`` (spec §3.6, §22.2) — the units ``β`` is measured in."""

    @pytest.fixture
    def lowered(self, fork: FluxPolytope, reduced: ReducedPolytope) -> ReducedObjective:
        return lower_objective(reduced, _objective(fork, 0.25))

    @pytest.fixture
    def warmup(self, reduced: ReducedPolytope) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.uniform(0.0, 10.0, size=(40, reduced.n_free))

    def test_warmup_range_is_j_star_minus_the_low_quantile(
        self, lowered: ReducedObjective, warmup: np.ndarray
    ) -> None:
        """``s_J = J* − Q_{0.05}(J(W))``, recomputed here from the definition."""
        j_star = 5.0
        _, _, j_warmup = lowered.evaluate_many(warmup)
        expected = j_star - float(np.quantile(j_warmup, 0.05))

        scale = choose_energy_scale(lowered, warmup, j_star=j_star, mode="warmup_range")

        assert scale.value == pytest.approx(expected)
        assert scale.mode == "warmup_range"
        assert scale.quantile == 0.05
        assert scale.n_warmup_points == 40
        assert not scale.fell_back

    def test_it_measures_the_FULL_j_not_the_reduced_one(self, warmup: np.ndarray) -> None:
        """The trap `choose_energy_scale` takes a `ReducedObjective` to avoid.

        ``J*`` comes from the LP over the *full* flux vector. If ``J(W)`` were computed from
        `L1Objective` — which is ``J`` minus the fixed reactions' contribution — then ``s_J`` would
        absorb that constant and silently rescale **every β on the ladder**. Here the constant is
        ``λ·|c| = 0.25 × 2 = 0.5``, so the wrong ``s_J`` is off by exactly that.
        """
        polytope = _fork(lower=[2.0, 0.0, 0.0, 0.0, 0.0], upper=[2.0, 10.0, 10.0, 10.0, 10.0])
        reduced = polytope.reduce()
        lowered = lower_objective(reduced, _objective(polytope, 0.25))
        points = warmup[:, : reduced.n_free]

        scale = choose_energy_scale(lowered, points, j_star=5.0, mode="warmup_range")

        full = np.array([lowered.evaluate(v).total for v in points])
        reduced_only = np.array([lowered.line.evaluate(v) for v in points])
        assert scale.value == pytest.approx(5.0 - float(np.quantile(full, 0.05)))

        wrong = 5.0 - float(np.quantile(reduced_only, 0.05))
        assert abs(wrong - scale.value) == pytest.approx(0.5), "the constant must actually bite"

    def test_a_declared_scale_is_used_verbatim(
        self, lowered: ReducedObjective, warmup: np.ndarray
    ) -> None:
        scale = choose_energy_scale(lowered, warmup, j_star=5.0, mode=2.5)

        assert scale.value == 2.5
        assert scale.mode == "declared"
        assert scale.quantile is None
        assert not scale.fell_back

    def test_a_degenerate_range_raises_unless_a_fallback_is_declared(
        self, lowered: ReducedObjective, reduced: ReducedPolytope
    ) -> None:
        """Spec §22.2 says to fall back to a "**declared** positive scale" — and a library default
        is
        not a declaration.

        A degenerate range means every support vertex has essentially the LP-optimal ``J``: the
        objective barely varies over this polytope and *no* β means much. Silently substituting
        ``s_J = 1`` there rescales every rung of this strain's ladder, so its β = 2 names a
        different
        selection pressure from every other strain's β = 2 — the one thing ``s_J`` exists to
        prevent.
        And it would arrive as a warning in a log nobody reads. So: no declaration, no run.
        """
        point = np.zeros((3, reduced.n_free))
        j_at_point = lowered.evaluate(point[0]).total

        with pytest.raises(DegenerateEnergyScaleError, match="not resolvable"):
            choose_energy_scale(lowered, point, j_star=j_at_point, mode="warmup_range")

        declared = choose_energy_scale(
            lowered, point, j_star=j_at_point, mode="warmup_range", fallback=2.5
        )
        assert declared.fell_back
        assert declared.value == 2.5
        assert declared.warmup_quantile_j == pytest.approx(j_at_point)

    def test_the_floor_is_a_cancellation_floor_not_a_magnitude_floor(
        self, lowered: ReducedObjective, reduced: ReducedPolytope
    ) -> None:
        """**Codex, M6 review.** The range ``J* − Q(J(W))`` is invariant when a constant is added to
        ``J``. The floor it is compared against must therefore not depend on ``|J*|`` in a way that
        is *not* — or a constant that provably cannot change any probability changes ``s_J``, and
        every β on the ladder with it.

        The old floor was ``1e-9·max(1, |J*|)``. Shift ``J`` by a large enough constant and a
        perfectly healthy ``s_J`` drops below it and is replaced by 1.0, rescaling every rung of the
        ladder. It is the M2 bug (*the absolute magnitude of J must never reach a probability*)
        wearing the calibration layer's hat.

        The floor is now the float64 **resolution of the subtraction itself** — ~64 ULPs of the
        operands — so it refuses only a range the arithmetic genuinely cannot support.

        **The shift has to be big enough that the OLD code actually fails.** The first version of
        this test used ``+1e6``, where the old floor is ``1e-3`` — a thousand times *below* a range
        of 12 — so the buggy code would have sailed straight through it. Codex caught that in round
        2, and it is the M4 lesson again: *a regression test the bug passes is not a regression
        test*. The assertion on `old_floor` below pins the premise, so this one cannot quietly go
        toothless a second time.
        """
        shift = 1e12
        warmup = np.array([[0.0] * reduced.n_free, [6.0] + [0.0] * (reduced.n_free - 1)])
        j_star = 12.0

        plain = choose_energy_scale(lowered, warmup, j_star=j_star, mode="warmup_range")

        shifted_objective = dataclasses.replace(lowered, mu_offset=shift)
        shifted = choose_energy_scale(
            shifted_objective, warmup, j_star=j_star + shift, mode="warmup_range"
        )

        # The premise. Without it this test proves nothing, which is exactly how it was wrong.
        old_floor = 1e-9 * max(1.0, abs(j_star + shift))
        assert plain.value <= old_floor, (
            f"the old magnitude floor ({old_floor:.3e}) does not reject a range of "
            f"{plain.value:.4g}, so this test cannot fail on the bug it exists to catch"
        )

        assert not plain.fell_back
        assert not shifted.fell_back, "an additive constant must not trigger the fallback"

        # The range survives the shift **to within the resolution of the arithmetic that computed
        # it** — which is the strongest claim available, and exactly the one the ULP floor licenses.
        # At a 1e12 baseline one ULP is 1.2e-4, so the shifted range differs in its low bits (by
        # 4.9e-5 here); demanding bit-equality would be demanding precision float64 does not have.
        # What must hold is that the discrepancy stays *below the floor*, or the floor would be
        # rejecting ranges it cannot itself resolve.
        assert shifted.resolution is not None
        assert abs(shifted.value - plain.value) < shifted.resolution
        assert shifted.resolution < plain.value  # …and the range clears that floor comfortably

    def test_the_resolution_it_used_is_recorded(
        self, lowered: ReducedObjective, warmup: np.ndarray
    ) -> None:
        """A range one ULP above the floor is technically resolvable and scientifically worthless.
        Only this number says which one you got."""
        scale = choose_energy_scale(lowered, warmup, j_star=5.0, mode="warmup_range")

        assert scale.resolution is not None
        assert scale.resolution == pytest.approx(
            energy_scale_resolution(5.0, scale.warmup_quantile_j)
        )
        assert scale.value > scale.resolution

    def test_a_nonpositive_or_unknown_mode_is_refused(
        self, lowered: ReducedObjective, warmup: np.ndarray
    ) -> None:
        for bad in (0.0, -1.0, float("inf")):
            with pytest.raises(ObjectiveError, match="finite and > 0"):
                choose_energy_scale(lowered, warmup, j_star=5.0, mode=bad)

        with pytest.raises(ObjectiveError, match="warmup_range"):
            choose_energy_scale(lowered, warmup, j_star=5.0, mode="pilot_range")

        with pytest.raises(ObjectiveError, match="quantile"):
            choose_energy_scale(lowered, warmup, j_star=5.0, quantile=1.0)

    def test_the_manifest_records_everything_needed_to_reproduce_beta(
        self, lowered: ReducedObjective, warmup: np.ndarray
    ) -> None:
        """Spec §3.6: no hidden scaling. ``β`` is meaningless without the ``s_J`` it was divided by,
        so the manifest must carry both, and the ``J*`` they are relative to."""
        manifest = choose_energy_scale(
            lowered, warmup, j_star=5.0, mode="warmup_range"
        ).manifest()

        assert set(manifest) >= {
            "energy_scale",
            "energy_scale_mode",
            "j_star",
            "energy_scale_quantile",
            "warmup_quantile_j",
            "energy_scale_fell_back",
        }


class TestTheObjectiveAndThePolytopeMustDescribeTheSameModel:
    """**Codex, M6 review rounds 3–4.** `check_compatible` — one guard, called at every join.

    An objective is a biomass index, a mask and a weight vector; a polytope is a matrix and some
    bounds. Neither knows the other exists, and an index is just an integer — so a mismatched pair
    is not *detectable* downstream, it is **computable**, and it computes confidently.

    The bugs were found one join at a time (M6 review rounds 3 and 4) until it became clear that
    patching joins is not the same as having an invariant. Now there is one function, and every
    public entry point calls it.
    """

    def _model(self, ids: tuple[str, ...], biomass: int) -> FluxPolytope:
        matrix = NativeCSC.from_dense(np.array([[1.0, 1.0, -1.0]], dtype=np.float64))
        return FluxPolytope(
            reaction_ids=ids,
            metabolite_ids=("m",),
            stoichiometry=matrix,
            lower_bounds=np.array([1.0, 2.0, 0.0]),
            upper_bounds=np.array([1.0, 2.0, 10.0]),  # a and b are FIXED, c is free
            biomass_index=biomass,
        )

    def test_a_different_reaction_set_is_refused_even_when_the_indices_agree(self) -> None:
        """Comparing biomass *indices* is not enough. Objective biomass ``"a"`` at index 0 and
        polytope biomass ``"x"`` at index 0 agree numerically while naming different reactions of
        different models — and every index in the objective then addresses the wrong reaction."""
        objective = _objective(self._model(("a", "b", "c"), 0), 0.0)
        other = self._model(("x", "b", "c"), 0).reduce()

        with pytest.raises(IncompatibleObjectiveError, match="same reaction set"):
            lower_objective(other, objective)

    def test_a_different_biomass_reaction_is_refused(self) -> None:
        """The round-3 case. ``a`` is fixed at 1.0 and ``b`` at 2.0, so ``μ`` **is** that constant —
        the entire objective is wrong, and every LP check still passes."""
        objective = _objective(self._model(("a", "b", "c"), 0), 0.0)  # biomass = a
        polytope = self._model(("a", "b", "c"), 1).reduce()  # biomass = b

        with pytest.raises(IncompatibleObjectiveError, match="biomass"):
            lower_objective(polytope, objective)

    @pytest.mark.parametrize("entry_point", ["lp", "solve", "biomass_max", "critical"])
    def test_every_public_join_refuses_it_not_merely_the_one_that_was_patched(
        self, entry_point: str
    ) -> None:
        """The point of a shared guard. Each of these was, at some stage of the review, the *only*
        one that checked — and each of the others let the same mismatch straight through."""
        objective = _objective(self._model(("a", "b", "c"), 0), 0.25)  # biomass = a
        polytope = self._model(("a", "b", "c"), 1).reduce()  # biomass = b

        call = {
            "lp": lambda: build_sparse_objective_lp(polytope, objective),
            "solve": lambda: solve_sparse_objective(polytope, objective),
            "biomass_max": lambda: biomass_maximum(polytope, objective),
            "critical": lambda: critical_l1_penalty(polytope, objective),
        }[entry_point]

        with pytest.raises(IncompatibleObjectiveError):
            call()

    def test_the_matching_pair_is_accepted(self) -> None:
        polytope = self._model(("a", "b", "c"), 0)
        reduced = polytope.reduce()
        objective = _objective(polytope, 0.25)

        check_compatible(reduced, objective)  # does not raise
        lowered = lower_objective(reduced, objective)

        assert lowered.binds_to(reduced)
        assert reduced.biomass_id == "a"
        assert lowered.mu_offset == pytest.approx(1.0)  # biomass 'a' is fixed at 1.0

    def test_a_reduced_polytope_from_a_different_canonical_one_is_refused(self) -> None:
        """**Codex, M6 review round 5.** `resolve_objective` reads `origin_is_feasible` off the
        **canonical** bounds and computes ``λ*`` off the **reduced** LP. Hand it a mismatched pair
        and BUILD_PLAN §1.7's sparsity-cliff guard *inverts*.

        Both polytopes here have the same reactions and the same biomass — they differ only in
        ``EX``'s lower bound. The forced-flux one (``l = 1``) has **no** feasible origin, so the
        guard permits ``λ̃ ≥ 1``. The relaxed one (``l = 0``) does, so the guard must **refuse** it.
        Pass the forced canonical with the relaxed reduction and ``λ̃ = 1.5`` sails through, as
        ``origin_is_feasible = False`` — while the polytope that actually gets sampled collapses to
        ``v* = 0``, the exact failure §1.7 exists to prevent (and which no LP check can see: optimal
        status, zero residual, ``z = |v|`` exactly).
        """
        matrix = NativeCSC.from_dense(np.array([[1.0, -1.0]], dtype=np.float64))
        common = dict(
            reaction_ids=("EX", "BIO"),
            metabolite_ids=("m",),
            stoichiometry=matrix,
            upper_bounds=np.array([10.0, 10.0]),
            biomass_index=1,
        )
        forced = FluxPolytope(**common, lower_bounds=np.array([1.0, 0.0]))  # type: ignore[arg-type]
        relaxed = FluxPolytope(**common, lower_bounds=np.array([0.0, 0.0]))  # type: ignore[arg-type]

        assert not origin_is_feasible(forced)  # a forced flux: the cell cannot shut down
        assert origin_is_feasible(relaxed)  # …but this one can, so the cliff is real

        with pytest.raises(IncompatibleObjectiveError, match="not the reduction"):
            resolve_objective(forced, relaxed.reduce(), ObjectiveConfig(l1_penalty_scaled=1.5))

    def test_omitting_the_reduced_polytope_derives_it_safely(self) -> None:
        """The safe default: derive it, so the two cannot disagree — §1.7's guard still bites."""
        matrix = NativeCSC.from_dense(np.array([[1.0, -1.0]], dtype=np.float64))
        relaxed = FluxPolytope(
            reaction_ids=("EX", "BIO"),
            metabolite_ids=("m",),
            stoichiometry=matrix,
            lower_bounds=np.zeros(2),
            upper_bounds=np.array([10.0, 10.0]),
            biomass_index=1,
        )

        resolved = resolve_objective(relaxed, config=ObjectiveConfig(l1_penalty_scaled=0.5))
        assert resolved.scale.origin_is_feasible
        assert resolved.scale.l1_penalty < resolved.scale.critical_l1_penalty

        with pytest.raises(ObjectiveError, match="sparsity cliff"):
            resolve_objective(relaxed, config=ObjectiveConfig(l1_penalty_scaled=1.5))
