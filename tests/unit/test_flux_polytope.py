"""`FluxPolytope` → `ReducedPolytope`: elimination, the affine RHS, and reconstruction.

The M1 gate wants *exact full-model reconstruction* and *elimination equivalence*. The case that
actually has teeth is the one the example model cannot provide: a reaction fixed at a **nonzero**
value, which makes the reduced mass balance affine (``S_F v_F = −S_fixed v_fixed ≠ 0``). The toy
network exists to supply it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.flux_polytope import (
    FluxPolytope,
    InvalidPolytopeError,
    ReducedPolytope,
)
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.native_csc import VALUE_DTYPE, NativeCSC

# ---- toy-network facts, derived by hand ---------------------------------------------------------
#
# Reactions (frozen order):  EX_A  R1  R2  R3  FIX  BLK  BIO
# Metabolites:               A  B  C
#
#   A:  EX_A − R1 − R3 − FIX             = 0
#   B:        R1 − R2         − BLK      = 0
#   C:             R2 + R3 + FIX + BLK − BIO = 0
#
# FIX is fixed at 2.0 (nonzero!), BLK at 0.0. Free: EX_A, R1, R2, R3, BIO.
# So the reduced RHS is −S_fixed·v_fixed = −(FIX column)·2 = (+2, 0, −2).
FREE_IDS = ("EX_A", "R1", "R2", "R3", "BIO")
EXPECTED_RHS = np.array([2.0, 0.0, -2.0])


@pytest.fixture
def toy(toy_canonical: CanonicalModel) -> FluxPolytope:
    return toy_canonical.polytope


@pytest.fixture
def reduced(toy: FluxPolytope) -> ReducedPolytope:
    return toy.reduce()


class TestCanonicalPolytope:
    def test_order_is_frozen_as_written_in_the_file(self, toy: FluxPolytope) -> None:
        assert toy.reaction_ids == ("EX_A", "R1", "R2", "R3", "FIX", "BLK", "BIO")
        assert toy.metabolite_ids == ("A", "B", "C")

    def test_stoichiometry_matches_the_hand_written_matrix(self, toy: FluxPolytope) -> None:
        expected = np.array(
            [
                [1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0],  # A
                [0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0],  # B
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],  # C
            ]
        )
        np.testing.assert_array_equal(toy.stoichiometry.to_dense(), expected)

    def test_fixed_mask_finds_both_fixed_reactions(self, toy: FluxPolytope) -> None:
        assert [toy.reaction_ids[i] for i in toy.fixed_indices] == ["FIX", "BLK"]
        assert [toy.reaction_ids[i] for i in toy.free_indices] == list(FREE_IDS)

    def test_biomass_resolved_from_the_objective(self, toy: FluxPolytope) -> None:
        assert toy.biomass_id == "BIO"


class TestReduction:
    def test_free_reactions_survive_in_order(self, reduced: ReducedPolytope) -> None:
        assert [reduced.reaction_ids[i] for i in reduced.free_indices] == list(FREE_IDS)
        assert reduced.n_free == 5
        assert reduced.n_fixed == 2

    def test_the_affine_rhs_is_nonzero(self, reduced: ReducedPolytope) -> None:
        """The whole point of the toy model. A homogeneous reduced system would be *wrong* here."""
        np.testing.assert_allclose(reduced.rhs, EXPECTED_RHS)
        assert np.any(reduced.rhs != 0.0)

    def test_fixed_values_are_carried(self, reduced: ReducedPolytope) -> None:
        np.testing.assert_allclose(reduced.fixed_values, np.array([2.0, 0.0]))

    def test_biomass_index_is_remapped_to_reduced_coordinates(
        self, reduced: ReducedPolytope
    ) -> None:
        assert reduced.biomass_index == 4  # BIO is the 5th free reaction, not the 7th overall
        assert reduced.reaction_ids[reduced.free_indices[reduced.biomass_index]] == "BIO"

    def test_bounds_are_the_free_reactions_bounds(
        self, toy: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        np.testing.assert_array_equal(reduced.lower_bounds, toy.lower_bounds[toy.free_indices])
        np.testing.assert_array_equal(reduced.upper_bounds, toy.upper_bounds[toy.free_indices])


class TestReconstruction:
    def test_to_full_equals_the_explicit_affine_map(self, reduced: ReducedPolytope) -> None:
        """`to_full` indexes rather than multiplying; prove it computes exactly ``R v_red + c``."""
        rng = np.random.default_rng(np.random.SeedSequence(7))
        R = reduced.reconstruction_matrix()

        for _ in range(20):
            v_red = rng.standard_normal(reduced.n_free)
            np.testing.assert_allclose(
                reduced.to_full(v_red), R.matvec(v_red) + reduced.offset, atol=1e-15
            )

    def test_round_trip_is_exact(self, reduced: ReducedPolytope) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(8))
        v_red = rng.standard_normal(reduced.n_free)
        np.testing.assert_array_equal(reduced.to_reduced(reduced.to_full(v_red)), v_red)

    def test_reconstruction_restores_the_fixed_fluxes(self, reduced: ReducedPolytope) -> None:
        v_full = reduced.to_full(np.zeros(reduced.n_free))
        fix_index = reduced.reaction_ids.index("FIX")
        blk_index = reduced.reaction_ids.index("BLK")
        assert v_full[fix_index] == 2.0
        assert v_full[blk_index] == 0.0

    def test_batches_reconstruct(self, reduced: ReducedPolytope) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(9))
        batch = rng.standard_normal((13, reduced.n_free))

        full = reduced.to_full(batch)
        assert full.shape == (13, reduced.n_full)
        for i in range(13):
            np.testing.assert_array_equal(full[i], reduced.to_full(batch[i]))

    def test_rejects_a_wrongly_sized_vector(self, reduced: ReducedPolytope) -> None:
        with pytest.raises(ValueError, match="expected trailing dimension"):
            reduced.to_full(np.zeros(reduced.n_free + 1))


class TestFeasibilityEquivalence:
    """A reduced point is feasible **iff** its reconstruction is feasible in the full model.

    This is the elimination-equivalence gate: the reduced polytope is the same set, not merely a
    similar one. Solved feasible points come from the toy network's own mass balance.
    """

    def _feasible_reduced_points(self, reduced: ReducedPolytope) -> list[np.ndarray]:
        # A: EX_A = R1 + R3 + 2 ;  B: R2 = R1 ;  C: BIO = R2 + R3 + 2
        points = []
        for r1, r3 in [(0.0, 0.0), (1.0, 0.0), (0.0, 3.0), (2.5, 4.5), (4.0, 4.0)]:
            ex_a = r1 + r3 + 2.0
            points.append(np.array([ex_a, r1, r1, r3, r1 + r3 + 2.0], dtype=VALUE_DTYPE))
        return points

    def test_feasible_reduced_points_are_feasible_in_the_full_model(
        self, toy: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        for v_red in self._feasible_reduced_points(reduced):
            assert reduced.contains(v_red), f"{v_red} should be feasible in the reduced polytope"
            assert toy.contains(reduced.to_full(v_red)), "…and therefore in the full model"

    def test_a_bound_violation_is_rejected_in_both_spaces(
        self, toy: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        # EX_A has upper bound 10; drive it past that while keeping mass balance.
        r1, r3 = 6.0, 6.0
        v_red = np.array([r1 + r3 + 2.0, r1, r1, r3, r1 + r3 + 2.0])  # EX_A = 14 > 10
        assert not reduced.contains(v_red)
        assert not toy.contains(reduced.to_full(v_red))

    def test_a_mass_balance_violation_is_rejected_in_both_spaces(
        self, toy: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        v_red = np.array([2.0, 1.0, 1.0, 0.0, 3.0])  # A is unbalanced: EX_A should be 3
        assert not reduced.contains(v_red)
        assert not toy.contains(reduced.to_full(v_red))

    def test_ignoring_the_affine_rhs_would_have_passed_an_infeasible_point(
        self, toy: FluxPolytope, reduced: ReducedPolytope
    ) -> None:
        """The regression this whole design guards against.

        If the reduced system were treated as homogeneous (``S_F v = 0`` instead of ``= rhs``), then
        the all-zero reduced vector would look feasible. It is not: FIX forces 2 units of A into C,
        which something must supply and something must consume.
        """
        v_red = np.zeros(reduced.n_free)

        assert np.allclose(reduced.stoichiometry.matvec(v_red), 0.0)  # "feasible" if homogeneous
        assert not reduced.contains(v_red)  # but it is not
        assert not toy.contains(reduced.to_full(v_red))  # and the full model agrees


class TestValidation:
    def test_rejects_stoichiometry_of_the_wrong_shape(self) -> None:
        with pytest.raises(InvalidPolytopeError, match="expected"):
            FluxPolytope(
                reaction_ids=("r0", "r1"),
                metabolite_ids=("m0",),
                stoichiometry=NativeCSC.from_dense(np.ones((1, 3))),
                lower_bounds=np.zeros(2),
                upper_bounds=np.ones(2),
                biomass_index=0,
            )

    def test_rejects_infinite_bounds(self) -> None:
        with pytest.raises(InvalidPolytopeError, match="must be finite"):
            FluxPolytope(
                reaction_ids=("r0",),
                metabolite_ids=("m0",),
                stoichiometry=NativeCSC.from_dense(np.ones((1, 1))),
                lower_bounds=np.zeros(1),
                upper_bounds=np.array([np.inf]),
                biomass_index=0,
            )

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(InvalidPolytopeError, match="exceeds upper bound"):
            FluxPolytope(
                reaction_ids=("r0",),
                metabolite_ids=("m0",),
                stoichiometry=NativeCSC.from_dense(np.ones((1, 1))),
                lower_bounds=np.array([5.0]),
                upper_bounds=np.array([1.0]),
                biomass_index=0,
            )

    def test_rejects_biomass_index_out_of_range(self) -> None:
        with pytest.raises(InvalidPolytopeError, match="biomass_index"):
            FluxPolytope(
                reaction_ids=("r0",),
                metabolite_ids=("m0",),
                stoichiometry=NativeCSC.from_dense(np.ones((1, 1))),
                lower_bounds=np.zeros(1),
                upper_bounds=np.ones(1),
                biomass_index=3,
            )


class TestEdgeCases:
    def _all_fixed(self) -> FluxPolytope:
        return FluxPolytope(
            reaction_ids=("r0", "r1"),
            metabolite_ids=("m0",),
            stoichiometry=NativeCSC.from_dense(np.array([[1.0, -1.0]])),
            lower_bounds=np.array([3.0, 3.0]),
            upper_bounds=np.array([3.0, 3.0]),
            biomass_index=1,
        )

    def test_a_fully_fixed_model_reduces_to_a_singleton(self) -> None:
        """M4's dim-0 path starts here: nothing free, so the polytope is one point."""
        reduced = self._all_fixed().reduce()

        assert reduced.is_singleton
        assert reduced.n_free == 0
        assert reduced.biomass_index is None  # biomass is itself fixed
        np.testing.assert_allclose(reduced.to_full(np.zeros(0)), np.array([3.0, 3.0]))
        assert reduced.contains(np.zeros(0))

    def test_bounds_that_merely_sit_close_together_are_not_treated_as_fixed(self) -> None:
        """Fixing on a tolerance would delete a real, if narrow, degree of freedom."""
        polytope = FluxPolytope(
            reaction_ids=("r0",),
            metabolite_ids=("m0",),
            stoichiometry=NativeCSC.from_dense(np.zeros((1, 1))),
            lower_bounds=np.array([1.0]),
            upper_bounds=np.array([1.0 + 1e-14]),
            biomass_index=0,
        )
        assert polytope.n_fixed == 0
        assert polytope.n_free == 1
