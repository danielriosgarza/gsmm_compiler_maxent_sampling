"""M3 gate: the LP layer, against the two real models.

The gate is *solver objective == directly recomputed J*. That equation is the whole point: HiGHS
optimizes a linearized surrogate over ``(v, z)`` on a reduced polytope with a constant folded into
an offset, while `SparseFluxObjective.evaluate` computes ``J`` from the full flux vector and knows
nothing about any of that. They agree only if every one of those transformations is right —
the ``z = |v|`` linearization, the fixed-variable elimination, the objective lowering, and the
constant. One number checks them all.

Also here: the finding that M3 turned up. On the example model the *default* λ makes the LP stop
growing altogether. See `TestTheLambdaCliff`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.flux_polytope import FluxPolytope
from gsmm_compiler.highs_backend import SolverFrozenError, total_solve_count
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.sparse_objective import (
    ObjectiveError,
    SparseFluxObjective,
    build_flux_lp,
    build_sparse_objective_lp,
    critical_l1_penalty,
    origin_is_feasible,
    resolve_objective,
    solve_sparse_objective,
)


def _objective(polytope: FluxPolytope, l1_penalty: float) -> SparseFluxObjective:
    """A *raw*-λ objective — these tests pin the mathematics of J directly."""
    return SparseFluxObjective.from_polytope(polytope, l1_penalty=l1_penalty)


class TestTheGateEquation:
    """solver objective == direct J, on both models, across the whole useful λ range."""

    @pytest.mark.parametrize("l1_penalty", [0.0, 1e-4, 1e-3, 0.01, 1.0])
    def test_the_example_model_agrees(
        self, example_canonical: CanonicalModel, l1_penalty: float
    ) -> None:
        polytope = example_canonical.polytope
        objective = _objective(polytope, l1_penalty)

        optimum = build_sparse_objective_lp(polytope.reduce(), objective).solve()

        # `solve` already raises on a mismatch; asserting it here makes the gate criterion legible.
        assert optimum.solver_objective == pytest.approx(optimum.value.total, rel=1e-9, abs=1e-9)
        assert optimum.value.total == pytest.approx(
            objective.evaluate(optimum.v_full).total, rel=1e-12
        )

    @pytest.mark.parametrize("l1_penalty", [0.0, 0.1, 1.0, 10.0])
    def test_the_toy_agrees_and_it_is_the_one_with_a_nonzero_fixed_flux(
        self, toy_canonical: CanonicalModel, l1_penalty: float
    ) -> None:
        """The toy is the only model that can catch a dropped objective constant: its FIX reaction
        is pinned at **2.0**, so ``J(c) = −2λ`` is genuinely nonzero. All 513 of the example
        model's fixed reactions sit at zero, and would let a forgotten offset pass unnoticed."""
        polytope = toy_canonical.polytope
        reduced = polytope.reduce()
        lp = build_sparse_objective_lp(reduced, _objective(polytope, l1_penalty))

        assert lp.program.offset == pytest.approx(-2.0 * l1_penalty)

        optimum = lp.solve()
        assert optimum.solver_objective == pytest.approx(optimum.value.total, rel=1e-9, abs=1e-9)

    def test_the_toy_optimum_is_the_one_computed_on_paper(
        self, toy_canonical: CanonicalModel
    ) -> None:
        """With FIX pinned at 2, mass balance gives ``EX_A = R1 + R3 + 2`` and ``BIO = R2 + R3 + 2``
        (BLK is closed, so R1 = R2). Writing ``R1 = R2 = a`` and ``R3 = b``, at λ = 1:

            J = BIO − (EX_A + R1 + R2 + R3 + FIX)
              = (a + b + 2) − [(a + b + 2) + a + a + b + 2]
              = −2a − b − 2

        which is maximized at a = b = 0. So **J\\* = −2**, reached by running the forced maintenance
        flux and nothing else: the cell is kept alive by FIX at a loss, and every optional reaction
        stays switched off because each one costs more L1 than the biomass it returns.
        """
        polytope = toy_canonical.polytope
        optimum = build_sparse_objective_lp(polytope.reduce(), _objective(polytope, 1.0)).solve()

        assert optimum.j_star == pytest.approx(-2.0)
        assert optimum.value.mu == pytest.approx(2.0)  # BIO, forced by FIX
        assert optimum.value.cost == pytest.approx(4.0)  # EX_A = 2 and FIX = 2

        by_id = dict(zip(polytope.reaction_ids, optimum.v_full, strict=True))
        assert by_id["EX_A"] == pytest.approx(2.0)
        assert by_id["BIO"] == pytest.approx(2.0)
        assert by_id["FIX"] == pytest.approx(2.0)
        for optional in ("R1", "R2", "R3"):
            assert by_id[optional] == pytest.approx(0.0, abs=1e-9)


class TestTheOptimumIsARealFluxVector:
    @pytest.mark.parametrize("l1_penalty", [0.0, 1e-3, 1.0])
    def test_v_star_is_feasible_in_the_full_polytope(
        self, example_canonical: CanonicalModel, l1_penalty: float
    ) -> None:
        """Checked against the *full* canonical polytope — mass balance over all 894 metabolites and
        all 773 bounds — not against the reduced system the LP was actually solved on."""
        polytope = example_canonical.polytope
        optimum = build_sparse_objective_lp(
            polytope.reduce(), _objective(polytope, l1_penalty)
        ).solve()

        assert optimum.v_full.shape == (773,)
        assert polytope.contains(optimum.v_full, tol=1e-7)

    def test_z_equals_abs_v_at_the_optimum(self, example_canonical: CanonicalModel) -> None:
        polytope = example_canonical.polytope
        reduced = polytope.reduce()
        lp = build_sparse_objective_lp(reduced, _objective(polytope, 1e-3))

        optimum = lp.solve()

        expected = np.abs(reduced.to_reduced(optimum.v_full)[lp.z_columns])
        np.testing.assert_allclose(optimum.z, expected, atol=1e-7)
        assert optimum.max_z_deviation < 1e-7

    def test_the_fixed_reactions_come_back_at_their_fixed_values(
        self, example_canonical: CanonicalModel
    ) -> None:
        """The LP never saw these 513 columns; `to_full` must restore them, not leave them empty."""
        polytope = example_canonical.polytope
        reduced = polytope.reduce()
        optimum = build_sparse_objective_lp(reduced, _objective(polytope, 1e-3)).solve()

        np.testing.assert_allclose(
            optimum.v_full[reduced.fixed_indices], reduced.fixed_values, atol=0.0
        )


class TestTheLambdaCliff:
    """**The M3 finding.** Above a model-specific λ*, the sparse optimum is the origin.

    ``J(v) = μ(v) − λC(v)`` is maximized by ``v = 0`` as soon as λ exceeds ``λ* = max_v μ(v)/C(v)``:
    the origin is always feasible (``S·0 = 0``), costs nothing and earns nothing, and that beats any
    growth whose L1 cost outruns its biomass. On this model μ_max ≈ 41.6 while ``C ≈ 4.5e4`` at the
    growth optimum, which puts λ* at roughly **1.5e-3** — so λ is not a dimensionless knob, and both
    our default (1.0) and the spec's suggested value (0.01) sit far past the cliff.

    The LP is not wrong when this happens. ``J`` is. Nothing inside the LP can tell: the status is
    optimal, the residual is zero, ``z == |v|`` exactly. Only ``μ_max`` standing next to ``μ(v*)``
    gives it away, which is why `solve_sparse_objective` always computes both.
    """

    def test_the_origin_is_feasible_which_is_what_makes_the_collapse_possible(
        self, example_canonical: CanonicalModel
    ) -> None:
        """The precondition, pinned — because it is a property of the *model*, not of the method.

        Not one reaction in this model is forced to carry flux: no ``l > 0``, no ``u < 0``, and no
        ATP-maintenance demand anywhere. So ``v = 0`` sits inside the polytope, and a large enough λ
        can retreat to it. A curated model with a maintenance reaction pinned above zero (``ATPM ≥
        8.39`` is the usual convention) **cannot** collapse this way — some flux is compulsory, and
        μ(v*) stays positive however hard the L1 term pushes. That is why our toy network cannot
        reproduce this failure and the genome-scale model can: FIX = 2.0 keeps the toy alive.
        """
        polytope = example_canonical.polytope
        forced = np.flatnonzero((polytope.lower_bounds > 0.0) | (polytope.upper_bounds < 0.0))

        assert forced.size == 0
        assert polytope.contains(np.zeros(polytope.n_reactions))

    def test_at_the_default_lambda_the_example_model_stops_growing_entirely(
        self, example_canonical: CanonicalModel
    ) -> None:
        polytope = example_canonical.polytope
        solution = solve_sparse_objective(polytope.reduce(), _objective(polytope, 1.0))

        assert solution.is_sparsity_dominated
        assert solution.optimum.value.mu == pytest.approx(0.0, abs=1e-9)
        assert solution.biomass_retention == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(solution.optimum.v_full, np.zeros(773), atol=1e-9)

        # ...while the polytope could have grown at 41.6 all along.
        assert solution.biomass_maximum > 40.0

    def test_below_the_cliff_it_grows_and_the_objective_is_sparse(
        self, example_canonical: CanonicalModel
    ) -> None:
        """λ = 1e-3 sits under λ*: biomass survives, and the L1 term does its actual job — the flux
        distribution is far sparser than the 131 reactions plain FBA lights up."""
        polytope = example_canonical.polytope
        solution = solve_sparse_objective(polytope.reduce(), _objective(polytope, 1e-3))

        assert not solution.is_sparsity_dominated
        assert solution.optimum.value.mu > 1.0
        assert 0.0 < solution.biomass_retention < 1.0

    def test_the_cliff_is_where_the_arithmetic_says_it_is(
        self, example_canonical: CanonicalModel
    ) -> None:
        """Bracket λ*: growth survives at 1e-3 and is gone by 3e-3.

        Pinned as a bracket rather than a single number because λ* is a property of *this* curated
        model, and a bracket is what a reader needs in order to choose λ for another one.
        """
        polytope = example_canonical.polytope
        reduced = polytope.reduce()

        alive = solve_sparse_objective(reduced, _objective(polytope, 1e-3))
        dead = solve_sparse_objective(reduced, _objective(polytope, 3e-3))

        assert alive.optimum.value.mu > 1.0
        assert not alive.is_sparsity_dominated
        assert dead.optimum.value.mu == pytest.approx(0.0, abs=1e-9)
        assert dead.is_sparsity_dominated

    def test_j_star_never_rises_with_lambda(self, example_canonical: CanonicalModel) -> None:
        """``J*(λ) = max_v (μ − λC)`` is a maximum of lines of slope ``−C ≤ 0``: concave and
        nonincreasing. A J* that climbed would mean the LP was not finding the optimum."""
        polytope = example_canonical.polytope
        reduced = polytope.reduce()

        j_stars = [
            build_sparse_objective_lp(reduced, _objective(polytope, lam)).solve().j_star
            for lam in [0.0, 1e-4, 1e-3, 3e-3, 0.01, 0.1, 1.0]
        ]

        assert np.all(np.diff(j_stars) <= 1e-9), j_stars
        assert j_stars[0] == pytest.approx(41.633, rel=1e-3)  # λ = 0 is plain FBA
        assert j_stars[-1] == pytest.approx(0.0, abs=1e-9)  # λ = 1 is the origin


class TestScaleReferencedLambda:
    """The resolution of §1.7: λ is expressed as ``λ̃ · λ*``, with ``λ*`` measured per model.

    This is the class that makes the cross-model comparison mean something. A raw λ of 1.0 is
    harmless on the toy network and catastrophic here; the same ``λ̃`` puts *every* strain at the
    same fraction of its own sparsity cliff, which is what "comparable selection pressure" (§1.1)
    has to mean if it is to mean anything.
    """

    def test_lambda_star_comes_from_one_lp_and_lands_where_the_search_said(
        self, example_canonical: CanonicalModel
    ) -> None:
        """λ* on this model is 1.89e-3. Computed exactly, by a single Charnes–Cooper LP — the
        earlier 40-step bisection agreed to 8 figures, but a bisection is not what we ship."""
        polytope = example_canonical.polytope
        critical = critical_l1_penalty(polytope.reduce(), _objective(polytope, 0.0))

        assert critical == pytest.approx(1.88987572e-3, rel=1e-6)

    def test_lambda_star_is_exactly_where_this_model_stops_growing(
        self, example_canonical: CanonicalModel
    ) -> None:
        """Not near the cliff — *at* it. A hair below, the cell grows; a hair above, it shuts down.
        This is the property that earns λ* the right to be the unit λ is measured in."""
        polytope = example_canonical.polytope
        reduced = polytope.reduce()
        critical = critical_l1_penalty(reduced, _objective(polytope, 0.0))

        below = solve_sparse_objective(reduced, _objective(polytope, 0.999 * critical))
        above = solve_sparse_objective(reduced, _objective(polytope, 1.001 * critical))

        assert below.optimum.value.mu > 0.0
        assert not below.is_sparsity_dominated
        assert above.optimum.value.mu == pytest.approx(0.0, abs=1e-9)
        assert above.is_sparsity_dominated

    def test_lambda_tilde_is_a_selection_pressure_dial(
        self, example_canonical: CanonicalModel
    ) -> None:
        """What the study actually needs from λ: turning λ̃ up must trade growth for sparsity,
        smoothly and monotonically, and never fall off the cliff.

        Measured on this model: λ̃ = 0 keeps 100% of μ_max (plain FBA), 0.25 keeps 95%, 0.5 keeps
        60%, 0.9 keeps 30%. A dial, not a trapdoor.
        """
        polytope = example_canonical.polytope
        reduced = polytope.reduce()

        retentions = []
        for scaled in (0.0, 0.25, 0.5, 0.9):
            resolved = resolve_objective(
                polytope, reduced, ObjectiveConfig(l1_penalty_scaled=scaled)
            )
            solution = solve_sparse_objective(reduced, resolved.objective)

            assert not solution.is_sparsity_dominated, f"λ̃ = {scaled} collapsed"
            retentions.append(solution.biomass_retention)

        assert retentions[0] == pytest.approx(1.0)
        assert np.all(np.diff(retentions) < 0.0), retentions
        assert retentions[-1] < 0.5  # λ̃ = 0.9 really is heavy pressure

    def test_the_default_config_grows(self, example_canonical: CanonicalModel) -> None:
        """The whole point of the change. The *default* config used to hand this model an objective
        whose optimum was zero flux; now it hands it one that keeps 60% of achievable growth."""
        polytope = example_canonical.polytope
        reduced = polytope.reduce()

        resolved = resolve_objective(polytope, reduced, ObjectiveConfig())
        solution = solve_sparse_objective(reduced, resolved.objective)

        assert resolved.scale.l1_penalty_scaled == 0.5  # the default λ̃
        assert resolved.scale.l1_penalty == pytest.approx(9.4494e-4, rel=1e-3)  # the raw λ it means
        assert not solution.is_sparsity_dominated
        assert solution.biomass_retention == pytest.approx(0.603, abs=0.01)

    def test_a_lambda_tilde_of_one_is_refused_on_this_model(
        self, example_canonical: CanonicalModel
    ) -> None:
        polytope = example_canonical.polytope

        with pytest.raises(ObjectiveError, match="sparsity cliff"):
            resolve_objective(
                polytope, polytope.reduce(), ObjectiveConfig(l1_penalty_scaled=1.0)
            )

    def test_the_same_lambda_tilde_means_different_raw_lambdas_on_different_models(
        self, example_canonical: CanonicalModel, toy_canonical: CanonicalModel
    ) -> None:
        """The heart of it. λ̃ = 0.5 resolves to λ = 9.4e-4 on the Bifido model and λ = 0.25 on the
        toy — a factor of **265** — because their μ/C scales differ by that much. Handing both the
        same *raw* λ would have meant wildly different selection pressures while looking, in the
        config file, like a controlled comparison.
        """
        config = ObjectiveConfig(l1_penalty_scaled=0.5)

        genome = resolve_objective(
            example_canonical.polytope, example_canonical.polytope.reduce(), config
        )
        toy = resolve_objective(toy_canonical.polytope, toy_canonical.polytope.reduce(), config)

        assert genome.scale.l1_penalty == pytest.approx(9.4494e-4, rel=1e-3)
        assert toy.scale.l1_penalty == pytest.approx(0.25)
        assert toy.scale.l1_penalty / genome.scale.l1_penalty == pytest.approx(265.0, rel=0.01)

    def test_the_toy_has_no_cliff_because_its_maintenance_flux_is_forced(
        self, toy_canonical: CanonicalModel
    ) -> None:
        """FIX = 2.0 keeps the toy alive: shutting down is not one of its options, so no λ can
        collapse it and λ̃ ≥ 1 is a legitimate request. The example model, with no forced flux
        anywhere, has no such protection — which is exactly why it found the bug and the toy could
        not have."""
        polytope = toy_canonical.polytope

        assert not origin_is_feasible(polytope)

        resolved = resolve_objective(
            polytope, polytope.reduce(), ObjectiveConfig(l1_penalty_scaled=2.0)
        )
        solution = solve_sparse_objective(polytope.reduce(), resolved.objective)

        assert not resolved.scale.origin_is_feasible
        assert solution.optimum.value.mu == pytest.approx(2.0)  # still alive, on FIX alone


class TestGeometryPremises:
    """What M4 is about to depend on."""

    def test_warm_starts_keep_the_geometry_lp_cheap_at_genome_scale(
        self, example_canonical: CanonicalModel
    ) -> None:
        """§14's flux-only LP, re-solved under 50 random objectives the way basis discovery will.

        The premise of a *sequential* geometry phase (§1.2) is that each re-solve costs a few
        pivots rather than a fresh solve. Asserted against the cold-start cost, not an absolute
        pivot count, which would only pin this HiGHS build's tie-breaking.
        """
        reduced = example_canonical.polytope.reduce()
        rng = np.random.default_rng(np.random.SeedSequence(11))
        program = build_flux_lp(reduced)

        cold = program.maximize(rng.standard_normal(reduced.n_free)).simplex_iterations
        warm = [
            program.maximize(rng.standard_normal(reduced.n_free)).simplex_iterations
            for _ in range(50)
        ]

        assert float(np.median(warm)) < cold, f"warm starts are not paying off: cold={cold} {warm}"

    def test_the_flux_lp_has_no_auxiliary_columns(self, example_canonical: CanonicalModel) -> None:
        """Geometry probes fluxes. A z column would enlarge every one of its hundreds of solves and
        contribute nothing — and, per spec §3.4, z must never be mistaken for part of the state."""
        reduced = example_canonical.polytope.reduce()

        assert build_flux_lp(reduced).n_cols == reduced.n_free == 260


class TestSamplingInvariants:
    """BUILD_PLAN §1.3 / §1.2 — enforced now, so M5 inherits them rather than adding them."""

    def test_evaluating_j_does_not_import_highspy(self) -> None:
        """A sampling worker evaluates ``J`` for its objective traces but must never load a solver
        (§1.2). `highs_backend` imports highspy *inside* the constructor for exactly this reason, so
        `sparse_objective` can be imported by a worker without dragging HiGHS into the process."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import gsmm_compiler.sparse_objective as s; "
                "print('highspy' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "False"

    def test_a_frozen_program_cannot_be_solved_once_sampling_starts(
        self, example_canonical: CanonicalModel
    ) -> None:
        """The mechanism M5's "zero inner-loop solves" gate will rely on."""
        reduced = example_canonical.polytope.reduce()
        program = build_flux_lp(reduced)
        program.maximize(np.zeros(reduced.n_free))

        program.freeze()
        before = total_solve_count()

        with pytest.raises(SolverFrozenError):
            program.solve()

        assert total_solve_count() == before
