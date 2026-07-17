"""The pilot DAG: re-rounding from a pilot, and ``s_J`` from the neutral ensemble's spread (M10).

Two claims are load-bearing here and they need different kinds of test.

**Re-rounding cannot move the target.** That is a theorem — ``range(diag(s)·B·L₁) =
range(diag(s)·B)`` for any invertible ``L₁`` — so the algebra is not what these tests are for. Codex
(M10 review round 3) named what actually breaks it: *"numerical feasibility tolerances, rank loss,
state carry-over, production-dependent rerunning, or residual adaptation are the actual ways the
implementation could break the argument."* Those are what is tested.

**``s_J = σ̂₀`` must be checked against arithmetic, not against itself.** A test that recomputes the
estimator the way the estimator computes it proves nothing. So the SD is checked against a
hand-written formula, the shift-invariance against an exact additive constant, and the degenerate
refusal against a premise assertion that fails loudly if the test ever goes toothless (the M4/M6
lesson: *a regression test the bug passes is not a regression test*).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest
from tests.conftest import synthetic_optimum

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.calibration import (
    GEOMETRY_PILOT_STAGE,
    SCALE_PILOT_STAGE,
    CalibrationError,
    GeometryPilot,
    ScalePilot,
    calibrate,
    run_geometry_pilot,
    run_scale_pilot,
)
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.maxent_sampler import run_chains
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.rounding import (
    RoundingError,
    build_transform,
    certify_reachable_mass_balance,
    reround_transform,
)
from gsmm_compiler.sparse_objective import (
    PILOT_SCALE_TARGET_RELATIVE_SE,
    DegenerateEnergyScaleError,
    IncompatibleObjectiveError,
    ObjectiveError,
    SparseFluxObjective,
    lower_objective,
    pilot_energy_scale,
)

PILOT = SamplerConfig(
    betas=(0.0,), n_chains=4, n_samples=250, burn_in=250, pilot_chains=4,
    pilot_samples=250, pilot_burn_in=250,
)

#       EX_A DIRECT SLOW1 SLOW2  BIO
FORK = [
    [1.0, -1.0, -1.0, 0.0, 0.0],  # A
    [0.0, 1.0, 0.0, 1.0, -1.0],  # B
    [0.0, 0.0, 1.0, -1.0, 0.0],  # C
]


def _fork_full() -> FluxPolytope:
    """The M3 fork network as a **canonical** polytope — `SparseFluxObjective.from_polytope` needs
    the full one, and building the objective through the real path rather than by hand is what keeps
    a lowering bug from cancelling against a test's own belief about the indices (M6)."""
    return FluxPolytope(
        reaction_ids=("EX_A", "DIRECT", "SLOW1", "SLOW2", "BIO"),
        metabolite_ids=("A", "B", "C"),
        stoichiometry=NativeCSC.from_dense(np.asarray(FORK, dtype=np.float64)),
        lower_bounds=np.zeros(5),
        upper_bounds=np.full(5, 10.0),
        biomass_index=4,
    )


@pytest.fixture(scope="module")
def fork_full() -> FluxPolytope:
    return _fork_full()


@pytest.fixture(scope="module")
def fork_reduced(fork_full: FluxPolytope) -> ReducedPolytope:
    return fork_full.reduce()


@pytest.fixture(scope="module")
def fork_lowered(fork_full: FluxPolytope, fork_reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    return lower_objective(
        fork_reduced, SparseFluxObjective.from_polytope(fork_full, l1_penalty=0.25)
    )


@pytest.fixture(scope="module")
def fork_setup(fork_reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(fork_reduced, model_id="fork")
    return geometry, build_transform(geometry, fork_reduced)


@pytest.fixture(scope="module")
def fork_certificate(fork_setup, fork_reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    """M9's proof for ``T₀``. `calibrate` requires it rather than recomputing it: a pilot is a
    chain, so an uncertified ``T₀`` walks off the manifold before production exists, and every
    artifact the DAG freezes descends from those draws (Codex, M10.2 review round 3)."""
    _, transform = fork_setup
    return certify_reachable_mass_balance(transform, fork_reduced)


@pytest.fixture(scope="module")
def simplex_setup(simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(simplex_polytope, model_id="simplex")
    return geometry, build_transform(geometry, simplex_polytope)


@pytest.fixture(scope="module")
def simplex_certificate(simplex_setup, simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    _, transform = simplex_setup
    return certify_reachable_mass_balance(transform, simplex_polytope)


@pytest.fixture(scope="module")
def simplex_pilot(  # type: ignore[no-untyped-def]
    simplex_setup, simplex_polytope: ReducedPolytope, simplex_certificate
) -> GeometryPilot:
    geometry, transform = simplex_setup
    return run_geometry_pilot(
        transform, simplex_polytope, config=PILOT, model_id="simplex",
        certificate=simplex_certificate,
    )


@pytest.fixture(scope="module")
def simplex_scale_pilot(  # type: ignore[no-untyped-def]
    simplex_setup, simplex_polytope: ReducedPolytope, simplex_certificate
) -> ScalePilot:
    geometry, transform = simplex_setup
    return run_scale_pilot(
        transform, simplex_polytope, config=PILOT, model_id="simplex",
        certificate=simplex_certificate,
    )


class TestRerounDingPreservesTheDirectionSpace:
    """The theorem, and then the implementation risks Codex named as the real failure modes."""

    def test_range_of_T1_equals_range_of_T0(
        self, simplex_setup, simplex_pilot: GeometryPilot, simplex_polytope: ReducedPolytope
    ) -> None:
        """``L₁`` is invertible, so ``T₁`` spans exactly what ``T₀`` spans — no more, no less.

        Projected both ways: one direction alone would miss a ``T₁`` spanning a strict *subspace*,
        which is precisely what a rank-deficient pilot covariance would produce — and a missing
        dimension produces no bad numbers, only absent ones (M5).
        """
        geometry, bootstrap = simplex_setup
        rerounded = reround_transform(
            geometry, simplex_polytope, bootstrap,
            pilot_coordinates=simplex_pilot.pooled_coordinates(),
        )

        for source, target in (
            (rerounded.transform, bootstrap.transform),
            (bootstrap.transform, rerounded.transform),
        ):
            basis_of_target, *_ = np.linalg.svd(target, full_matrices=False)
            residual = source - basis_of_target @ (basis_of_target.T @ source)
            assert np.abs(residual).max() < 1e-10

    def test_the_rerounded_transform_has_full_column_rank(
        self, simplex_setup, simplex_pilot: GeometryPilot, simplex_polytope: ReducedPolytope
    ) -> None:
        """Rank loss is Codex's named failure mode, and it is silent: a ``T`` that dropped a column
        still produces feasible samples, positive chords and exact mass balance — of a
        lower-dimensional slice."""
        geometry, bootstrap = simplex_setup
        rerounded = reround_transform(
            geometry, simplex_polytope, bootstrap,
            pilot_coordinates=simplex_pilot.pooled_coordinates(),
        )
        singular = np.linalg.svd(rerounded.transform, compute_uv=False)
        assert np.count_nonzero(singular > singular[0] * 1e-10) == geometry.dimension

    def test_q_equals_L0_times_y_is_an_identity_not_a_projection(
        self, simplex_setup, simplex_polytope: ReducedPolytope
    ) -> None:
        """``q = L₀·y`` must reproduce the geometry coordinates the fluxes independently give.

        Two routes to ``q`` that share no arithmetic: the algebraic identity this module uses, and
        a projection of the same states' *fluxes* through the geometry. They agree only if the
        identity is real.

        Driven straight off `run_chains` rather than off a pilot (M10.2b). The identity is a fact
        about ``L₀`` and the geometry — no pilot appears in it — and a `GeometryPilot` keeps only
        coordinates, so reaching for one here would need a *second* pilot whose fluxes are draws
        from a different stream: the two sides would then disagree for a reason that has nothing to
        do with the identity under test.
        """
        geometry, bootstrap = simplex_setup
        result = run_chains(
            bootstrap, simplex_polytope, config=PILOT, model_id="simplex", beta=0.0,
            stage=GEOMETRY_PILOT_STAGE,
        )
        chain = result.chains[0]
        via_identity = chain.coordinates @ bootstrap.cholesky.T
        via_geometry = geometry.to_coordinates(chain.fluxes)

        assert np.abs(via_identity - via_geometry).max() < 1e-9

    def test_support_coordinates_stay_the_support_hull_not_the_pilot_draws(
        self, simplex_setup, simplex_pilot: GeometryPilot, simplex_polytope: ReducedPolytope
    ) -> None:
        """Chain starts must stay dispersed over the **vertices**, not over interior pilot points.

        The natural bug is to reuse whatever point set built the covariance. It would produce no
        error: starts from a hull of interior draws are perfectly feasible — just clustered near the
        centre, exactly when the dispersion exists to make R̂ able to detect retained
        initialization.
        """
        geometry, bootstrap = simplex_setup
        rerounded = reround_transform(
            geometry, simplex_polytope, bootstrap,
            pilot_coordinates=simplex_pilot.pooled_coordinates(),
        )
        assert rerounded.support_coordinates.shape[0] == geometry.support_points.shape[0]
        assert rerounded.support_coordinates.shape[0] != simplex_pilot.recipe.n_chains * (
            simplex_pilot.recipe.n_draws
        )
        # and they really are the support vertices, in the new frame
        lifted = rerounded.to_flux(rerounded.support_coordinates)
        assert np.abs(lifted - geometry.support_points).max() < 1e-8

    def test_the_rerounded_transform_records_its_estimator(
        self, simplex_setup, simplex_pilot: GeometryPilot, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry, bootstrap = simplex_setup
        rerounded = reround_transform(
            geometry, simplex_polytope, bootstrap,
            pilot_coordinates=simplex_pilot.pooled_coordinates(),
        )
        assert bootstrap.diagnostics.covariance_source == "support_points"
        assert rerounded.diagnostics.covariance_source == "pilot"
        assert rerounded.manifest()["covariance_source"] == "pilot"


class TestRerounDingRefusesArtifactsThatNeverMet:
    """M6's bug class, at the one new join M10 introduces."""

    def test_a_bootstrap_from_another_geometry_is_refused(
        self, simplex_setup, simplex_pilot: GeometryPilot, simplex_polytope: ReducedPolytope,
        coupled_box_polytope: ReducedPolytope,
    ) -> None:
        """``q = L₀·y`` is the right change of coordinates only for *this* geometry's ``L₀``."""
        geometry, _ = simplex_setup
        other_geometry = build_geometry(coupled_box_polytope, model_id="other")
        other_bootstrap = build_transform(other_geometry, coupled_box_polytope)

        with pytest.raises(RoundingError, match="not built from this polytope"):
            reround_transform(
                geometry, simplex_polytope, other_bootstrap,
                pilot_coordinates=simplex_pilot.pooled_coordinates(),
            )

    def test_pilot_coordinates_of_the_wrong_width_are_refused(
        self, simplex_setup, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry, bootstrap = simplex_setup
        with pytest.raises(RoundingError, match="expected"):
            reround_transform(
                geometry, simplex_polytope, bootstrap,
                pilot_coordinates=np.zeros((10, geometry.dimension + 1)),
            )

    def test_a_single_pilot_draw_cannot_make_a_covariance(
        self, simplex_setup, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry, bootstrap = simplex_setup
        with pytest.raises(RoundingError, match="at least 2"):
            reround_transform(
                geometry, simplex_polytope, bootstrap,
                pilot_coordinates=np.zeros((1, geometry.dimension)),
            )

    def test_non_finite_pilot_coordinates_are_refused(
        self, simplex_setup, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry, bootstrap = simplex_setup
        bad = np.zeros((5, geometry.dimension))
        bad[2, 0] = np.nan
        with pytest.raises(RoundingError, match="finite"):
            reround_transform(
                geometry, simplex_polytope, bootstrap, pilot_coordinates=bad
            )


class TestThePilotIsFrozen:
    """Spec §17.4/§18.3: an adaptive ``T`` makes the kernel depend on the chain's own history."""

    def test_pilot_arrays_are_physically_read_only(
        self, simplex_pilot: GeometryPilot, simplex_scale_pilot: ScalePilot
    ) -> None:
        """`@dataclass(frozen=True)` freezes the binding, not the buffer (M5's lesson)."""
        with pytest.raises(ValueError):
            simplex_pilot.coordinates[0, 0, 0] = 1.0
        with pytest.raises(ValueError):
            simplex_scale_pilot.fluxes[0, 0, 0] = 1.0
        # The spawn keys are evidence, so they are frozen for the same reason the draws are.
        with pytest.raises(ValueError):
            simplex_pilot.spawn_keys[0, 0] = 1

    def test_the_sampler_cannot_import_calibration(self) -> None:
        """The structural guard, mirroring M7's reweighting↔sampler separation.

        `calibration` imports `maxent_sampler`; the reverse must be impossible, or a production
        chain could re-derive ``T`` or ``s_J`` from its own state mid-run — and its samples would
        then not be from ``π_β`` at all. Checked in a **subprocess** so this test cannot be fooled
        by a module another test already imported.
        """
        code = (
            "import sys; import gsmm_compiler.maxent_sampler;"
            "assert 'gsmm_compiler.calibration' not in sys.modules, "
            "'maxent_sampler pulled in calibration';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_the_two_pilot_stages_draw_from_different_streams(
        self, simplex_setup, simplex_polytope: ReducedPolytope, simplex_certificate
    ) -> None:
        """Stream separation is what makes pilot-seed sensitivity attributable (Codex, M10 r2).

        The RNG is keyed on ``(model_id, stage, β_index, chain_index)``, so the stage *name* is the
        only thing separating the geometry pilot from the scale pilot. If they shared a name they
        would share their draws, and "the rounding got unlucky" could never be told apart from "the
        scale got unlucky" — which is exactly what M10.2a found, `run_chain` having keyed every
        stream on a hardcoded ``"sample"``.

        🔴 **M10.2b rewrote this test, and the rewrite is the finding.** It used to assert
        ``not np.allclose(geometry_pilot.coordinates, scale_pilot.coordinates)``, which proves
        **non-identity**, not independence — and its name claimed independence. Worse, the property
        it named is not even true of the artifacts: ``T₁`` is derived from the geometry pilot and
        the scale pilot runs under ``T₁``, so the two pilots are *causally dependent*. What is
        independent is each pilot's **RNG stream given its inputs**, and the spawn key is the direct
        evidence for it rather than a downstream shadow of it. (Codex, M10.2b review round 1.)
        """
        _, transform = simplex_setup
        geometry_pilot = run_geometry_pilot(
            transform, simplex_polytope, config=PILOT, model_id="m",
            certificate=simplex_certificate,
        )
        scale_pilot = run_scale_pilot(
            transform, simplex_polytope, config=PILOT, model_id="m",
            certificate=simplex_certificate,
        )
        # The stage coordinate — spawn key column 1 — is what separates them, chain for chain.
        assert not np.array_equal(geometry_pilot.spawn_keys, scale_pilot.spawn_keys)
        assert np.array_equal(geometry_pilot.spawn_keys[:, 0], scale_pilot.spawn_keys[:, 0]), (
            "same model, so the model coordinate must agree — otherwise this test would pass for "
            "the wrong reason"
        )
        assert not np.array_equal(geometry_pilot.spawn_keys[:, 1], scale_pilot.spawn_keys[:, 1])

    def test_a_pilot_is_reproducible_from_its_semantic_key(
        self, simplex_setup, simplex_polytope: ReducedPolytope, simplex_certificate
    ) -> None:
        """Same coordinates ⇒ same draws, bit for bit. The RNG is named, not positional."""
        _, transform = simplex_setup
        first = run_scale_pilot(
            transform, simplex_polytope, config=PILOT, model_id="m",
            certificate=simplex_certificate,
        )
        second = run_scale_pilot(
            transform, simplex_polytope, config=PILOT, model_id="m",
            certificate=simplex_certificate,
        )
        assert np.array_equal(first.fluxes, second.fluxes)
        assert first.recipe.content_key() == second.recipe.content_key()

    def test_a_pilot_whose_stage_disagrees_with_its_class_is_inexpressible(
        self, simplex_pilot: GeometryPilot
    ) -> None:
        """The stage is a `ClassVar`, and the recipe's stage must match it.

        M10.2a's `run_neutral_pilot` took a ``stage`` argument, so the caller chose the RNG stream
        *and* the key while the payload came from somewhere else entirely. Under M10.2b's split a
        `GeometryPilot` keyed as a scale pilot would hold coordinates under a key naming fluxes —
        the artifact would stop being a function of its key, in the milestone about that. There is
        no ``stage`` parameter to get wrong any more; this pins the one remaining way to state the
        contradiction. (Codex, M10.2b review round 2.)
        """
        with pytest.raises(CalibrationError, match="would be keyed as"):
            dataclasses.replace(
                simplex_pilot,
                recipe=dataclasses.replace(simplex_pilot.recipe, stage=SCALE_PILOT_STAGE),
            )

    def test_a_pilot_that_drew_from_streams_its_recipe_does_not_name_is_refused(
        self, simplex_pilot: GeometryPilot
    ) -> None:
        """The M10.2a defect, caught by construction rather than by a statistical test.

        `run_chain` keyed every stream on a hardcoded ``"sample"``, so the two pilots drew identical
        numbers. The spawn keys a pilot's chains *actually consumed* are stored; the keys its recipe
        *names* are recomputed here from ``(model_id, STAGE, 0, chain_index)``; they must agree. Had
        this guard existed, M10.2a's bug would have raised on the first pilot ever built — the
        stored keys would have carried ``hash("sample")`` where the recipe names
        ``hash("geometry_pilot")``.

        This is why there is no stored flux fingerprint: evidence you **recompute** is evidence,
        evidence you store and read back is a claim. (Codex, M10.2b review round 2, refusing its own
        round-1 proposal.)
        """
        foreign = np.array(simplex_pilot.spawn_keys, dtype=np.int64)
        foreign[:, 1] += 1  # as if the stage had not reached `run_chain`
        with pytest.raises(CalibrationError, match="streams its recipe does not name"):
            dataclasses.replace(simplex_pilot, spawn_keys=foreign)


class TestThePilotKeyCoversWhatCanChangeItsBytes:
    """BUILD_PLAN §1.1 — and Codex's round-3 catch that polytope+objective+stream is not enough."""

    def test_the_key_covers_the_input_transform(self, simplex_pilot: GeometryPilot) -> None:
        """A pilot run under one ``T`` must not be reusable under another: its coordinates are only
        meaningful in the frame that produced them."""
        recipe = simplex_pilot.recipe
        other = dataclasses.replace(recipe, transform_key="a-different-transform")
        assert other.content_key() != recipe.content_key()

    def test_the_key_covers_the_schedule(self, simplex_pilot: GeometryPilot) -> None:
        recipe = simplex_pilot.recipe
        assert (
            dataclasses.replace(recipe, burn_in=recipe.burn_in + 1).content_key()
            != recipe.content_key()
        )
        assert (
            dataclasses.replace(recipe, n_chains=recipe.n_chains + 1).content_key()
            != recipe.content_key()
        )

    def test_the_key_covers_the_stage(self, simplex_pilot: GeometryPilot) -> None:
        recipe = simplex_pilot.recipe
        other = dataclasses.replace(recipe, stage=SCALE_PILOT_STAGE)
        assert other.content_key() != recipe.content_key()

    def test_the_key_covers_the_feasibility_tolerance(self, simplex_pilot: GeometryPilot) -> None:
        """M10.2b: the pilots did not honour ``geometry.feasibility_tol`` at all — they silently
        took `run_chains`' 1e-9 default while production used the configured value. It reaches start
        selection, chord construction and refresh validation, so it moves the draws; the moment the
        pilot honours it, an unhashed tolerance is a false-hit generator."""
        recipe = simplex_pilot.recipe
        other = dataclasses.replace(recipe, feasibility_tol=recipe.feasibility_tol * 100.0)
        assert other.content_key() != recipe.content_key()

    def test_the_pilot_carries_no_objective_key(self, simplex_pilot: GeometryPilot) -> None:
        """The β=0 law is objective-independent, so **one pilot serves every objective** on a
        polytope — which is what lets M7's base and reweighted objectives be calibrated against the
        *same* neutral ensemble. An objective key here would be a lie about the dependency."""
        assert not any("objective" in f.name for f in dataclasses.fields(simplex_pilot.recipe))
        assert "objective_key" not in simplex_pilot.manifest()


# ---- s_J = σ̂₀ ---------------------------------------------------------------------------------


def _pilot_fluxes_with_known_j(
    lowered, n_chains: int = 4, n_draws: int = 200, seed: int = 3
):  # type: ignore[no-untyped-def]
    """Random reduced fluxes, plus the ``J`` the objective assigns them.

    Returning both is the point: every assertion below compares `pilot_energy_scale`'s answer with
    one computed *here*, from `evaluate_many` and plain NumPy — never with a second call into the
    thing under test.
    """
    rng = np.random.default_rng(seed)
    fluxes = rng.uniform(0.0, 10.0, size=(n_chains, n_draws, lowered.n_free))
    _, _, j = lowered.evaluate_many(fluxes.reshape(-1, lowered.n_free))
    return fluxes, np.asarray(j).reshape(n_chains, n_draws)


class TestThePilotScaleIsTheNeutralSpread:
    def test_s_J_is_the_standard_deviation_of_J(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """Checked against the arithmetic, not against the implementation."""
        fluxes, j = _pilot_fluxes_with_known_j(fork_lowered)
        scale = pilot_energy_scale(
            fork_lowered, fluxes,
            optimum=synthetic_optimum(fork_lowered, 12.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert scale.mode == "pilot_sd"
        assert scale.value == pytest.approx(float(np.std(j.reshape(-1), ddof=1)), rel=1e-12)

    def test_J_star_cannot_move_the_scale_by_one_ulp(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """The structural gain over `choose_energy_scale`: ``J*`` is reported, never in the axis.

        ``s_J = J* − Q_q(J(W))`` subtracts three model-derived artifacts, and M6 and M7 each found
        an unguarded join among exactly those three. Here a wrong ``J*`` corrupts a *reported
        diagnostic* (Δ₀, G) and nothing else.
        """
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered)
        low = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        high = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 1e6),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert low.value == high.value, "J* moved s_J — it must not enter the scale at all"
        assert low.pilot is not None and high.pilot is not None
        assert low.pilot.gap != high.pilot.gap, "Δ₀ must track J*: it is the reported observable"

    def test_the_scale_is_invariant_to_an_additive_constant_of_J(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """``s_J`` must not move when ``J → J + c`` (§1.6.4), because a constant provably cannot
        change any probability. A **spread** has this for free where a *range to J\\** needs both
        ends to shift together — one more reason the estimand is the SD.

        The shift is 1e12, big enough that a magnitude-based floor would have fired: the premise is
        asserted below, so this test cannot quietly go toothless (the M4/M6 lesson).
        """
        shift = 1e12
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered)

        plain = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        shifted_objective = dataclasses.replace(fork_lowered, mu_offset=shift)
        shifted = pilot_energy_scale(
            shifted_objective, fluxes,
            optimum=synthetic_optimum(shifted_objective, 12.0 + shift),
            pilot_polytope_key=shifted_objective.polytope_key,
        )

        old_floor = 1e-9 * max(1.0, 12.0 + shift)
        assert plain.value <= old_floor, (
            f"a magnitude floor ({old_floor:.3e}) does not reject a spread of {plain.value:.4g}, "
            "so this test cannot fail on the bug it exists to catch"
        )

        # The tolerance is the arithmetic's, not a fudge: at a 1e12 baseline one ULP is 1.2e-4, so
        # each shifted J carries ~1 ULP of *representation* error against a spread of ~3.3 — about
        # 4e-5 relative per draw, averaging down over the pilot to ~1e-6. **The estimand is exactly
        # shift-invariant; its float64 computation cannot be**, which is the whole reason the
        # resolution floor exists. (M9 recorded the same effect for the warm-up range.) A bar tight
        # enough to reject this would be testing float64, not the estimator.
        ulp_noise = float(np.spacing(shift)) / plain.value
        assert shifted.value == pytest.approx(plain.value, rel=1e-4), (
            "an additive constant of J moved s_J by more than the representation error it forces "
            f"(one ULP at this baseline is {ulp_noise:.2e} of the spread)"
        )

    def test_a_spread_below_float64_resolution_raises(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """Undefined ≠ imprecise. A pilot with no variation in ``J`` has no neutral fluctuation for
        β to be measured against, so **every** rung would be meaningless — and unlike a precision
        shortfall, no amount of extra sampling fixes it.

        The refusal reuses M6's predeclared 64-ULP mechanism rather than a bespoke bar: Codex's
        round-3 catch was that an invented "is σ̂₀ too small" criterion quietly becomes the very
        noise-floor gate this design rejects.
        """
        fluxes = np.zeros((4, 50, fork_lowered.n_free))  # every draw identical ⇒ sd exactly 0
        with pytest.raises(DegenerateEnergyScaleError, match="not resolvable"):
            pilot_energy_scale(
                fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
                pilot_polytope_key=fork_lowered.polytope_key,
            )

    def test_there_is_no_fallback_for_a_degenerate_pilot(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """`choose_energy_scale` takes a declared fallback; this deliberately does not.

        A degenerate *warm-up range* can mean the vertices happen not to spread. A degenerate
        *pilot* means the chain saw no variation in ``J`` at all — there is nothing to declare a
        scale relative to, and substituting one would name a selection pressure with no referent.
        """
        import inspect

        assert "fallback" not in inspect.signature(pilot_energy_scale).parameters


class TestThePilotScaleReportsItsOwnPrecision:
    def test_relative_se_uses_the_centered_square_ess_and_pearson_kurtosis(  # type: ignore[no-untyped-def]
        self, fork_lowered
    ) -> None:
        """``se(σ̂)/σ ≈ √(K−1)/(2√ESS_{(J−μ)²})``, recomputed here from the definition.

        The Gaussian ``1/√(2·ESS_J)`` is wrong twice over — it fixes ``K = 3`` and reads the ESS of
        the wrong series (Codex, M10 r1). On the real model the two ESSs differ by 2.17×, so this is
        not a rounding detail.
        """
        from gsmm_compiler.diagnostics import effective_sample_size

        fluxes, j = _pilot_fluxes_with_known_j(fork_lowered)
        scale = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert scale.pilot is not None

        centered = j - float(np.mean(j))
        m2 = float(np.mean(centered**2))
        m4 = float(np.mean(centered**4))
        kurtosis = m4 / m2**2
        ess = float(effective_sample_size((centered**2)[:, :, np.newaxis])[0])
        expected = np.sqrt(kurtosis - 1.0) / (2.0 * np.sqrt(ess))

        assert scale.pilot.relative_se == pytest.approx(expected, rel=1e-9)
        assert scale.pilot.excess_kurtosis == pytest.approx(kurtosis - 3.0, rel=1e-9)
        assert scale.pilot.ess_centered_square == pytest.approx(ess, rel=1e-9)

    def test_imprecision_warns_and_does_not_refuse(self, fork_lowered, caplog) -> None:  # type: ignore[no-untyped-def]
        """A precision bar would let an unlucky pilot seed reject a correct run — this repo's
        worst-ever defect in a new coat (BUILD_PLAN §1.4.2). A short pilot must still *produce* a
        scale, and say it is imprecise."""
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered, n_chains=2, n_draws=8)
        scale = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert scale.pilot is not None
        assert scale.value > 0.0, "an imprecise scale is still a scale"
        assert scale.pilot.relative_se > PILOT_SCALE_TARGET_RELATIVE_SE
        assert scale.pilot.precision_warning is True

    def test_R90_is_one_for_gaussian_J_and_is_only_a_diagnostic(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """``R₉₀ = (Q₉₅−Q₀₅)/(3.289707·σ̂)`` is 1 for a Gaussian population.

        It is reported, never used to *pick* an estimator: switching to a robust scale wherever the
        tails look inconvenient would make β mean different things in different strains — the exact
        failure ``s_J`` exists to prevent — and would forfeit the ``I₀ = 1`` property that motivates
        the SD (Codex, M10 r2).
        """
        rng = np.random.default_rng(11)
        n_chains, n_draws = 4, 4000
        # Drive J Gaussian directly: J is linear in v off the L1 kink, so a Gaussian v with a
        # positive mean well away from zero gives a Gaussian J.
        fluxes = rng.normal(5.0, 0.5, size=(n_chains, n_draws, fork_lowered.n_free))
        scale = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 50.0),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert scale.pilot is not None
        assert scale.pilot.robust_width_ratio == pytest.approx(1.0, abs=0.05)
        assert abs(scale.pilot.excess_kurtosis) < 0.3

    def test_headroom_G_is_the_gap_in_neutral_sds(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """``G = Δ₀/σ̂₀`` — the per-strain observable ``pilot_sd`` exists to unhide."""
        fluxes, j = _pilot_fluxes_with_known_j(fork_lowered)
        j_star = 40.0
        scale = pilot_energy_scale(
            fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, j_star),
            pilot_polytope_key=fork_lowered.polytope_key,
        )
        assert scale.pilot is not None
        expected_gap = j_star - float(np.mean(j))
        assert scale.pilot.gap == pytest.approx(expected_gap, rel=1e-12)
        assert scale.pilot.headroom == pytest.approx(expected_gap / scale.value, rel=1e-12)


class TestThePilotScaleRefusesArtifactsThatNeverMet:
    def test_a_pilot_from_the_wrong_polytope_is_refused(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """The pilot fluxes are a bare array with no identity of their own — the M7 round-3 defect,
        one mode further on."""
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered)
        with pytest.raises(IncompatibleObjectiveError, match="different polytope"):
            pilot_energy_scale(
                fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
                pilot_polytope_key="some-other-polytope",
            )

    def test_an_optimum_from_another_objective_is_refused(
        self, fork_full: FluxPolytope, fork_reduced: ReducedPolytope, fork_lowered
    ) -> None:  # type: ignore[no-untyped-def]
        """It cannot move ``s_J`` — but it would corrupt Δ₀ and G, the numbers a cross-strain
        comparison actually reads."""
        other = lower_objective(
            fork_reduced, SparseFluxObjective.from_polytope(fork_full, l1_penalty=0.75)
        )
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered)

        with pytest.raises(IncompatibleObjectiveError, match="different objective"):
            pilot_energy_scale(
                fork_lowered, fluxes, optimum=synthetic_optimum(other, 12.0),
                pilot_polytope_key=fork_lowered.polytope_key,
            )

    def test_a_single_chain_pilot_is_refused(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """R̂ and the between-chain σ̂ spread are the only checks here that can see retained
        initialization — and an ESS cannot (Codex, M10 r3)."""
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered, n_chains=1)
        with pytest.raises(ObjectiveError, match="at least"):
            pilot_energy_scale(
                fork_lowered, fluxes, optimum=synthetic_optimum(fork_lowered, 12.0),
                pilot_polytope_key=fork_lowered.polytope_key,
            )

    def test_flat_pilot_fluxes_are_refused(self, fork_lowered) -> None:  # type: ignore[no-untyped-def]
        """``(n_chains, n_draws, n_free)`` is required, not a convenience: a pooled ``(N, n_free)``
        array would silently discard the chain structure R̂ needs."""
        fluxes, _ = _pilot_fluxes_with_known_j(fork_lowered)
        pooled = fluxes.reshape(-1, fork_lowered.n_free)
        with pytest.raises(ObjectiveError, match="expected"):
            pilot_energy_scale(
                fork_lowered, pooled, optimum=synthetic_optimum(fork_lowered, 12.0),
                pilot_polytope_key=fork_lowered.polytope_key,
            )


class TestCalibrateRunsOnlyThePilotsItWasAsked:
    """Both stages are opt-in switches, and `calibrate` must run exactly what was asked."""

    def test_neither_switch_leaves_v1_untouched(
        self, fork_setup, fork_reduced: ReducedPolytope, fork_lowered, fork_certificate
    ) -> None:  # type: ignore[no-untyped-def]
        """With both stages off, `calibrate` must return ``T₀`` and the support-vertex scale
        **unchanged** — so a caller can route every run through it without changing v1's numbers."""
        geometry, bootstrap = fork_setup
        result = calibrate(
            geometry, fork_reduced, bootstrap, fork_lowered,
            model_id="fork",
            optimum=synthetic_optimum(fork_lowered, 12.0),
            bootstrap_certificate=fork_certificate,
            sampler=dataclasses.replace(PILOT, pilot_reround=False, energy_scale="warmup_range"),
        )
        assert result.transform is bootstrap
        assert result.geometry_pilot is None
        assert result.scale_pilot is None
        assert result.energy_scale.mode == "warmup_range"

    def test_reround_alone_runs_one_pilot(
        self, fork_setup, fork_reduced: ReducedPolytope, fork_lowered, fork_certificate
    ) -> None:  # type: ignore[no-untyped-def]
        geometry, bootstrap = fork_setup
        result = calibrate(
            geometry, fork_reduced, bootstrap, fork_lowered,
            model_id="fork",
            optimum=synthetic_optimum(fork_lowered, 12.0),
            bootstrap_certificate=fork_certificate,
            sampler=dataclasses.replace(PILOT, pilot_reround=True, energy_scale="warmup_range"),
        )
        assert result.geometry_pilot is not None
        assert result.scale_pilot is None
        assert result.transform is not bootstrap
        assert result.transform.diagnostics.covariance_source == "pilot"
        assert result.energy_scale.mode == "warmup_range"

    def test_the_scale_pilot_runs_under_the_rerounded_transform(
        self, fork_setup, fork_reduced: ReducedPolytope, fork_lowered, fork_certificate
    ) -> None:  # type: ignore[no-untyped-def]
        """Step 3 of the DAG: the scale pilot uses ``T₁``, so σ̂₀ is estimated from the
        better-mixing chain — while the geometry pilot stays bound to ``T₀``, the frame it ran in.

        A poor ``T₀`` cannot deform the neutral *target*, only the efficiency of estimating the
        scale from it, which is why the errors do not compound as target deformation (Codex, r2).
        """
        geometry, bootstrap = fork_setup
        result = calibrate(
            geometry, fork_reduced, bootstrap, fork_lowered,
            model_id="fork",
            optimum=synthetic_optimum(fork_lowered, 12.0),
            bootstrap_certificate=fork_certificate,
            sampler=dataclasses.replace(PILOT, pilot_reround=True, energy_scale="pilot_sd"),
        )
        assert result.scale_pilot is not None
        assert result.geometry_pilot is not None
        assert result.scale_pilot.recipe.transform_key == result.transform.content_key()
        assert result.geometry_pilot.recipe.transform_key == bootstrap.content_key()
        assert (
            result.scale_pilot.recipe.transform_key
            != result.geometry_pilot.recipe.transform_key
        )
        assert result.energy_scale.mode == "pilot_sd"
        assert result.energy_scale.pilot is not None

    def test_the_two_pilots_run_on_different_streams(
        self, fork_setup, fork_reduced: ReducedPolytope, fork_lowered, fork_certificate
    ) -> None:  # type: ignore[no-untyped-def]
        """The shipped orchestration must give its two pilots different streams — checked on the
        **minimal-payload path production actually takes**, not on a test-only variant of it.

        🔴 **M10.2b rewrote this too.** It asserted ``not np.allclose(geometry.fluxes,
        scale.fluxes)``, which (a) proves non-identity rather than independence and (b) was
        comparing two pilots run under **different transforms**, so it would have passed on a shared
        stream anyway — the frames alone move the numbers. The stage coordinate of the spawn key is
        the thing that actually separates them, and `NeutralPilot.__post_init__` has already proved
        those keys are the ones each recipe names. (Codex, M10.2b review round 1.)
        """
        geometry, bootstrap = fork_setup
        result = calibrate(
            geometry, fork_reduced, bootstrap, fork_lowered,
            model_id="fork",
            optimum=synthetic_optimum(fork_lowered, 12.0),
            bootstrap_certificate=fork_certificate,
            sampler=dataclasses.replace(PILOT, pilot_reround=True, energy_scale="pilot_sd"),
        )
        assert result.geometry_pilot is not None and result.scale_pilot is not None
        assert result.geometry_pilot.recipe.stage != result.scale_pilot.recipe.stage
        assert not np.array_equal(
            result.geometry_pilot.spawn_keys[:, 1], result.scale_pilot.spawn_keys[:, 1]
        )

    def test_the_reround_improves_conditioning(
        self, fork_setup, fork_reduced: ReducedPolytope, fork_lowered, fork_certificate
    ) -> None:  # type: ignore[no-untyped-def]
        """The whole reason spec §17.4 exists. Reported so the improvement is *shown*, not asserted:
        `bootstrap_condition_number` is kept beside the final one."""
        geometry, bootstrap = fork_setup
        result = calibrate(
            geometry, fork_reduced, bootstrap, fork_lowered,
            model_id="fork",
            optimum=synthetic_optimum(fork_lowered, 12.0),
            bootstrap_certificate=fork_certificate,
            sampler=dataclasses.replace(PILOT, pilot_reround=True, energy_scale="warmup_range"),
        )
        assert result.bootstrap_condition_number == bootstrap.diagnostics.condition_number
        assert result.manifest()["rerounded"] is True
        assert result.manifest()["final_condition_number"] == (
            result.transform.diagnostics.condition_number
        )

    def test_an_objective_from_another_polytope_is_refused(
        self, simplex_setup, simplex_polytope: ReducedPolytope, fork_lowered,
        simplex_certificate,
    ) -> None:  # type: ignore[no-untyped-def]
        """The M6 "two artifacts that never met" join, guarded at the DAG's own entrance.

        A genuinely well-formed objective — lowered from the fork onto *the fork's* polytope, so it
        passes every check of its own — handed to `calibrate` alongside the simplex. Nothing about
        either artifact is malformed; they simply were never computed against each other, which is
        exactly the join that computes confidently and describes the wrong model.
        """
        geometry, bootstrap = simplex_setup
        with pytest.raises(CalibrationError, match="never met|not lowered"):
            calibrate(
                geometry, simplex_polytope, bootstrap, fork_lowered,
                model_id="simplex",
                optimum=synthetic_optimum(fork_lowered, 1.0),
                bootstrap_certificate=simplex_certificate,
                sampler=PILOT,
            )
