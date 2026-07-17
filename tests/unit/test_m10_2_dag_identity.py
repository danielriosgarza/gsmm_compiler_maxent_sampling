"""M10.2 — the pilot DAG's identity: what each artifact's key must name, and what must be refused.

Every test here guards a defect that was **live in the package** when it was written, and every one
of them was found by asking one question of an artifact this repo already had: *is it a function of
its key?* Four times the answer was no.

The through-line is BUILD_PLAN §1.1's asymmetry, which is the reason these are correctness tests and
not hygiene tests: **a false miss only recomputes; a false hit corrupts.** So an incomplete key is
strictly worse than an absent one — absent means no cache, incomplete means a cache that
confidently returns the wrong bytes.

Three of these could not have been caught by testing behaviour, only identity. A pilot run with the
wrong start hint is a *valid* pilot; a chain resumed from another experiment is a *valid* chain; a
transform loaded past its gate samples a *feasible-looking* polytope. Each is individually correct
and collectively wrong — M6's disease, which is why the tests are about keys.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import (
    GeometryError,
    ReducedGeometry,
    build_geometry,
)
from gsmm_compiler.batch import (
    BatchError,
    _already_done,
    build_l3_bundle,
    geometry_cache_key,
    prepare_model,
    sample_recipe_key,
)
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.calibration import (
    GeometryPilot,
    run_geometry_pilot,
    run_scale_pilot,
)
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.rounding import (
    ReachabilityCertificate,
    RoundingError,
    build_transform,
    certify_reachable_mass_balance,
    require_certified_transform,
)

PILOT = SamplerConfig(
    betas=(0.0,), n_chains=2, n_samples=60, burn_in=60,
    pilot_chains=2, pilot_samples=60, pilot_burn_in=60,
)


@pytest.fixture(scope="module")
def simplex_geometry(simplex_polytope: ReducedPolytope) -> ReducedGeometry:
    return build_geometry(simplex_polytope, model_id="simplex")


@pytest.fixture(scope="module")
def simplex_transform(simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    return build_transform(simplex_geometry, simplex_polytope)


@pytest.fixture(scope="module")
def simplex_certificate(simplex_transform, simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    return certify_reachable_mass_balance(simplex_transform, simplex_polytope)


@pytest.fixture(scope="module")
def simplex_pilot(  # type: ignore[no-untyped-def]
    simplex_transform, simplex_polytope: ReducedPolytope, simplex_certificate
) -> GeometryPilot:
    return run_geometry_pilot(
        simplex_transform, simplex_polytope, config=PILOT, model_id="simplex",
        certificate=simplex_certificate,
    )


# ---- (b) the pilot's key --------------------------------------------------------------------


class TestTheNeutralPilotIsAFunctionOfItsKey:
    """The M10.1 defect: `NeutralPilot` *claimed* objective-independence its code did not have.

    `calibrate` fed both β=0 pilots ``optimum_coordinates`` — the objective's own LP optimum —
    while `content_key` hashed no objective and no start. Measured on the example model, two pilots
    differing in nothing else: **identical key**, max |Δy| = 2.79, ``T₁`` cond 7198 vs 9663,
    ``s_J`` 2.6287 vs 2.4995.

    Not bias — both are honest draws from one β=0 law and the gap is Monte Carlo noise. Worse: the
    artifact was not a function of its key, so M7's two-objectives-on-one-polytope case would take
    the first cache hit and never know. The mechanism is sharper than "a different start": the hint
    changes the support hull's cardinality, hence the Dirichlet draw's dimension, hence RNG
    consumption on *every* subsequent transition — the streams desynchronise.
    """

    def test_a_neutral_pilot_cannot_be_given_objective_state(self) -> None:
        """Structural, not conventional — the fix is that the parameter does not exist.

        Passing ``None`` correctly is a thing a caller can forget; a parameter that is absent is a
        thing a caller cannot get wrong. Same guard shape M7 uses to keep reweighting out of the
        sampler and M10.1 uses to keep `calibration` out of `maxent_sampler`.
        """
        for builder in (run_geometry_pilot, run_scale_pilot):
            parameters = inspect.signature(builder).parameters
            assert "optimum_coordinates" not in parameters
            assert "objective" not in parameters
            assert "optimum" not in parameters
            # M10.2b: nor a `stage`, which is what let a class and its key disagree.
            assert "stage" not in parameters

    def test_the_key_separates_pilots_that_draw_different_numbers(
        self, simplex_transform, simplex_polytope: ReducedPolytope,  # type: ignore[no-untyped-def]
        simplex_certificate,
    ) -> None:
        """Each field is checked by *moving it* — a key that omits one returns here as a false hit.

        `seed` is the `SeedSequence` entropy (`provenance.stream_seed`) and `refresh_interval` sets
        the float64 refresh phase, which M5 settled is part of this chain's state. Both change the
        bytes; neither was in the key.
        """
        base = run_geometry_pilot(
            simplex_transform, simplex_polytope, config=PILOT, model_id="simplex",
            certificate=simplex_certificate,
        )
        for field, value in (("seed", PILOT.seed + 1), ("refresh_interval", 7)):
            moved = run_geometry_pilot(
                simplex_transform, simplex_polytope,
                config=dataclasses.replace(PILOT, **{field: value}),
                model_id="simplex",
                certificate=simplex_certificate,
            )
            assert moved.recipe.content_key() != base.recipe.content_key(), (
                f"moving {field} changed the draws but not the key — a false cache hit"
            )
        # M10.2b: the tolerance reaches start selection, chord construction and refresh validation,
        # so it moves the draws. It was not a parameter at all until this milestone.
        retoleranced = run_geometry_pilot(
            simplex_transform, simplex_polytope, config=PILOT, model_id="simplex",
            certificate=simplex_certificate, feasibility_tol=1e-7,
        )
        assert retoleranced.recipe.content_key() != base.recipe.content_key()

    def test_the_premise_that_moving_those_fields_moves_the_draws(
        self, simplex_transform, simplex_polytope: ReducedPolytope,  # type: ignore[no-untyped-def]
        simplex_certificate,
    ) -> None:
        """The control the test above needs to mean anything.

        If `seed` did *not* move the draws, the assertion that it moves the key would be testing a
        key that is merely *fussy*, not one that is *complete* — and this repo has shipped a
        regression test that could not fail on its own bug before (M6's `s_J` floor). So: prove the
        premise.
        """
        base = run_geometry_pilot(
            simplex_transform, simplex_polytope, config=PILOT, model_id="simplex",
            certificate=simplex_certificate,
        )
        reseeded = run_geometry_pilot(
            simplex_transform, simplex_polytope,
            config=dataclasses.replace(PILOT, seed=PILOT.seed + 1),
            model_id="simplex",
            certificate=simplex_certificate,
        )
        assert not np.array_equal(base.coordinates, reseeded.coordinates)

    def test_the_pilot_carries_no_objective_key(self, simplex_pilot: GeometryPilot) -> None:
        """The independence is the *point*, once it is true.

        M7 puts a base and a reweighted objective on one polytope; they must be calibrated against
        the **same** neutral ensemble or their β axes are not comparable. That reuse is only sound
        because nothing objective-derived reaches the pilot — which is what the signature test
        above now enforces.
        """
        manifest = simplex_pilot.manifest()
        assert not any("objective" in key for key in manifest)


# ---- (a) the geometry is serializable, and its key names its bytes ---------------------------


class TestTheGeometryRoundTripsThroughItsKey:
    def test_a_rebuilt_geometry_is_the_geometry(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        arrays, meta = simplex_geometry.to_bundle()
        rebuilt = ReducedGeometry.from_bundle(arrays, meta, simplex_polytope)

        assert rebuilt.content_key() == simplex_geometry.content_key()
        for name in ("scaling", "basis", "center", "support_points"):
            assert np.array_equal(getattr(rebuilt, name), getattr(simplex_geometry, name))
        # The certificate and diagnostics round-trip as *objects*, not as `as_dict`'s report keys —
        # `as_dict` renames fields and injects the derived `resolution`, so feeding it back would
        # not even construct.
        assert rebuilt.certificate == simplex_geometry.certificate
        assert rebuilt.diagnostics == simplex_geometry.diagnostics

    def test_the_basis_comes_back_fortran_contiguous(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        """A performance contract the dataclass documents — and never an identity one.

        `provenance.hash_array` C-orders before hashing, so order cannot change the key. That is
        exactly why it has to be asserted separately: a reconstruction that silently returned a
        C-ordered basis would keep every key green while making each column read strided.
        """
        arrays, meta = simplex_geometry.to_bundle()
        rebuilt = ReducedGeometry.from_bundle(arrays, meta, simplex_polytope)
        assert rebuilt.basis.flags.f_contiguous

    def test_corrupted_bytes_cannot_reproduce_the_key(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        """The guard that makes 'content-addressed' mean something.

        A cache is content-addressed or it is a filing cabinet. An artifact that does not hash to
        the key it was stored under is not the artifact the key names, whatever its arrays look
        like — and this perturbation is deliberately *plausible*: the geometry stays orthonormal,
        every shape agrees, and only the key knows.
        """
        arrays, meta = simplex_geometry.to_bundle()
        arrays = {**arrays, "center": np.asarray(arrays["center"]) + 1e-9}
        with pytest.raises(GeometryError, match="does not reproduce its own cached content key"):
            ReducedGeometry.from_bundle(arrays, meta, simplex_polytope)

    def test_a_geometry_cannot_be_rebuilt_against_another_polytope(
        self, simplex_geometry: ReducedGeometry, coupled_box_polytope: ReducedPolytope
    ) -> None:
        arrays, meta = simplex_geometry.to_bundle()
        with pytest.raises(GeometryError, match="different polytope"):
            ReducedGeometry.from_bundle(arrays, meta, coupled_box_polytope)

    def test_support_points_are_part_of_the_geometrys_identity(
        self, simplex_geometry: ReducedGeometry
    ) -> None:
        """They were not, and they are load-bearing twice over.

        `rounding.build_transform` takes ``C_q`` from the support hull, and
        `sparse_objective.choose_energy_scale` reads ``warmup_range``'s ``s_J`` off it. So two
        geometries agreeing on ``B`` and disagreeing on their hull produce a different transform
        *and* a different β axis — while hashing identically. The M6 join, one artifact further on.
        """
        moved = dataclasses.replace(
            simplex_geometry, support_points=simplex_geometry.support_points * 0.5
        )
        assert moved.content_key() != simplex_geometry.content_key()


# ---- (c)+(d) the certificate gates every transform that gets sampled -------------------------


class TestNoTransformIsSampledUncertified:
    """M9's gate was installed in one place: the *miss* path of `batch._load_or_build_geometry`.

    So a cache hit skipped it, and `maxent build-geometry --cache-dir` wrote a bundle under the
    same key after printing ``REFUSED`` for it. Warm the cache, then sample: an uncertified
    transform, no error, every downstream check green. That gate cost a two-round collab review and
    Codex's counterexample killed the first fix for it — and it could be walked past with two CLI
    commands.
    """

    def test_an_uncertified_certificate_is_refused(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        transform = build_transform(simplex_geometry, simplex_polytope)
        real = certify_reachable_mass_balance(transform, simplex_polytope)
        # A residual just past the contract: the smallest lie the gate must still catch.
        failed = dataclasses.replace(real, worst_absolute=real.contract * 1.001)
        with pytest.raises(RoundingError, match="must not be sampled"):
            require_certified_transform(failed, transform, simplex_polytope)

    def test_a_certificate_for_another_transform_is_refused(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        """The check M10 makes necessary — and `polytope_key` cannot make it.

        ``T₀`` and ``T₁`` are two transforms of one polytope, so they share a ``polytope_key``
        *exactly*. Before M10 there was only ever one transform per polytope and the certificate
        needed no transform of its own; the pilot DAG makes a second, and the certificate is a
        statement about one materialized matrix (it recomputes ``E = S·T`` and ``Ω`` from ``T⁺``).
        """
        transform = build_transform(simplex_geometry, simplex_polytope)
        certificate = certify_reachable_mass_balance(transform, simplex_polytope)
        other = dataclasses.replace(
            transform, transform=np.asfortranarray(transform.transform * 2.0)
        )
        assert certificate.polytope_key == other.polytope_key, (
            "premise: the polytope key cannot tell these apart — which is why transform_key exists"
        )
        with pytest.raises(RoundingError, match="different transform"):
            require_certified_transform(certificate, other, simplex_polytope)

    def test_the_verdict_is_re_derived_and_not_read_off(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        """`to_cache` stores the evidence; `as_dict` stores the verdict. The cache uses `to_cache`.

        `is_certified` is a *derived* property (``worst_absolute <= contract``). Caching it would
        let a bundle assert innocence; caching the fields makes that inexpressible — the loader
        computes the verdict from the numbers. (M9: never trust a reading, check the bound.)
        """
        transform = build_transform(simplex_geometry, simplex_polytope)
        certificate = certify_reachable_mass_balance(transform, simplex_polytope)
        cached = certificate.to_cache()
        assert "is_certified" not in cached
        assert "reachable_is_certified" not in cached

        # A bundle that *claims* it passed while its evidence says otherwise.
        cached["worst_absolute"] = certificate.contract * 10.0
        assert not ReachabilityCertificate.from_cache(cached).is_certified

    def test_a_poisoned_cache_entry_cannot_be_sampled(
        self, tmp_path, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope  # type: ignore[no-untyped-def]
    ) -> None:
        """The end-to-end bypass, reproduced at the layer that used to believe it.

        This is what two CLI commands could do before M10.2: put a bundle in the store whose
        certificate does not hold, and let the reader trust the key. The reader now re-derives the
        verdict on **every** path, so the hit is refused exactly like the miss would have been.
        """
        from gsmm_compiler.config import Config

        config = Config()
        transform = build_transform(simplex_geometry, simplex_polytope)
        certificate = certify_reachable_mass_balance(transform, simplex_polytope)
        assert certificate.is_certified, "premise: this geometry is genuinely samplable"

        geometry_arrays, geometry_meta = simplex_geometry.to_bundle()
        transform_arrays, transform_meta = transform.to_bundle()
        poisoned = dataclasses.replace(
            certificate, worst_absolute=certificate.contract * 100.0
        )

        cache = ArtifactCache(tmp_path / "cache")
        cache.store(
            "L3",
            geometry_cache_key(simplex_polytope, config, model_id="simplex"),
            arrays={
                **{f"geometry_{k}": v for k, v in geometry_arrays.items()},
                **transform_arrays,
            },
            meta={
                "geometry": geometry_meta,
                "transform": transform_meta,
                "geometry_manifest": simplex_geometry.manifest(),
                "reachability_certificate": poisoned.to_cache(),
            },
        )

        from gsmm_compiler.batch import _load_or_build_geometry

        with pytest.raises(RoundingError, match="must not be sampled"):
            _load_or_build_geometry(
                simplex_polytope, config, model_id="simplex", cache=cache
            )

    def test_calibrate_cannot_be_given_an_uncertified_bootstrap(
        self,
        simplex_geometry: ReducedGeometry,
        simplex_polytope: ReducedPolytope,
        simplex_flux_polytope,  # type: ignore[no-untyped-def]
    ) -> None:
        """The pilots are chains. An uncertified ``T₀`` walks off the manifold *before* production.

        `bootstrap_certificate` is a required argument rather than an internal computation: the
        proof already exists (`build_l3_bundle` made it), recomputing it would be 334 wasted LPs on
        the same matrix, and demanding it makes an uncertified transform unable to enter the pilot
        DAG at all. Passing it is what `prepare_model` does; there is no way to not pass it.
        """
        from tests.conftest import synthetic_optimum

        from gsmm_compiler.calibration import calibrate
        from gsmm_compiler.sparse_objective import SparseFluxObjective, lower_objective

        # Built from one source, so the objective and the polytope are the same model — M6's
        # `check_compatible` refuses the alternative, which is the point of it.
        reduced = simplex_flux_polytope.reduce()
        geometry = build_geometry(reduced, model_id="simplex_flux")
        transform = build_transform(geometry, reduced)
        real = certify_reachable_mass_balance(transform, reduced)

        assert "bootstrap_certificate" in inspect.signature(calibrate).parameters
        assert (
            inspect.signature(calibrate).parameters["bootstrap_certificate"].default
            is inspect.Parameter.empty
        ), "a defaulted proof is a proof a caller can forget"

        lowered = lower_objective(
            reduced, SparseFluxObjective.from_polytope(simplex_flux_polytope, l1_penalty=0.25)
        )
        failed = dataclasses.replace(real, worst_absolute=real.contract * 1.001)
        with pytest.raises(RoundingError, match="must not be sampled"):
            calibrate(
                geometry, reduced, transform, lowered,
                model_id="simplex_flux",
                optimum=synthetic_optimum(lowered, 1.0, reduced.content_key()),
                sampler=PILOT, bootstrap_certificate=failed,
            )

    def test_the_reported_certificate_names_the_transform_that_is_sampled(
        self, tmp_path, toy_path  # type: ignore[no-untyped-def]
    ) -> None:
        """Under re-rounding the run manifest must describe ``T₁``, because ``T₁`` is what ran.

        The regression this catches was introduced *by this milestone*: `prepare_model` reported
        the L3 bundle's certificate, which is ``T₀``'s, while production sampled ``T₁``. A manifest
        describing an artifact that was not used is this package's signature bug — and the report
        must be in `as_dict` form, not the cache's field-named `to_cache`, or a reader looking for
        ``reachable_is_certified`` finds nothing.
        """
        from gsmm_compiler.batch import ModelSpec
        from gsmm_compiler.config import Config

        config = dataclasses.replace(
            Config(),
            sampler=SamplerConfig(
                betas=(0.0,), n_chains=1, n_samples=20, burn_in=20, pilot_reround=True,
                pilot_chains=2, pilot_samples=40, pilot_burn_in=40,
            ),
        )
        plan = prepare_model(ModelSpec(model_path=str(toy_path)), config)
        report = plan.reports["reachability_certificate"]

        assert "reachable_is_certified" in report, "the report lost `as_dict`'s derived verdict"
        assert report["reachable_is_certified"] is True
        assert report["reachable_transform_key"] == plan.transform.content_key(), (
            "the manifest certifies a transform other than the one this run samples"
        )

    def test_a_pilot_cannot_be_run_on_an_uncertified_transform(
        self, simplex_transform, simplex_polytope: ReducedPolytope, simplex_certificate  # type: ignore[no-untyped-def]
    ) -> None:
        """Gating `calibrate` alone left the public pilot builder — the same chain — open.

        A caller could step past the guard without fabricating anything (Codex, round 4). A pilot is
        not a lesser chain: every artifact the DAG freezes descends from its draws, so it takes the
        same gate production does.
        """
        failed = dataclasses.replace(
            simplex_certificate, worst_absolute=simplex_certificate.contract * 1.001
        )
        for builder in (run_geometry_pilot, run_scale_pilot):
            with pytest.raises(RoundingError, match="must not be sampled"):
                builder(
                    simplex_transform, simplex_polytope, config=PILOT, model_id="simplex",
                    certificate=failed,
                )

    def test_a_certificate_cannot_relax_its_own_bar(
        self, simplex_geometry: ReducedGeometry, simplex_polytope: ReducedPolytope
    ) -> None:
        """Re-deriving the verdict is not enough if the artifact picks the bar it is judged against.

        `certify_reachable_mass_balance` accepts any positive contract, so a caller can ask for
        ``contract=1.0``, get a **truthful** `is_certified`, and walk the gate — no fabrication, no
        corruption, just a proof of a different and useless proposition. M9's finding was that there
        is **one** declared definition of mass-balanced (η = 1e-9, the same one emitted samples are
        held to), so the gate judges against *the policy* and the certificate's own contract is
        provenance. (Codex, round 5.)
        """
        from gsmm_compiler.rounding import DEFAULT_MASS_BALANCE_CONTRACT

        transform = build_transform(simplex_geometry, simplex_polytope)
        real = certify_reachable_mass_balance(transform, simplex_polytope)

        # Truthful under its own absurd bar, and it says so: `is_certified` is True.
        relaxed = dataclasses.replace(
            real, contract=1.0, worst_absolute=DEFAULT_MASS_BALANCE_CONTRACT * 1000.0
        )
        assert relaxed.is_certified, "premise: it passes *its own* declared contract"
        with pytest.raises(RoundingError, match="not this package's declared contract"):
            require_certified_transform(relaxed, transform, simplex_polytope)

        # A *stricter* bar must still pass: worst_absolute is what is tested, and a smaller
        # residual clears a larger bar.
        stricter = dataclasses.replace(real, contract=DEFAULT_MASS_BALANCE_CONTRACT / 1000.0)
        require_certified_transform(stricter, transform, simplex_polytope)

    def test_corrupt_evidence_cannot_certify(self) -> None:
        """The hole in the `to_cache`-stores-evidence design, which the design itself opened.

        Storing the fields rather than the verdict stops a bundle *asserting* it passed — but only
        if the fields are evidence. ``worst_absolute = −1`` is not a claim, it is nonsense, and it
        makes ``worst_absolute <= contract`` **true**: a corrupted certificate certified, through
        the very mechanism built to stop a fabricated one. This is the repo's stated corruption
        model (accidental damage); an adversary who hand-builds a *plausible* certificate defeats
        any Python-level proof object, and this does not pretend otherwise.
        """
        good = dict(
            worst_absolute=3.8e-11, worst_row=0, worst_row_id="X", contract=1e-9,
            n_rows=10, n_rows_certified=10, n_lps=20, elapsed_seconds=0.5,
            polytope_key="p", transform_key="t",
        )
        assert ReachabilityCertificate(**good).is_certified, "premise: this one is genuine"

        for what, damage in (
            ("a negative residual passes `<= contract`", {"worst_absolute": -1.0}),
            ("a NaN residual", {"worst_absolute": float("nan")}),
            ("a non-finite contract", {"contract": float("inf")}),
            ("a negative LP count", {"n_lps": -5}),
            ("more rows certified than exist", {"n_rows_certified": 11}),
        ):
            with pytest.raises(RoundingError, match="certificate"):
                ReachabilityCertificate(**{**good, **damage})  # type: ignore[arg-type]
                pytest.fail(f"corrupt evidence accepted: {what}")

    def test_the_l3_writer_refuses_to_return_an_uncertified_bundle(self) -> None:
        """One writer, so the schema cannot drift into two.

        The CLI's `--cache-dir` path and `batch` now share `build_l3_bundle` — which raises rather
        than returning an uncertified bundle, so neither caller can store one even by trying.
        """
        source = inspect.getsource(build_l3_bundle)
        assert "require_certified_transform" in source
        cli_source = inspect.getsource(
            __import__("gsmm_compiler.cli", fromlist=["_cmd_maxent_build_geometry"])
            ._cmd_maxent_build_geometry
        )
        assert "build_l3_bundle" in cli_source
        assert "to_bundle" not in cli_source, "the CLI must not assemble a second L3 schema"


# ---- (f) a COMPLETE marker names an experiment, not just a chain -----------------------------


class TestRestartCannotMixTwoExperiments:
    """§1.1 always specified this key (``L2 + L3 + β + chain seed coords + …``). Nothing built it.

    Restart skipped on ``COMPLETE`` alone and `store_chain` recorded only the ``polytope_key`` — so
    a results directory reused after any change that moves the numbers resumed the units it had and
    sampled the rest from a different law. Two experiments in one tree, stacked into one
    cross-model table, every per-chain diagnostic green: each chain really is correct, which is
    what makes the mixture invisible.

    M10 forced the issue rather than created it. Before the pilot DAG, ``T`` and ``s_J`` were pure
    functions of the polytope and the config; now they descend from a *pilot*, so two runs of an
    unchanged config against an unchanged model can legitimately disagree.
    """

    @staticmethod
    def _plan(tmp_path, toy_path):  # type: ignore[no-untyped-def]
        from gsmm_compiler.batch import ModelSpec
        from gsmm_compiler.config import Config

        config = dataclasses.replace(
            Config(),
            sampler=SamplerConfig(betas=(0.0,), n_chains=1, n_samples=20, burn_in=20),
        )
        return prepare_model(ModelSpec(model_path=str(toy_path)), config), config

    def test_the_recipe_key_moves_with_everything_that_moves_the_samples(
        self, tmp_path, toy_path  # type: ignore[no-untyped-def]
    ) -> None:
        plan, _ = self._plan(tmp_path, toy_path)
        base = sample_recipe_key(plan, beta_index=0, chain_index=0)

        # `energy_scale_value` is the pilot's whole fingerprint on production: under `pilot_sd` it
        # *is* σ̂₀, so a re-run with a different pilot lands here and nowhere else.
        moved = {
            "energy_scale_value": dataclasses.replace(
                plan, energy_scale_value=plan.energy_scale_value * 1.01
            ),
            "schedule": dataclasses.replace(
                plan, sampler=dataclasses.replace(plan.sampler, n_samples=21)
            ),
            "seed": dataclasses.replace(
                plan, sampler=dataclasses.replace(plan.sampler, seed=plan.sampler.seed + 1)
            ),
            "refresh_interval": dataclasses.replace(
                plan, sampler=dataclasses.replace(plan.sampler, refresh_interval=3)
            ),
            # `model_id` keys the RNG stream (`provenance.stream_seed`), so `--model-id` against
            # one file draws different numbers — and nothing else in the key names it.
            "model_id": dataclasses.replace(plan, model_id=plan.model_id + "_v2"),
            # Consumed by start selection and chord validation.
            "feasibility_tol": dataclasses.replace(plan, feasibility_tol=1e-8),
            # Changes the stored `trace_near_zero_counts` arrays — the artifact's own bytes.
            "near_zero_thresholds": dataclasses.replace(
                plan, near_zero_thresholds=(1e-5,)
            ),
            # The start hint. M7 settled it must NOT be keyed on the objective and `s_J`, because
            # it cannot move the invariant target. This key asks a different question — are these
            # bytes the same artifact? — and by that one it is no different from `seed`.
            "optimum_coordinates": dataclasses.replace(
                plan,
                optimum_coordinates=(
                    None
                    if plan.optimum_coordinates is None
                    else plan.optimum_coordinates + 1e-6
                ),
            ),
            # Storage decides the bytes on disk. M9 measured that `reduced` round-trips to 1.1e-13
            # and *not* bit-exactly (gemv per row vs gemm per block), so byte-identity holds within
            # a mode, not across one.
            "store_mode": dataclasses.replace(
                plan, storage=dataclasses.replace(plan.storage, mode="reduced")
            ),
        }
        for what, changed in moved.items():
            assert sample_recipe_key(changed, beta_index=0, chain_index=0) != base, (
                f"{what} changes the samples but not the recipe key"
            )
        assert sample_recipe_key(plan, beta_index=0, chain_index=1) != base

    def test_the_recipe_names_the_writer_as_well_as_the_kernel(self) -> None:
        """§1.1: "provenance in every key: parser + code + artifact-schema versions".

        The key carried `SAMPLER_IMPL_VERSION`, whose own docstring scopes it to the **transition
        kernel**. But `store_chain` decides the arrays written, their names, the casts and the
        manifest's fields — and had no version at all, so an output-only change kept the recipe key
        identical and left stale units looking resumable (Codex, round 4). Two pieces of code decide
        these bytes; two versions name them.
        """
        source = inspect.getsource(sample_recipe_key)
        assert "sampler_impl_version" in source
        assert "output_impl_version" in source, (
            "the writer that lays down the bytes is unnamed in the key that identifies them"
        )

    def test_the_start_hint_premise_it_really_does_move_the_draws(
        self, tmp_path, toy_path  # type: ignore[no-untyped-def]
    ) -> None:
        """The control for keying `optimum_coordinates` — and it is not a formality.

        The exclusion had a *good* argument behind it (M7's, which Codex conceded in that review):
        a start hint cannot move the invariant target. If the hint did not move the draws either,
        keying it would be superstition. So prove it moves them, in the production kernel, exactly
        as it was proved for the pilots.
        """
        from gsmm_compiler.maxent_sampler import run_chain

        plan, _ = self._plan(tmp_path, toy_path)
        if plan.optimum_coordinates is None:
            pytest.skip("this model has no start hint to move")

        def draw(hint):  # type: ignore[no-untyped-def]
            return run_chain(
                plan.transform, plan.reduced, config=plan.sampler, model_id=plan.model_id,
                chain_index=0, beta=0.0, beta_index=0, optimum_coordinates=hint,
            ).coordinates

        assert not np.array_equal(
            draw(plan.optimum_coordinates), draw(plan.optimum_coordinates * 0.5)
        )

    def test_a_complete_unit_from_another_recipe_is_refused(
        self, tmp_path, toy_path  # type: ignore[no-untyped-def]
    ) -> None:
        """Refuse, do not recompute.

        A results tree is the user's output, not a cache. Silently overwriting a foreign unit
        destroys a run someone may still want; silently keeping it reports a wrong number. This
        package refuses in that situation (M9's worker sweep did the same).
        """
        from gsmm_compiler.batch import SampleJob

        chain_dir = tmp_path / "unit"
        chain_dir.mkdir()
        (chain_dir / "manifest.json").write_text(json.dumps({"recipe_key": "a-different-run"}))
        (chain_dir / "COMPLETE").touch()

        job = SampleJob(
            chain_dir=chain_dir, model_id="m", beta=0.0, beta_index=0, chain_index=0,
            reduced=None, transform=None, objective=None,  # type: ignore[arg-type]
            energy_scale_value=1.0, j_star=0.0, optimum_coordinates=None,
            movable=np.zeros(0, dtype=np.intp), storage=None, sampler=None,  # type: ignore[arg-type]
            near_zero_thresholds=(), feasibility_tol=1e-9, recipe_key="this-run",
        )
        with pytest.raises(BatchError, match="different recipe"):
            _already_done(job)

        matching = dataclasses.replace(job, recipe_key="a-different-run")
        assert _already_done(matching), "the same recipe must still resume — this is restart"
