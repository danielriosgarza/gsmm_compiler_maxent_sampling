"""M10.2b: the pilots enter the content-addressed store — the DAG's expensive node, at last.

The arithmetic §1.6.7 recorded and this milestone re-measured on the example model:

=========================================  ==========  ========================
``build_geometry`` (~1100 LPs)             1.17 s      cached since M8
``build_transform`` → ``T₀``               0.005 s     — (derived)
**the two β=0 pilots**                     **19.31 s** **not cached until now**
``reround_transform`` → ``T₁``             0.009 s     — (derived)
``certify_reachable_mass_balance``         0.54 s      — (derived)
=========================================  ==========  ========================

So `--cache-dir` cached the cheap stage and recomputed the expensive one, 16.4× upside-down: a
restart under ``pilot_reround`` re-ran 19.3 s of pilots before it could resume one chain, and M9's
Amdahl ceiling of 24.9× fell to ~3.65×. §1.1's rule is **cache what is expensive, derive what is
cheap, key everything**, and the pilot is the only node here on the expensive side of it.

**What makes this a correctness milestone and not a performance one.** §1.1's asymmetry — a false
miss only recomputes, a false hit corrupts — means wiring a store to a key is where an incomplete
key stops being harmless. `PilotRecipe.content_key` existed through M10.2a with nothing able to hit
it; that is exactly why M10.2a could *find* it incomplete and why M10.2b must not leave it so. Two
questions run through every test below:

1. **Is the artifact a function of its key?** (`TestThePilotIsAFunctionOfItsKey`)
2. **Does a hit accept what a miss would refuse?** (`TestACacheHitIsNotAWayPastTheGate` — the
   M10.2a defect, which lived in a ``compute()`` closure that ran only on a miss.)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gsmm_compiler import calibration
from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.calibration import (
    PILOT_CACHE_LAYER,
    CalibrationError,
    GeometryPilot,
    ScalePilot,
    _load_or_run_pilot,
    pilot_recipe,
)
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.rounding import (
    RoundingError,
    build_transform,
    certify_reachable_mass_balance,
)

PILOT = SamplerConfig(
    betas=(0.0,), n_chains=2, n_samples=60, burn_in=60,
    pilot_chains=2, pilot_samples=60, pilot_burn_in=60,
)


@pytest.fixture(scope="module")
def setup(simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(simplex_polytope, model_id="simplex")
    transform = build_transform(geometry, simplex_polytope)
    certificate = certify_reachable_mass_balance(transform, simplex_polytope)
    return geometry, transform, certificate


def _run(pilot_type, setup, polytope, *, cache=None, certificate=None):  # type: ignore[no-untyped-def]
    _, transform, good = setup
    return _load_or_run_pilot(
        pilot_type, transform, polytope,
        config=PILOT, model_id="simplex",
        certificate=good if certificate is None else certificate,
        feasibility_tol=1e-9,
        cache=cache,
    )


# ---- the hit must be the miss -------------------------------------------------------------------


class TestACacheHitIsIndistinguishableFromAMiss:
    """The one thing a cache can get wrong that no test of the *numbers* would notice.

    Every property here is a property of the **object**, not of its values: dtype, contiguity and
    writeability decide whether the second run of a pipeline behaves like the first. This is also
    why the payload projection lives in the builder rather than in the store — if `run_*_pilot`
    returned both arrays and the cache kept one, a hit would hand back an object missing a field
    the miss had, and the crash would be reachable only on a re-run.
    """

    def test_the_three_paths_agree_bit_for_bit(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        cache = ArtifactCache(tmp_path / "cache")
        uncached = _run(GeometryPilot, setup, simplex_polytope)
        miss = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        hit = _run(GeometryPilot, setup, simplex_polytope, cache=cache)

        for other in (miss, hit):
            assert np.array_equal(uncached.coordinates, other.coordinates)
            assert np.array_equal(uncached.spawn_keys, other.spawn_keys)
            assert uncached.recipe == other.recipe
            assert other.coordinates.dtype == uncached.coordinates.dtype
            assert other.spawn_keys.dtype == uncached.spawn_keys.dtype
            assert other.coordinates.flags.c_contiguous
            assert not other.coordinates.flags.writeable, (
                "a cached pilot must be as frozen as a fresh one — spec §17.4/§18.3 turns on the "
                "pilot being unable to move once production starts"
            )

    def test_the_premise_that_the_second_call_really_was_a_hit(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        """The control every test in this class needs, and the milestone's whole point besides.

        Without it, "the hit equals the miss" would pass just as well against a cache that never
        stored anything and recomputed a deterministic chain twice — and this repo has shipped a
        regression test its own bug passed before (M6's ``s_J`` floor).
        """
        cache = ArtifactCache(tmp_path / "cache")
        first = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        assert cache.is_cached(PILOT_CACHE_LAYER, first.recipe.content_key())

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("the chains were re-run on a cache hit — 19.3 s wasted")

        monkeypatch.setattr(calibration, "run_chains", explode)
        second = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        assert np.array_equal(first.coordinates, second.coordinates)


# ---- the gate --------------------------------------------------------------------------------


class TestACacheHitIsNotAWayPastTheGate:
    """🔴 M10.2a's defect, one node along — and the highest-risk line in this diff.

    M9's mass-balance gate lived in the ``compute()`` closure of `batch._load_or_build_geometry`,
    which runs **only on a miss**; on a hit nothing read the certificate, so warming the cache and
    then sampling walked straight past it. Wiring the pilot as
    ``get_or_compute(layer, key, lambda: run_pilot(...))`` reproduces that defect exactly, and it
    would look completely natural. So the gate is in the caller, before the dispatch.
    """

    def test_a_cached_pilot_still_refuses_an_uncertified_transform(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        _, _, good = setup
        cache = ArtifactCache(tmp_path / "cache")
        warm = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        # The premise: without a *hit* here, the gate would be running on the miss path and this
        # test would prove nothing about the hit path.
        assert cache.is_cached(PILOT_CACHE_LAYER, warm.recipe.content_key())

        failed = dataclasses.replace(good, worst_absolute=good.contract * 1.001)
        with pytest.raises(RoundingError, match="must not be sampled"):
            _run(GeometryPilot, setup, simplex_polytope, cache=cache, certificate=failed)

    def test_the_gate_also_refuses_on_a_miss(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """Both paths, one acceptance semantics — and an empty cache is the miss path."""
        _, _, good = setup
        failed = dataclasses.replace(good, worst_absolute=good.contract * 1.001)
        cache = ArtifactCache(tmp_path / "cache")
        with pytest.raises(RoundingError, match="must not be sampled"):
            _run(GeometryPilot, setup, simplex_polytope, cache=cache, certificate=failed)

    def test_a_cached_pilot_still_refuses_a_transform_from_another_polytope(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """🔴 The hole behind the repair — found by `/collab` round 3 *in M10.2b's own fix*.

        M10.2b hoisted the certificate gate out of ``compute()`` and **left the polytope relation
        behind in `_run_pilot_chains`**, i.e. on the miss path only. That looks mitigated by the
        recipe, which hashes both `polytope_key` and `transform_key` — and it is not:
        `RoundedTransform.content_key` does **not** hash the transform's own `polytope_key`. So a
        transform whose `polytope_key` is a lie keys *identically*, the certificate gate passes
        (both of its comparisons are against unchanged values), and a hit serves the pilot while a
        miss refuses it. Measured exactly that before the fix.

        The lie has to be told this way — `dataclasses.replace` on the field alone — precisely
        because any *honest* wrong polytope changes the keys and is caught by three other checks.
        That is what makes the asymmetry invisible without an adversary.
        """
        _, transform, good = setup
        cache = ArtifactCache(tmp_path / "cache")
        warm = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        assert cache.is_cached(PILOT_CACHE_LAYER, warm.recipe.content_key())

        liar = dataclasses.replace(
            transform, polytope_key="a-polytope-this-transform-never-met"
        )
        # The premise, without which this test proves nothing: the lie is invisible to the key, so
        # the lookup really does land on the warmed entry.
        assert liar.content_key() == transform.content_key()

        with pytest.raises(CalibrationError, match="not built from this polytope"):
            _load_or_run_pilot(
                GeometryPilot, liar, simplex_polytope,
                config=PILOT, model_id="simplex", certificate=good,
                feasibility_tol=1e-9, cache=cache,
            )

    def test_a_refused_pilot_leaves_nothing_in_the_store(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """M10.2a's other half: `maxent build-geometry --cache-dir` stored its bundle **after
        printing REFUSED**, so the next run hit a cache entry the gate had already rejected."""
        _, transform, good = setup
        failed = dataclasses.replace(good, worst_absolute=good.contract * 1.001)
        cache = ArtifactCache(tmp_path / "cache")
        recipe = pilot_recipe(
            transform, simplex_polytope, config=PILOT, model_id="simplex",
            stage=GeometryPilot.STAGE, feasibility_tol=1e-9,
        )
        with pytest.raises(RoundingError):
            _run(GeometryPilot, setup, simplex_polytope, cache=cache, certificate=failed)
        assert not cache.is_cached(PILOT_CACHE_LAYER, recipe.content_key())


# ---- artifact identity ------------------------------------------------------------------------


class TestThePilotIsAFunctionOfItsKey:
    """§1.6.7's question, asked of the thing M10.2b is about to make hittable."""

    def test_the_two_stages_are_two_artifacts(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """Same transform, same schedule, same seed — different stage, so different bytes and
        therefore different keys. A shared key here would serve a geometry pilot's coordinates to a
        scale pilot asking for fluxes."""
        cache = ArtifactCache(tmp_path / "cache")
        geometry_pilot = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        scale_pilot = _run(ScalePilot, setup, simplex_polytope, cache=cache)
        assert geometry_pilot.recipe.content_key() != scale_pilot.recipe.content_key()
        assert cache.is_cached(PILOT_CACHE_LAYER, geometry_pilot.recipe.content_key())
        assert cache.is_cached(PILOT_CACHE_LAYER, scale_pilot.recipe.content_key())

    def test_each_pilot_stores_exactly_its_stage_s_array(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """The payload rule, checked on disk rather than in the docstring.

        Measured on the example model: coordinates 2.94 MB, fluxes 16.64 MB, so store-both would be
        39.2 MB per model against the split's 19.6 MB. That matters most exactly where it was meant
        to — in ``reduced`` storage mode a model's whole production output is 11.8 MB, so store-both
        pilots would be 3.3× the deliverable they calibrate.
        """
        cache = ArtifactCache(tmp_path / "cache")
        geometry_pilot = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        scale_pilot = _run(ScalePilot, setup, simplex_polytope, cache=cache)

        for pilot, expected in (
            (geometry_pilot, {"coordinates.npy", "spawn_keys.npy"}),
            (scale_pilot, {"fluxes.npy", "spawn_keys.npy"}),
        ):
            directory = cache.artifact_dir(PILOT_CACHE_LAYER, pilot.recipe.content_key())
            assert {path.name for path in directory.glob("*.npy")} == expected

    def test_a_pilot_cannot_be_loaded_as_the_other_stage(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """``pilot_kind`` is checked against the class the caller asked for and **never used to
        choose it** — a store that picks the type from the bytes it loaded lets the bytes decide
        what they are."""
        cache = ArtifactCache(tmp_path / "cache")
        geometry_pilot = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        artifact = cache.load(PILOT_CACHE_LAYER, geometry_pilot.recipe.content_key())
        with pytest.raises(CalibrationError, match="not the .* pilot it was looked up as"):
            ScalePilot.from_bundle(artifact.arrays, artifact.meta, geometry_pilot.recipe)

    def test_a_pilot_that_does_not_reproduce_its_key_is_refused(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """The store is content-addressed, so bytes that do not re-derive their key are not that
        key's artifact, whatever their arrays look like (the `ReducedGeometry.from_bundle` pattern).
        """
        cache = ArtifactCache(tmp_path / "cache")
        pilot = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        artifact = cache.load(PILOT_CACHE_LAYER, pilot.recipe.content_key())
        tampered = {**artifact.meta, "content_key": "not-the-key-it-is-stored-under"}
        with pytest.raises(CalibrationError, match="does not reproduce the key"):
            GeometryPilot.from_bundle(artifact.arrays, tampered, pilot.recipe)

    def test_a_pilot_built_from_another_recipe_is_refused(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        cache = ArtifactCache(tmp_path / "cache")
        pilot = _run(GeometryPilot, setup, simplex_polytope, cache=cache)
        artifact = cache.load(PILOT_CACHE_LAYER, pilot.recipe.content_key())
        foreign = dataclasses.replace(pilot.recipe, model_id="a-different-strain")
        with pytest.raises(CalibrationError, match="different recipe"):
            GeometryPilot.from_bundle(artifact.arrays, artifact.meta, foreign)

    def test_freezing_is_a_class_invariant_not_a_builder_convention(
        self, setup, simplex_polytope: ReducedPolytope  # type: ignore[no-untyped-def]
    ) -> None:
        """`_frozen` runs in `__post_init__`, so **every** path produces a frozen pilot.

        It used to run only in `run_*_pilot` and `from_bundle` — the two well-behaved constructors —
        so plain dataclass construction yielded a pilot whose draws a later line could mutate, and
        spec §17.4/§18.3 turns entirely on a pilot being unable to move once production starts. An
        invariant that two callers agree to uphold is a convention. (Codex, round 3.)
        """
        pilot = _run(GeometryPilot, setup, simplex_polytope)
        hand_built = GeometryPilot(
            recipe=pilot.recipe,
            spawn_keys=np.array(pilot.spawn_keys, dtype=np.int64),  # writable on the way in
            coordinates=np.array(pilot.coordinates, dtype=np.float64),  # writable on the way in
        )
        with pytest.raises(ValueError):
            hand_built.coordinates[0, 0, 0] = 1.0
        with pytest.raises(ValueError):
            hand_built.spawn_keys[0, 0] = 1

    def test_a_payload_that_is_not_the_recipe_s_shape_is_refused(
        self, setup, simplex_polytope: ReducedPolytope  # type: ignore[no-untyped-def]
    ) -> None:
        """The recipe says how many chains drew how many samples; the payload must be that array.
        `from_bundle` checked neither dimension, so a right-dtype wrong-shape bundle rebuilt happily
        and failed later, somewhere else. (Codex, round 3.)"""
        pilot = _run(GeometryPilot, setup, simplex_polytope)
        with pytest.raises(CalibrationError, match="but its recipe names"):
            dataclasses.replace(pilot, coordinates=pilot.coordinates[:, :-1, :])

    def test_a_pilot_under_another_transform_is_a_different_key(
        self, setup, simplex_polytope: ReducedPolytope  # type: ignore[no-untyped-def]
    ) -> None:
        """A pilot's coordinates are only meaningful in the frame that produced them, and a wrong
        frame is same-shape, different-geometry, every check green."""
        _, transform, _ = setup
        recipe = pilot_recipe(
            transform, simplex_polytope, config=PILOT, model_id="simplex",
            stage=GeometryPilot.STAGE, feasibility_tol=1e-9,
        )
        other = dataclasses.replace(recipe, transform_key="some-other-transform")
        assert other.content_key() != recipe.content_key()

    def test_the_lookup_key_is_computable_before_the_pilot_exists(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        """`cache.ArtifactCache` looks an artifact up by a function of its **inputs**, so the key
        cannot be a method on the built pilot — which is what `PilotRecipe` exists to fix, and why
        the recipe is carried *by* the artifact rather than recomputed beside it (two writers of one
        key is the defect M10.2a removed from the L3 bundle)."""
        _, transform, _ = setup
        predicted = pilot_recipe(
            transform, simplex_polytope, config=PILOT, model_id="simplex",
            stage=GeometryPilot.STAGE, feasibility_tol=1e-9,
        ).content_key()
        built = _run(GeometryPilot, setup, simplex_polytope, cache=ArtifactCache(tmp_path / "c"))
        assert built.recipe.content_key() == predicted


# ---- the DAG's own output must not move --------------------------------------------------------


class TestTheCacheDoesNotTouchTheNumbers:
    """M8 asserted this of execution mode and M9 confirmed it through a second instrument. The
    cache is the same claim about a different axis: a store is a performance decision, and the day
    it moves ``T₁`` or ``σ̂₀`` it has become a scientific one."""

    def test_calibrate_gives_the_same_transform_and_scale_either_way(
        self, setup, simplex_polytope: ReducedPolytope, tmp_path  # type: ignore[no-untyped-def]
    ) -> None:
        from tests.conftest import synthetic_optimum

        from gsmm_compiler.flux_polytope import FluxPolytope
        from gsmm_compiler.native_csc import NativeCSC
        from gsmm_compiler.sparse_objective import SparseFluxObjective, lower_objective

        geometry, transform, certificate = setup
        # The same reaction axis the fixture's polytope uses — M6's join guard is right that the IDs
        # *are* the coordinate system, and a test that dodged it would be testing a different model.
        full = FluxPolytope(
            reaction_ids=simplex_polytope.reaction_ids,
            metabolite_ids=simplex_polytope.metabolite_ids,
            stoichiometry=NativeCSC.from_dense(np.ones((1, 3))),
            lower_bounds=np.zeros(3),
            upper_bounds=np.ones(3),
            biomass_index=simplex_polytope.biomass_full_index,
        )
        lowered = lower_objective(
            simplex_polytope, SparseFluxObjective.from_polytope(full, l1_penalty=0.25)
        )
        sampler = dataclasses.replace(PILOT, pilot_reround=True, energy_scale="pilot_sd")

        def run(cache: ArtifactCache | None) -> tuple[str, float]:
            result = calibration.calibrate(
                geometry, simplex_polytope, transform, lowered,
                model_id="simplex",
                optimum=synthetic_optimum(lowered, 1.0, simplex_polytope.content_key()),
                sampler=sampler,
                bootstrap_certificate=certificate,
                cache=cache,
            )
            return result.transform.content_key(), result.energy_scale.value

        uncached_key, uncached_scale = run(None)
        cache = ArtifactCache(tmp_path / "cache")
        miss_key, miss_scale = run(cache)
        hit_key, hit_scale = run(cache)

        assert uncached_key == miss_key == hit_key, "the cache moved T₁"
        assert uncached_scale == miss_scale == hit_scale, "the cache moved s_J = σ̂₀"
