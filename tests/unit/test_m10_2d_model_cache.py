"""M10.2d: L0 enters the store — and a warm run stops importing the parser.

After M10.2b keyed the pilots, warm `batch.prepare_model` was **1.21 s and essentially all
`load_canonical_model`** — a stage `cache.py`'s own docstring and DEVELOPMENT_STATUS both described
as cached, and which nothing stored. M9's "the code never implemented its own documentation", a
third time.

**The arithmetic decided the scope, as it has every time.** Measured warm on the example model:

===============================  ===========  =========================
`load_canonical_model`           **1.157 s**  cached here
`.reduce()` → L1                 **0.001 s**  **derive it**
objective + LP → L2              **0.045 s**  **derive it**
===============================  ===========  =========================

§1.1 names a four-layer DAG; keying a 1 ms stage would be §1.6.7's "16.4× upside-down" mistake with
the numbers changed. **L0 only.**

**And the prize is bigger than the parse.** `load_canonical_model` is 1.157 s on the first call and
0.52 s on later ones: the gap is cobra's own **0.65 s import**, 54% of the cost. A cache that
skipped the parse but still imported cobra — to read `cobra.__version__` for its key, say — would
recover barely half of what it was built for. So the lookup key reads cobra's version from package
**metadata**, and `TestAWarmRunNeverImportsTheParser` pins the result.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.model_input import (
    MODEL_CACHE_LAYER,
    CanonicalModel,
    ModelCacheError,
    load_canonical_model,
    model_lookup_key,
)

TOY = "examples/toy_network.json"


@pytest.fixture
def cache(tmp_path: Path) -> ArtifactCache:
    return ArtifactCache(tmp_path / "cache")


# ---- the architectural claim -------------------------------------------------------------------


class TestAWarmRunNeverImportsTheParser:
    """The point of the layer, and the reason its key had to be designed for it.

    cobra is 0.65 s of `load_canonical_model`'s 1.157 s. Skipping the *parse* while still importing
    the parser would leave more than half the cost on the table — so this is not a nice property of
    the implementation, it is the requirement the lookup key was built around. In a **subprocess**,
    because a module another test already imported would make this pass for free.
    """

    def test_a_cache_hit_does_not_import_cobra(self, tmp_path: Path) -> None:
        code = (
            "import sys;"
            "from gsmm_compiler.cache import ArtifactCache;"
            "from gsmm_compiler.model_input import load_canonical_model;"
            f"cache = ArtifactCache({str(tmp_path / 'c')!r});"
            # miss: parses, and necessarily imports cobra
            f"first = load_canonical_model({TOY!r}, None, cache=cache);"
            "assert 'cobra' in sys.modules, 'a miss must parse';"
            "sys.exit(0)"
        )
        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0

        code = (
            "import sys;"
            "from gsmm_compiler.cache import ArtifactCache;"
            "from gsmm_compiler.model_input import load_canonical_model;"
            f"cache = ArtifactCache({str(tmp_path / 'c')!r});"
            # hit: must not import cobra — that is 0.65 s of the 1.157 s
            f"model = load_canonical_model({TOY!r}, None, cache=cache);"
            "assert model.polytope.n_reactions > 0;"
            "assert 'cobra' not in sys.modules, 'a cache hit imported cobra';"
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_the_lookup_key_does_not_import_cobra(self, tmp_path: Path) -> None:
        """The key is computed *before* the decision to parse, so if computing it imported cobra the
        hit path could never avoid it. `provenance._installed_version` reads package metadata."""
        code = (
            "import sys;"
            "from gsmm_compiler.model_input import model_lookup_key;"
            f"key = model_lookup_key({TOY!r}, None);"
            "assert key;"
            "assert 'cobra' not in sys.modules, 'computing the lookup key imported cobra';"
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


# ---- hit == miss --------------------------------------------------------------------------------


class TestACacheHitIsIndistinguishableFromAMiss:
    def test_the_three_paths_agree(self, cache: ArtifactCache) -> None:
        uncached = load_canonical_model(TOY, None)
        miss = load_canonical_model(TOY, None, cache=cache)
        hit = load_canonical_model(TOY, None, cache=cache)

        for other in (miss, hit):
            assert other.l0_key == uncached.l0_key
            assert other.l1_key == uncached.l1_key
            assert other.model_id == uncached.model_id
            assert other.source_sha256 == uncached.source_sha256
            assert other.source_path == uncached.source_path
            assert other.polytope.reaction_ids == uncached.polytope.reaction_ids
            assert other.polytope.metabolite_ids == uncached.polytope.metabolite_ids
            assert other.polytope.biomass_index == uncached.polytope.biomass_index
            assert np.array_equal(other.polytope.lower_bounds, uncached.polytope.lower_bounds)
            assert np.array_equal(other.polytope.upper_bounds, uncached.polytope.upper_bounds)
            assert np.array_equal(other.exchange_mask, uncached.exchange_mask)
            assert np.array_equal(
                other.polytope.stoichiometry.values, uncached.polytope.stoichiometry.values
            )

    def test_the_premise_that_the_second_call_really_was_a_hit(self, cache: ArtifactCache) -> None:
        """Without this, "the hit equals the miss" would pass against a cache that stored nothing
        and re-parsed a deterministic file twice."""
        first = load_canonical_model(TOY, None, cache=cache)
        assert cache.is_cached(MODEL_CACHE_LAYER, model_lookup_key(TOY, None))
        assert first.l0_key == load_canonical_model(TOY, None, cache=cache).l0_key

    def test_the_stored_provenance_is_kept_not_recaptured(self, cache: ArtifactCache) -> None:
        """It describes *the parse that produced these arrays* — cobra's version is even folded into
        `l0_key`. Recapturing it on load would make a hit differ from a miss and report this run's
        environment as the one that did work it did not do."""
        miss = load_canonical_model(TOY, None, cache=cache)
        hit = load_canonical_model(TOY, None, cache=cache)
        assert hit.provenance == miss.provenance


# ---- artifact identity --------------------------------------------------------------------------


class TestTheModelIsAFunctionOfItsKey:
    """§1.6.7's question, asked of the layer M10.2d makes hittable."""

    def test_the_lookup_key_covers_the_biomass_override(self) -> None:
        """It selects a different reaction, so a different IR — and the same file's bytes."""
        base = load_canonical_model(TOY, None)
        other_id = next(r for r in base.polytope.reaction_ids if r != base.polytope.biomass_id)
        assert model_lookup_key(TOY, other_id) != model_lookup_key(TOY, None)

    def test_the_lookup_key_covers_the_source_path(self, tmp_path: Path) -> None:
        """🔴 Not redundant with the file's sha256, and this is the subtle one.

        `build_canonical_model` falls back to ``source.stem`` when a model carries no ``id`` of its
        own — so **two identical files under different names are two different `model_id`s**, and
        `model_id` keys the RNG streams (`provenance.stream_seed`). A key over the bytes alone would
        serve one strain's IR under another's name. Two copies of one file are parsed twice instead:
        a false miss costs 0.5 s, a false hit corrupts.
        """
        copy = tmp_path / "a-different-name.json"
        copy.write_bytes(Path(TOY).read_bytes())
        assert model_lookup_key(copy, None) != model_lookup_key(TOY, None)

    def test_the_lookup_key_follows_the_file_s_bytes(self, tmp_path: Path) -> None:
        original = Path(TOY).read_text()
        mutated = tmp_path / Path(TOY).name
        mutated.write_text(original.replace("1000", "999", 1))
        assert Path(TOY).read_bytes() != mutated.read_bytes()
        # Same *name*, different bytes: the sha256 must separate them.
        same_name = tmp_path / "copy" / Path(TOY).name
        same_name.parent.mkdir()
        same_name.write_text(original)
        assert model_lookup_key(mutated, None) != model_lookup_key(same_name, None)

    def test_a_model_that_does_not_reproduce_its_content_key_is_refused(
        self, cache: ArtifactCache
    ) -> None:
        """The lookup key is a proxy over the file's bytes; `l0_key` is the proof over the IR.

        §1.1: "validates the loaded artifact's content L0 key on load — so a false lookup hit is
        caught, never trusted." This is also what makes it safe to put the reaction IDs in `meta`,
        which `ArtifactCache` does **not** hash: tamper with them and the re-derived key moves.
        """
        load_canonical_model(TOY, None, cache=cache)
        artifact = cache.load(MODEL_CACHE_LAYER, model_lookup_key(TOY, None))
        ids = list(artifact.meta["reaction_ids"])
        tampered = {**artifact.meta, "reaction_ids": [f"not_{name}" for name in ids]}
        with pytest.raises(ModelCacheError, match="does not reproduce its own content key"):
            CanonicalModel.from_bundle(artifact.arrays, tampered)

    def test_an_unknown_envelope_schema_is_refused(self, cache: ArtifactCache) -> None:
        load_canonical_model(TOY, None, cache=cache)
        artifact = cache.load(MODEL_CACHE_LAYER, model_lookup_key(TOY, None))
        with pytest.raises(ModelCacheError, match="envelope schema"):
            CanonicalModel.from_bundle(
                artifact.arrays, {**artifact.meta, "cache_schema_version": 999}
            )


# ---- the file-hash lie the store must not tell ---------------------------------------------------


def test_build_canonical_model_has_no_cache_parameter() -> None:
    """Structural, per this repo's rule that an absent parameter cannot be passed by mistake.

    M8's opening defect: a file hash **cannot** prove an in-memory model came from the file whose
    bytes it hashes, so a file-keyed lookup would be a lie on `build_canonical_model`'s path — it
    accepts a model that may have been assembled or mutated. `load_canonical_model` hashes and
    parses in *one call*, which is what makes the correspondence real and the store honest. The
    guard is that there is no parameter to misuse.
    """
    import inspect

    from gsmm_compiler.model_input import build_canonical_model

    assert "cache" not in inspect.signature(build_canonical_model).parameters
    assert "cache" in inspect.signature(load_canonical_model).parameters


def test_l1_and_l2_are_deliberately_not_cached() -> None:
    """The scope decision, pinned so a later reader does not "finish" the four-layer DAG.

    Measured warm: `.reduce()` **1 ms**, objective + LP **45 ms**, against `load_canonical_model`'s
    **1.157 s**. §1.1's numbered layers describe the *dependency* structure; they are not a shopping
    list of stores. Caching a 1 ms stage is §1.6.7's "16.4× upside-down" mistake with new numbers:
    **cache what is expensive, derive what is cheap, key everything.**
    """
    import gsmm_compiler.batch as batch
    import gsmm_compiler.calibration as calibration
    import gsmm_compiler.model_input as model_input

    layers = {
        model_input.MODEL_CACHE_LAYER,
        batch.GEOMETRY_CACHE_LAYER,
        calibration.PILOT_CACHE_LAYER,
    }
    assert layers == {"L0", "L3", "pilot"}, (
        "a new cache layer appeared; if it is L1 or L2, check the arithmetic first — they cost "
        "1 ms and 45 ms"
    )


def test_the_reduced_polytope_is_derived_not_stored(cache: ArtifactCache) -> None:
    """`l1_key` names the reduced IR for manifests; nothing looks it up (`.reduce()` is 1 ms)."""
    model = load_canonical_model(TOY, None, cache=cache)
    assert model.l1_key == model.polytope.content_key()
    assert not cache.is_cached("L1", model.l1_key)
    assert dataclasses.is_dataclass(model.polytope.reduce())
