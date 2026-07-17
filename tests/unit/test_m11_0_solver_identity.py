"""The L3 **lookup** key must name the solver that produces the bytes it is about to look up.

`highs_backend.BACKEND_IMPL_VERSION`'s own docstring says "Bumped when a change here can alter the
bytes of a solve. Feeds the L2/L3 cache keys (§1.1)." It feeds `ReducedGeometry.content_key`, the
**content** key — and it did not feed `batch.geometry_cache_key`, the **pre-build lookup** key.
That is the gap: the content key can only be computed from an artifact that already exists, so it
is the lookup key that decides whether to build at all. The code never implemented its own
documentation (M11.0; the repo's signature bug, found in §1.6.7, §1.6.8, M10.2d and here).

**Two distinct consequences, and only the second is silent.** `GEOMETRY_IMPL_VERSION`'s docstring
already records the mechanism for its own bump: because the lookup key does not fold in
`ReducedGeometry.content_key`, a stale bundle is found by the deficient lookup key and then dies on
a content-key/schema mismatch. §1.1's rule is that a schema change must **miss**, never error on
stale bytes — so a backend bump alone would turn a rebuild into a hard failure. The *silent* path
is the HiGHS version, which was absent from **both** identities: upgrade highspy, and a cache warmed
by the old solver is served as if the new one had produced it. A false miss only recomputes; a false
hit corrupts (§1.1).

The two tests below are written to fail on the shipped code, per M10.2c's rule: a test that cannot
fail on the broken code is not evidence, and "it passed" is exactly how it hides. Both read the
version through the **module object** rather than a from-import, so `monkeypatch.setattr` reaches
the value `geometry_cache_key` actually consults; against the unfixed key both return an unchanged
digest and the assertion fails cleanly, rather than erroring on a missing attribute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import dense_polytope

from gsmm_compiler import highs_backend, provenance
from gsmm_compiler.batch import _load_or_build_geometry, geometry_cache_key
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.config import Config
from gsmm_compiler.flux_polytope import ReducedPolytope


def _triangle() -> ReducedPolytope:
    """``v0 = v1 + v2``, all in [0, 10]. Any polytope does — the key is what is under test."""
    return dense_polytope([[1.0, -1.0, -1.0]], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])


@pytest.fixture(name="key_inputs")
def _key_inputs() -> tuple[ReducedPolytope, Config]:
    return _triangle(), Config()


def test_the_l3_lookup_key_moves_when_the_backend_version_moves(
    key_inputs: tuple[ReducedPolytope, Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solver change that can alter a solve's bytes must **miss** the cache, not reuse it.

    Fails on the shipped code: `geometry_cache_key` never consulted `BACKEND_IMPL_VERSION`, so the
    two digests below were equal and a backend bump reached the lookup as a hit.
    """
    reduced, config = key_inputs
    before = geometry_cache_key(reduced, config, model_id="strain_1")

    bumped = highs_backend.BACKEND_IMPL_VERSION + 1
    monkeypatch.setattr(highs_backend, "BACKEND_IMPL_VERSION", bumped)
    after = geometry_cache_key(reduced, config, model_id="strain_1")

    assert before != after, (
        "the L3 lookup key ignores BACKEND_IMPL_VERSION, so a solver change that its own docstring "
        "says 'can alter the bytes of a solve' is served from a cache built by the old solver"
    )


def test_the_l3_lookup_key_moves_when_the_highs_version_moves(
    key_inputs: tuple[ReducedPolytope, Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent path: highspy's version was in **neither** identity.

    `BACKEND_IMPL_VERSION` is ours and we can bump it; the solver's own version is not, and nothing
    bumps on a `uv sync`. The geometry's support points are LP outputs, so a different HiGHS can
    move them under a fixed key with nothing to notice.
    """
    reduced, config = key_inputs
    before = geometry_cache_key(reduced, config, model_id="strain_1")

    monkeypatch.setattr(
        provenance, "_installed_version", lambda dist: "0.0.0-not-the-installed-highs"
    )
    after = geometry_cache_key(reduced, config, model_id="strain_1")

    assert before != after, (
        "the L3 lookup key ignores the installed HiGHS version, so upgrading highspy serves a "
        "cache warmed by the previous solver — a false hit, which corrupts (§1.1)"
    )


def test_the_lookup_key_does_not_import_the_solver_it_names() -> None:
    """The boundary that makes the fix the right shape — and it must run in a **subprocess**.

    §1.1: "The basis hash belongs to content identity and manifests, never to the pre-build lookup
    key: hashing the artifact to decide whether to build the artifact is circular." So the lookup
    key names the solver by a **version**, knowable before the solve, never by its output. The
    version therefore has to come from `importlib.metadata` and not from ``highspy.__version__`` —
    otherwise merely asking "should I build?" would drag the solver in, and the M10.2d property
    that a warm run stays solver-free would be spent on the key that exists to avoid the work.

    In a subprocess, for the reason M10.2d's own cobra test states: "a module another test already
    imported would make this pass for free." The in-process version of this test passed alone and
    failed in the suite — it was asserting global interpreter state, not anything about the key.
    """
    code = (
        "import sys;"
        "from gsmm_compiler.batch import geometry_cache_key;"
        "from gsmm_compiler.config import Config;"
        "from tests.conftest import dense_polytope;"
        "reduced = dense_polytope([[1.0, -1.0, -1.0]], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);"
        "key = geometry_cache_key(reduced, Config(), model_id='strain_1');"
        "assert isinstance(key, str) and key;"
        "assert 'highspy' not in sys.modules, 'computing the L3 lookup key imported the solver';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_a_backend_bump_rebuilds_rather_than_dying_on_stale_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The behaviour §1.1 actually requires, end to end — not just "the digest moved".

    "A schema change must **miss**, never error on stale bytes" (`GEOMETRY_IMPL_VERSION`'s
    docstring). The two key tests above prove the *name* changes; this one proves what the name is
    for, through `_load_or_build_geometry` — the function the batch and the CLI both go through.

    Non-vacuous on the shipped code twice over: the `key_before != key_after` assertion fails,
    and had it not, the warm entry would have been served under its old name and then died in
    `ReducedGeometry.from_bundle` on the content key — a hard failure where §1.1 wants a rebuild.
    """
    reduced, config = _triangle(), Config()
    cache = ArtifactCache(tmp_path / "cache")

    _, _, warm_meta = _load_or_build_geometry(reduced, config, model_id="strain_1", cache=cache)
    key_before = geometry_cache_key(reduced, config, model_id="strain_1")
    assert warm_meta["numerical_identity"]["recipe_key"] == key_before

    monkeypatch.setattr(
        highs_backend, "BACKEND_IMPL_VERSION", highs_backend.BACKEND_IMPL_VERSION + 1
    )
    key_after = geometry_cache_key(reduced, config, model_id="strain_1")
    assert key_before != key_after, "a backend bump must rename the artifact it can change"

    # The cache still holds the entry under `key_before`. This must rebuild, not raise.
    _, _, rebuilt_meta = _load_or_build_geometry(reduced, config, model_id="strain_1", cache=cache)
    assert rebuilt_meta["numerical_identity"]["recipe_key"] == key_after, (
        "the bumped backend was served the geometry built by the previous one"
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]
"""The subprocess above imports ``tests.conftest``, so it needs the repo root on its path."""
