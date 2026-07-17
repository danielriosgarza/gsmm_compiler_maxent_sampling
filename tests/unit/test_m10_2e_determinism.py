"""M10.2e: the L3 artifact was not a function of its key — ambient threads picked the basis.

For ten milestones one L3 key named two different bases. Multi-threaded OpenBLAS reduces in a
different order, so the Gram-Schmidt re-orthogonalization landed 2.7e-15 apart depending on an
environment variable nobody had set, and §1.1's rule — *an artifact must be a function of its key* —
was violated in v1's own geometry. **It failed closed**: the two bases differ by a few ULPs, both
certify, and neither is wrong. But the pilot amplifies that gap to O(1) draws, so ``T₁``, ``s_J``
and the whole β axis inherited it, and one of the two bases produced a ``T₁`` whose certificate LP
returned ``kUnknown`` — which is how the CLI came to fail from a clean cache under default
threading, three layers from the cause.

Nobody noticed because every recorded CLI verification reused a cache warmed by a threads-pinned
probe.

**What is proved here, in the order the defect has to be closed:**

* the basis and ``T₀`` are the *same bytes* whatever the ambient BLAS thread count is — and the
  test proves it is not vacuous by reproducing the defect through ``__wrapped__`` first;
* the policy is **forced**, not defaulted: a caller who exports 8 threads still gets the artifact
  the key names;
* the policy is **scoped**: importing this library does not take the caller's process hostage;
* the L3 key **moves** with the policy version, because a cache warmed before M10.2e holds an
  ambient-thread basis that this policy will not reproduce, and without the bump those entries stay
  valid hits;
* a bundle records what built it, and a hit checks that account.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import threadpoolctl

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.batch import (
    BatchError,
    _report_numerical_identity,
    build_l3_bundle,
    geometry_cache_key,
)
from gsmm_compiler.config import Config
from gsmm_compiler.numerics import (
    DETERMINISM_POLICY_VERSION,
    L3_BLAS_THREADS,
    deterministic_blas,
    effective_runtime_profile,
    numerical_runtime_profile,
)
from gsmm_compiler.provenance import hash_array
from gsmm_compiler.rounding import build_transform


def _ambient_blas_threads() -> int:
    """The thread count BLAS is using right now — an observation, not a policy."""
    pools = threadpoolctl.threadpool_info()
    return max((int(pool["num_threads"]) for pool in pools), default=1)


def _many_threads() -> int:
    """The 'lots of threads' side of every comparison here — the machine's cores, **not** the
    ambient count.

    Deliberate, and the difference is the whole grip of this file. Keyed off the ambient count,
    every test below would **skip** under `OMP_NUM_THREADS=1` — an environment a careful user or CI
    is *especially* likely to set — because there would be no second thread to compare against. Four
    silently-skipped tests in a green suite read exactly like four passing ones, which is a small
    copy of the disease M10.2e is about. `threadpoolctl` can raise the limit *above* an env pin
    (measured: pinned to 1, raised to 14), so the comparison is available in either environment and
    these tests are asked to prove themselves in both.
    """
    return os.cpu_count() or 1


@pytest.fixture(scope="module")
def reduced(example_canonical):  # type: ignore[no-untyped-def]
    return example_canonical.polytope.reduce()


class TestTheArtifactIsAFunctionOfItsKey:
    """The defect itself: one key, two bases, chosen by an environment variable."""

    @pytest.mark.slow
    def test_the_basis_and_T0_are_the_same_bytes_at_any_ambient_thread_count(self, reduced) -> None:
        """The regression test for M10.2e, and it **proves it is not vacuous before it passes.**

        A machine whose BLAS happens to reduce identically at every thread count would let a broken
        policy pass this silently — the failure mode of every "it's deterministic now" test. So the
        undecorated constructor runs first, through `functools.wraps`'s ``__wrapped__``: if the
        defect does not reproduce *here*, on *this* CPU, there is nothing to protect against and
        the test says so instead of claiming a pass it did not earn.
        """
        many = _many_threads()
        if many < 2:
            pytest.skip(f"this machine has {many} core; the defect needs at least 2 to appear")

        protected = build_geometry(reduced, model_id="bifido")

        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"):
            unprotected = build_geometry.__wrapped__(reduced, model_id="bifido")
            still_protected = build_geometry(reduced, model_id="bifido")

        if hash_array(unprotected.basis) == hash_array(protected.basis):
            pytest.skip(
                f"this BLAS reduces identically at 1 and {many} threads, so the M10.2e defect "
                "does not reproduce on this CPU and this test would prove nothing"
            )

        # The defect is live on this machine (above), and the policy is what closes it.
        assert hash_array(still_protected.basis) == hash_array(protected.basis)
        assert np.array_equal(still_protected.basis, protected.basis)

        # `build_transform` is sensitive *independently* of the basis — measured, not assumed: hold
        # the basis fixed and T₀ still moves. A policy scoped to the basis alone would pass the
        # assertions above and leave half the defect in place.
        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"):
            t0_threaded = build_transform(protected, reduced)
            t0_unprotected = build_transform.__wrapped__(protected, reduced)
        t0_pinned = build_transform(protected, reduced)

        assert hash_array(t0_threaded.transform) == hash_array(t0_pinned.transform)
        assert hash_array(t0_unprotected.transform) != hash_array(t0_pinned.transform), (
            "the rounding is no longer thread-sensitive on this CPU — if that is real, "
            "`build_transform`'s scope could be dropped; more likely this test lost its grip"
        )

    @pytest.mark.slow
    def test_the_support_points_were_never_the_problem(self, reduced) -> None:
        """HiGHS is already pinned to ``threads=1``, so the LPs were always reproducible.

        Worth pinning because it is what localises the defect to NumPy: the support points enter the
        basis, and if *they* moved, `deterministic_blas` would be treating a symptom.
        """
        many = _many_threads()
        if many < 2:
            pytest.skip("needs more than one core to be meaningful")

        # One thread (by policy) vs many (unprotected) — a real comparison. Comparing two
        # *unprotected* builds would run at whatever the ambient count is on both sides and assert
        # nothing at all.
        pinned = build_geometry(reduced, model_id="bifido")
        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"):
            threaded = build_geometry.__wrapped__(reduced, model_id="bifido")

        assert hash_array(threaded.support_points) == hash_array(pinned.support_points)
        assert hash_array(threaded.basis) != hash_array(pinned.basis), (
            "premise: the bases must differ here, or this test is not observing the defect it "
            "exists to localise away from HiGHS"
        )


class TestThePolicyIsForcedAndScoped:
    def test_it_overrides_a_caller_who_asked_for_more_threads(self) -> None:
        """**Forced**, not defaulted — the difference between the two thread policies.

        `batch._limit_thread_env` uses ``setdefault`` and is right to: it is a performance hint, and
        a user who exports ``OMP_NUM_THREADS=4`` may have them. Reproducibility of a keyed artifact
        is not a hint, so honouring the request here would mean the key describes bytes the
        caller's environment chose.
        """
        many = _many_threads()
        if many < 2:
            pytest.skip("needs more than one core to be meaningful")

        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"), deterministic_blas():
            inside = _ambient_blas_threads()

        assert inside == L3_BLAS_THREADS

    def test_it_restores_the_callers_threads_afterwards(self) -> None:
        """**Scoped**: a library that pins BLAS process-wide to protect its own 1.2 s build would
        take the caller's whole program hostage. That is why this is not an ``os.environ`` mutation
        at import time — which would not work anyway, since BLAS reads those when it loads."""
        before = _ambient_blas_threads()

        with deterministic_blas():
            pass

        assert _ambient_blas_threads() == before

    def test_nesting_is_safe(self) -> None:
        """`build_l3_bundle` calls two separately-scoped constructors; they must not fight."""
        with deterministic_blas():
            with deterministic_blas():
                assert _ambient_blas_threads() == L3_BLAS_THREADS
            assert _ambient_blas_threads() == L3_BLAS_THREADS


class TestTheRuntimeProfileIsRecordedHonestly:
    def test_the_effective_profile_reports_the_threads_the_build_actually_used(self) -> None:
        """A manifest describing work that did not happen is this package's signature bug (M10.2b).

        Read outside the policy's scope, the profile reports the *ambient* count — describing a
        build that never ran. `effective_runtime_profile` owns the scope so the two call sites that
        compare profiles cannot disagree; if they could, the mismatch warning would fire on every
        cache hit and teach the user to ignore it.
        """
        profile = effective_runtime_profile()

        assert profile["determinism_policy_version"] == DETERMINISM_POLICY_VERSION
        assert profile["l3_blas_threads"] == L3_BLAS_THREADS
        for pool in profile["blas_pools"]:
            assert pool["num_threads"] == L3_BLAS_THREADS

    def test_it_records_what_decides_the_last_bit(self) -> None:
        """One thread does not buy cross-machine byte identity — OpenBLAS ``DYNAMIC_ARCH`` selects
        kernels by runtime CPU detection. What cannot be eliminated is recorded instead."""
        with deterministic_blas():
            profile = numerical_runtime_profile()

        pools = profile["blas_pools"]
        if not pools:
            pytest.skip("no BLAS pool detected in this build")
        for pool in pools:
            assert set(pool) == {
                "user_api",
                "internal_api",
                "version",
                "threading_layer",
                "architecture",
                "num_threads",
            }
        assert "filepath" not in str(pools), "a home-directory path is not a numerical fact"

    def test_the_two_profiles_a_cache_hit_compares_are_equal_on_one_machine(self) -> None:
        """The false-alarm guard. If these disagreed, every hit would warn."""
        assert effective_runtime_profile() == effective_runtime_profile()


class TestTheKeyMovesWithThePolicy:
    def test_the_policy_version_is_in_the_L3_key(self, reduced, monkeypatch) -> None:
        """Without this the fix reaches nothing that was already warmed.

        A cache written before M10.2e holds an **ambient-thread** basis, which a build under this
        policy will not reproduce. If the key did not move, those entries would stay valid hits —
        and §1.1's asymmetry is the whole point: a false miss only recomputes, a false hit corrupts.
        """
        config = Config()
        before = geometry_cache_key(reduced, config, model_id="bifido")

        monkeypatch.setattr(
            "gsmm_compiler.batch.DETERMINISM_POLICY_VERSION", DETERMINISM_POLICY_VERSION + 1
        )
        after = geometry_cache_key(reduced, config, model_id="bifido")

        assert before != after


class TestTheBundleAccountsForItself:
    @pytest.mark.slow
    def test_it_records_the_key_it_answers_to_and_what_built_it(self, reduced) -> None:
        config = Config()
        _, meta = build_l3_bundle(reduced, config, model_id="bifido")
        identity = meta["numerical_identity"]

        assert identity["recipe_key"] == geometry_cache_key(reduced, config, model_id="bifido")
        assert identity["runtime"] == effective_runtime_profile()
        assert len(identity["basis_hash"]) == 64
        assert len(identity["transform_hash"]) == 64

    def test_a_bundle_found_under_the_wrong_key_is_refused(self) -> None:
        """A bundle must answer to the key it is stored under. Two writers disagreeing about a
        schema is M10.2a's defect; a damaged store is the other reading. Neither is sampled from."""
        meta = {
            "numerical_identity": {
                "recipe_key": "a-key-this-bundle-was-not-fetched-under",
                "basis_hash": "0" * 64,
                "runtime": effective_runtime_profile(),
            }
        }

        with pytest.raises(BatchError, match="says it was built for"):
            _report_numerical_identity(meta, key="the-key-it-was-fetched-under")

    def test_a_bundle_with_no_account_of_itself_is_refused(self) -> None:
        with pytest.raises(BatchError, match="carries no `numerical_identity`"):
            _report_numerical_identity({}, key="some-key")

    def test_a_foreign_runtime_warns_rather_than_refuses(self, caplog) -> None:
        """**Visibility beside elimination, not instead of it**, and deliberately not a gate.

        A bundle built under a different BLAS is not wrong — it is what an honest cross-machine
        cache looks like, and refusing it would make caches unshareable, which is exactly the cost
        that ruled out putting the thread count in the key. But it is the condition under which this
        key's promise thins from *these bytes* to *these bytes, on the profile that wrote them*, so
        the user hears it. Had this line existed, M10.2e's defect would have read "one recipe key,
        two contents" on the second run.
        """
        local = effective_runtime_profile()
        foreign = dict(
            local,
            blas_pools=[
                dict(pool, architecture="a-different-cpu") for pool in local["blas_pools"]
            ],
        )
        meta = {
            "numerical_identity": {
                "recipe_key": "k",
                "basis_hash": "b" * 64,
                "runtime": foreign,
            }
        }

        with caplog.at_level("WARNING"):
            _report_numerical_identity(meta, key="k")  # returns; does not raise

        assert "a different numerical runtime" in caplog.text
        assert "a-different-cpu" in caplog.text


class TestTheSamplerNeedsNoScopeOfItsOwn:
    """The scope is bounded by measurement in **both** directions, which is why it stops here."""

    @pytest.mark.slow
    def test_the_chain_is_thread_invariant_given_its_inputs(self, reduced) -> None:
        """Measured: hold the geometry and ``T₀`` fixed and the draws are bit-identical at 1 thread
        and at 14. The inner loop is chord arithmetic on short vectors, below any BLAS threading
        threshold — so the chains inherit determinism from their inputs, and wrapping them would be
        cost without a property. `numerics` claims this; this test is why it may.

        **``fluxes`` is asserted beside ``coordinates``, and that is the review's finding, not a
        flourish.** The M10.2e round-2 review went looking for the hole in the repair and proposed
        exactly this one: the scopes cover ``T₀``/``T₁`` *construction* but not the **keyed chains
        that consume them**, and `RoundedTransform.to_fluxes` is a matrix product that produces the
        pilot and production sample bytes. It is the right place to look — and the answer is that it
        is invariant. But the original version of this test hashed only ``coordinates``, so it was
        defending a claim broader than the one it checked. The `to_fluxes` path now has to hold too.
        """
        from gsmm_compiler.config import SamplerConfig
        from gsmm_compiler.maxent_sampler import run_chains

        many = _many_threads()
        if many < 2:
            pytest.skip("needs more than one core to be meaningful")

        geometry = build_geometry(reduced, model_id="bifido")
        transform = build_transform(geometry, reduced)
        config = SamplerConfig(n_chains=2, n_samples=150, burn_in=150, refresh_interval=100)

        pinned = run_chains(transform, reduced, config=config, model_id="bifido", beta=0.0)
        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"):
            threaded = run_chains(transform, reduced, config=config, model_id="bifido", beta=0.0)

        assert hash_array(threaded.coordinates) == hash_array(pinned.coordinates)
        assert hash_array(threaded.fluxes) == hash_array(pinned.fluxes)

    @pytest.mark.slow
    def test_the_certificate_is_thread_invariant_although_it_sits_outside_the_scope(
        self, reduced
    ) -> None:
        """The other place the round-2 review went looking, and it is a fair question.

        `certify_reachable_mass_balance` runs in `build_l3_bundle` **after** `build_transform`'s
        decorator has exited, and its numbers are stored in ``meta["reachability_certificate"]`` —
        so they are part of the L3 bundle's bytes, under the L3 key. If they moved with the thread
        count, the artifact would still not be a function of its key and this milestone would have
        missed its own target.

        They do not: the work is 334 HiGHS LPs (already pinned to ``threads=1``, §1.2) and a `pinv`
        on a matrix far too small to thread. **Pinned by a test rather than left to a comment**,
        because it is the premise on which the certificate is left unwrapped — and a premise nobody
        checks is how this defect lasted ten milestones.
        """
        from gsmm_compiler.rounding import certify_reachable_mass_balance

        many = _many_threads()
        if many < 2:
            pytest.skip("needs more than one core to be meaningful")

        geometry = build_geometry(reduced, model_id="bifido")
        transform = build_transform(geometry, reduced)

        pinned = certify_reachable_mass_balance(transform, reduced)
        with threadpoolctl.threadpool_limits(limits=many, user_api="blas"):
            threaded = certify_reachable_mass_balance(transform, reduced)

        assert threaded.worst_absolute == pinned.worst_absolute
        assert threaded.worst_row_id == pinned.worst_row_id

        # Everything the certificate *asserts*, exactly — but not `elapsed_seconds`, and the
        # exclusion is a finding rather than a convenience. `to_cache()` goes into the L3 meta and
        # carries a **wall-clock** field, so two builds under one key write different bytes no
        # matter how deterministic the arithmetic is. **"An artifact is a function of its key" is
        # therefore a claim about its numbers, not about every byte of its manifest** — §1.1 means
        # the former, and a test asserting the latter fails on a stopwatch and teaches nothing.
        volatile = {"elapsed_seconds"}
        assert {k: v for k, v in threaded.to_cache().items() if k not in volatile} == {
            k: v for k, v in pinned.to_cache().items() if k not in volatile
        }


class TestTheTwoThreadPoliciesStaySeparate:
    def test_the_worker_policy_is_a_default_and_the_L3_policy_is_forced(self, monkeypatch) -> None:
        """Two requirements that share a mechanism are still two requirements — the M10.2e lesson,
        and the reason this is a test rather than a comment. `_limit_thread_env` is about worker
        oversubscription and yields to an explicit user choice; `deterministic_blas` is about a
        keyed artifact and does not. A later edit that "unifies" them breaks one of the two.
        """
        from gsmm_compiler.batch import _limit_thread_env

        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        _limit_thread_env()

        import os

        assert os.environ["OMP_NUM_THREADS"] == "4", "the performance policy yields to the user"

        if _many_threads() >= 2:
            with deterministic_blas():
                assert _ambient_blas_threads() == 1, "the determinism policy does not"
