"""The numerical-runtime determinism policy: what the L3 key can honestly promise (§1.1, §1.2).

**Two requirements share one mechanism, and they are still two requirements.** §1.2 pins
BLAS/OpenMP threads to 1 in the sampling workers — a *performance* policy, guarding against
oversubscription when 14 solver-free workers each start a nested thread pool.
`batch._limit_thread_env` implements it, before the spawn pool starts, and it works. What nothing
ever asked for is the
*correctness* requirement discovered in M10.2e: the geometry build, which runs in the **parent** at
whatever thread count the ambient environment happens to have, must be reproducible. Conflating the
two is why the second went unstated for ten milestones — so this module is the second one, named
separately, and `_limit_thread_env` keeps the first.

**The defect it closes.** §1.1 requires that an artifact be a function of its key. It was not:
multi-threaded OpenBLAS reduces in a different order, so the ambient thread count silently selected
between two bases under **one** L3 key. Measured on the example model (Bifido, d = 46), and the
support points are identical throughout because HiGHS is already pinned to ``threads=1`` — the
sensitivity is entirely NumPy's:

===================================  ===============  ===============
regime                               basis            ``T₀``
===================================  ===============  ===============
ambient (14 threads)                 ``f775d464d1``   ``b2cebda97b``
env-pinned before the NumPy import   ``55d39f6b87``   ``8e587b6ad5``
ambient + `deterministic_blas`       ``55d39f6b87``   ``8e587b6ad5``
===================================  ===============  ===============

The third row is why this module exists rather than an import-time mutation of ``os.environ``: a
**runtime** limit reproduces the env-pinned bytes *exactly*, without a library reaching into its
caller's environment, and it holds regardless of what the caller exported. The two bases differ by
2.7e-15 and both certify — this is a reproducibility defect, not a wrong-answer one — but the pilot
amplifies that to O(1) draws, so ``T₁``, ``s_J`` and the whole β axis inherit it.

**The scope is measured, not assumed, in both directions.** `build_transform` is sensitive
*independently of the basis*: hold the basis fixed at ``55d39f6b87`` and ``T₀`` still moves
(``8e587b6ad5`` pinned vs ``9d334b3f31`` ambient), so scoping the basis alone would have left half
the defect in place. The sampler is **not**: hold the geometry and ``T₀`` fixed and 200 draws × 2
chains are bit-identical at 1 thread and at 14 (``2b13baec26``). Its inner loop is chord arithmetic
on short vectors, below any BLAS threading threshold. So the policy covers exactly the three
constructors of thread-sensitive keyed artifacts — `build_geometry`, `build_transform`,
`reround_transform` — and the chains inherit determinism from their inputs rather than from a scope
of their own.

**What the key may promise, and what it may not.** ``L3_BLAS_THREADS`` is folded into the L3 key
through `DETERMINISM_POLICY_VERSION`, so caches holding an ambient-thread basis miss rather than
serve bytes this policy would not reproduce. But one thread does **not** buy cross-machine byte
identity: this NumPy ships OpenBLAS ``0.3.31 DYNAMIC_ARCH``, which selects kernels at *runtime* by
CPU detection (``architecture`` below reads ``neoversev2`` here), so another CPU can differ in the
last bit at the same thread count. **Within a declared numerical-runtime profile** a recipe key
rebuilds deterministically; **across profiles byte equality is not promised** — unrestricted
cross-machine cache sharing and strict byte identity cannot both be had from ordinary
floating-point libraries. `numerical_runtime_profile` records the profile so that a violation is
*visible* rather than inferred: it is what makes "one recipe key produced two content keys" a thing
a user can read off two manifests, instead of a thing discovered when a certificate fails closed ten
milestones later.

Implemented in **M10.2e** — see BUILD_PLAN.md §1.1, §1.2, §1.6.8.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final, ParamSpec, TypeVar

import threadpoolctl

_P = ParamSpec("_P")
_R = TypeVar("_R")

DETERMINISM_POLICY_VERSION: Final = 1
"""Bump when this policy changes what bytes an L3 build produces — it is folded into the L3 key.

Load-bearing rather than bookkeeping, and M10.2e is itself the demonstration: a cache written
before this module existed holds an **ambient-thread** basis, which a build under this policy will
not reproduce. Without the bump those entries stay valid hits and the fix reaches nothing that was
already warmed.
"""

L3_BLAS_THREADS: Final = 1
"""The forced BLAS thread count for L3 construction — a *reproducibility* choice, not a tuning one.

One thread is the only count reproducible **across machines with different core counts**, which is
what a shared cache key implicitly claims.

**And it costs nothing — it pays.** Measured on the example model rather than assumed acceptable:
`build_geometry` is **1.170 s pinned vs 1.488 s at 14 threads (0.79×, 21% faster)**, and L3 as a
whole is 0.317 s *cheaper* under this policy. A 260×46 Gram-Schmidt is far too small for 14 threads
to repay their dispatch overhead, so the nondeterminism bought nothing and charged 0.3 s for it.
Threads for *performance* live in `batch._limit_thread_env` — a different policy for a different
reason, and the reason they are two things (§1.6.8).
"""


@contextmanager
def deterministic_blas() -> Iterator[None]:
    """Force BLAS to `L3_BLAS_THREADS` for the duration — the scoped, forced reproducibility policy.

    **Scoped**, because a library that mutates its caller's ``os.environ`` at import time to get
    this would take the whole process hostage for one 1.2 s build, and would not even work: BLAS
    reads those variables when it loads, so a `setdefault` after NumPy is imported changes nothing,
    and a caller who exported ``OMP_NUM_THREADS=4`` would still get the ambient basis.

    **Forced**, not defaulted, and that is the deliberate part: this overrides an explicit user
    thread choice inside its scope. `batch._limit_thread_env`'s ``setdefault`` is right for a
    performance policy — a user who asks for 4 threads may have them — and wrong for this one, where
    honouring the request would mean the key describes bytes the caller's environment chose.

    ``user_api="blas"`` is the *measured* minimal scope, not a guess: it reproduces an
    all-variables env-pin (``OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1``) bit-for-bit
    on this platform. Nesting is safe and cheap — threadpoolctl restores the prior limits on exit,
    so `build_l3_bundle`'s three scoped constructors do not fight each other.
    """
    with threadpoolctl.threadpool_limits(limits=L3_BLAS_THREADS, user_api="blas"):
        yield


def under_deterministic_blas(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Declare that a function constructs a keyed artifact and must do so reproducibly.

    A decorator rather than a `with` block inside each body, for the reason M10.2a learned about
    gates: *a guard is only as total as its weakest entrance.* The scope belongs to the
    **function**, so no later edit can add a BLAS call below it, and no caller reaches the
    primitive around it.
    That last part is the whole point — `build_l3_bundle` is L3's one writer, but the M5 test
    fixtures call `build_geometry` and `build_transform` **directly**, and a policy scoped to the
    orchestrator would have left those on the ambient basis while claiming the geometry was
    deterministic. Gating an orchestrator does not gate its primitive.
    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with deterministic_blas():
            return function(*args, **kwargs)

    return wrapper


def numerical_runtime_profile() -> dict[str, Any]:
    """The runtime facts that decide the last bit — recorded so a divergence is *visible*.

    Visibility **beside** elimination, not instead of it. `deterministic_blas` removes the thread
    count as a variable; this records what is left, and what is left is real: OpenBLAS's
    ``DYNAMIC_ARCH`` picks its kernel by CPU detection at load time, so ``architecture`` and
    ``version`` can move under a fixed thread count and take the last bit with them.

    Written into the L3 meta and the geometry manifest, where it answers the question this
    package could not answer for ten milestones: *did these two artifacts, sharing a recipe key,
    come from the same numerical runtime?* ``filepath`` is deliberately dropped — it is an absolute
    path into somebody's home directory, which is provenance noise, not a numerical fact.

    This is a **manifest**, not a gate. The basis hash beside it is content identity; neither may
    enter the pre-build lookup key, which would be circular — you cannot hash the artifact to
    decide whether to build the artifact.
    """
    pools = [
        {
            "user_api": pool.get("user_api"),
            "internal_api": pool.get("internal_api"),
            "version": pool.get("version"),
            "threading_layer": pool.get("threading_layer"),
            "architecture": pool.get("architecture"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in threadpoolctl.threadpool_info()
    ]
    return {
        "determinism_policy_version": DETERMINISM_POLICY_VERSION,
        "l3_blas_threads": L3_BLAS_THREADS,
        "threadpoolctl_version": threadpoolctl.__version__,
        "blas_pools": pools,
    }


def effective_runtime_profile() -> dict[str, Any]:
    """`numerical_runtime_profile` as the L3 constructors see it — under the policy's own scope.

    The scope belongs to this function rather than to its callers, and that is not tidiness: read
    *outside* `deterministic_blas`, ``num_threads`` reports the ambient count, so a profile written
    at build time and a profile read back for comparison would disagree on every single cache hit —
    a warning that fires always, which is a warning nobody reads. Two call sites agreeing to
    remember the scope is a convention; one function that cannot be called without it is a property.
    """
    with deterministic_blas():
        return numerical_runtime_profile()
