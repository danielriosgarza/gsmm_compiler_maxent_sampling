# M10.2e — the L3 artifact is not a function of its key: BLAS threads pick the basis

Read `src/gsmm_compiler/affine_geometry.py` (basis construction) and `src/gsmm_compiler/batch.py`
(`_limit_thread_env`, `geometry_cache_key`, `run_batch`). Do **not** read anything under `.claude/`.
Focus: pick the fix. This is geometry, which BUILD_PLAN makes math-critical.

## Measured, reproducible, on the example model (Bifido, d=46)

Same polytope, same `model_id`, same seed, same config — **one L3 cache key `e9d6fc28673a`**:

| OMP/OPENBLAS/MKL_NUM_THREADS | support points | basis | T0 |
|---|---|---|---|
| 1 | `a0d32f1c02` | **`d35fe4fccf`** | `539da9cfc640` |
| unset | `a0d32f1c02` | **`970f8dddac`** | `d0f3bee2ce8f` |
| 4 | `a0d32f1c02` | **`970f8dddac`** | `d0f3bee2ce8f` |

Deterministic per setting; reproducible across processes. Support points come from HiGHS, which
`build_geometry` pins to `threads=1` — identical. The **basis** is NumPy/BLAS
(`residual -= basis @ (basis.T @ residual)`, Gram-Schmidt re-orthogonalization at
`affine_geometry.py:483`, also 350/991/994), and multi-threaded BLAS reduces in a different order.

**Both bases are valid.** Identical M4 span certificates (exhaustive=True, complement_complete=True,
`max_width` 1.808e-12 vs 1.804e-12, 0 inconclusive, leakage ~6e-16), same dimension 46, same
`cond(C_q)` 1.537e4, and both T0 pass `certify_reachable_mass_balance` (3.93e-11 / 4.06e-11). This is
a **reproducibility** defect, not a wrong-answer one.

**But it decides whether the package runs.** Basis `970f8dddac` yields a T1 (after the M10 pilot
re-rounding) whose `certify_reachable_mass_balance` LP returns **kUnknown**, so the run refuses:

    CLI, fresh cache, OMP_NUM_THREADS=1   -> complete (4/4 units), T1 certified 3.87e-11
    CLI, fresh cache, ambient threads     -> failed: LPNotOptimalError ... kUnknown

Present at HEAD before this milestone. It **fails closed** (M9's certificate refuses rather than
sampling something wrong), so no incorrect numbers were ever produced. Nobody noticed because every
recorded CLI verification reused a cache warmed by a threads-pinned probe.

**Agreed scope:** fix the *reproducibility*. The certify-kUnknown fragility is recorded, not chased.

## The plan already mandates a fix, and the code disagrees

BUILD_PLAN §1.2: "Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` **before** NumPy
import". `batch._limit_thread_env` does `os.environ.setdefault(...)` **inside `run_batch`**, i.e.
after numpy is imported at module scope, and its own docstring says it exists so "the
freshly-imported NumPy in each **spawned worker** inherits it". The parent's geometry build was never
covered. (This is the 4th "BUILD_PLAN settles it and the code drifted" this session.)

## The fork — which, and why?

**A. Force the env in `gsmm_compiler/__init__.py` before numpy is imported.** Literally what §1.2
says. Costs: importing a *library* mutates the user's `os.environ` process-wide; and `setdefault`
would not give determinism at all (a user with `OMP_NUM_THREADS=8` still gets basis `970f8dddac`),
so it must *force*, overriding an explicit user choice.

**B. `threadpoolctl` around the geometry build only.** Scoped, no global mutation, deterministic
regardless of the caller's env. Costs: a new runtime dependency (wheel-only is fine here), and it
must wrap every BLAS call that touches the basis.

**C. Put the effective thread count in `geometry_cache_key`.** Honest — §1.1 says a key must hash
everything that changes the bytes. Does **not** fix the CLI failure (the default path still builds
the unlucky basis) and makes caches non-shareable across machines.

**D. Make the basis construction itself thread-invariant** (e.g. deterministic reduction). Is this
achievable at acceptable cost for a 260x46 Gram-Schmidt, or is it a fantasy?

**E. Something I have not thought of.**

## Attack

1. Which option, and what does it cost that I have not priced?
2. Is pinning to 1 thread even *sufficient* for determinism, or does BLAS remain non-reproducible
   across machines/versions at fixed thread count (different SIMD kernels, different blocking)? If it
   is not sufficient, is "the L3 artifact is a function of its key" achievable at all — and if not,
   what should the key *promise*, and what should §1.1 say instead?
3. Does A/B contradict §1.2's own reasoning that thread pinning exists to prevent worker
   oversubscription (a *performance* concern) rather than to make geometry reproducible (a
   *correctness* concern)? Are these two different requirements that happen to share a mechanism —
   and does treating them as one hide something?
4. Is there a case for making the geometry's nondeterminism *visible* (e.g. hashing the basis into
   the run manifest) rather than eliminated?

End with exactly:
VERDICT: AGREE | DISAGREE
CONTESTED: <numbered list of points you dispute, or 'none'>
