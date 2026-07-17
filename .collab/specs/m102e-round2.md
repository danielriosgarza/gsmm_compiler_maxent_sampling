# M10.2e round 2 — attack the BUILT diff, not the plan

Round 1 (`.collab/specs/m102e-threads.md`) agreed a design; this is the **built** result. Read
`src/gsmm_compiler/numerics.py` (new), the diffs in `src/gsmm_compiler/batch.py`,
`affine_geometry.py`, `rounding.py`, and `tests/unit/test_m10_2e_determinism.py` (new). Do **not**
read anything under `.claude/`.

**Precedent is why you are being asked.** M10.2a's round 4 and M10.2b's round 3 each opened DISAGREE
**on the built diff** and found a defect the repair itself had introduced. Assume this diff has one.

## What was built

1. `numerics.deterministic_blas()` — a **forced** (`threadpool_limits(limits=1, user_api="blas")`),
   **scoped** context manager, applied via the `@under_deterministic_blas` decorator to exactly three
   functions: `build_geometry`, `build_transform`, `reround_transform`.
2. `DETERMINISM_POLICY_VERSION = 1` folded into `batch.geometry_cache_key`, so caches holding an
   ambient-thread basis miss.
3. `batch._numerical_identity` → `meta["numerical_identity"]` = {recipe_key, basis_hash,
   transform_hash, support_points_hash, runtime}. On a cache **hit**, `_report_numerical_identity`
   **raises** if the recorded `recipe_key` != the key it was fetched under (or the block is absent),
   and **warns** if the runtime profile differs.
4. `effective_runtime_profile()` captures the profile *inside* the policy scope, so the written and
   the compared profile are commensurable.
5. `_limit_thread_env` unchanged (still `setdefault`) — kept as the separate *performance* policy.
6. The M5 `samples` fixture: 1500 → **4000** draws.

## Measured, reproducible (Bifido, d=46, 14 cores, OpenBLAS 0.3.31 DYNAMIC_ARCH neoversev2)

| regime | support | basis | T0 |
|---|---|---|---|
| ambient (14) | `ae13c4c3c0` | `f775d464d1` | `b2cebda97b` |
| env-pinned before numpy import | `ae13c4c3c0` | `55d39f6b87` | `8e587b6ad5` |
| ambient + `deterministic_blas` | `ae13c4c3c0` | `55d39f6b87` | `8e587b6ad5` |

- **`build_transform` is thread-sensitive independently of the basis** (basis fixed: `8e587b6ad5`
  pinned vs `9d334b3f31` ambient) — round 1's framing said the basis was the sensitive artifact.
- **The sampler is not**: geometry+T0 frozen → draws bit-identical at 1 and 14 threads (`2b13baec26`).
- Pinning is **21% faster** (build_geometry 1.170 s vs 1.488 s).
- 954 tests pass at ambient **and** under `OMP_NUM_THREADS=1` (0 skipped, both). ruff + mypy strict
  clean. `maxent sample` completes from a clean cache under default threading (was the blocker).
- CLI-warmed cache (`build-geometry --cache-dir`) + `sample` cross-process: hits, no spurious warning.

## The R̂ change, which is the part I am least sure of

Pinning made `test_the_chains_mix_and_the_diagnostics_say_so` fail (R̂ 1.1654 vs a 1.15 bar). I then
measured across **8 seeds** at the fixture's 1500 draws: R̂ ∈ [1.089, 1.177], min ESS ∈ [10.2, 50.7] —
i.e. the bars sit inside the distribution of *valid* runs and seed 0 fails **both**. I concluded the
thread count was never the cause (seeds toss the same coin) and fixed the **schedule**: 4000 draws →
R̂ ∈ [1.033, 1.059], ESS ∈ [59.7, 155.0] over 5 seeds. Cost: +12 s, once, module-scoped.

## Attack

1. **Find the hole in the repair.** Where does this diff let an artifact still not be a function of
   its key? Specifically: is the *decorator* placement (`build_geometry`, `build_transform`,
   `reround_transform`) actually total? What other code path constructs or mutates a **keyed** array
   with BLAS — `to_coordinates`, `dispersed_start`, `precompute`, the `features`/`diagnostics`
   aggregation, `movable_reactions`, anything in `calibration` outside `reround_transform`?
2. **The identity gate.** `_report_numerical_identity` raises on a `recipe_key` mismatch. Is that
   reachable in a legitimate scenario (two writers, concurrent claim dir, a resumed run) where it
   would refuse a *correct* bundle? I gate a mismatch and only warn on a profile difference — is that
   split right, or does it fail open where it matters? Note `ArtifactCache` hashes arrays and
   **trusts meta**, so `numerical_identity` is unhashed trusted metadata: what can a corrupted or
   hand-edited meta do here that it could not before?
3. **Is `user_api="blas"` sufficient?** It reproduces an all-variable env pin bit-for-bit on *this*
   platform. Does it hold where OpenBLAS is built with the OpenMP threading layer, or where MKL/BLIS
   is the provider, or where NumPy dispatches through OpenMP directly? Should the scope be `limits=1`
   with no `user_api` filter?
4. **The R̂ schedule change.** Is fixing the *schedule* the honest lever, or am I hiding a slow-mixing
   model behind a longer chain — and does a bar cleared by 2.5× margin over 5 seeds still catch the
   regression it exists to catch? Would you have moved the bar, the estimator, or the schedule?
5. **`DETERMINISM_POLICY_VERSION` in the L3 key but not the pilot key.** The pilots key on the
   transform's *content*, so a changed T₀ moves them transitively. Is that transitivity real and
   complete, or does some pilot/sample key need the policy version explicitly?
6. Anything I have recorded as measured that you think is measured wrong, or concluded too widely.

End with exactly:
VERDICT: AGREE | DISAGREE
CONTESTED: <numbered list of points you dispute, or 'none'>
