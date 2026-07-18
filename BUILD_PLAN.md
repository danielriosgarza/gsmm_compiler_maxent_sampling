# GSMM-Compiler MaxEnt Sampler — Build Plan

Companion to `GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md`. The spec fixes the
mathematics and module layout. This plan resolves the cross-cutting engineering concerns the
spec leaves open — **speed, cached/restartable outputs, parallelization, scientific accuracy** —
and sequences the work into gated milestones. Produced by a Claude × Codex design collaboration
(2 rounds, converged); the decision log is in `.collab/specs/collab-outcome.md`.

---

## 0. What the example model tells us

`models/GCF_000010425_1_ASM1042v1_..._noO2.json`: 773 reactions, 894 metabolites, biomass
`bio1`, all bounds finite (±1000 or fixed), **513 fixed reactions (l==u, all at zero)**, 65 boundary
reactions (63 `EX_` exchanges + 2 `SK_` sinks — M0 corrected an earlier "65 exchanges").

Measured geometry facts that reframe every performance decision:

| Quantity | Value | Consequence |
|---|---|---|
| Free reactions (l<u) | 260 | inner-loop `n_active` ≤ 260, not 773 |
| rank(S over free cols) | 205 | equality constraints bind hard |
| **Affine sampling dimension d** | **≤ 55** | geometry + MCMC are *tiny* here |
| Geometry `B`, `T` memory | ~0.34 MB | memory guards matter only for future GSMMs |
| Geometry LP count | ~55–205 | sequential warm-started simplex is plenty fast |

**Implication:** for this model the whole pipeline is small. The caching/parallelism/batched-LP
machinery earns its keep on *larger* genome-scale models and on *batch* runs across many strains.
Build correctness-first at this scale, then let benchmarks (not speculation) drive the scaling work in M9.

### Locked scope decisions

- **Batch-aware from v1.** The CLI ingests a *models manifest* (one row per strain: model path,
  biomass reaction, optional per-model overrides — mirrors `metabolicSubcommunities/metadata/strains.tsv`)
  and produces per-model results **plus cross-model summaries**. One shared worker pool spans all
  `(model, β, chain)` units; geometry is computed and cached per model. See §1.1–§1.2, M8.
- **Reweighted-L1 is in v1** as **M7**, right after the positive-β sampler (M6), with weights frozen
  before production sampling — never a v2 afterthought.
- **Sample storage is configurable** (full-flux float64 default / float32 / reduced-state+summaries).
  See §1.3.
- **Python 3.11**, matching the proven-working `metabolicSubcommunities/.venv` (cobra 0.31.1) on this Jetson.

---

## 1. Cross-cutting decisions (the part that goes beyond the spec)

### 1.1 Caching — a 4-layer content-addressed DAG

Each layer is an immutable artifact keyed by the hash of everything upstream that can change its
bytes. Geometry is β-independent, so it is computed **once** and reused across the whole β-ladder.

```
source file ──sha256──┐
                      ▼
[L0] Parsed model IR      key = content_key(model_id, polytope content, exchange mask,
                                            cobra_version, parser_schema_version)   (M8: content-addressed)
   frozen IDs, coeffs, bounds, metadata (raw file hash alone is NOT enough — parser semantics matter)
                      ▼
[L1] Reduced polytope IR  key = hash(canonical IDs, CSC arrays, bounds, user overrides,
                                     fixed-var elimination, dtype/endianness, schema_version)
   canonical S (CSC) + full→reduced map  v_full = R·v_reduced + c   (see §1.5)
        ├────────────────────────────────────────────────┐
        ▼                                                 ▼
[L2] Objective + LP optimum                         [L3] Geometry
   key = L1 + biomass_idx + λ + penalty_indices          key = L1 + scaling + tolerances
       + weights + obj_impl_version + energy_policy           + geom_seed + discovery_algo_version
   holds J*, v*, μ*, C*, and s_J                              + numpy_version + solver_settings
   (s_J is objective-dependent → lives HERE, not L3)     holds B, support_points, center,
                                                          L (Cholesky), T, dimension, span certificate
        └───────────────────────┬────────────────────────┘
                                ▼
   Samples: one artifact per (β, chain) unit
   key = L2 + L3 + β + chain_seed_coords + sampler_version + burn/thin/n_samples
        (M10.2: `batch.sample_recipe_key` — written into every unit's manifest and matched on
         restart. It was specified here from the start and computed nowhere; see §1.6.7.)
```

**M10.2 corrections to this diagram** (§1.6.7 has the reasoning):

- **L3 stores the `ReducedGeometry`** — `s`, `B`, centre, support points, span certificate — *and* the
  `T₀` built from it, plus `T₀`'s reachability certificate. It long stored only the transform, which
  is why `reround_transform` (needing `B` and `s`) could not run on a cache hit. `build_l3_bundle` is
  the **one writer** of that schema; two writers is what let the CLI cache an uncertified bundle.
- **The pilot and `T₁` are not a new layer.** Derive `T₁` (9 ms) from L3 and a pilot; the thing worth
  keying is the **pilot** (19.2 s), which is M10.2b. `NeutralPilot.content_key` already exists and is
  now complete.
- **L2 was never a strict layer**: `warmup_range`'s `s_J` is nominally L2 but reads L3's support
  points, while the stated L2 key omits L3. `pilot_sd` makes the edge explicit. The numeric labels are
  becoming less useful than named immutable nodes; recorded, not yet acted on.

**M10.2e correction — what a key may promise** (§1.6.8 has the reasoning). Everything above assumes
that fixing a key fixes the bytes. **It does not, and it cannot.** The ambient BLAS thread count
selected between two valid bases under one L3 key for ten milestones, because multi-threaded OpenBLAS
reduces in a different order. `numerics.deterministic_blas` removes the thread count as an input, and
`DETERMINISM_POLICY_VERSION` is in the L3 key so caches warmed before it miss. But this NumPy ships
OpenBLAS `DYNAMIC_ARCH`, which picks kernels by **runtime CPU detection**, so a different machine can
still differ in the last bit at the same thread count. So the honest statement of the rule is:

> **Within a declared numerical-runtime profile**, a recipe key rebuilds deterministically. **Across
> profiles, byte equality is not promised.** Unrestricted cross-machine cache sharing and strict byte
> identity cannot both be had from ordinary floating-point libraries — this is a choice about which
> to give up, not a defect to fix.

And one more precision, because M10.2e's own test tripped on it: *"an artifact is a function of its
key"* is a claim about an artifact's **numbers**, not about every byte of its manifest. The L3 meta
embeds `ReachabilityCertificate.to_cache()`, which carries `elapsed_seconds` — a **wall clock**. Two
builds of one key therefore write different bundle bytes however deterministic the arithmetic is, and
that is correct: the timing is provenance. Read the rule as *the arrays and the numbers derived from
them*, which is what a false hit would corrupt; a stopwatch in a manifest corrupts nothing.

The profile is *recorded* rather than keyed (`numerical_identity` in every L3 bundle: the recipe key,
the basis/`T₀`/support hashes, the policy version, and the BLAS vendor/version/architecture/threads).
Keying it would make caches unshareable across machines; recording it makes a divergence readable —
which is precisely what nobody could do when one recipe key quietly produced two contents. The basis
hash belongs to **content identity and manifests, never to the pre-build lookup key**: hashing the
artifact to decide whether to build the artifact is circular.

This does not weaken the sampled law. `s_J = σ̂₀` is reproducible **in distribution** across machines,
not bit-for-bit, and M10 already reports that as calibration uncertainty (±2.6% relative SE, §1.6.6):
another machine's σ̂₀ is another honest draw from the same pilot law.

Rules (all adopted from the collaboration):
- **L0 is content-addressed, not file-hash-addressed** *(M8; refines the original `sha256(file)`
  key)*. `build_canonical_model` accepts a cobra `Model` that may have been assembled or mutated in
  memory, and a file hash cannot prove such a model came from the file whose bytes it hashes — so the
  old key let a model inherit **another file's L0 identity** (a model loaded, mutated, and re-frozen
  against the same path would keep the pristine file's key). The L0 key now fingerprints the IR the
  model *actually holds* (`model_id`, the polytope's `content_key`, the exchange mask) folded with the
  cobra + parser versions. A file's `sha256` is still recorded as **provenance** on the trusted
  `load_canonical_model` path (which both hashes and parses one file, so the correspondence is real),
  but it is never the identity. M8's cache computes a **file-lookup key** (`sha256(file)+cobra+schema`)
  separately, to skip re-parsing across runs, and validates the loaded artifact's content L0 key on
  load — so a false lookup hit is caught, never trusted.
- **Provenance in every key**: parser + code + artifact-schema versions, dtype/endianness, numpy
  version. A refactor that changes array semantics must miss the cache, not silently load stale bytes.
- **Validate on load**: shape, dtype, finite-check, and a stored content hash for every array.
- **Storage format**: `.npy` (memory-mappable) for large arrays; `.npz`/JSON for small bundles +
  manifest. Compressed `.npz` cannot be zero-copy mapped — don't use it for the big matrices.
- **Restartability**: per-stage **and per-chain** completion markers, not one top-level `COMPLETE`.
  A 31-of-32-chain failure resumes only the missing chain.
- **Concurrency**: a writer-claim directory (atomic `mkdir`) so two jobs don't compute the same key;
  write into a temp dir, `fsync` files + parent, atomic rename within one filesystem, create
  `COMPLETE` last.
- **Batch layout & aggregation**: `results/<batch>/<model_id>/…` per strain (full per-model run dir),
  plus `results/<batch>/cross_model/` holding aggregated tables — β-summary, reaction-activity, and
  exchange-conversion matrices stacked across strains. This is what powers the comparative question
  in spec §2 ("do two species retain different amounts of metabolic flexibility at comparable
  selection pressure"). The aggregation stage only *reads* per-model artifacts, so a partial batch
  still yields a valid cross-model table over the strains that finished.

### 1.2 Parallelization

- **Geometry (L3): sequential.** Basis discovery orthogonalizes each probe against the current
  basis, so it is inherently serial. Use one persistent HiGHS instance with simplex **warm starts**,
  `threads=1`. A **batched-LP** variant (solve K random-objective LPs concurrently, one rank-revealing
  QR on the differences) stays behind a benchmark gate — adopt only if it gives ≥1.5× wall-clock at
  the *same validated dimension*. At d≤55 here, sequential wins by default.
  - **Geometry (L3): BLAS pinned to 1 thread, forced and scoped** *(M10.2e — a requirement this plan
    did not state for ten milestones; §1.6.8)*. `threads=1` above pins **HiGHS**, so the support
    points were always reproducible; the **basis** is NumPy, and the ambient thread count silently
    chose between two of them under one L3 key. `numerics.deterministic_blas` wraps `build_geometry`,
    `build_transform` and `reround_transform` — the three constructors of thread-sensitive keyed
    artifacts, each verified to need it and the sampler verified not to. **This is a different policy
    from the worker thread limit below**, sharing only a mechanism (see §1.6.8): it is *forced*
    (a keyed artifact may not depend on what the caller exported) and *scoped* (a library does not
    seize its caller's process). It is also free — measured, pinning is **21% faster** here, because
    a 260×46 Gram-Schmidt is far too small for 14 threads to repay their dispatch overhead.
- **Sampling: process pool over `(β, chain)` units.** Given frozen geometry these are independent.
  A worker receives **only** frozen NumPy arrays (`T_active`, `center_active`, bounds, objective
  arrays, index maps) + a semantic RNG seed. A worker **never imports cobra or HiGHS**.
  - Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` **before** NumPy import (the real
    oversubscription risk in solver-free workers is BLAS/OpenMP, not HiGHS). *This bullet is a
    **performance** policy and is implemented and working — `run_batch` pins the env before creating
    the spawn pool, so each worker's freshly-imported NumPy inherits it. M10.2e initially misread it
    as also mandating geometry determinism; it does not, and reading it that way is what hid the real
    gap for ten milestones (§1.6.8). `setdefault` is correct **here** — a resource hint may yield to
    a user who exports 4 — and wrong for a keyed artifact, which is why L3's policy is separate.*
  - Workers **write their own** `.npy` files; never ship flux matrices back through IPC.
  - **Benchmark worker count {1, 2, 4, 7, 14}** on the Jetson by ESS-per-wall-second. 14 can lose to
    4–7 under memory-bandwidth/thermal limits; pick empirically.
- **Batch scheduling.** Across many strains the unit of work is `(model, β, chain)`. Process models
  so their per-model geometry (sequential, cached) can overlap the *sampling* of earlier models, but
  feed **one global worker pool** sized once for the machine — never a pool per model, which would
  oversubscribe the 14 cores. Each model gets its own result subdir; a final aggregation stage reads
  the per-model summaries into cross-model tables (§1.1).
  - **Implemented in M10.2c** as a *one-model lookahead* — submit `i`'s units, prepare `i+1`, drain
    `i` (§1.6.9). Measured A/B at M=3, cold, 8-rung ladder: **131.6 s → 93.1 s (1.41×)**; the
    asymptote is **1.93×** and `speedup = M(P+S)/(M·P + S)`, so **quote the M, not the limit**.
  - ⚠️ **This bullet's premise is `P ≲ S`, and it holds only at a real ladder.** Measured, `P` (parent:
    parse + geometry + **pilots**) is 23.1 s against `S` (pooled sampling) of **21.5 s at 8 rungs** but
    **1.3 s at the `betas=(0.0,)` default** — where the same overlap is worth ~5%. M10's pilot DAG made
    `P` 20× more expensive and nobody re-derived this bullet. *Before extending it, re-measure both
    terms: a scheduling decision is a claim about a ratio, and it expires when either side moves
    (§1.6.6b).*
- **RNG**: derive streams from stable semantic coordinates `(model_id, stage, β_index, chain_index)`
  via `SeedSequence`, and store the spawn keys. A flat `spawn()` sequence renumbers every downstream
  stream when task count changes — reproducibility death. Keying on `model_id` keeps each strain's
  streams stable regardless of batch composition or ordering.
- **Independence caveat**: "embarrassingly parallel" describes compute, not mixing. Cold high-β chains
  can stay trapped near init. Start chains from **dispersed convex combinations** of support points and
  `v*`. Sequential-ladder warm starts and parallel tempering are a *separate later mode* (M10), never a
  replacement for independent-chain diagnostics.

### 1.3 Inner-loop speed

All work happens in the reduced/active space (`n_active ≤ 260`, `d ≤ 55`), pure NumPy, zero Python
loops over reactions.

Per-coordinate precompute (once, frozen), for each reduced coordinate `k`:
- `d_vec = T_active[:, k]` and its nonzero structural support;
- `1/d` on the support (chord), `-1/d` on penalized∩support (breakpoints);
- slope drops `2·λ·w_r·|d_r|`; biomass component `d_b`; bound constants.

Per step:
1. pick `k ∈ {1..d}` uniformly; `d = d_vec`;
2. chord `[t_lo,t_hi]` from **all nonzero** components (see §1.6 correctness note);
3. `β==0` → `t ~ U(t_lo,t_hi)`; else build the piecewise-linear concave `J(t)`, choose a segment by
   log-mass (custom logsumexp, `expm1`/`log1p`), sample within it by stable inverse-CDF;
4. `y[k] += t`; `v_active += t·d_vec` (incremental); maintain `μ, C, J(0)` incrementally too;
5. every `refresh_interval`: rebuild `v_active = center_active + T_active @ y`, recheck
   bounds + mass balance, and reconcile incremental `μ,C,J` against a fresh evaluation.

A HiGHS **solve counter** asserts zero solver calls after sampling begins (integration test).
Profile temporary-array allocation and breakpoint sorting before micro-optimizing — at n~200–800
those, not "Python-ness," dominate.

**Sample storage is a config choice** (`output.store_flux_dtype` + `output.store_mode`), all three
selectable per run:
- `full_flux` / `float64` — full 773-length vectors, best fidelity, ~1 GB for an 8×4×5000 run (default).
- `full_flux` / `float32` — same shape, half the disk; calculations stay float64, only the stored copy narrows.
- `reduced` — store reduced `y`-states + a selected flux/exchange summary block; reconstruct full
  vectors on demand via `v = center + T·y`. Smallest on disk, best for large batches; the geometry
  artifact must be retained to reconstruct. Objective traces (μ, C, J, log-energy) are always stored
  regardless of mode.

### 1.4 Scientific accuracy — the guarantees

- **Deterministic span certificate.** After basis discovery, probe an orthonormal **basis of
  range(B)ᗮ** (within the active-coordinate space), obtained by pivoted QR, ordered by residual norm.
  `n_active − d` LP-pairs. This is a *complete* certificate: a missed feasible direction `w ⊥ range(B)`
  gives `p_iᵀw = w_i ≠ 0` for some probe, forcing positive width — impossible to hide. Random probes
  are a cheap pre-pass only. Capped runs on huge models record `span_certificate_exhaustive=false`
  and are called a *randomized partial check*, never an unconditional guarantee.
  - **Refined by M4 (collab, 6 rounds).** The exact-arithmetic proof is right, but a float64 LP can
    only say "flatter than this", so the certificate is **resolution-bounded** and reports what it
    licenses: `resolution = √k·√(1+leakage)·max_j width_upper(p_j) + leakage·diameter`. The `√k` is
    not optional — width is subadditive, so a direction tilted across all `k` probes hides that
    factor from each one. Flatness rests on a **weak-duality** upper bound (assumes nothing of the
    returned point, not even feasibility), never on the primal width, which is a *lower* bound and
    the wrong end of the interval. The licensed claim is: *every exact-polytope direction has its
    component orthogonal to `range(B)` bounded in width by `resolution`* — **not** "cannot
    under-count". See `.collab/specs/collab-outcome.md` § M4.
- **Exactness of the 1D conditional.** `expm1`/`log1p` inverse-CDF is the primary path across all κ.
  The small-|κL| uniform form is only a below-float64-eps series limit, documented — not a silent
  approximation sold as "exact." Sign-aware log-mass formulas (no `log(expm1(x))` for x<0).
- **No snapping.** Never round small sampled fluxes to zero; thresholds apply only in analysis
  (`features.py`), never to chain state.
- **z is LP-only.** Auxiliary absolute-value variables never enter the sampled state.
- **Reproducibility, scoped honestly.** Byte-identical traces are promised only within a locked
  binary + hardware environment. Across NumPy/BLAS/HiGHS/CPU changes, require matching *statistical*
  results + recorded provenance.

### 1.4.1 FVA-blocked reactions are structural zeros of the direction space  *(M4 finding; SETTLED)*

The example model has **61 free reactions (of 260) that cannot carry any flux at all** — the file
leaves `l < u`, but mass balance pins them. So the naive `n_free − rank(S) = 55` is only an *upper
bound*; the true affine dimension is **d = 46**, confirmed by an independent FVA+rank oracle.

This is a correctness requirement, not bookkeeping. If `max vᵢ == min vᵢ` over `P`, then every
feasible direction has `dᵢ = 0` **identically**, so a nonzero `B[i,:]` is numerical error. Left in, it
is not harmless: a basis row of ~1e-15 in a coordinate whose centre sits ~1e-13 *outside* its own
bound (both solver noise) divides into **a chord limit of order 0.03–0.5**, squarely inside the
legitimate chord. Measured, the chord at the centre came out `[−0.54, −0.39]` — *excluding `t = 0`* —
and `line_geometry` correctly refuses to sample it. **M5 could not have started.**

So the blocked components are projected out of every candidate direction, exactly. This is *not* the
forbidden snapping of small fluxes (§1.6): no flux is rounded, and a pinned reaction keeps its value;
what is zeroed is a component of the *direction space* that an LP measured as zero. It is
**numerically fixed at resolution `blocked_tol`**, not provably constant — a true 5e-16-wide dimension
would be dropped, and the separation guard would not object. The three resolutions must not
contradict each other: `scale_floor ≥ blocked_tol/span_tol`, `‖r_blocked/s_blocked‖₂ ≤ span_tol`, and
the SVD rank cutoff ≥ the LP's `feasibility_tol`.

### 1.4.2 Mass balance is gated on **reachable states**, not on `‖S·T‖`  *(M9 finding; SETTLED)*

**M5's rounding gate rejected a valid genome-scale geometry ~33% of the time, and the `model_id`
*string* decided which.** `model_id` keys the RNG (`stream_seed`), which drives the span-certificate
probes and the support-LP discovery directions → different support points → different covariance →
different `L` → different `T`. Across 24 streams on the *same polytope*, **8 raised `RoundingError`**.

The instrument was the problem. `rounding._transform_mass_balance` computes `max_k max_i |S_i·T_k| /
(|S|·|T_k|)_i` — a max over per-**(column,row)** ratios. (Its docstring claimed
`max_k ‖S·T_k‖_∞ / ‖|S|·|T_k|‖_∞`, a per-column ratio of *norms*; the two are not equal and differ
here by five orders of magnitude. The code never implemented its own documentation.)

| measure | min | median | max | spread | fails `span_tol=1e-9` |
|---|---|---|---|---|---|
| `‖S·T‖` **absolute** | 3.139e-12 | 3.851e-12 | 5.535e-12 | **1.8×** | — |
| `‖S·T‖` **relative** (the gate) | 9.588e-11 | 4.835e-10 | 3.038e-08 | **373×** | **8/24** |

The absolute residual is nearly constant; only the relative one swings. **The residual is not
generated by the row's own multiply — it is an absolute floor inherited from the basis
construction.** The discriminator: a log-log fit of residual against row cancellation scale over
61009 (column, row) pairs has slope **+0.165**, not the **+1** a locally-generated error requires;
across ≥4 decades of row scale the median residual rises only **6.6×** where ~1e4× would be expected.
At large scale `r/q → 2.4·eps` (honest arithmetic); at small scale `r/q → 1.8e5·eps`. So the gate
divided a fixed ~1e-13 floor by a per-row scale as small as 1e-5 — **the M4 lesson ("never divide by
a small number that is noise") one module further on.** M5's reassurance that it "passes on its own
merits (3.5e-10 vs 1e-9)" was one draw from a distribution whose 79th percentile crosses the bar.

**The replacement asks the operative question.** A per-direction bar cannot answer it in any case:
what matters is not how far a *unit* step along `T_k` leaves the manifold, but whether any state the
chain can **reach** violates mass balance. With `E = S·T`, `r_c = S·c − b`, `Y = {y : l ≤ c+T·y ≤ u}`:

```
R_i = max( |r_c,i + min_{y∈Y} E_i·y| , |r_c,i + max_{y∈Y} E_i·y| )     # 2 LPs per metabolite
certified  ⟺  max_i R_i ≤ η
```

Four things this gets right, each earned against a counterexample from the M9 `/collab`:

1. **The maximum over `Y` is exact, not a box bound.** `Σ_k |E_ik|·ρ_k` over per-coordinate radii is
   sound but unboundedly loose, because `Y` is coupled, not a box: for
   `Y = {|y₁|≤1, |y₂|≤1, |y₁−y₂|≤δ}` and `E_i = (1,−1)`, the box bound is `2` while the truth is `δ`.
2. **`η` is the contract `diagnostics.feasibility_report` already applies to emitted samples**
   (`|S_i·v − b_i| / max((|S|·|v|)_i, 1) ≤ 1e-9`), not a second tolerance fitted to a measurement.
   One declared definition of "mass balanced", proved a priori and checked a posteriori. The LP's own
   feasibility tolerance is an implementation *capability* that must support the contract, never
   define it.
3. **The bound comes from weak duality, never a primal reading** — M4's lesson, which named M5/M6 as
   where the temptation returns and where M9 duly walked into it. A returned `objective_value` is a
   *lower* bound on the max, so a solve that stops short reports the reachable residual too **small**
   and certifies a transform that reaches further. `_reachable_extreme` bounds
   `max e·y ≤ Σ_j max(π_j lo_j, π_j hi_j) + Σ_k |d_k|·Ω_k`, `d = e − Tᵀπ`, for **any** `π`. The `Ω`
   term is unavoidable with `y` free (any `d ≠ 0` sends the sup to `+∞`), and comes from a provable
   outer box via a **freshly recomputed** `T⁺` — an artifact does not vouch for itself.
4. **The objective is normalized before it reaches HiGHS.** `E_i` is ~1e-13; raw, it sits under the
   dual feasibility tolerance and every reduced cost reads as zero. Measured: a 1e-10 coefficient
   beside 1.0 coefficients is **dropped from HiGHS's scaled matrix entirely** — it reports that row's
   activity as 0.0 where the truth is 133.3, with `max_primal_infeasibility = 0.0`.

**One deliberate conservatism.** The contract's denominator is `max((|S|·|v|)_i, 1) ≥ 1`, so proving
`R_i ≤ η` proves the contract for every reachable `v` without computing it — which keeps `Y` fixed
across all solves (only the objective moves) and so keeps M3's warm start. A model whose fluxes earn
a denominator `≫ 1` is therefore held to a stricter bar than the contract demands. That errs toward
**refusing** a good transform, never toward admitting a bad one.

**Measured on the example model**: certified on every RNG stream, `max_i R_i` = **3.6e-11 … 5.1e-11**
(a **1.41×** spread against the old gate's 373×), **20–28× inside** the contract, **334 LPs / ~0.5 s**
— only 167 of 894 metabolite rows have `E_i` structurally nonzero. It sits just above M5's
independently measured 2.6e-11 emitted-sample residual, exactly as an upper bound on a superset must.

**`RoundingDiagnostics.transform_mass_balance_error` survives as a reported diagnostic and never
raises.** It is a genuine Oettli–Prager componentwise backward error — `|S_i·T_k| / q` is the smallest
componentwise relative perturbation of `S_i` making `(S_i+ΔS_i)·T_k = 0` exactly — and it catches one
corruption the certificate deliberately misses: for `S=[1]`, `T=[δ]`, `|v| ≤ δ = 1e-12`, the true
polytope has dimension **zero** but `T` invents motion; the diagnostic reports `1.0`, the certificate
correctly reports a reachable residual of only 1e-12 and passes. Structural invariant vs reachable
amplitude — two questions, two instruments, one of them a gate.

> ⚠️ **Still open (M10):** the span certificate is a *second* RNG-marginal gate — `build_geometry`
> raises "not exhaustive (214/214 probes, 1 inconclusive)" on ~1–2 of 20 streams. Same shape
> (a tolerance at the noise floor), not yet diagnosed, and **not** touched by M9.

### 1.5 Fixed-variable elimination (correctness + speed)

The 513 `l==u` reactions are removed from the sampled state in the **reduced polytope IR** (L1),
while the full canonical model IR (L0) is retained for identity. Store the affine reconstruction
explicitly: `v_full = R·v_reduced + c`, with mass balance becoming `S_F v_F = −S_fixed v_fixed`
(nonzero fixed fluxes create a real affine RHS — handle it). Objective lowering must fold in the
fixed-variable `L1` contribution and any constant term. **Saved samples stay full 773-length**, with
reaction IDs and fixed-status metadata intact. Test feasible-set *and* objective equivalence against
the full-`n` path on small models, plus round-trip reconstruction (full-space bounds + mass balance).

### 1.6 Correctness deltas vs the spec (call these out in code comments + tests)

Deltas 1–5 came from the design collaboration; **6–9 were added by the M2 collab review**, which found
two distribution-corrupting bugs in code that already passed 264 tests. Full reasoning and the measured
triggers are in `.collab/specs/collab-outcome.md`.

1. **Chord must keep every nonzero component.** The spec's "ignore |dᵢ|<tol" in the chord is a
   feasibility bug: a tiny `dᵢ` with `vᵢ` near its bound still binds a short, finite limit. Dropping it
   samples outside bounds. **No tolerance enters the chord at all** — see delta 6.
2. **Breakpoints: keep distinct cuts.** Merging unequal-but-close breakpoints changes `J(t)` and thus
   the target. Group only *exactly* coincident cuts (summing their slope drops).
3. **s_J belongs to the objective layer**, not geometry (it is evaluated through `J`).
4. **J\* is not a strict numeric upper bound.** Solver tolerance can make `(J(v)−J*)/s_J` slightly
   positive; the log-density code must not assume ≤ 0. (In the line kernel `J*` is now absent
   entirely — see delta 7.)
5. **Span validation is deterministic** (§1.4), stronger than the spec's random probes.
6. **A degenerate chord is a SELF-LOOP, never a redraw** *(overrides spec §19)*. Redrawing a different
   coordinate makes coordinate selection **state-dependent** and breaks the random-scan Gibbs
   stationarity argument: the kernel is the uniform mixture `(1/d)·Σₖ Pₖ` only because `k` is chosen
   independently of the state. When the feasible set on the line is a single point, its exact
   conditional is the point mass there, so the chain moves to it (`t = 0` for an on-bounds state).
   Consequently there is **no minimum chord width** — a 1e-13-wide chord is simply sampled. A *raw
   crossed* chord is different: the feasible set is empty, so `v` is infeasible and it raises.
7. **The absolute magnitude of `J` must never reach a probability.** `J*` and any constant offset cancel
   out of `p(t)` algebraically but *not* numerically. Store knot heights **relative to the peak of `J`**
   (accumulated from the slopes, never through the absolute value) and anchor each segment's mass
   integral at its **higher endpoint**. The line kernel takes no `J*`. Getting this wrong reversed which
   segment the sampler favoured, from slopes that were themselves exactly right.
8. **The opening slope is fixed by side, never by midpoint** *(overrides spec §20.3 step 6)*. On a
   one-ULP first segment the midpoint rounds onto the cut, where `sgn(0) = 0` yields a subgradient —
   measured 2× off in 10.5% of such configurations.
9. **Scaling parameters are validated, not trusted.** `β/s_J` is computed once and rejected if it
   overflows *or underflows to zero* (which would silently flatten the tilt). Computing it once also
   guarantees the mass stage and the sampling stage use the same `κ`.

### 1.6.1 (M5) Rounding cannot move the target, and the sampler's honest float64 claim

Two things the M5 collab review forced into precise language.

**The transform is a preconditioner and provably nothing more.** `L` is `d × d` and invertible, so
`range(T) = range(diag(s)·B·L) = range(diag(s)·B)` *exactly*, and `y ↦ v = centre + T·y` is affine
and injective with a constant Jacobian. So uniform-in-`y` is uniform-on-the-polytope and `π_β` in `y`
is `π_β` in flux, for **every** invertible `L`. The ridge is therefore free to be an engineering
parameter. But the identity is an *assumption* until something checks it: `rounding` now takes an SVD
of the computed `T` and refuses a rank below `d`. **A `T` that quietly lost a column produces no bad
numbers, only absent ones** — every sample feasible, every chord positive, mass balance exact, and
part of the support never visited.

**In float64 the chain is not Markov in `y` alone.** Its state is `(y, cache error, refresh phase)`,
because `v` is maintained incrementally. Exact Gibbs invariance is claimed **only in exact
arithmetic**; a measured drift is *not* a bound on the error induced in the stationary law (that needs
a spectral-gap argument we do not have). What is claimed is that the perturbation is small, corrigible
and *observed*: `v` is rebuilt exactly from `y` on a fixed schedule, the stored flux is the exact
`centre + T·y` of the stored state, and the discrepancy is measured at every refresh **and every
sample** (max 2.1e-11 on the example model, against fluxes of ~1e3).

**Scaling and rounding do different jobs.** `diag(s)` fixes the axes' *units* — an axis-aligned
1000:1 stretch is absorbed before rounding is reached. `L` fixes their *correlations*, which no
diagonal matrix can see. On the genome-scale model rounding takes the shortest chord at the centre
from 0.018 to 0.744 (41×) and the spread across axes from 77× to 3.8×.

**The relative mass-balance floor belongs to fluxes and must not touch directions.** A sampled *flux*
carries solver noise at the FVA-blocked reactions, so a metabolite row touched only by blocked
reactions divides a noise value by itself and reports a relative residual of exactly 1.0 — hence the
`scale_floor = 1.0` in `NativeCSC.relative_residual`. A *direction* carries no such noise (`T`'s
blocked rows are exactly `0.0`), so the transform's own check is **unfloored**. Correct where the
noise exists, absent where it cannot be.

### 1.6.2 (M6) The tilt adds inputs, not machinery — and three places the spec invites an error

M6 changes **nothing** in the transition kernel. At `β > 0` the only difference is that `sample_line`
builds M2's piecewise-exponential conditional instead of drawing uniformly, and every word of the
§1.6.1 invariance argument survives verbatim — it never mentioned the conditional's *shape*, only
that it is **exact**. What M6 supplies is four inputs: the objective in **reduced** coordinates, the
energy scale `s_J`, the β-ladder, and the traces.

Three deviations from the spec, each one a place a subtle error is invited:

1. **`J` is *not* maintained incrementally**, though spec §1.3 step 4 suggests it. Nothing needs it:
   `build_piecewise_j` derives every slope from `v` and the direction on the spot, and the conditional
   depends on `J` only through peak-relative *heights*. A running `J` would be a second cache to
   drift, reconcile and mistrust — M5 paid that price for `v`, which is genuinely needed for the
   chord, and there is no reason to pay it twice for a quantity that is only *reported*. Traces are
   computed **exactly**, after the fact, from the stored fluxes.
2. **The objective is lowered onto the reduced polytope, and is therefore `J` up to an additive
   constant.** That is not a defect: the constant (the fixed reactions' `μ` and L1 cost) provably
   cancels from `p(t)`, and it must never reach a probability — the same fact that keeps `J*` out of
   the kernel. `ReducedObjective` carries the constant *separately*, for reporting, because a trace of
   `J` has to be comparable with the LP's `J*` and a probability must not be.
3. **Mean-`J` monotonicity is a theorem, not a hope.** With `κ = β/s_J` and `π_κ ∝ e^{κJ}`,
   `dE_κ[J]/dκ = Var_κ(J) ≥ 0`. So a *violation* is never physics — it is noise or a bug, and the
   check exists to tell those apart. It measures each drop in **Monte-Carlo standard errors**
   computed from the ESS **of the `J` trace itself** (not the coordinates, not `√N`), and reports
   **R̂(`J`)** alongside, because an ESS says nothing about retained initialization.

### 1.6.3 (M6) Three artifacts meet in the sampler, and they must be *bound*, not merely passed

*(M6 collab finding — the nastiest failure mode in the package so far.)*

`run_ladder` takes the L1 polytope, the L3 transform and the L2 objective. They are all just arrays,
and until M6 nothing checked that they had ever been computed against each other.

Hand it an objective lowered from a **different model of the same size** and the chain tilts by the
reactions *that* objective names — while `ReducedObjective.evaluate_many` reports *those same
reactions* as `μ` and `C`. So the trace of `J` **rises monotonically with β, exactly as the theorem
demands**, because the chain really is maximizing the thing the trace is measuring. Every diagnostic
agrees and every one describes the wrong model. Feasibility, mass balance, the chords and R̂ cannot
help: **none of them knows which reaction `J` is supposed to be about.**

Not hypothetical once M8 exists: L2 and L3 are *separate cache artifacts*, and a stale key is all it
takes to load two that never met. So `ReducedPolytope.content_key()` is the public L1 key, every
downstream artifact carries it, and `run_ladder` refuses a mismatched pair. One string comparison.

### 1.6.4 (M6) `s_J` is a *range*, so its floor must be a **resolution** and not a magnitude

*(M6 collab finding.)* `s_J = J* − Q₀.₀₅(J(W))` (spec §22.2) is invariant when a constant is added to
`J`. Any floor it is compared against must be too — or a constant that provably cannot change a
probability changes `s_J`, and with it **every rung of the ladder**.

The original floor was `1e-9·max(1, |J*|)`. Shift `J` by `+1e16` and a healthy `s_J = 12` fell below a
floor of `1e7` and was silently replaced by 1.0, making every positive rung **12× hotter**. This is
M2's delta 7 (*the absolute magnitude of `J` must never reach a probability*) wearing the calibration
layer's hat, and it is the fourth time in this project a **magnitude** has been used where a
**resolution** was needed.

The floor is now the float64 **resolution of the subtraction itself** — 64 ULPs of `max(|J*|, |Q|)` —
which asks the question that has an answer: *does this difference have any significant digits left?*
It cuts both ways, which is how you know it is right: at `|J*| = 1e5` the old floor was `1e-4` and the
new one `9.3e-10`, so a real range of `1e-6` is now **kept**.

**And a degenerate range now raises.** Spec §22.2 says to fall back on a "**declared** positive
scale", and *a library default is not a declaration*. A silent `s_J = 1` would make this strain's
`β = 2` name a different selection pressure from every other strain's — the exact failure `s_J` exists
to prevent — as a log line nobody reads. `sampler.energy_scale_fallback` defaults to `None`, and
`None` means stop.

### 1.6.5 (M7) every input to `s_J` is keyed on **both** the objective and the polytope

*(M7 collab finding.)* M7 is the first milestone with **two objectives on one polytope** (base vs
reweighted), which is the M6 "two artifacts never computed against each other" bug given fresh fuel:
on the toy, `s_J` is **0.68** under the base objective and **0.0068** under the reweighted one, and
M6's guard (`energy_scale.polytope_key`) could not tell them apart, because the two share a
`polytope_key` exactly. `s_J = J* − Q_q(J(W))` is a subtraction of three model-derived inputs — the
optimum's `J*`, the objective that evaluates `J(W)`, and the warm-up array `W` — and it is only a
*range* if all three come from one objective on one polytope.

So all three are keyed and cross-checked before a single `J(W)` is formed:
- `LPOptimum` carries `objective_key` **and** `polytope_key`. `objective_key` alone is insufficient —
  it hashes the objective's params, *not* the polytope's bounds, so two polytopes differing only in
  bounds hash identically and `J*(A) − Q(J_B(W))` would pass every objective check (Codex, r2).
- `ReducedObjective` carries both keys; `choose_energy_scale` requires a **`warmup_polytope_key`** and
  checks it, because the warm-up array is a bare `(K, n_free)` matrix with no identity of its own — a
  same-shaped set from the wrong polytope silently changes `s_J` (Codex, r3).
- `run_ladder` re-checks the `EnergyScale` and the transform against the objective.

`optimum_coordinates`, by contrast, is **deliberately not keyed**: it is a start *hint* (one vertex of
a Dirichlet hull, then made feasible or the run raises), it enters only the initial state and never
the kernel/objective/`s_J`/traces, so it **cannot change the invariant target** — a wrong hint only
seeds a poorer start, which is observable via feasibility and R̂/ESS. Keying it would imply it defines
the distribution, which it does not (Codex conceded, r5). The boundary is documented instead.

> **Refined by M10.2 — this is about *target* identity, and there is a second question.** The
> sentence above is correct and it settles exactly one thing: whether the hint belongs in the keys
> that name a **distribution** (the objective, `s_J`). It does not. But "are these bytes the same
> **artifact**?" is a different question with a different answer: a finite chain started elsewhere is
> a different chain. The hint therefore **is** hashed by `batch.sample_recipe_key` and **is** kept out
> of the β=0 pilots entirely — see §1.6.7. Importing this paragraph's reasoning into an artifact key
> is precisely the error M10.2 made and Codex caught: a recipe key already hashes `seed` and
> `chain_index`, which define no law either.

### 1.6.6 (M10) `s_J` from the pilot's **spread**, not its range to `J*` — and M6's remedy was wrong

*(M10 collab finding, 4 rounds, converged AGREE.)* M6 recorded a **prerequisite** — that the β axis
is uncalibrated — together with a remedy, a mechanism and a magnitude: use spec §22.2's "support **or
pilot** points", and the ladder "tilts ~12× harder". **The diagnosis was exactly right and all three
parts of the cure were wrong**, which nobody could see because nobody had done the arithmetic.

Measured (Bifido, d = 46, λ̃ = 0.5, `J*` = 9.4664, 4 chains × (3000+3000), N = 12000):

| candidate `s_J` | value | `dE/dβ|₀` | β to close the gap (linear response) |
|---|---|---|---|
| **A** `J* − Q₀₅(J(support))` — M6's | 32.51 | 0.183 | 117 |
| **B** `J* − Q₀₅(J(pilot))` — **spec §22.2 literal** | 25.41 | 0.234 | 91 |
| **C** `J* − mean(J(pilot))` | 21.40 | 0.278 | 77 |
| **E** `sd(J(pilot))` | **2.44** | **2.44** | **8.8** |

Spec §22.1's ladder tops out at **β = 16**. Swapping the point set *inside the spec's formula*
(A → B) buys **1.28×**, not 12× — the `J*` anchor dominates, so the fix does essentially nothing.
M6's "12×" is `32.5/2.44`: **a ratio between an anchored range and a spread, two different
quantities.** The remedy that works abandons the formula.

**Decision: `s_J = σ̂₀`, the SD of `J` over a frozen β=0 pilot, as a NEW mode
`sampler.energy_scale = "pilot_sd"`.** `warmup_range` keeps its semantics and its label; v1's
results keep their scale method. Measured, the *identical* ladder now closes **75.8%** of the gap at
β = 16 (E[J] −12.18 → +4.24, monotone, R̂ ≤ 1.06) where `warmup_range` closed **13%**.

**What may be claimed:** `I₀ = 1` and `KL(π_β‖π_0) = ½β² + O(β³)` — β is the **local**
Fisher-standardized coordinate, and `β = 1` shifts `E[J]` by one neutral SD to first order.
**Exact at the *estimand* level only:** the implemented coordinate uses the frozen plug-in, so
`I₀ = σ₀²/σ̂₀²`. **What may NOT be claimed:** a universal finite-β axis; Fisher–Rao arc length at
finite β (that is `ℓ(β) = ∫₀^β √(Var_t(J))/σ₀ dt`, equal to β only infinitesimally); that the ladder
"spans". This is M6's own "engine validated, scale not calibrated" distinction, one layer deeper.

**No scalar is universal, so σ₀ sets the axis and Δ₀ is *reported*.** If the neutral deficit
`X = J* − J` has a density of states `g(x) ~ C·x^{r−1}`, the tilted law is `e^{−κx}·g(x)`:
measure-zero is what *produces* the `x^{r−1}` power and hence `r/κ`, so `1 − q(κ) ~ r/(κΔ₀)` and the
anchored coordinate **does** govern fractional gap closure in the sharp regime (entropy modifies it,
it does not defeat it). E is natural in the *weak* regime, C in the *sharp* one. So the run reports
`Δ₀ = J* − E₀[J]`, `G = Δ₀/σ̂₀` (9.03 here — "the strain's headroom in neutral standard deviations"),
`β·G` and `q(β)`: the anchored view stays recoverable as a **derived observable** instead of being
baked into the x-axis, where it would hide the very cross-strain quantity §1.1 exists to compare.

**The pipeline is sequential, and the two pilots are independent streams.**

```
1. geometry pilot at β=0 under T₀   (OBJECTIVE-INDEPENDENT)
2. freeze its covariance → build T₁   (spec §17.4; measured cond(C_q) 1.54e4 → 5.97e3, 2.57×)
3. INDEPENDENT scale pilot at β=0 under T₁   (better mixing → better ESS for σ̂₀)
4. freeze σ̂₀ → production chains on independent streams
```

One shared pilot would be *valid* — the transform cannot move the stationary law and both artifacts
are frozen — but it would make pilot-seed sensitivity **unattributable**, since geometry quality and
the selected target would move together. Separating them separates *random efficiency calibration*
from *random target calibration*. A poor `T₀` cannot deform the neutral **target**, only the
efficiency of estimating σ̂₀ from it, so the stages do not compound as target deformation.
**The β=0 law is objective-independent, so one neutral pilot serves every objective on a polytope** —
which matters directly, because M7 puts a base *and* a reweighted objective on one. The pilot
artifact carries **no objective key**; the derived scale artifacts do.

**Precision warns; validity refuses.** `se(σ̂)/σ ≈ √(K−1)/(2·√ESS_{(J−μ)²})` with **Pearson**
kurtosis and the ESS of the **centered-square** series — not the Gaussian `1/√(2·ESS_J)`, which fixes
`K = 3` and reads the wrong series (measured: the two ESSs differ by **2.17×**). Target ~2%; above it
a **warning, never a gate** — a precision bar on an MCMC estimate would refuse a correct run for an
unlucky pilot seed, which is §1.4.2's defect in a new coat. **But** nonpositive / non-finite / below
`64·ulp(max|J|)` still **raises**: those make the target *undefined*, a different failure from
imprecise. The refusal reuses M6's predeclared `ENERGY_SCALE_ULP_MARGIN` rather than inventing a bar —
a bespoke "is σ̂₀ too small" criterion is exactly how the noise-floor gate would re-enter.
The estimand is **predeclared as the SD and never switched per strain** after seeing diagnostics;
`R₉₀ = (Q₉₅−Q₀₅)/(3.289707·σ̂)` (1.0 for a Gaussian; **1.015** measured), skew and excess kurtosis are
reported *as diagnostics*, not as estimator selectors — switching would forfeit `I₀ = 1` and make β
mean different things in different strains.

**What the DAG guarantees, precisely.** Freezing `T₁` and `σ̂₀` before production gives a
**time-homogeneous kernel with a fixed conditional invariant law**. It does *not* give stationarity
from iteration zero — burn-in gives convergence, not stationarity. And conditional on the pilot the
invariant target is `π_{β/σ̂₀}`, not the ideal `π_{β/σ₀}`; marginalising over pilot randomness gives a
**mixture of calibrated targets**. That is *calibration uncertainty*, not an invariance failure.
Range-invariance alone is **not** the clean condition either: `T₁` must be a nonsingular affine
coordinate change **on the affine hull**. The algebra was never in doubt — the real risks are
feasibility tolerances, **rank loss**, state carry-over and residual adaptation, which is what the
tests target.

#### `r_eff(κ)` — a falsifiable prediction, now a diagnostic  *(and the ladder's real ceiling)*

For a piecewise-linear `J` near an optimal face of dimension `f`, with `c = d − f`, Laplace gives
`Z(κ) ~ e^{κJ*}·C·κ^{−c}`, hence `J* − E_κ[J] ~ c/κ` and

```
r_eff(κ) := κ·[J* − E_κ J] → c      (corroborator: κ²·Var_κ(J) → the same c)
```

an **integer-ish plateau under regular local geometry** — not an unconditional expectation. At small
κ, `r_eff = κΔ₀ − κ²σ₀² + O(κ³)` starts at **zero**, so non-constancy *before* the asymptotic region
is expected. **Measured, that expansion is confirmed to three digits** (κ=0.104: predicted 2.20,
measured 2.182; κ=0.209: predicted 4.27, measured 4.263).

Measured plateau: `r_eff` = 35.4 (β=16) → **37.4 ± 1.9 (β=32) → 37.0 ± 3.6 (β=64)** — flat within
MCSE, and the corroborator agrees where it should (`κ²Var = 38.6` at β=32). So **c ≈ 37–39 and the
optimal face has dimension f ≈ 7–9** in a d=46 polytope — tentative, and under-powered.

🔴 **Above β=64 the numbers measure mixing failure, not geometry.** R̂ climbs 1.22 → 1.39 → 1.79 →
**1.91** and ESS collapses to **4**. The proof it is not physics is M6's own theorem
(`dE_β[J]/dβ = Var_β(J)/s_J ≥ 0`): `E[J]` *falls* 8.6357 → 8.6109 from β=128 to β=256. A drop is
never physics. Codex's `J*`-indictment signature (a linear drift, `r_eff` 44 → 91 as κ doubles) duly
fires there — and is **unattributable**, because the diagnostic's precondition is a converged chain.
**Practical consequence: under `pilot_sd`, β = 16 is the working top rung at a 4×(2000+2000)
schedule** (q = 0.76, R̂ = 1.06); β ≥ 32 needs a far longer one, because the tilted chain concentrates
and its chords shorten.

### 1.6.6b (M10.2b) A recorded **measurement** goes stale when a later milestone moves its premise

§1.6.6 and three docstrings recorded re-rounding's gain as `cond(C_q)` 1.54e4 → **5.36e3 (2.87×)**
(and 5.11e3 elsewhere). The shipped code produces **5.97e3 (2.57×)**, and has since M10.2a. The cause
is M10.2a's own fix: removing the objective's `optimum_coordinates` start hint from the pilots
**changes every pilot's draws** — its `CALIBRATION_IMPL_VERSION = 2` note says exactly that, in order
to justify a cache-invalidating version bump — and `cond(C_q)` is a *function of those draws*. Nobody
re-measured. Confirmed by re-running the old path: **with the hint, 5304; without it, as shipped,
5969.**

Nothing is wrong with the code, and the *finding* is intact: re-rounding really does improve
conditioning, by 2.57× rather than 2.87×. What was wrong is the **status of a number**. This repo
already knows that a tracker's *forward-looking remedies* are conjecture until measured (§1.6.6, M6's
"12×"). This is the sharper sibling: **a recorded measurement is a claim with a premise, and it
expires silently when the premise moves.** A version bump that announces "this changes every draw" is
a bell that should ring for every derived number in the docs — the bump was made and the bell was not
heard, in the milestone whose whole subject was artifacts drifting from their keys.

### 1.6.7 (M10.2) An artifact must be a **function of its key** — and §1.1's L3/samples were not

*(M10.2 collab finding, 4 rounds, converged AGREE.)* M10.1 recorded the CLI wiring as blocked on a
**design fork §1.1 does not settle**: the cache returns a `RoundedTransform` with no `ReducedGeometry`,
so re-rounding on a hit needs "the pilot and `T₁` to enter the DAG as a new layer". The arithmetic
says otherwise, and it was never done:

| stage | measured (Bifido, d = 46, serial) | cached before M10.2? |
|---|---|---|
| `build_geometry` (~1100 LPs) | **1.168 s** | yes — as `T₀`'s bundle |
| `build_transform` → `T₀` | 0.005 s | — |
| the two β=0 pilots | **19.202 s** | **no** |
| `reround_transform` → `T₁` | 0.009 s | **no** |

A layer for `T₁` would exist to avoid rebuilding a **1.17 s** stage while costing **19.2 s** to fill —
16.4× upside-down. The rule is the one §1.1 already implies: **cache what is expensive, derive what is
cheap, key everything.** `prepare_model` goes 2.388 s → 21.6 s of *serial parent* work, which is
Amdahl's term, not a cache question — it is **M10.2b** (pilot caching + two-phase pool dispatch), and
it is why restart under `pilot_reround` re-runs 19.2 s of pilots before resuming one chain.

**The blocker itself was plan/code drift, not a fork.** §1.1 has always said L3 "holds B,
support_points, center, L (Cholesky), T, dimension, span certificate". `RoundedTransform.to_bundle`
holds `T`, `L`, centre and support coordinates — no `B`, no `s`, no reconstructable certificate — and
`ReducedGeometry` had **no serializer at all**. M9's "the code never implemented its own
documentation", one layer up. (Codex's correction, conceded: it cached a *non-reconstructible hybrid*,
not "the transform". And repairing it does **not** dissolve the topology question — §1.1's L2 was
already not a strict layer, since `warmup_range`'s `s_J` is nominally L2 but reads L3's support points
while the stated L2 key omits L3. `pilot_sd` only makes that edge impossible to ignore.)

**The through-line, and the reason these are correctness fixes.** §1.1's asymmetry — *a false miss only
recomputes; a false hit corrupts* — means **an incomplete key is strictly worse than none**: absent
means no cache, incomplete means a store that confidently returns the wrong bytes. Asking "is this
artifact a function of its key?" of things this repo already had returned **no** four times:

- **The neutral pilot was objective-dependent and said it wasn't.** `NeutralPilot`'s docstring —
  "**objective-independent**, and that is load-bearing … one neutral pilot serves every objective on a
  polytope" — was false when written: `calibrate` fed both β=0 pilots `optimum_coordinates`, derived
  from the objective's own LP optimum, while `content_key` hashed no objective and no start. Measured,
  two pilots differing in *nothing else*: **identical `content_key`**, max |Δy| = 2.79, `T₁` cond 7198
  vs 9663, `s_J` 2.6287 vs 2.4995. Not bias — both are honest draws from one β=0 law and the gap is
  Monte Carlo noise. The defect is that **the artifact was not a function of its key**, so M7's
  two-objectives-on-one-polytope case takes the first hit and never knows. Codex's mechanism is
  sharper than "a different start": the hint changes the support hull's cardinality, hence the
  Dirichlet draw's dimension, hence **RNG consumption on every later transition** — the streams
  desynchronise. Fixed **structurally**: `run_neutral_pilot` has no such parameter. The claim's true
  form is "…every objective sharing this polytope, **transform and pilot recipe**".
- **M9's mass-balance gate was bypassable through the package's own cache-warming path.** It lived in
  the `compute()` closure of `batch._load_or_build_geometry` — which runs **only on a miss**. On a hit
  nothing read the certificate; and `maxent build-geometry --cache-dir` assembled its *own* bundle
  under `batch`'s key, omitted the certificate from it, and **stored it after printing `REFUSED`**.
  Two writers of one schema is the defect. `batch.build_l3_bundle` is now the one writer, it raises
  rather than returning an uncertified bundle, and `require_certified_transform` runs on **every**
  load path. It checks three things, each refusing a different lie: the polytope (M6's join), the
  **transform** (new — `T₀` and `T₁` share a `polytope_key` *exactly*, so `ReachabilityCertificate`
  gained a `transform_key`), and the **verdict, re-derived** from `worst_absolute` vs `contract`
  rather than read off a stored boolean. Hence `to_cache` stores the fields and `as_dict` the verdict:
  a bundle asserting innocence beside contrary evidence is inexpressible. (M9: never trust a reading,
  check the bound.)
- **`T₁` was sampled uncertified — and must be certified before the *scale pilot*, not production.**
  The scale pilot is itself a chain stepping in `T₁`'s frame; an uncertified `T₁` lets it walk off the
  manifold and `σ̂₀` is then read off off-manifold fluxes. The exact-arithmetic theorem does **not**
  transfer `T₀`'s certificate: `range(T₁) = range(T₀)` exactly (§1.6.1), so the true worst residual is
  the same number, but the certificate is a *numerical* bound recomputing `E = S·T₁` and `Ω` from a
  fresh `T₁⁺`, and `fl(B·L₀)` and `fl(B·L₁)` need not share a floating-point column space. Measured:
  `T₁` certifies at **3.86e-11**, inside M9's independently measured `T₀` range of 3.6e-11 … 5.1e-11 —
  two certificates, two matrices, no shared computation, agreeing where the theorem says they must.
  Order: certify `T₀` → geometry pilot → `T₁` → **certify `T₁`** → scale pilot → production. And
  `calibrate` takes `bootstrap_certificate` as a **required argument** rather than recomputing it: the
  proof exists already, and demanding it makes an uncertified transform unable to enter the DAG.
- **A `COMPLETE` marker named a chain, not an experiment.** §1.1 has always specified the sample key
  (`L2 + L3 + β + chain seed coords + sampler_version + burn/thin/n_samples`). **Nothing computed it**:
  restart skipped on the marker alone and `store_chain` recorded only `polytope_key`. So a results
  directory reused after any change that moves the numbers resumed the units it had and sampled the
  rest **from a different law** — two experiments in one tree, stacked into one cross-model table,
  every per-chain diagnostic green *because each chain really is correct*. M10 forced this rather than
  created it: `T` and `s_J` were once pure functions of the polytope and config; now they descend from
  a pilot, so two runs of one unchanged config can honestly disagree. `batch.sample_recipe_key` now
  computes it and `_already_done` **refuses** rather than recomputing — a results tree is the user's
  output, not a cache.

**The criterion, stated once because getting it wrong is easy:** an artifact key asks *"are these bytes
the same artifact?"*, **not** *"is this the same distribution?"* M10.2 initially excluded
`optimum_coordinates` from the sample recipe by importing §1.6.5's target-identity reasoning — while
having just fixed the identical defect for the pilots. Codex's refutation is decisive and general: the
recipe key already hashes `seed`, `chain_index`, `schedule` and `storage_mode`, **none of which define
the stationary law**. Both keys are right; they answer different questions. `movable` is the one
exclusion that survives, being an exact function of a transform already hashed.

### 1.6.8 (M10.2e) Two requirements that share a mechanism are still **two requirements**

§1.6.7 asked "is this artifact a function of its key?" of four things and got **no** four times. Asked
of v1's own geometry, the answer was also no — and this one had been shipping since M4.

**The defect.** One L3 key (`e9d6fc28673a`), two bases, selected by an environment variable nobody
set. The support points are identical — §1.2 pins HiGHS to `threads=1` — but the basis is NumPy:
`residual -= basis @ (basis.T @ residual)`, and multi-threaded OpenBLAS reduces in a different order.

| `OMP_NUM_THREADS` | basis | Δ | `T₁` cond | `certify(T₁)` |
|---|---|---|---|---|
| **1** | `d35fe4fccf` | — | 5969 | **3.873e-11 OK** |
| unset / 2 / 4 / 8 | `970f8dddac` | **2.7e-15** | **5352** | 🔴 **kUnknown** |

Four separate findings, and only the first is fixed here:

1. **The artifact was not a function of its key.** §1.1's rule, violated in the geometry it describes.
2. **The two bases are the same basis** — 2.7e-15 apart, a few ULPs, identical span certificates
   (`max_width` 1.80e-12, 0 inconclusive), same d = 46, both `T₀` certify. *Nothing here is wrong.*
3. **The pilot amplifies 2.7e-15 to 2.601** — O(1) on a coordinate range of [−2.48, 1.95]. Not a bug;
   an MCMC being chaotic, at a gain of ~10¹⁵. But it makes geometry reproducibility the precondition
   for the pilot DAG meaning anything twice.
4. 🔴 **The `T₁` that fails `certify` is *better* conditioned than the one that passes** (5352 vs
   5969), so the failure cannot be blamed on unlucky geometry: `certify_reachable_mass_balance` is
   **fragile**. Pinning the threads makes that basis unreachable by default and **hides** this rather
   than fixing it — any model or seed can still land on it. Recorded, deliberately not chased; it
   needs the certificate's LP formulation looked at, not the thread count. Reproduction: build with
   `OMP_NUM_THREADS` unset (basis `970f8dddac`), then `certify_reachable_mass_balance(T₁, reduced)`.

**The lesson, and it is not the one I first wrote down.** The tracker's first draft of this section
claimed *"§1.2 already mandates the fix and the code drifted — the fourth time this session"*. That is
**false**, and Codex refused it. §1.2's thread rule is a sub-bullet of *"Sampling: process pool"*
whose own parenthetical names its purpose — oversubscription "in solver-free workers" — and it is
**implemented and works**: `run_batch` pins the env before the spawn pool exists, so each worker's
fresh NumPy inherits it. There was no drift. **There was a gap**: nothing ever asked for *parent-side
geometry determinism*, because worker oversubscription (performance) and geometry reproducibility
(correctness) are **two requirements that happen to share one mechanism**, and treating them as one
is what let the second go unstated for ten milestones. *I pattern-matched a rule onto a case it does
not cover, in the session where that rule had just paid off three times* — which is this repo's own
recorded failure mode about confident prose, committed while recording it. **Corollary: a rule that
has just paid off three times is exactly the rule you will over-apply next.**

Being two policies, they are implemented as two, and differ where the requirements differ:
`_limit_thread_env` **defaults** (a resource hint yields to a user who exports 4) and is applied
pre-spawn; `numerics.deterministic_blas` **forces** (a keyed artifact may not depend on the caller's
environment) and is **scoped** to the L3 constructors (a library does not seize its caller's process,
and could not fix this by mutating `os.environ` anyway — BLAS reads those at load time, so a
`setdefault` after NumPy is imported changes nothing).

**Building it moved two of the design's own premises**, both by measurement:

- **The scope was wrong in the spec, in both directions.** The collab framing named *the basis* as the
  sensitive artifact. But hold the basis fixed and `T₀` **still** moves (`8e587b6ad5` pinned vs
  `9d334b3f31` ambient) — the covariance and Cholesky are BLAS in their own right, so scoping the
  basis alone would have left half the defect in place. Conversely the **sampler needs no scope**:
  hold the geometry and `T₀` fixed and the draws are bit-identical at 1 thread and 14, its inner loop
  being chord arithmetic on short vectors. Three constructors, verified individually.
- **The policy is free — it *pays*.** `build_geometry` is **1.170 s pinned vs 1.488 s at 14 threads
  (0.79×, 21% faster)**; L3 total −0.317 s. A 260×46 Gram-Schmidt is far too small for 14 threads to
  repay dispatch overhead. The nondeterminism bought nothing and cost 0.3 s.

**And the R̂ bar it exposed was never a threading problem at all.** Pinning the threads made
`test_the_chains_mix_and_the_diagnostics_say_so` fail (R̂ 1.1654 vs a 1.15 bar), which looked like the
fix breaking a test. Measured across 8 seeds at the fixture's own 1500 draws, the truth is worse and
simpler: **R̂ spans 1.089–1.177 and min ESS spans 10.2–50.7** — the bars sit *inside* the distribution
of valid runs, 2 of 8 seeds fail, and the fixture's own seed 0 fails **both**. The thread count was
never the cause; it was one way to toss a coin that seeds toss just as well. This is M9's *a bar a
valid input clears only 2 times in 3 is not a tolerance, it is a coin flip*, for the third time.
R̂ → 1 as the chain grows is a theorem, so the **schedule** is the honest lever, not the bar: at 4000
draws R̂ is 1.033–1.059 and min ESS 59.7–155.0 across 5 seeds — the same bars, now with 2.5× and 3×
margin, catching a regression that breaks mixing instead of sampling the noise.

### 1.6.9 (M10.2c) The mandate was real, the *premise* had expired — and the arithmetic decides which

**§1.2's batch-scheduling bullet says exactly what the tracker claimed it says**, and this time there
is no over-application: it is a *top-level* bullet whose subject **is** scheduling across models
("process models so their per-model geometry can overlap the *sampling* of earlier models, but feed
one global worker pool"), and `run_batch` was `for spec in specs: _run_one_model(...)`, which prepares,
submits and **drains** in one breath. So the code disagreed with the plan. **That is not sufficient
reason to build it, and the difference is the milestone.** §1.6.7's rule — *cache what is expensive,
derive what is cheap* — has a scheduling twin: **overlap what is long, and measure which term that
is.** A mandate names a mechanism; only arithmetic says whether it pays.

**Measured first (Bifido, d = 46, 14 cores, cold cache):**

| | `betas=(0.0,)` — the default | the 8-rung ladder — production |
|---|---|---|
| `P` prepare (parent, **1 core**) | 23.1 s | 23.1 s |
| `S` sampling (pool, **14 cores**) | **1.3 s** (4 units) | **21.5 s** (32 units) |
| `P/S` | **18×** | **1.08×** |
| overlap is worth | **~5%** | **1.93×** (asymptotic) |

**The default config would have sent this milestone to the wrong lever.** At `betas=(0.0,)` the
parent's 23.1 s dwarfs 1.3 s of sampling, the pilots are 86.7% of `prepare_model`, and the obvious
conclusion is that §1.2's overlap is noise while the *pilots* need the pool. At the ladder the package
actually exists to run, `P ≈ S` and the conclusion inverts. **Which lever wins is a function of which
config you measure, and the default is not the production case.**

**And that inversion vindicates the tracker's "weaker remedy" label for pilot-chain dispatch, for a
reason it did not state.** `prepare_model` is 23.1 s of which the two β=0 pilots are **20.1 s
(86.7%)** — 8 chains (2 pilots × 4) walking one at a time in the parent while 13 cores idle, and
`run_chains`' own docstring says they are poolable and that a pool "draws the *same numbers*". It
reads like an obvious win. It is not, **because the parent and the pool are different cores**: once
prepare overlaps sampling the pilots are **free — hidden behind the pool** — so pooling them buys a
further **0.7%** (23.1 → 22.9 s/model) against a **23.2 s/model** all-cores floor. Overlap alone lands
on that floor. *Two levers on the same 20 s can differ by 100× in value depending only on what else is
running.*

| per model, 8-rung ladder | s/model | speedup |
|---|---|---|
| today (serial) | 44.6 | 1.00× |
| **+ cross-model overlap** | **23.1** | **1.93×** |
| + pilot pooling only | 29.6 | 1.51× |
| + both | 22.9 | 1.94× |

**The win is a function of batch size, and quoting the asymptote alone would be the M6 "12×" error
again.** Only the *last* model's sampling has nothing to hide behind, so `speedup = M(P+S)/(M·P + S)`:
**1.32× at M=2, 1.47× at M=3, 1.77× at M=10, 1.91× at M=100**, → 1.93×. Measured A/B at M=3, both arms
run: **131.6 s → 93.1 s = 1.41×**, which is **96% of the 1.47× achievable at that M** (the residual is
the parent becoming a 15th process on 14 cores). *1.93× is the limit, not the number a 3-strain batch
sees.*

**Shape of the fix.** `_run_one_model` split into `_prepare_one` → `_submit_one` → `_settle_one`, and
the loop became a **one-model lookahead**: submit `i`'s units, prepare `i+1`, *then* drain `i`. One
deep, never two — after the fix the parent is the bottleneck, so a second lookahead would queue rather
than overlap while holding a third strain's arrays live. The overlap is safe because §1.2's own RNG
rule already earned it: a stream is named by `(model_id, stage, β, chain)` and **never by when it
ran**, so scheduling cannot move a draw — asserted at M=2 in `test_overlapping_the_strains_does_not_
move_a_single_draw`. `_prepare_one` **returns** its failure rather than raising: preparation now sits
next to *another* model's in-flight units, and a raise would sink work that has nothing to do with the
bad strain.

**A timing claim cannot be a test, so what is tested is the shape that produces it** — that `prepare(b)`
precedes `drain(a)`. Both overlap tests were **run against the serial order and fail there** (M10.2e's
lesson: 4 silent skips read exactly like 4 passes). That probe earned its keep immediately: the first
version of the lookahead-depth test **passed on the serial code**, because with two specs depth-1 and
depth-2 emit identical event orders. It takes **three** specs to see the difference, and a two-spec
version of that test asserted nothing at all.

### 1.6.10 (M10.2c) 🔴 The span certificate refuses **12.5% of valid strains**, and only the seed decides

**Found while measuring M10.2c, caused by none of it.** A 3-strain batch over the *same model file*
came back `['complete', 'failed', 'complete']`. The failure reproduces with **no batch, no pool, no
cache and no overlap** — `prepare_model` alone, one `model_id` at a time — so it is a **v1 defect**,
open since M4:

```
GeometryError: the span certificate is not exhaustive
(214/214 probes, 2 inconclusive, complement complete=True)
```

**The only thing that varies is `model_id`, and it varies nothing but the RNG stream.** Measured over
16 ids on one unchanged file: **14 certify, 2 do not — 12.5%.** Every run agrees on the answer:
`d = 46`, `max_width ≈ 1.85e-12` (i.e. the complement is flat to 12 digits, every time). The
certificate **computes the right geometry and then declines to say so**, because 1–2 probes of 214
were noise-swamped.

| `model_id` | exhaustive | inconclusive / 214 | d | max_width |
|---|---|---|---|---|
| 14 of 16 | ✅ | 0 | 46 | ~1.85e-12 |
| `strain_1` | 🔴 | **2** | 46 | 1.869e-12 |
| `strain_11` | 🔴 | **1** | 46 | 1.838e-12 |

**This is M9's lesson for the fourth time, and the first time it is load-bearing for the package's
purpose**: *a bar a valid input clears only 7 times in 8 is not a tolerance, it is a coin flip.* The
other three instances were a test bar (§1.6.8), an R̂ bar (§1.6.8), and `certify_reachable_mass_balance`
(the standing *Carried, not chased* item). **This one is different in kind: it is not a flaky test, it
is a batch of 100 strains silently returning 88.** `metabolicSubcommunities` is exactly that batch.

**What is *not* yet known — and the arithmetic that would settle it is not done.** `exhaustive` is
`failing is None and complete and not capped and n_inconclusive == 0`. A probe is inconclusive when its
noise swamps the configured tolerance (`NOISE_SAFETY`, §1.6). So the open question is whether an
inconclusive probe on a complement direction that **every other seed measures at ~1e-12** is evidence
of anything at all, or whether the gate is conflating *"this probe was uninformative"* with *"the span
may be incomplete"* — those are different propositions, and `complement complete=True` says the sweep
was not truncated. **Do not "fix" this by widening the tolerance or by retrying the probe until it
passes** — that is choosing the bar to get the verdict, which §1.6.7's fourth review round already
rejected once. Per this plan's own rule, the remedy here is a **hypothesis**: measure whether an
inconclusive probe ever coincides with a genuinely incomplete span before touching the gate.
Deliberately **not chased in M10.2c** — one milestone at a time — but it outranks the remaining M10
extensions, and `geometry.exhaustive_span_certificate=false` is a *workaround that disables the
check*, not a fix.

### 1.6.11 (M11) 🔴 The package runs on **4 of the 40 strains it exists for** — and §1.6.10 was the *smallest* of four failures

**Every measured premise in this plan is a claim about one anaerobe.** `models/` holds one file.
*Bifidobacterium adolescentis* is the **only anaerobe of the 40** curated strains in
`metabolicSubcommunities/models/gapfilled/method_3_curated`; the other 39 are aerobes, and the first
one measured has **479 free reactions to Bifido's 260**. d = 46, n_free = 260, 214 probes, the "12.5%
of strains" of §1.6.10, the worker sweep, §1.2's "at d≤55 sequential wins by default" — all of it is
one organism. The gate-fragility *pattern* was diagnosed with real precision (§1.6.8, §1.6.10) and its
*incidence* was measured on a sample of one.

Swept `build-geometry` over all 40 with the default config — verified to be the intended input: the
repo's example model is **byte-identical** to the curated one, and `ModelSpec` carries only
path/biomass/id, so there is no medium to apply.

| outcome | strains | recorded? |
|---|---|---|
| ✅ succeeds | **4** (10%) | — |
| 🔴 `blocked/moving split is ambiguous` | **24** (60%) | **nowhere** |
| 🔴 `kUnknown` on the `flux_only` LP | **9** (22.5%) | **nowhere** |
| 🔴 span certificate not exhaustive | **2** (5%) | ranked **#1** (§1.6.10) |
| 🔴 `kUnknown` on `reachable_mass_balance` | **1** (2.5%) | *Carried, not chased* |

**The tracker's #1 was the smallest of four, and the two largest were unrecorded.**

#### The mechanism, after three `/collab` rounds (13 points conceded, two of my remedies killed)

> **Primal quality is history-sensitive under warm starts, while these dual constructions stay sound
> independently of it — but their *tightness* varies, and must be judged by the resulting bound.**

That is Codex's wording and it is narrower than mine. My "chaotic primal, *good duals*" is falsified
by this milestone's own FVA data, which shows duals that are sound but **loose**. Measured
(*B. pumilus*, n_free = 479): the FVA sweep dies at column 468 after **938 warm-started solves**,
while the *identical* LP on a fresh instance returns `kOptimal` in 117 iterations. Measured
(*E. gilvus*, n_free = 446), decomposing `dual_upper_bound` into raw + allowance, warm vs cold:

| idx | WARM `U` | raw | COLD `U` | raw | warm/cold |
|---|---|---|---|---|---|
| 260 — the "narrowest **moving**" | 3.3811e-09 | **3.3358e-09** | **4.5297e-11** | **0.0** | **74.6×** |
| 307 — genuinely moving | 5.1093e-01 | 5.1093e-01 | 5.1093e-01 | 5.1093e-01 | **1.000×** |

**Two necessary causes, not one** (Codex, r2 — and this is why the fix is not "fix the solver"):

1. **Warm reuse supplies a loose certificate.** Weak duality is sound for *any* multipliers, so a
   stale-for-this-objective `π` gives a true but slack bound.
2. **The classifier turns absence of a proof into proof of the opposite.** `dual_upper_bound`
   *deliberately* promises soundness for arbitrary duals — so a loose certificate is an
   **anticipated input**, not a solver violation. But `blocked_reactions` reads `U > blocked_tol` as
   **"moving"**, when it licenses only *"not certified blocked by this dual."* Two states where three
   exist. **A perfect solver would not fix this.**

**The guard was right to refuse.** Cold, *L. lactis* classifies **4 more** reactions blocked
(126 → 130) and *Latilactobacillus* **8 more** (92 → 100) — so the warm mask calls "moving" a set of
reactions the cold instrument certifies flat, and §1.4.1 is explicit that carrying a blocked
coordinate's noise into the direction space is what "divides into a chord limit of order 0.03–0.5,
squarely inside the legitimate chord". It failed closed; it merely blamed the tolerance instead of
the instrument.

> ⚠️ **This paragraph must not be read further than it goes, and its first draft was.** *(Codex,
> M11.1 review — corrected before the build, having already been committed.)* A changed `n_blocked`
> does **not** imply a changed `d`: basis discovery projects candidates through `DirectionSpace`, so
> a reaction wrongly called "moving" may still contribute nothing to the basis. **The census measured
> the classifier's mask, not the basis** — so "warm was about to admit noise *dimensions*" is
> unsupported, and "0 regressions" is a statement about the mask, not about L3. What is measured is
> that the *mask* differs. Whether `d`, the basis, the support hashes or the span certificate differ
> is **M11.1's gate**, not an established fact. Nor does *Hafnia*'s `U ≈ 2e-9` establish that its
> true width is near 2e-9 — only that this certificate cannot resolve it at this arithmetic floor.

#### The paired census (Codex refused to let one model generalise, and was right)

40 models, warm vs cold FVA, controls included:

| warm → cold | count |
|---|---|
| AMBIGUOUS → **OK** | **22** |
| `kUnknown` → **OK** | **9** |
| OK → OK (controls) | 7 |
| AMBIGUOUS → AMBIGUOUS | **2** |
| **regressions (OK → not-OK)** | **0** |

So: **the mechanism explains 22 of 24**, not "all". The residue is both *Hafnia alvei* strains — the
two largest models, `n_blocked` **identical** warm and cold, `U ≈ 2e-9` against a 1e-9 bar. Cold
cannot help them: they are **genuinely unresolved**, which is exactly what the third state is for.
**And that is the acceptance criterion**: the disease was *a gate whose verdict depends on the RNG
stream rather than on the input*; after the fix the only refusals are two strains of one species,
**deterministically** — a property of the model, which is what a gate is supposed to test.

#### The design (Codex's, adopted verbatim — mine was unsound)

> **Use primal signals for primal witnesses, dual bounds for upper-bound claims, and achieved bound
> tightness for acceptance.**

- **`blocked_reactions` gets a third state**, because a sound loose bound cannot decide "moving".
  `kUnknown` yields *no witness*, so the crash (9) and the misclassification (22) are **the same
  case**: unresolved → escalate to one cold solve → if still unresolved, **report the resolution
  rather than choose a class**. Not "retry until it passes": the acceptance criterion never moves,
  and the retry is a *declared different computation* (the inherited basis is removed), bounded at
  one. Fixed-period `clearSolver` is **rejected** — there is no justified `K`, and reaction 380 is
  tight *after* loose predecessors, so this is path sensitivity, not age.
  - ⚠️ **The two states are not symmetric, and calling them both "certified" was my error** (Codex,
    M11.1 review). `U_i ≤ T` rests on weak duality and assumes nothing of any returned point, so
    **BLOCKED is certified**. The other side has only `W_i = v̂⁺ᵢ − v̂⁻ᵢ` from points that are merely
    *tolerance*-feasible, and turning a constraint residual into a distance from the exact feasible
    set needs a **Hoffman constant this package does not have** — a fact `probe_direction` already
    records at its own noise bar. So the state is **resolution-qualified MOVING** (`L_i > T` strictly,
    with `L_i = W_i − δ_i` rounded outward, `δ_i` the module's existing `NOISE_SAFETY·2·admitted`
    bar), never "certified moving". This is §1.4.1's own honesty — *"numerically fixed at resolution
    `blocked_tol`, not provably constant"* — finally applied to **both** sides of the split. And the
    fallback I reached for instead (treat every `U_i > T` as unresolved) is **unworkable**: nothing
    would ever supply a lower bound, so every genuinely moving reaction would be unresolved forever.
  - 🔴 **`ranges[i] = max(dual_bound, primal_width)` must go, and not for tidiness.** `U` and `L`
    *bracket* the true width, so a `max` over them **silently repairs a contradiction**: if the
    tolerance-qualified `L_i` ever exceeds the rigorous `U_i`, the arithmetic contract is broken and
    it must be **loud**, not maxed away. Store them separately and refuse on `L_i > U_i`.
  - **`min_separation` becomes a diagnostic, and my rationale for that was wrong too.** UNRESOLVED
    asks whether an interval *straddles a fixed threshold*; separation asks whether `d` is
    *sensitive to moving* the threshold. They are different properties, so retiring the guard
    genuinely **forfeits the tolerance-stability claim** — acceptable only because this module is
    already declared resolution-bounded, and only if the diagnostic is recomputed from the certified
    interval endpoints rather than today's conflated `ranges`. `_check_blocked_span` and the
    complement certificate stay **gates**.
  - 🔴 **The escalation must be *fully* cold — a fresh instance per solve — and the census is what
    taught me that.** My first cut passed one fresh instance for a reaction's `max` and `min`, so
    `min` warm-started off `max`'s basis: one step of inherited history, which is the very thing the
    escalation exists to remove. Measured, it falsely refused **10 of 40** strains; per-solve-fresh
    refuses **2** (both *Hafnia*). The discriminator that proves the fix is not just "the number went
    down": *Hafnia* stays unresolved either way (`U = 2.05e-9`, admitted `1.5e-12`, so the allowance
    is `3e-11`, not inflated — the *dual bound itself* is genuinely above the bar), while *L. brevis*
    resolves the instant the second solve stops inheriting the first's basis. The fix separates the
    warm-start artifact from the real certificate floor exactly.
  - ⚠️ **What the gate actually is — and I stated it too strongly first** (Codex, closing review).
    "No reaction unresolved after escalation" is **not** met: *Hafnia*'s two reactions have a
    **repeatable certificate floor** (`U > blocked_tol` for a near-zero-width reaction), and `U > tol`
    proves only *uncertified-blocked*, never *proven-nonzero*. What M11.1 fixes is the **disease** —
    the *RNG-marginal* refusal. `blocked_reactions` takes no seed, so its refusal is now a
    **structurally deterministic** function of the model, and the seed lottery (§1.6.10, §1.6.11's
    24-strain cluster) is gone. The two *Hafnia* strains are a **separate, deferred** problem: a
    stronger blocked certificate, or documented `blocked_tol` guidance letting the user declare the
    resolution. Calling that "done" would be the overclaim this repo exists to catch.
  - **M11.2 inherits a scope this milestone measured into being.** The 8 `kUnknown@flux_only` + 4
    span refusals are the same degradation in support-LP discovery and the span sweep, which reuse
    the same persistent program. So M11.2 is the span gate change (`resolution ≤ span_tol`, §1.6.11)
    **and** a **shared** escalation mechanism for those stages — escalating *before* an unknown result
    commits to accumulated support/span state, else the stage needs a full cold restart. And the
    shared cause is a **measurement M11.2 must make first** (the paired warm-vs-fresh experiment),
    not a fact the census established.

#### M11.2 — the span gate, the build-wide solve session, and a floor the sweep *split* in two

**Built.** (A) the span certificate's `exhaustive` is `resolution ≤ span_tol`, not `n_inconclusive
== 0` — the flatness claim rests only on each probe's rigorous `width_upper`, so a noise-swamped
probe (a *primal* discovery signal) is no reason to refuse (§1.6.10's open question, answered *no*).
One extracted `_span_resolution` feeds both the report and the gate. (B) a build-wide `_SolveSession`
owns the LP instance: warm until the first `kUnknown`, cold-only after — fixing the leak Codex found,
where M11.1's `warm = None` was function-local to `blocked_reactions` while `build_geometry` kept
warm-starting the later stages off the same degraded instance. A `resolution > span_tol` refusal is
re-confirmed by a **fully-cold re-sweep** before it is final.

**Measured, build-geometry over all 40: OK 4 → 20 → 30; `kUnknown@flux_only` 8 → 0** (the session
eliminated every one; the paired warm-vs-fresh experiment confirmed each is warm-start history, as
required before building). The control (Bifido) never leaves the warm path (`n_cold_solves = 0`).

**The closing `/collab` found four places the diff did not match the approved design, all fixed:**
(1) `blocked_reactions` caught *every* `LPNotOptimalError`, not only `kUnknown` — an infeasible or
unbounded status (a model verdict a fresh basis cannot change) would have been silently cold-retried;
`_reraise_unless_kunknown` now re-raises all but `kUnknown`. (2) the cold pair bypassed the approved
`solve_fresh_once`; it uses it now. (3) `degraded_at`/`n_cold_solves` and `n_lp_solves` counted
different solve populations without saying so — `degraded_at` (a session index that read 0 when the
degradation was in `blocked_reactions`) became `degraded: bool`, and each counter now documents
exactly what it counts and the serialized-build assumption behind the global one.

🔴 **(4) is the one that changed a conclusion, and it is the reason to run the measurement Codex
demanded.** I had called the 2 span refusals "genuine, deterministic, confirmed cold." A cold
re-sweep *keeps the RNG-discovered basis*, so a different `model_id` gives a different resolution — I
had measured one seed each. The 8-seed sweep **split them**:
  - **pumilus: 8/8 seeds refuse** (resolution 2.35e-9…5.20e-9, d = 88). A **genuine √k floor** — the
    axis-wise complement certificate cannot certify it to 1e-9 under *any* seed. `resolution =
    √k·max_width` with `k = n_free − d = 391` and `max_width ≈ 1.4e-10`; the √k (subadditivity, §1.4)
    is not optional, so this is a real limit of *this* certificate, deferred like *Hafnia*.
  - **Liquorilactobacillus: 3/8 seeds *pass*** (resolution 7.67e-10…4.57e-9, d = 56). **RNG-marginal,
    not a floor.** `max_width` is a property of the *discovered basis*, so the resolution straddles
    span_tol by seed — §1.6.10 resurfacing at the *resolution* level. The fix is a **tighter,
    basis-independent** span certificate (the true worst width over `range(B)ᗮ` is basis-independent;
    `√k·max_width` over an arbitrary orthonormal basis is a loose, basis-dependent estimate of it),
    which is genuine new work, **not** the gate change M11.2 made. Deferred.

**So M11.2 closes on its stated scope** — the `kUnknown` crashes and the noise-swamped span refusals
are fixed — and leaves a refined, honest residue: pumilus (a genuine √k floor) and Liquorilactobacillus
(the √k certificate's basis-dependence). The "samples are valid to resolution `R`" contract (accept a
coarser-than-`span_tol` geometry and *report* `R`) is a **separate milestone**, because a small
orthogonal-width bound does not by itself bound the sampled distribution's error — a thin polytope's
cross-sectional volume can vary strongly along the retained directions, and in total variation the
slice law can be singular against the full-dimensional target (Codex). *Getting a bar to pass is not
the same as earning the claim behind it — the milestone's own recurring lesson, one level up.*
- **Span gates on `resolution ≤ span_tol`**, replacing `n_inconclusive == 0`. See §1.6.10's open
  question: the answer is **no**, an inconclusive probe is evidence of nothing about the span.
  Measured — the two failing directions are **conclusive** when re-probed cold *and* when re-probed
  after 30 unrelated solves, same basis and direction; and across 46 constructed truncations,
  detection was 46/46 on **conclusive** probes with `n_inconclusive == 0` in every sweep. The two
  signals are **disjoint by ~4.8e4×**. ⚠️ **Dropping the clause alone would be unsound**, and this
  is Codex's best catch: `exhaustive` bounds the resolution **nowhere**, so `n_inconclusive` is all
  that stands between a terrible dual vector's sound-but-enormous bound and a certificate calling
  itself exhaustive. Measured: all 12 seeds — **including both that fail today** — have
  `resolution` 2.6e-11…3.1e-11 against `span_tol` 1e-9, a 37× margin, and the failing seeds sit
  *inside* the passing range. The resolution gate is **stronger** than today's rule and it closes
  the marginal-truncation hole automatically: a missed direction wider than `span_tol` forces
  `resolution > span_tol`. `SpanCertificate.resolution` is currently read by **no gate in `src/`** —
  only by `tests/integration/test_m4_geometry.py:116`. The suite already asserts the bar the code
  never enforced.
- **Reachability gets a caller-specific dual-witness path.** `_reachable_extreme`'s docstring says
  "Weak duality assumes nothing about the returned point: not optimality, not even feasibility", and
  it obeys — reading **only** `row_duals` — then routes through a backend gate that raises before
  `getSolution()`. **It refuses on the one output it never reads.** Measured: HiGHS reports
  `primal_solution_status=INFEASIBLE` but `dual_solution_status=FEASIBLE`,
  `max_dual_infeasibility=0.0`; the bound from those discarded duals is **1.2836396355893382e-12** vs
  **1.2836396131404284e-12** optimal — 8 digits, and *larger* (the sound direction) — and the full
  replay is **CERTIFIED at 3.677e-11, 27× inside the contract**, inside the band of the 15 passing
  streams. ⚠️ **Do not loosen `HighsLinearProgram.solve()`**: `sparse_objective.critical_l1_penalty`
  needs `LPNotOptimalError` to detect a genuinely unbounded `J*`. One caller earns an exemption its
  own math proves. Premises get asserted *before* the solve (finite `Ω`; `row_lower ≤ 0 ≤ row_upper`,
  so `y = 0 ∈ Y`); `kInfeasible`/`kUnbounded` contradict the constructed compact LP and stay hard
  failures. **Refuted as causes, each by measurement**: normalization (`max|E_i|`=5.4e-14, normalized
  nonzeros span 0.0209…1.0), unboundedness, cycling (30 iterations), the fresh `T⁺`, conditioning
  (s6 fails at cond 5643 *and* 5849 while s5 at 6120 passes), and the tolerance — which is
  **non-monotone**: 1 failure at 1e-9, **0 at 1e-8, 3 at 1e-10, 0 at 1e-12**. `simplex_scale_strategy=0`,
  `presolve=off` and `solver=ipm` each clear it, and **reaching for any of them would be choosing the
  option that gives the verdict, on n=1** — the thing this plan has rejected four times.

#### Two process notes

**A remedy is a hypothesis until measured — mine died twice.** (1) I proposed deriving `blocked_tol`
from `dual_upper_bound`'s rounding allowance, on the claim that the 3.38e-9 *was* that allowance. It
is **1.3%** allowance and 98.7% raw bound: the derived bar would sit at ~4.5e-11 and 3.38e-9 would
**still** classify as moving. **The remedy would have failed outright.** (2) I proposed simply
dropping `n_inconclusive`, which would have made the certificate **unsound**. Both were caught by
review, not by me — and (1) died on a measurement I ran only because Codex refused the claim.

**And the primal-lower-bound error is in this file already.** I argued reaction 260 "does not move"
from a primal width of −2.9e-12. `blocked_reactions`' own comment says an LP that stopped short
reports zero for a wide-open reaction; §1.4.2 records M9 walking into the same trap after M4 wrote
the warning. All the measurement licensed was `0 ≤ W₂₆₀ ≤ 3.381e-9`. **Third instance, and this time
against a warning printed ten lines above the code I was reading.**

#### M11.3 — the caller-specific dual-witness path, built

**Built.** `HighsLinearProgram.solve_dual_witness(*, accept)` returns a narrow `LPDualWitness`
(`model_status, run_status, row_duals, elapsed`) whenever HiGHS reports a whitelisted status,
reading the duals `solve()` discards on a non-optimal one — **without** loosening `solve()`, whose
`LPNotOptimalError` `sparse_objective.critical_l1_penalty` still reads to detect an unbounded `J*`.
`_reachable_extreme` calls it with `_REACHABLE_WITNESS_STATUSES = {kOptimal, kUnknown}` and returns
`(bound, from_unknown)`; `certify_reachable_mass_balance` counts `n_unknown_witnesses` onto the
`ReachabilityCertificate` (telemetry, not a gate — the bound is valid for any finite duals).
`ROUNDING_IMPL_VERSION 2 → 3` for the certificate's new field.

**The census's "1" was the tail of a queue, and the fix cleared the whole tail.** The §1.6.11 table
above recorded `kUnknown@reachable_mass_balance` as **1 of 40** because the other 39 failed *earlier*
(blocked-split, `flux_only`, span) and never reached the certificate. Once M11.0–M11.2 cleared those
stages, **6** strains reach reachability and fail there — not a regression, a queue draining onto its
last gate. Measured on all 6 (the premise, before building): the only non-optimal status is
`kUnknown` (never `kUnbounded`/`kInfeasible` across ~2838 solves); `max_dual_infeasibility = 0.0` on
every one; the discarded-dual bound agrees with a cold optimal re-solve to **7.8–9.0 digits**; the
completed certificate is **CERTIFIED 12–27× inside 1e-9**; and the `kUnknown` row is never the
binding row — accepting its duals lets the loop *finish*, it does not change the verdict.

| strain | d | kUnknown solves | warm-vs-cold agreement | worst_absolute | margin |
|---|---|---|---|---|---|
| Rahnella aquatilis | 145 | 1 | 8.3 digits | 7.60e-11 | 13.2× |
| Lentilactobacillus kefiri | 58 | 1 | 8.4 digits | 5.95e-11 | 16.8× |
| Lactococcus lactis BIA2553 | 51 | 1 | 9.0 digits | 3.75e-11 | 26.6× |
| Lactiplantibacillus pentosus | 71 | 1 | 8.2 digits | 6.04e-11 | 16.6× |
| Lentilactobacillus buchneri | 67 | 2 | 7.8 digits | 5.61e-11 | 17.8× |
| Lactiplantibacillus plantarum | 97 | 1 | 8.4 digits | 8.52e-11 | 11.7× |

**`/collab`: design-review DISAGREE ×3 (all adopted), closing AGREE.** Codex's three: (1) return a
narrow `LPDualWitness`, not `LPSolution`, so "holding an `LPSolution` proves `kOptimal`" stays true —
and validate `row_duals.shape` only, since `HighsSolution::clear` can empty the dual vector (verified
at the HiGHS 1.15.1 source); (2) exact `status.name` match, not the substring test that also matches
`kUnboundedOrInfeasible`, with the accept tokens validated and an accurate "expected one of {…}"
message; (3) durable telemetry + the version bump. **Deferred, conceded real:** the
objective-normalization residual `Σ|E_i − ‖E_i‖·unit|·Ω` is uncharged and the final `*norm`/`+offset`/`max`
round inward — but it is **dual-independent** (identical for `kOptimal` and `kUnknown`, so orthogonal
to this change) and **pre-existing** (M9), and measured at **2.7e-25 / 4.2e-26** (~16 orders below the
contract, ~8e-17 relative). M11.3 changes zero bound *values*; the residual hardening is its own small
step. (Codex caveat: the measurement is prioritization evidence, not a universal bound over future models.)

🔴 **Result: build-geometry OK 34 → 40 of 40 on this machine** — but state the two parts honestly. The
**6 reachability fixes are durable and machine-independent**: the fix removes the failure mode, so
those strains build whether or not a reachable solve degrades to `kUnknown`. The **other 4** (2
*Hafnia*, pumilus, Liquorilactobacillus) that §1.6.11 deferred were already passing in the M11.3
baseline (**34, not 30** — the pre-M11.3 sweep this session) as **basis/RNG-marginal** per §1.6.10; on
another basis they can flip. So "40/40" is earned for the 6 and observed-here for the 4. **979 tests
green; ruff + mypy clean. And geometry is still only the first stage — rounding, the pilot DAG and the
sampler have never run on an aerobe** (M11.4).

### 1.7 λ is scale-referenced: `λ = λ̃ · λ*`  *(M3 finding; decision SETTLED)*

`J(v) = μ(v) − λ·C(v)` compares a **biomass flux** with a **sum of hundreds of absolute fluxes**.
Those two quantities are not on the same scale, and their ratio is a property of each model:

| | example model (Bifido) | toy network |
|---|---|---|
| `μ_max` | 41.63 | 10.0 |
| `C(v)` at the growth optimum | ≈ 4.5 × 10⁴ | 4.0 |
| **critical λ\*** | **1.89 × 10⁻³** | ∞ (cannot collapse) |

Above `λ* = max_v μ(v)/C(v)` the LP optimum is **exactly the origin**: `v = 0` is feasible
(`S·0 = 0`), it costs nothing and earns nothing, and that beats any growth whose L1 cost outruns its
biomass. On the example model this means:

- our default `l1_penalty = 1.0` is **529× past the cliff**;
- the spec's own suggested `l1_penalty = 0.01` (§8) is **5.3× past it**.

At those values `J* = 0`, `v* = 0`, and every downstream stage — `s_J`, the β-ladder, the reweighting
loop — would tilt toward a distribution concentrated on *no metabolism at all*. **The LP is not wrong
when this happens; `J` is.** Nothing inside the LP can tell: status optimal, residual zero, `z = |v|`
exactly. Only `μ_max` standing next to `μ(v*)` gives it away, so `solve_sparse_objective` always
computes both and `SparseObjectiveSolution.is_sparsity_dominated` flags it.

The collapse needs a feasible origin. This model has **no forced-flux reaction at all** (no `ATPM`
lower bound), so it retreats to zero; a model with a maintenance demand pinned above zero cannot.
That is also why the toy network cannot reproduce the failure — `FIX = 2.0` keeps it alive — and why
it took the genome-scale model to find it.

**Decision (settled 2026-07-13): λ is scale-referenced.** The config takes a **dimensionless `λ̃`**
(`objective.l1_penalty_scaled`, default 0.5) and the raw penalty is resolved *per model* as

```
λ = λ̃ · λ*        λ* = max_{v ∈ P} μ(v)/C(v)        (resolve_objective)
```

- `λ̃ = 0` is plain FBA; `λ̃ → 1` is the most sparsity pressure the model can carry while still
  growing. `λ̃ ≥ 1` is **refused** when the origin is feasible (it is a guaranteed collapse), and
  allowed when it is not (a forced-flux model has no cliff).
- **`λ*` is computed exactly by one LP**, not by a search. `max μ/C` is a linear-fractional program;
  the Charnes–Cooper substitution `y = v·t, t = 1/C(v)` linearizes it into "maximize `μ(y)` subject
  to a unit cost budget `C(y) ≤ 1`" — the bounds homogenize into rows `l·t ≤ y ≤ u·t`, and the
  absolute value linearizes with the same `z ≥ ±y` trick as §12. Verified against a 40-step
  bisection (agrees to 8 figures) and against a toy whose `λ* = 1/2` is derivable on paper.
- **No hidden scaling** (spec §3.6): `λ̃`, `λ*`, the raw `λ`, and `origin_is_feasible` all go into
  the manifest, so the raw λ the mathematics used is always recoverable.

Why this and not a raw λ: **the cross-model comparison is the point of the batch design** (§1.1 —
*"do two species retain different amounts of metabolic flexibility at comparable selection
pressure"*). λ̃ = 0.5 resolves to λ = 9.4e-4 on the Bifido model and λ = 0.25 on the toy — a factor
of **265** — because their μ/C scales differ by that much. A shared *raw* λ would have meant wildly
different selection pressures across strains while looking, in the config file, like a controlled
comparison. Measured λ̃ ladder on the example model: `λ̃ = 0 → 100%` of μ_max retained, `0.25 → 95%`,
`0.5 → 60%`, `0.9 → 30%`. A dial, not a trapdoor.

**Settled by M7 — λ is re-resolved every iteration (`λ_k = λ̃·λ*(w_k)`).** Reweighting changes `w`,
and `λ*` is a function of `w` (doubling every weight halves `λ*`), so M7 had to choose whether the raw
λ stays frozen at its base-weight value or is re-resolved from the current weights. **Measurement
closed it, not preference:** one reweighting step moves `λ*` from 1.9e-3 to ~4e2 (default clip) or
~2.3e5 (wider) because `C_w` changes *units* — a sum of absolute fluxes becomes very nearly a count of
active reactions. Freezing λ collapses the effective pressure `λ/λ*(w)` from 0.5 to ~4e-6 **and
crashes M3's `z == |v|` LP gate by the second iteration** (deviation 25 at the default clip). So λ is
re-resolved: `λ̃` stays the user's dial and goes on meaning the same selection pressure across the loop
and the batch. This also makes the median renormalization a mathematical **no-op** — `w → cw` sends
`λ* → λ*/c`, so `λw` (the only thing `J` uses) is invariant — which is why step-4 normalization is a
*conditioning* step that cannot move the target, and why a frozen λ would have made it a *modelling*
step that rescaled the pressure by an arbitrary median every iteration. Recorded in
`.collab/specs/collab-outcome.md` § M7.

---

## 2. Milestones and acceptance gates

"Build from the mathematics outward": the 1D math oracle (M2) and a packaging spike (M0) come
**before** any parallelism or cache complexity. Each gate must pass before the next milestone starts.

| # | Milestone | Deliverables | Acceptance gate |
|---|---|---|---|
| **M0** | Platform & packaging spike | `uv` venv, wheel-only install of highspy/cobra/numpy, `pip install -e .`, import + load example model + solve one native-array LP, verify multiprocessing + thread-limit env | Installs on aarch64/Jetson **from wheels only**; example model loads; 1 LP solves; production core imports **no scipy**; `uv tree` pinned |
| **M1** | Canonical + reduced IR | load/validate/freeze order, native CSC (no scipy), content hashing + provenance, **mandatory l==u elimination** into reduced IR w/ `v_full=R·v_red+c`, `model inspect` CLI | hand-checked CSC on toy; exact full-model reconstruction; elimination equivalence (feasible-set + objective) on toy |
| **M2** | 1D kernel (math oracle) | chord, breakpoints, segment masses, categorical selection, stable truncated-exp inverse-CDF | analytic + property tests across κL ∈ {0, ±1e-16, ±1e-12, ±1e-8, ±1, ±100, ±1000}; continuity at breakpoints; nonincreasing slopes; t=0 / endpoint / duplicate / one-ULP / narrow-chord cases |
| **M3** | Native LP layer | flux-only LP, (v,z) sparse-objective LP, biomass-only diagnostic LP, direct-`J` verification, `z=|v|` checks, one-shot solution extraction | solver objective == direct `J`; feasibility on degenerate toys; z=\|v\| within tol; no scipy |
| **M4** | Affine geometry | sequential warm-started basis discovery (scaled active coords), center from support points, **deterministic span certificate**, geometry diagnostics, memory guard | known toy dims recovered; **truncated basis rejected**; ‖S·diag(s)·B‖≈0; scale-sensitive narrow example classified right; dim-0 singleton path returns constant sample |
| **M5** | Rounding + β=0 sampler | support-covariance Cholesky rounding (ridge escalation), coordinate hit-and-run at β=0, multi-chain, feasibility + convergence diagnostics | uniform analytic targets reproduced; transform-invariance of moments; positive chords at start; ‖ST‖≈0; **zero inner-loop HiGHS solves** |
| **M6** | Positive-β maxent sampler | exact piecewise-exp line conditional, explicit β-ladder, objective traces (μ,C,J, norm log-energy), concentration tests | truncated-exponential + truncated-Laplace analytic targets; mean `J` nondecreasing in β within MC uncertainty; large-β stress; 1D quadrature cross-check in reduced coord |
| **M7** ✅ | Reweighted-L1 (frozen weights) | iterative reweighting `w_r ← w_base/(\|v_r\|+ε)` with clipping + median-renormalization, save every weight vector + LP solution, **freeze final weights before sampling**, rebuild objective/LP-optimum/`s_J` (L2 cache) from frozen weights. **λ re-resolved each iteration** (`λ_k = λ̃·λ*(w_k)`, §1.7); every `s_J` input keyed on objective+polytope (§1.6.5) | deterministic weights for fixed seed; active-set + **weight fixed point** converge; weights frozen ⇒ objective `J` unchanged during MCMC (reweighter cannot import sampler); labeled experimental (not exact cardinality); sampler reproduces analytic targets under the reweighted `J`. **PASSED 2026-07-16** (733 tests; `/collab` 5 rounds AGREE) |
| **M8** ✅ | Cache, restart, batch orchestration & production | 4-layer cache, per-chain markers + writer-claim locking, atomic rename + fsync, **batch runner over a models manifest**, one global process pool over `(model, β, chain)`, worker thread-limit env, per-model run dirs + **cross-model aggregation**, manifests + diagnostics + `COMPLETE` | kill-and-resume resumes only missing `(model,chain)` units; partial batch yields valid cross-model tables; concurrent-writer safe; corrupted-artifact rejected; same-env deterministic traces; full batch runs on ≥2 strains with documented resources. **PASSED 2026-07-16** (content-addressed cache store with atomic-mkdir writer claim; `spawn` pool workers import no solver; serial==pool byte-identical; L0 key made content-addressed) |
| **M9** | Performance & GSMM hardening | `benchmark.py` (new module) + `maxent benchmark` CLI → [benchmarks/M9_REPORT.md](benchmarks/M9_REPORT.md); worker-count sweep {1,2,4,7,14} by **ESS(J)/wall-sec**; allocation + sort profiling; `reduced` storage-mode validation; **the reachable-state mass-balance certificate (§1.4.2)** — scope added mid-milestone when the benchmark's own worker sweep could not run | benchmark report produced; all performance assertions hold (no per-step HiGHS, no scipy, no Python loop in chord, no element-wise highspy extraction, no full reconstruction every step) |
| **M10** | Deferred extensions | **(1) pilot rerounding + pilot-based `s_J` — DONE**, as one DAG (bootstrap `T₀` → geometry pilot → `T₁` → scale pilot → `σ̂₀`), `energy_scale="pilot_sd"` additive beside `warmup_range` (§1.6.6). **(2a) wire the DAG into `batch`/CLI — DONE**; the recorded "fork §1.1 does not settle" was plain/code drift (§1.6.7). **(2b) key the pilots into the cache — DONE** (§1.6.6b, §1.6.7): the pilots are the DAG's only expensive node (19.3 s vs geometry's 1.17 s). **(2c) overlap `prepare_model` with sampling across models — DONE** (§1.6.9): §1.2 did mandate it *and* the arithmetic backed it, but only at a real ladder — at the `betas=(0.0,)` default the same overlap is worth ~5%, and the default is what a reader would have measured. A one-model lookahead (submit `i` → prepare `i+1` → drain `i`); measured A/B at M=3 **131.6 s → 93.1 s (1.41×**, 96% of the 1.47× achievable at that M; asymptote 1.93×). The recorded "two-phase pool dispatch" was indeed the **weaker** remedy — but for an unstated reason: once prepare overlaps sampling the pilots are *free*, so pooling them adds **0.7%**. **(2d) L0 cached — DONE** (M10.2d): warm `prepare_model` 1.21 s → 0.645 s, and a warm run never imports cobra. Then: β→performance calibration (spec §22.3, now cheap — `q(β)` and `r_eff(κ)` are already computed); parallel tempering; slice line kernel; downstream mode-feature extraction | each behind its own tests; none alters the validated v1 target distribution. **(1) PASSED 2026-07-16** (37 new tests; `/collab` 4 rounds AGREE; ladder closes 75.8% of the gap at β=16 vs 13% before, cond(C_q) 2.57× better — *see §1.6.6b: the 2.87× recorded here was pre-M10.2a and stale*). **(2b) PASSED 2026-07-17** (18 new tests; `/collab` 3 rounds — round 1 refuted my payload design, round 3 found a hit/miss asymmetry **inside my own repair**; `prepare_model` 22.9 s → 1.2 s warm, `T₁`/`s_J` bit-identical). **(2c) PASSED 2026-07-17** (4 new tests, both overlap tests proved non-vacuous against the serial order; 1.41× at M=3, draws bit-identical to the un-overlapped path). 🔴 **(2c) also measured a live defect it did not cause: the span certificate refuses 12.5% of valid strains (2 of 16 `model_id`s) — see §1.6.10** |

| **M11** | 🔴 **v1 on the real batch** — the 40 curated strains the package exists for (§1.6.11) | **(0) the L3 lookup key names its solver — DONE** (M11.0): `highs_backend.solver_identity` (`BACKEND_IMPL_VERSION` + the installed HiGHS version, read from `importlib.metadata` so the key never imports the solver) folded into `batch.geometry_cache_key`. **Forced first**: every remaining item changes solve bytes, and without this a bumped backend was a *hit* that then died on the content key (§1.1 wants a miss), while the HiGHS version was in **neither** identity — a `uv sync` silently reused another solver's geometry. Then: **(1)** `blocked_reactions`' third state + one bounded cold escalation (22 + 9 strains); **(2)** span gates on `resolution ≤ span_tol` (2 strains, and the ~12% seed lottery); **(3)** reachability's caller-specific dual-witness path (1 strain); **(4)** re-run the 40-strain census; **(5)** release: LICENSE, README/docs, a manifest that reads the real `strains.tsv` | `build-geometry` succeeds on the curated 40, or refuses for a reason that is a property of the **model** rather than of the RNG stream — the disease being a gate whose verdict the seed decides. Every remedy names the measurement that confirms its premise *first*. **(0) PASSED 2026-07-17** (962 tests, +4; the two key tests fail on the shipped code with an identical digest, and the subprocess isolation test is proved non-vacuous by sabotage). **(1) PASSED 2026-07-17** (965 tests; `/collab` DISAGREE×3, all adopted): the three-state classifier + fully-cold escalation takes **build-geometry OK from 4 → 20 of 40**; `blocked_reactions` now refuses only the **2 *Hafnia*** strains, structurally deterministic (it takes no seed) — the RNG-marginal *disease* is fixed. 🔴 The stated "zero-unresolved" gate is **not** met (Codex): *Hafnia* is a repeatable certificate floor, deferred, not closed. **(2) PASSED 2026-07-18** (span gate `resolution ≤ span_tol` + a build-wide solve session; `/collab` design-review + DISAGREE×1 closing, 4 discrepancies fixed): **build-geometry OK 20 → 30 of 40**, `kUnknown@flux_only` **8 → 0** (the session fixed the leak where M11.1's abandonment did not cross stages), the §1.6.10 noise-swamped span refusals gone. The 8-seed sweep Codex demanded **split** the 2 span residues: **pumilus is a genuine √k floor** (8/8 refuse), **Liquorilactobacillus is RNG-marginal** (3/8 pass — a basis-dependence needing a tighter certificate). Both deferred; the "valid to resolution R" contract is a separate milestone. **(3) PASSED 2026-07-18** (reachability's caller-specific dual-witness path; `/collab` design-review DISAGREE×3 all adopted, closing AGREE): `solve_dual_witness` reads a `kUnknown` solve's sound duals without loosening `solve()`; **build-geometry OK 30 → 40 of 40** — the **6** `reachable_mass_balance` failures (the census's "1" was the tail of a queue the earlier stages masked) cleared **durably** (the fix removes the failure mode). Premise measured on all 6 first: only `kUnknown` ever arises, duals dual-feasible and 8-digit-tight, CERTIFIED 12–27× inside 1e-9. The 4 deferred (2 *Hafnia*, pumilus, Liquorilactobacillus) pass on this machine as basis-marginal. Deferred, conceded real: the dual-independent objective-normalization residual (~1e-25, its own step). ⚠️ **Still measures geometry only** — rounding, the pilot DAG and the sampler have **never run on an aerobe**, and §1.2's "at d≤55 sequential wins" is a claim about the one anaerobe. **(4)** the end-to-end census (rounding/pilots/sampler on an aerobe) and **(5)** release remain |

### 2.1 What M6 ships, and what it does not  *(M6 finding; SETTLED — and it constrains M10)*

**M6 ships a validated maximum-entropy *engine* with an *uncalibrated* β scale.** The distinction is
not pedantry; it was forced by measurement.

The tilt is exact — analytic targets: a truncated exponential, an asymmetric truncated Laplace with an
interior bend, a coupled `(1−x)·e^{γx}` marginal, and a reduced-coordinate quadrature cross-check that
evaluates `J` straight from its definition. Its **magnitude** is pinned against the linear-response
identity `dE_β[J]/dβ = Var_β(J)/s_J`. Mean-`J` rises monotonically along the ladder, with R̂(`J`)
confirming the rise is not retained initialization. All of that is about the *sampler*, and it holds.

But on the example model `s_J = 31.3` while the β=0 chain explores only `sd(J) = 2.6`. The warm-up
range is taken over the geometry's **support-LP vertices** — extreme points, where the L1 cost is
enormous and `J` runs down to −28 — while the chain lives in the interior at `J ≈ −12`. So `s_J` is
calibrated to a range **12× wider** than the one actually sampled, the linear response is only 0.22
per unit β, and **the top rung of spec §22.1's own ladder (β = 16) closes just 13% of the gap to
`J*`.** The ladder is a fine-tuning knob, not a switch.

That is a fact about the **calibration**, not the sampler, and the remedy is one spec §22.2 already
gestures at when it says to set `s_J` from "support **or pilot** points": **M10's pilot-based `s_J`**,
which reads the scale off a β=0 pilot chain's own `J` spread (2.6) and would tilt ~12× harder for the
identical ladder. It changes *what β names*, not the target at any given β — the distribution M6
validates is untouched either way.

**Consequently: M10's pilot-based `s_J` is a prerequisite for presenting the β-ladder as spanning
neutral-to-strongly-selected regimes.** Until it lands, a run reports what it measured and does not
pretend the β axis means more than itself. Recorded here rather than left as folklore, because it is
exactly the kind of claim a downstream paper would make by accident.

> ✅ **Discharged by M10 (§1.6.6), but not by the remedy named above.** `sampler.energy_scale =
> "pilot_sd"` closes **75.8%** of the gap at β = 16 where `warmup_range` closed 13%. **The paragraph
> above got the diagnosis right and the cure wrong**: spec §22.2's formula with pilot points buys
> **1.28×**, not the 12× claimed — the "12×" was a ratio between an *anchored range* and a *spread*,
> which are different quantities. The lesson is worth more than the fix: **a deferred remedy is a
> hypothesis, not a plan.** M10 also bounds what may now be said — β is a *local* Fisher-standardized
> coordinate (`I₀ = 1`, `KL ≈ ½β²`), exact at the **estimand** level, and **no scalar `s_J` is a
> universal finite-β axis**. A run reports `q(β)`, `Δ₀`, `G` and `r_eff(κ)` alongside β so the claim
> stays checkable.

---

## 3. Test plan (mapped to gates)

- **Unit** (`tests/unit`): native CSC (starts/indices/values, matvec/rmatvec, malformed rejection);
  COBRA adapter (order preservation, biomass-by-ID, missing/duplicate/NaN/inf detection); sparse-obj
  LP; reduced-IR reconstruction round-trip; chord (positive/negative/zero components, t=0, zero-length
  redraw); piecewise objective (vs direct eval on a grid, continuity, monotone slopes, duplicate/no
  breakpoints); 1D distributions (uniform, `e^{κt}`, `e^{−α|t|}` moments/quantiles with fixed seeds);
  reweighted-L1 (weight-update formula, clipping, median-renormalization, deterministic for fixed
  seed, weights frozen before sampling).
- **Statistical** (`tests/statistical`): 2D box `J=−λ(|x|+|y|)` → two truncated Laplaces, compare
  marginal moments; equality-constrained polygon vs 1D quadrature; mean-`J` monotonic in β.
- **Integration** (`tests/integration`): toy JSON end-to-end (every output file); COBRApy textbook
  model (load→LP→geometry→β=0→β>0→all samples feasible); **no-solver-in-inner-loop** counter;
  kill-and-resume; concurrent-writer; corrupted-artifact rejection; **batch over ≥2 strains** →
  per-model dirs + valid cross-model aggregate (including with one strain deliberately failed).
- **Performance** (`tests/performance`, slow/scheduled): the M9 benchmark suite + assertions.
- **No-SciPy gate**: run the core test subset in a venv without scipy; static scan of core imports.

---

## 4. Packaging (aarch64 / Jetson) — resolved in M0

- **Python 3.11 (locked).** The sibling `metabolicSubcommunities/.venv` runs cobra 0.31.1 on Python
  3.11 on this Jetson today — a proven-good baseline. Pin the uv venv to 3.11 rather than gambling on
  3.12 wheel availability. Reevaluate 3.12 only if a benchmark or dependency demands it.
- **cobra** (0.31.1, platform-independent wheel) pulls the optlang/GLPK stack even for parsing;
  `swiglpk` ships an aarch64 wheel. **highspy** ships a manylinux aarch64 wheel. NumPy too.
- **Gate**: `uv` resolves the *entire* graph wheel-only (no source builds); `uv tree` inspected and
  the full lock pinned; JSON load works on 3.11; loading selects no unwanted solver; the production
  numerical core imports zero scipy modules. Phrase the requirement as "no SciPy in the numerical
  path," and treat "installs without SciPy at all" as an empirical result of M0, not an assumption
  (cobra's extras may transitively want it).

---

## 5. Immediate next step

Execute **M0** now: scaffold the `src/gsmm_compiler` package + `pyproject.toml`, create the `uv`
venv on **Python 3.11**, and prove wheel-only install + example-model LP on this Jetson. That single
spike de-risks aarch64 wheels + the no-scipy numerical path before any math is written.

### Milestone dependency graph (v1 = M0–M9; M10 deferred)

```
M0 spike ─► M1 IR ─► M2 1D-oracle ─► M3 LPs ─► M4 geometry ─► M5 β=0 ─► M6 β>0 ─► M7 reweighted-L1 ─► M8 cache+batch ─► M9 perf
                     (math-first, before any parallelism/cache)                    (frozen weights)   (multi-strain)
```
