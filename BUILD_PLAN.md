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
[L0] Parsed model IR      key = sha256(file) + cobra_version + parser_schema_version
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
```

Rules (all adopted from the collaboration):
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
- **Sampling: process pool over `(β, chain)` units.** Given frozen geometry these are independent.
  A worker receives **only** frozen NumPy arrays (`T_active`, `center_active`, bounds, objective
  arrays, index maps) + a semantic RNG seed. A worker **never imports cobra or HiGHS**.
  - Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` **before** NumPy import (the real
    oversubscription risk in solver-free workers is BLAS/OpenMP, not HiGHS).
  - Workers **write their own** `.npy` files; never ship flux matrices back through IPC.
  - **Benchmark worker count {1, 2, 4, 7, 14}** on the Jetson by ESS-per-wall-second. 14 can lose to
    4–7 under memory-bandwidth/thermal limits; pick empirically.
- **Batch scheduling.** Across many strains the unit of work is `(model, β, chain)`. Process models
  so their per-model geometry (sequential, cached) can overlap the *sampling* of earlier models, but
  feed **one global worker pool** sized once for the machine — never a pool per model, which would
  oversubscribe the 14 cores. Each model gets its own result subdir; a final aggregation stage reads
  the per-model summaries into cross-model tables (§1.1).
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
- **Exactness of the 1D conditional.** `expm1`/`log1p` inverse-CDF is the primary path across all κ.
  The small-|κL| uniform form is only a below-float64-eps series limit, documented — not a silent
  approximation sold as "exact." Sign-aware log-mass formulas (no `log(expm1(x))` for x<0).
- **No snapping.** Never round small sampled fluxes to zero; thresholds apply only in analysis
  (`features.py`), never to chain state.
- **z is LP-only.** Auxiliary absolute-value variables never enter the sampled state.
- **Reproducibility, scoped honestly.** Byte-identical traces are promised only within a locked
  binary + hardware environment. Across NumPy/BLAS/HiGHS/CPU changes, require matching *statistical*
  results + recorded provenance.

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

### 1.7 λ is not a dimensionless knob — the sparsity cliff  *(M3 finding; decision OPEN)*

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

**Open decision (blocks M6, not M4/M5).** β=0 ignores `J` entirely and geometry is λ-independent, so
M4 and M5 are unaffected. Before M6 tilts by `J`, we must settle whether λ stays **raw** (user picks
per model, guarded by the diagnostic) or becomes **scale-referenced** (e.g. `λ = λ̃ · μ_max/C_ref`,
recorded explicitly — no hidden scaling, spec §3.6). The batch/cross-model goal (§1.1: *"comparable
selection pressure"* across strains) argues hard for the second: a single raw λ means a different
selection pressure in every strain, which would make the cross-model comparison meaningless.

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
| **M7** | Reweighted-L1 (frozen weights) | iterative reweighting `w_r ← w_base/(\|v_r\|+ε)` with clipping + median-renormalization, save every weight vector + LP solution, **freeze final weights before sampling**, rebuild objective/LP-optimum/`s_J` (L2 cache) from frozen weights | deterministic weights for fixed seed; active-set converges within tol; weights frozen ⇒ objective `J` unchanged during MCMC (never updated from chain state); labeled experimental (not exact cardinality); sampler still reproduces analytic targets under the reweighted `J` |
| **M8** | Cache, restart, batch orchestration & production | 4-layer cache, per-chain markers + writer-claim locking, atomic rename + fsync, **batch runner over a models manifest**, one global process pool over `(model, β, chain)`, worker thread-limit env, per-model run dirs + **cross-model aggregation**, manifests + diagnostics + `COMPLETE` | kill-and-resume resumes only missing `(model,chain)` units; partial batch yields valid cross-model tables; concurrent-writer safe; corrupted-artifact rejected; same-env deterministic traces; full batch runs on ≥2 strains with documented resources |
| **M9** | Performance & GSMM hardening | benchmark suite (parse→CSC→passModel→first LP→warm-start LPs→sparse LP→geometry→rounding→β=0 sps→β>0 sps→breakpoint dist→output), worker-count sweep {1,2,4,7,14} across the batch, allocation + sort profiling, `reduced` storage-mode validation | benchmark report produced; all performance assertions hold (no per-step HiGHS, no scipy, no Python loop in chord, no element-wise highspy extraction, no full reconstruction every step) |
| **M10** | Deferred extensions | pilot rerounding + **pilot-based s_J** (split bootstrap-geometry → β=0 pilot → {final T, s_J} DAG); β→performance calibration; parallel tempering; slice line kernel; downstream mode-feature extraction | each behind its own tests; none alters the validated v1 target distribution |

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
