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
```

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
| **M9** | Performance & GSMM hardening | benchmark suite (parse→CSC→passModel→first LP→warm-start LPs→sparse LP→geometry→rounding→β=0 sps→β>0 sps→breakpoint dist→output), worker-count sweep {1,2,4,7,14} across the batch, allocation + sort profiling, `reduced` storage-mode validation | benchmark report produced; all performance assertions hold (no per-step HiGHS, no scipy, no Python loop in chord, no element-wise highspy extraction, no full reconstruction every step) |
| **M10** | Deferred extensions | pilot rerounding + **pilot-based s_J** (split bootstrap-geometry → β=0 pilot → {final T, s_J} DAG); β→performance calibration; parallel tempering; slice line kernel; downstream mode-feature extraction | each behind its own tests; none alters the validated v1 target distribution |

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
