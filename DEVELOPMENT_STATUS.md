# Development Status — GSMM-Compiler MaxEnt Sampler

**This file is the single source of truth for build progress.** It pairs with:
- [BUILD_PLAN.md](BUILD_PLAN.md) — design, milestones, acceptance gates (the *what/why*).
- [GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md](GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md) — original math spec.
- [.collab/specs/collab-outcome.md](.collab/specs/collab-outcome.md) — locked cross-cutting decisions.

## ▶ How to resume ("continue package development")

1. Read **Current state** below, then the ACTIVE milestone's row in [BUILD_PLAN.md](BUILD_PLAN.md).
2. Run the **Verify current state** commands — trust observed test results, not just the checkboxes.
3. Execute the next unchecked task under the ACTIVE milestone. **Build one milestone at a time.**
4. 🤝 **Math-critical milestones (M2, M4, M5, M6, M7) require a `/collab` adversarial review** as a
   gate step — Codex independently attacks the correctness of the distribution before the gate closes.
   See the protocol + review targets in [CLAUDE.md](CLAUDE.md). Also use `/collab` for unsettled design
   forks and for non-obvious failing statistical tests (debug mode).
5. A milestone is DONE only when its **acceptance gate passes** (tests green + gate criteria met +
   collab review clean where required). Then: check its boxes, advance **Current state** to the next
   milestone, append a **Session log** line, and `git commit`.
6. Keep [BUILD_PLAN.md](BUILD_PLAN.md) authoritative for design. If a decision changes, update it there
   and record it in [.collab/specs/collab-outcome.md](.collab/specs/collab-outcome.md).

---

## Current state

| Field | Value |
|---|---|
| **Active milestone** | **M6 — Positive-β maximum-entropy sampler** |
| **Status** | ⬜ NOT STARTED (M0–M5 gates passed; 569 tests green) |
| **Next action** | Wire M2's exact piecewise-exp line conditional into the sampler for β>0. The plumbing already exists — `run_chain(beta=..., objective=..., energy_scale=...)` calls `sample_line`, and it *refuses* β>0 without an objective. M6 must supply: the `L1Objective` in **reduced** coordinates, `s_J` from the support points (`warmup_range`), the explicit β-ladder, and the objective traces. |
| **Blockers** | none |
| **Last updated** | 2026-07-14 |

> 🤝 **M6 requires a `/collab` adversarial review as a gate step** — that the sampled law is exactly
> π_β (no hidden approximation); `s_J`/`J*` handling; mean-J monotonicity. See CLAUDE.md.

### ⚠️ Settled by M5: the sampler is correct but this model MIXES SLOWLY — plan the schedule for it

`step_scale_ratio = 0.008`: the shortest rounded axis is 125× shorter than the longest, so coordinate
hit-and-run crawls along it. Measured on the example model, **4 chains × (N burn-in + N sampling)
sweeps**:

| sweeps | max R̂ | min ESS | median ESS | of |
|---|---|---|---|---|
| 500 + 500 | 1.32 | 6 | 29 | 2000 |
| 2000 + 2000 | 1.08 | 26 | 119 | 8000 |
| 3000 + 3000 | 1.08 | 34 | 154 | 12000 |
| 8000 + 8000 | 1.04 | 251 | 397 | 32000 |

ESS grows about linearly and R̂ falls monotonically — the signature of a correct-but-slow chain, not a
stuck one (no coordinate has zero within-chain variance at any length). **The `SamplerConfig` defaults
(1000/1000) are not enough for this model to reach R̂ < 1.01.** Report R̂/ESS with every run and set the
schedule from them, rather than trusting a default.

**The known remedy is spec §17.4 pilot rounding, and it is measured, not hypothetical.** Re-rounding
from the covariance of an actual β=0 pilot chain (instead of the support-LP vertices, which are only
an initial approximation) improves `cond(C)` from 1.56e4 → 5.14e3 and **ESS by ~2.5×** (median
174 → 428 at 3000+3000). It is deferred to **M10** by BUILD_PLAN, deliberately — it is a separate DAG
stage, not a change to the target. If M6/M8 wall-clock becomes the binding constraint, this is the
lever to pull, and it does not touch the validated distribution.

### ✅ Settled by M5: rounding cannot move the target — and `diag(s)` and `L` do different jobs

`range(T) = range(diag(s)·B·L) = range(diag(s)·B)` exactly for **any** invertible `L`, so the ridge is
a free engineering parameter and *transform-invariance is a testable claim*. It is tested: two
transforms whose ridges differ by seven orders of magnitude sample the same flux distribution, checked
in units of **Monte-Carlo standard error** (max |z| < 5, mean z² < 2 over 199 reactions) rather than in
units of σ — the naive "means agree to 0.25 σ" bar failed at 0.39 σ on two runs that are perfectly
consistent, because at ESS ≈ 1% the standard error of a mean is already ~0.1 σ.

**Scaling and rounding are not the same job.** `diag(s)` fixes the axes' *units*: an axis-aligned
1000:1 stretch is absorbed before rounding is reached (measured — the unrounded chords differ by 1.41,
not 1000). `L` fixes their *correlations*, which no diagonal matrix can see. That is where the 41×
improvement in the worst chord comes from.

### ✅ Settled by M4: the sampling dimension is 46, not 55 — and the geometry hands M5 a usable start

**61 of the 260 free reactions cannot carry flux at all.** The model file leaves `l < u`, but mass
balance pins them, so `n_free − rank(S) = 55` is only an *upper bound* and the true affine dimension
is **d = 46** (confirmed by an independent FVA+rank oracle sharing no code with the geometry).

That is not bookkeeping. A blocked reaction is an *exact structural zero* of the direction space, and
a basis row of ~1e-15 there — divided by a centre sitting ~1e-13 outside its own bound, both pure
solver noise — produces **a chord limit of order 0.03–0.5**, right inside the legitimate chord. The
measured chord at the centre was `[−0.54, −0.39]`, which **excludes `t = 0`**; `line_geometry` refused
it, so **M5 could not have started from that centre.** Blocked components are now projected out
exactly. See BUILD_PLAN §1.4.1.

**What M5 inherits, already checked:** every basis direction's chord through the centre contains
`t = 0` with positive length (min 0.018); the centre is *exactly* bound-feasible; and the support
points span all 46 directions (rank 46/46) — so M5's covariance ridge cannot quietly conceal a
singular covariance instead of failing on it.

### ⚠️ What the span certificate does and does not claim (M4 collab, 6 rounds)

It licenses exactly this: **every feasible direction of the exact polytope has its component
orthogonal to `range(B)` bounded in width by `SpanCertificate.resolution`** (2.78e-11 scaled = 5.6e-8
flux units on the example model). It is **not** "cannot under-count a dimension" — a direction thinner
than the resolution can be missed, and `blocked_tol` drops one narrower than itself.

The asymmetry runs the *safe* way: the geometry may **over**-count (admit an ε-feasible direction) but
cannot omit a direction it had the resolving power to see. Over-counting is benign for a sampler — the
chain explores a slightly larger set, and every sample is still checked. Omitting a wide direction
would silently delete part of the support, and no downstream test would ever see the samples that
were never drawn.

### ✅ Settled by M3: λ is scale-referenced (`λ = λ̃ · λ*`)

`J = μ − λC` compares a biomass flux with a sum of hundreds of absolute fluxes; above a
model-specific `λ* = max_v μ(v)/C(v)` the LP optimum is **exactly the origin** and the cell stops
growing. The old default `λ = 1.0` was **529× past that cliff** on the example model. So the config
now takes a **dimensionless `λ̃`** (`objective.l1_penalty_scaled`, default 0.5) and `resolve_objective`
computes `λ = λ̃ · λ*` per model — `λ*` exactly, from one Charnes–Cooper LP. `λ̃ = 0` is plain FBA;
`λ̃ → 1` is maximum sparsity pressure that still grows; `λ̃ ≥ 1` is refused when the origin is
feasible. **Same `λ̃` = same selection pressure in every strain**, which is what the cross-model
comparison needs. Full reasoning: BUILD_PLAN §1.7. **M7 must still decide** whether λ stays frozen at
its base-weight value through the reweighting loop or is re-resolved from the frozen final weights
(`λ*` depends on `w`).

### Platform facts established by M0 (build on these, don't re-derive)

- Python **3.11.15**, aarch64/Jetson. Wheel-only install verified with `uv pip install -e ".[dev]" --only-binary=:all:` — **no source builds**. Pinned in `uv.lock` (47 packages).
- Resolved versions: **cobra 0.31.1 · highspy 1.15.1 · numpy 2.4.6**.
- **SciPy is absent from the venv entirely** — stronger than the gate required (§4 anticipated cobra might pull it transitively; it does not). The no-scipy gate is enforced by `tests/unit/test_no_scipy.py` at both runtime and source level.
- HiGHS accepts native `int32` CSC index/start arrays + `float64` values via `passModel`. The biomass LP matches cobra's own FBA optimum to `rel=1e-6` — the CSC assembly is verified, not merely accepted.
- ⚠️ **highspy attribute reads return Python `list`, not NumPy views** (the pybind layer copies). M3's LP layer must keep its own float64 arrays and extract solutions in one `np.asarray` shot — never element-wise. Pinned by `test_highspy_returns_python_lists_not_arrays`.

### Facts established by M5 (the sampler — build on these, don't re-derive)

- **The β>0 plumbing is already in place and *refuses* to run half-configured.** `run_chain` takes
  `beta`, `objective` and `energy_scale` and passes them straight to M2's `sample_line`; it raises if
  `β > 0` arrives without an objective or with a non-positive `s_J`. M6 supplies the `L1Objective` in
  **reduced** coordinates (not full — the sampler's `v` is length `n_free`), `s_J`, the ladder and the
  traces. It does not need to touch the transition kernel.
- **Time is measured in *sweeps*, one sweep = `d` coordinate updates.** `burn_in`, `n_samples`, `thin`
  and `refresh_interval` are all in sweeps, so a schedule means the same thing across a batch of
  models with different `d`. Counting single updates would make `burn_in = 1000` mean 21 passes here
  and 1000 on a 1-D model.
- **A relative residual needs a *floor* as well as a denominator.** `NativeCSC.relative_residual`
  divides by `max(|S|·|v|, 1.0)`. The floor is not slack: 24 metabolite rows of this model are touched
  by a **single** free reaction, where `residual = |S_ij·v_j|` and `scale = |S_ij|·|v_j|` are *the same
  number*, so an unfloored ratio is identically 1.0 for any nonzero flux. One such row (`cpd02375_c0`,
  both its reactions FVA-blocked) made the sampler report a relative mass-balance error of **1.0** from
  an absolute one of 3.4e-14. **But the floor must not touch a *direction*:** `T`'s blocked rows are
  exactly `0.0`, so there is no noise to divide by, and flooring the transform's own check would only
  weaken it (measurably — on 2049 of 41124 rows). Correct where the noise exists, absent where it can't.
- **`np.min(x, initial=0.0)` is not an empty-array guard.** For a *min* reduction `initial` is a
  candidate, so it floors the answer at 0 — `ConvergenceReport.min_ess` reported an ESS of **0** for a
  sample whose every entry was 8000. Same trap in the mirror direction for `np.max(..., initial=1.0)`.
- **`@dataclass(frozen=True)` freezes the binding, not the buffer.** And `np.ascontiguousarray` returns
  its argument *unchanged* when it is already contiguous and float64 — so freezing an array in place
  can reach back through an alias and freeze the caller's object underneath it. `rounding` copies
  before it freezes.
- **A missing dimension produces no bad numbers, only absent ones.** Nothing in a feasibility check,
  a chord, or a mass-balance residual can see a `T` that has silently lost a column — the chain just
  samples a lower-dimensional slice of the polytope, perfectly cleanly. Hence the explicit SVD rank
  check. M6 and M7 should assume the same about anything they add: *the dangerous failure is the one
  that removes states, not the one that produces wrong ones.*

### Facts established by M4 (the geometry — build on these, don't re-derive)

- **`d = 46`, and the certificate cost 1089 LPs / 1.2 s.** FVA is 520 of them, discovery 140, the
  complement sweep 428. Warm-started simplex; ~4800 total pivots. Cheap enough to leave alone.
- **A width has two ends and needs two instruments.** The *primal* width (objective difference of two
  returned endpoints) is a **lower** bound: it proves a direction exists. Certifying a direction
  *flat* needs an **upper** bound, and only weak duality gives one that assumes nothing about the
  solver — not optimality, not even primal feasibility. Never certify flatness from a primal reading;
  M5/M6 will face the same temptation with `s_J` and the energy traces.
- **A mass-balance residual must be judged on a *relative* bar.** `S·v` sums terms of size ~1e5 here,
  so evaluating it costs ~1e-10 of rounding before any solver error. An absolute 1e-9 bar charges that
  to the solver and fails a perfectly good geometry (measured). `NativeCSC.cancellation_scale`
  (`|S|·|x|`) gives the scale to divide by — and it *cannot* be had from `matvec(abs(x))`, which
  re-applies the signed `S` and cancels all over again.
- **Never divide by a small number that is noise.** Two of M4's bugs were the same shape: a ~1e-15
  basis row divided into a ~1e-13 bound violation gave a chord limit of 0.03–0.5, and a Gram-Schmidt
  residual of ~1e-3 divided into a 1e-12 LP row residual gave a basis error of 8e-10. Both were
  measured, neither was visible in any test that passed.
- **`stream_seed(model_id, stage, β_index, chain_index)`** (in `provenance`) is the RNG keying M5's
  chains must use. It hashes with sha256, **not** Python's `hash()`, which is salted per interpreter —
  a spawn key built from that would name a different stream in every worker.

### Facts established by M3 (the LP layer — build on these)

- **`HighsLinearProgram` is the only place that touches `highspy`**, and it imports it *inside* the
  constructor. So `sparse_objective` can be imported by an MCMC worker without loading a solver
  (§1.2's "a worker never imports HiGHS"). Pinned by a subprocess test.
- **HiGHS adds `lp.offset_` to the reported objective with its sign intact under `kMaximize`**
  (probed, then pinned). That is how the fixed reactions' contribution to `J` — which no LP variable
  can express — reaches the solver's objective, making "solver objective == directly recomputed J"
  one *complete* equation rather than one missing a constant.
- **A `z` column exists only where `λ·w_r > 0`.** A zero-cost `z` has nothing pushing it down onto
  `|v_r|`; it would settle anywhere in `[|v_r|, z_max]` and fail its own `z == |v|` check on a
  solution that is perfectly correct in `v`. So `λ = 0` builds no auxiliaries at all and the LP
  collapses to the flux-only model — which is also the right answer.
- **Solve counting is process-global, and programs can be `freeze()`d.** Per-instance counts would
  let a sampler evade the "no solver in the inner loop" assertion by building a fresh LP. M5's gate
  can now assert *and* prohibit.
- **Warm starts pay off at genome scale**: 50 random objectives on the §14 flux LP re-solve in well
  under the cold-start pivot count. The sequential geometry phase (§1.2) is on solid ground.
- **λ is scale-referenced, and `λ*` is exact.** `critical_l1_penalty` gets `λ* = max_v μ(v)/C(v)` from
  **one** LP — `max μ/C` is linear-fractional, and Charnes–Cooper (`y = v·t, t = 1/C(v)`) turns it
  into "maximize μ(y) subject to a unit cost budget". Don't bisect it. `resolve_objective` is the
  config-driven entry point; `SparseFluxObjective.from_polytope` stays pure and takes a **raw** λ.
- ⚠️ **`λ*` depends on the weights** (doubling `w` halves `λ*`). M7 must decide whether λ stays frozen
  at its base-weight value or is re-resolved from the frozen final weights — it moves `J` either way.

### Facts established by M2 (the line kernel is now a trusted oracle — build on it)

- **The line kernel takes no `J*`.** It cancels out of `p(t)`, and carrying it invites catastrophic
  cancellation. M6 will still need `s_J` and `J*` for *diagnostics* (energy traces), but must not feed
  `J*` into the draw.
- **M5 must NOT redraw a coordinate when a chord is degenerate** — that makes coordinate selection
  state-dependent and breaks stationarity. `sample_line` already returns the correct self-loop; just
  apply it (`y_k += 0`). See BUILD_PLAN §1.6 delta 6.
- **M5 builds the per-coordinate precompute itself.** An unvalidated caller-supplied `support` array was
  removed from `feasible_chord`: a truncated one silently reintroduces the §1.6.1 tolerance bug. The
  precompute must be *derived* from `T`, so the invariant holds by construction.
- **`PiecewiseLinearJ.values`/`evaluate()` return absolute `J` and are for reporting only.** Never form a
  probability from them; the mass path uses `heights` (peak-relative) exclusively.
- Weights, `λ`, `β` and `s_J` are frozen inputs to the kernel. M7's reweighting must complete *before*
  sampling starts — a weight that moves mid-chain retargets every conditional and destroys stationarity.

### Facts established by M1

- **CSC index width is `int32`, not the spec's `int64`.** `highspy.kHighsIInf == 2**31 - 1`, i.e. HiGHS is built with a 32-bit `HighsInt`; int64 arrays are *accepted* but silently narrowed on every `passModel`. `native_csc.INDEX_DTYPE = np.int32`, and the 2³¹ nnz ceiling is a construction-time check. Pinned by `test_highs_index_width_is_what_native_csc_stores`.
- **cobra rejects NaN bounds, inverted bounds and duplicate IDs itself** — but happily accepts **infinite** bounds, which we must reject (an unbounded polytope has nothing to sample). Our own guards still matter for models built/mutated in memory rather than parsed.
- **New module `provenance.py`** (not in spec §6): L0/L1 content keys were needed long before M8's cache exists, and the core must hash arrays without importing cobra.
- The toy network (`examples/toy_network.json`) exists to supply the case the example model **cannot**: a reaction fixed at a **nonzero** value (`FIX = 2.0`), making the reduced mass balance genuinely affine. All 513 of the example model's fixed reactions sit at zero, so it would never catch a homogeneous-RHS bug.

## Verify current state

```bash
cd /home/mcpu/GitHub/gsmm_compiler_maxent_sampling
.venv/bin/python -V                                    # expect 3.11.15
.venv/bin/python -m pytest -q | tail -3                # expect 569 passed
.venv/bin/ruff check . && .venv/bin/mypy               # expect clean
.venv/bin/gsmm-compiler model inspect examples/toy_network.json     # affine RHS: nonzero
.venv/bin/gsmm-compiler model inspect models/GCF_000010425_1_ASM1042v1_protein_non_gapfilled_latest_gapfilled_noO2.json
```

---

## Milestone checklist  (v1 = M0–M9 · M10 deferred)

Legend: ⬜ todo · 🟨 in progress · ✅ done (gate passed)

### ✅ M0 — Platform & packaging spike  *(gate passed 2026-07-13)*
Gate: wheel-only install on Python 3.11/aarch64 · example model LP solves · core imports no scipy.
- [x] `pyproject.toml` — src layout, `requires-python==3.11.*`, deps cobra/highspy/numpy; dev: pytest, pytest-cov, ruff, mypy; `gsmm-compiler` console script
- [x] package skeleton `src/gsmm_compiler/` with module stubs (per BUILD_PLAN §6 / spec §6) + `__init__.py` + `py.typed`
- [x] `uv venv` on Python 3.11 + `uv pip install -e .`; `uv.lock` pinned, **no source builds** (`--only-binary=:all:`)
- [x] smoke: load Bifido model with cobra + build one native-array `highspy` LP via `passModel` + solve — **and cross-check the optimum against cobra's own FBA**
- [x] no-scipy scan: assert the numerical core modules import zero scipy (runtime + static AST scan)
- [x] `gsmm-compiler model inspect models/GCF_000010425_1_..._noO2.json` prints basic report

### ✅ M1 — Canonical + reduced model IR  *(gate passed 2026-07-13)*
Gate: hand-checked CSC · exact full-model reconstruction · l==u elimination equivalence on toy.
- [x] `config.py` — TOML load/resolve + CLI overrides + resolved-config echo (`gsmm-compiler config show`); unknown keys are errors
- [x] `model_input.py` — load, freeze reaction/metabolite order, validate, `model_report.json`, L0 key
- [x] `native_csc.py` — `NativeCSC` dataclass, column-wise builder, validation, `matvec`/`rmatvec`, **int32** index width (see facts above)
- [x] `flux_polytope.py` — `FluxPolytope` + **reduced polytope IR**: eliminate l==u, `v_full = R·v_red + c`, affine RHS `S_F v_F = -S_fixed v_fixed`
- [x] L0/L1 content hashing + provenance keys (parser/cobra/schema versions, dtype **and byte order**) → `provenance.py`
- [x] unit tests: CSC hand-check + order preservation + NaN/inf/dup detection + elimination & reconstruction round-trip
- [x] **gate**: LP optima identical full-vs-reduced under 200 random objectives (toy) and 10 (genome-scale) — feasible-set *and* objective equivalence, incl. the fixed-flux objective constant

### ✅ M2 — One-dimensional kernel (math oracle)  *(gate passed 2026-07-13)*
Gate: analytic + property tests across κL ∈ {0,±1e-16,±1e-12,±1e-8,±1,±100,±1000}; continuity; monotone slopes; edge cases.
- [x] `line_geometry.py` — feasible chord `[t_lo,t_hi]`; **keep all nonzero direction components**; `nextafter` inward; **self-loop (not redraw) on a degenerate chord**; raises on an empty or non-finite chord
- [x] `line_distribution.py` — breakpoints `τ_r=-v_r/d_r`; piecewise-linear concave `J(t)` via slope drops `2λw_r|d_r|`; **peak-anchored heights + higher-endpoint mass anchoring**; segment log-masses (`expm1`/`log1p` + custom logsumexp); categorical segment choice; stable truncated-exp inverse-CDF
- [x] unit tests: chord (±/zero comps, tiny-but-binding comps, t=0, degenerate); piecewise J vs direct grid eval, continuity, nonincreasing slopes, duplicate/near-duplicate/no breakpoints; `log φ` vs a **60-digit `decimal` oracle** across every regime
- [x] statistical tests: uniform (β=0), `e^{κt}`, `e^{-α|t|}` — KS vs exact analytic CDFs + an independent fine-grid quadrature of the *directly evaluated* `J`; fixed seeds, p-floor 1e-3
- [x] 🤝 **`/collab` adversarial review — 4 rounds, converged (AGREE, contested: none).** Found **two distribution-corrupting bugs that passed 264 tests**. See `.collab/specs/collab-outcome.md`; BUILD_PLAN §1.6 deltas 6–9 are new.

### ✅ M3 — Native HiGHS LP layer  *(gate passed 2026-07-13)*
Gate: solver objective == direct J · feasibility on degenerate toys · z=|v| within tol · no scipy.
- [x] `highs_backend.py` — `HighsLinearProgram` adapter: build `HighsLp`, `passModel`, `set_objective`/`set_maximize`, `solve` w/ status+model-status checks, one-shot solution extraction, basis get/set, **solve counter** (process-global) + `freeze()`
- [x] `sparse_objective.py` — `SparseFluxObjective`; (v,z) LP by **direct CSC assembly** (no scipy blocks); z=|v| checks; biomass-only diagnostic LP; recompute + compare J; `SparseObjectiveSolution` (the L2 bundle: J*, v*, μ*, C*, μ_max, retention, **sparsity-dominated flag**)
- [x] unit tests: analytic toy optimum, solver-obj==direct-J, z=|v|, feasibility, λ sensitivity, biomass excluded by default, custom weights
- [x] **gate**: solver obj == direct J on toy + genome-scale across λ ∈ {0, 1e-4, 1e-3, 0.01, 1}; v* feasible in the *full* 773-reaction polytope; degenerate toys (singleton / infeasible / unique-point / tied optima / λ=0) all classified right; no scipy
- [x] **addendum (settled by user, 2026-07-13)**: λ is **scale-referenced** — `critical_l1_penalty` (exact λ*, one Charnes–Cooper LP), `origin_is_feasible`, `resolve_objective`; config takes dimensionless `l1_penalty_scaled`. See BUILD_PLAN §1.7

### ✅ M4 — Affine geometry + span certificate  *(gate passed 2026-07-13)*
Gate: known toy dims recovered · truncated basis rejected · ‖S·diag(s)·B‖≈0 · singleton path.
- [x] `affine_geometry.py` — scaled coords `s_i`; orthonormal basis (2-pass projection, Fortran-contig, block-allocated); support-LP discovery w/ warm starts; memory guard before **every** allocation
- [x] **deterministic span certificate** — probes an orthonormal basis of range(B)ᗮ (pivoted Gram-Schmidt over the axes, ordered by residual norm); random probes are the pre-pass; a failing sweep *hands back the missing direction*; capped/inconclusive runs are refused unless explicitly downgraded
- [x] **`blocked_reactions` + `DirectionSpace`** (beyond spec) — FVA finds the 61 free reactions that cannot move; their components, and any non-mass-balanced ones, are projected out of every candidate exactly. Without this, the chord at the centre excluded `t = 0` and M5 could not start
- [x] feasible center from support points; **exactly** bound-feasible (clamp bounded by the LP tolerance, mass balance re-verified); chords at the centre validated; geometry diagnostics + manifest
- [x] tests: known toy dims (triangle/narrow/singleton/pinned), orthonormality, mass-balanced directions, **truncated basis rejected (all 46 columns, one at a time)**, dim-0 singleton returns a constant, memory guard, determinism
- [x] **gate**: `d = 46` on the genome-scale model, matching an **independent FVA+rank oracle** that shares no code with the geometry; certificate exhaustive (214/214 probes, 0 inconclusive); a 500-step hit-and-run walk stays feasible in the *full* 773-reaction polytope with **zero** HiGHS solves
- [x] 🤝 **`/collab` adversarial review — 6 rounds.** Found a crash, a silently dropped dimension, a dual-side blind spot, an accumulating basis error, an unsound bound formula, and **two test bugs** (one made a test unable to fail). All reproduced before fixing. See `.collab/specs/collab-outcome.md` § M4

### ✅ M5 — Rounding + β=0 sampler  *(gate passed 2026-07-14)*
Gate: uniform analytic targets reproduced · transform-invariance · ‖ST‖≈0 · **zero inner-loop HiGHS solves**.
- [x] `rounding.py` — support-point covariance, ridge (rel. to trace/d) + geometric escalation, Cholesky `L`, `T=diag(s)BL`, ‖ST‖ check (relative, unfloored), **SVD rank check on `T`**, physically read-only arrays, per-coordinate precompute derived from `T` and validated against it
- [x] `maxent_sampler.py` (β=0) — coordinate hit-and-run, reduced state `y` + incremental `v` on the structural support only, periodic exact refresh, **stored flux is the exact `centre+T·y` of the stored state**, SeedSequence keyed on `(model_id,stage,β,chain)`
- [x] multi-chain (4, dispersed Dirichlet(0.5) starts over the support hull); `diagnostics.py` split-R̂ / Geyer ESS / autocorr / feasibility
- [x] tests: **the 2-simplex** (marginal `2(1−x)`, *not* uniform — the target a box cannot distinguish), coupled box, anisotropic box, transform-invariance of moments (ESS-normalized z, not σ), positive chords at start, solve-counter==0
- [x] 🤝 **`/collab` adversarial review — 3 rounds, converged (AGREE, contested: none).** Found **six defects that 553 tests passed over**. See `.collab/specs/collab-outcome.md` § M5; BUILD_PLAN §1.6.1 is new.

### ⬜ M6 — Positive-β maximum-entropy sampler  *(ACTIVE)*
Gate: truncated-exp + truncated-Laplace targets · mean J nondecreasing in β · 1D quadrature cross-check.
- [ ] wire exact piecewise-exp line conditional (M2) into the sampler for β>0
- [ ] explicit β-ladder; energy scale `s_J` from support points (`warmup_range`); objective traces (μ,C,J, norm log-energy, near-zero counts)
- [ ] feasible starts (dispersed convex combos of support pts + v*); map start flux → reduced coords via `numpy.linalg.solve`
- [ ] tests: analytic log-concave targets; empirical mean-J monotone in β; 2D box → two truncated Laplaces; large-β stress
- [ ] 🤝 **`/collab` adversarial review (required gate step)** — that the sampled law is exactly π_β (no hidden approximation); `s_J`/`J*` handling; mean-J monotonicity

### ⬜ M7 — Reweighted-L1 (frozen weights)
Gate: deterministic weights (fixed seed) · active-set converges · weights frozen before MCMC · targets reproduced under reweighted J.
- [ ] reweighting loop `w_r ← w_base/(|v_r|+ε)`, clip to limits, median-renormalize, stop on active-set+solution tol
- [ ] save every weight vector + LP solution; **freeze final weights**, rebuild objective/LP-opt/`s_J` (L2)
- [ ] label experimental (not exact cardinality); guard: weights never updated from MCMC state
- [ ] tests: weight formula, clipping/renorm, determinism, frozen-before-sampling invariant
- [ ] 🤝 **`/collab` adversarial review (required gate step)** — weights frozen before sampling; `J` never changes mid-chain (would invalidate stationarity)

### ⬜ M8 — Cache, restart, batch orchestration & production
Gate: resume only missing (model,chain) units · partial batch → valid cross-model tables · concurrent-writer safe · deterministic same-env traces.
- [ ] `output.py` — run-dir layout, atomic temp+rename+fsync, per-chain + run `COMPLETE` markers, configurable storage (`full_flux` f64/f32 · `reduced`)
- [ ] 4-layer content-addressed cache + writer-claim (atomic mkdir) locking + on-load validation
- [ ] **batch runner** over models manifest; one **global** process pool over `(model,β,chain)`; set OPENBLAS/OMP/MKL threads=1 in workers; workers write own files
- [ ] **cross-model aggregation** (`results/<batch>/cross_model/`: β-summary, reaction-activity, exchange matrices)
- [ ] `diagnostics.py` (feasibility/objective/mcmc/geometry/solver JSON) + `features.py` (raw/activity/exchange/pathway)
- [ ] full CLI: `maxent solve-lp | build-geometry | sample | diagnose`
- [ ] tests: kill-and-resume, concurrent-writer, corrupted-artifact rejection, batch ≥2 strains + one deliberately failed

### ⬜ M9 — Performance & GSMM hardening
Gate: benchmark report produced · all performance assertions hold.
- [ ] benchmark suite (parse→CSC→passModel→first LP→warm-start LPs→sparse LP→geometry→rounding→β=0 sps→β>0 sps→breakpoint dist→output)
- [ ] worker-count sweep {1,2,4,7,14} on Jetson by ESS/wall-sec across the batch
- [ ] allocation + breakpoint-sort profiling; validate `reduced` storage mode
- [ ] assert: no per-step HiGHS, no scipy, no Python loop in chord, no element-wise highspy extraction, no full reconstruction every step

### (deferred) M10 — Extensions
Only after M0–M9: pilot rerounding + pilot-based `s_J` (bootstrap→pilot→final DAG); β→performance calibration; parallel tempering; slice-based line kernel; downstream mode-feature discovery. Each behind its own tests; none alters the validated v1 target.

---

## Session log  (append one line per working session, newest last)

- 2026-07-13 — Design collaboration (Claude×Codex, 2 rounds) → BUILD_PLAN.md. Investigated blocked reactions (513 = FVA-blocked under anaerobic medium; fixed-var elimination provably correct). Locked scope: batch-aware v1, reweighted-L1 in v1 (M7), configurable storage, Python 3.11. Created this tracker + project CLAUDE.md. **No code yet — next session starts M0.**
- 2026-07-13 — **M0 gate PASSED.** Scaffolded `pyproject.toml` (hatchling, src layout) + all 17 modules of the spec §6 skeleton; `uv` venv on 3.11.15 with a wheel-only install (`uv.lock`, 47 pkgs, zero source builds). Smoke test builds the biomass LP from native int32/float64 CSC arrays, passes it to `highspy.passModel`, solves to `kOptimal`, and **matches cobra's own FBA optimum to rel=1e-6** — so the CSC assembly is verified, not just accepted. 25 tests green; ruff + mypy --strict clean. Findings: (a) **scipy is absent from the venv entirely**, so the no-scipy path is free rather than fought for; (b) **highspy attribute reads return Python lists, not NumPy views** — recorded for M3's one-shot extraction design; (c) BUILD_PLAN §0's "65 exchanges" was 63 `EX_` + 2 `SK_` sinks — corrected in place. **Next: M1** (`config.py` first).
- 2026-07-13 — **M1 gate PASSED.** Built `native_csc` (int32 CSC + `matvec`/`rmatvec` via `bincount`), `flux_polytope` (canonical + reduced IR with the explicit affine RHS), `model_input` (validation → canonical L0 IR + `model_report.json`), `config` (TOML + overrides, unknown keys rejected), and a new `provenance` module for L0/L1 content keys. Added `examples/toy_network.json` — 7 reactions, with **FIX pinned at 2.0** so the reduced mass balance is genuinely affine, the case the Bifido model cannot exercise (all 513 of its fixed reactions sit at zero). Gate closed by LP-level equivalence: full-vs-reduced optima agree under 200 random objectives on the toy and 10 on the genome-scale model, including the fixed-flux objective constant that §1.5 warns about. 164 tests green; ruff + mypy --strict clean. Findings: (a) **HiGHS uses a 32-bit `HighsInt`** (`kHighsIInf == 2**31-1`), so the spec's `int64` CSC would be silently narrowed on every `passModel` — we store int32 and bound-check nnz; (b) **cobra already rejects NaN/inverted/duplicate-ID models but accepts infinite bounds**, so that is the one malformation our parser layer must catch itself. `.collab/specs/collab-outcome.md` is referenced by three docs but **does not exist** (the directory is empty) — decisions survive in BUILD_PLAN §1. **Next: M2** (math-critical — `/collab` review required at the gate).
- 2026-07-13 — **M2 gate PASSED (math-critical).** Built `line_geometry` (feasible chord) and `line_distribution` (breakpoints → piecewise-linear concave `J` → segment log-masses → categorical choice → truncated-exp inverse CDF). 287 tests green; ruff + mypy --strict clean. `log φ = log(expm1(x)/x)` is checked against a **60-digit `decimal` oracle** rather than a restatement of itself, and the statistical tests KS the sampler against exact analytic CDFs *and* against an independent fine-grid quadrature of the **directly evaluated** `J` — a reference that never touches the piecewise machinery, so a misplaced breakpoint cannot cancel out against itself. **The `/collab` review ran 4 rounds and earned its keep: it found two bugs that corrupted the sampled distribution while 264 tests passed.** (a) *The absolute magnitude of `J` reached the probabilities.* `h_a = β(J−J*)/s_J` built from absolute knot values cancels catastrophically; with a biomass flux of 1e16 the true segment probabilities [0.387, 0.613] came back as [0.632, 0.368] — **the favoured segment reversed**, from slopes that were themselves exactly right. Fixed by storing knot heights relative to the **peak** of `J` and anchoring each segment's mass integral at its **higher endpoint**; `J*` left the API entirely (it provably cancels). (b) *Redrawing a coordinate on a degenerate chord breaks stationarity* — spec §19 and BUILD_PLAN §1.6 both prescribed it, but it makes coordinate selection state-dependent and collapses the random-scan Gibbs invariance argument. Now a **self-loop**, and there is no minimum chord width at all. Round 2 is the one to remember: **both of its findings were defects in round 1's fixes**, not in the original code. Also corrected: the opening slope (spec §20.3's midpoint rule reads `sgn(0)=0` on a one-ULP first segment and is 2× off in a measured 10.5% of them); `UNIFORM_LIMIT = eps/4`, not `eps/2`, because float spacing is asymmetric about 1.0; `β/s_J` validated against a *silent* underflow. I rebutted one Codex claim with measurements (the `MIN_NORMAL` "collapse" was ≤1 ULP on a probability-1e-15 set, not distribution corruption) and it conceded. BUILD_PLAN §1.6 gains **deltas 6–9**; `.collab/specs/collab-outcome.md` now exists and records all four rounds. **Next: M3** (`highs_backend.py` — not math-critical, no collab gate).
- 2026-07-13 — **M3 gate PASSED.** Built `highs_backend` (the only module that touches `highspy`) and `sparse_objective` (the objective, the (v,z) LP, the §14 flux-only LP, the biomass-only diagnostic). 369 tests green; ruff + mypy --strict clean. The gate is one equation — *solver objective == directly recomputed J* — and it is load-bearing: HiGHS optimizes a linearized surrogate over `(v,z)` on a **reduced** polytope with a constant folded into an **offset**, while `evaluate` computes `J` from the full 773-flux vector and knows none of that. They agree only if the linearization, the fixed-variable elimination, the objective lowering *and* the constant are all right. Verified on both models across λ ∈ {0, 1e-4, 1e-3, 0.01, 1}. Design notes: (a) HiGHS adds `lp.offset_` to the reported objective **sign-intact under `kMaximize`** — probed, then pinned by a test — so the fixed reactions' L1 cost reaches `J*` instead of being quietly dropped (only the toy, with `FIX = 2.0`, can catch that; all 513 of the example model's fixed reactions sit at zero); (b) a `z` column is built **only where `λw_r > 0`**, because a zero-cost `z` has nothing pushing it down onto `|v_r|` and would fail its own check on a solution that is perfectly correct in `v`; (c) the solve counter is **process-global** and programs can be `freeze()`d, so M5 can both assert and *prohibit* an inner-loop solve; (d) `highspy` is imported inside the constructor, so a worker can import the objective without loading a solver. **🔴 The milestone's real find is scientific, not structural: at the default λ = 1.0 the genome-scale LP optimum is *the origin* — `v* = 0`, `J* = 0`, zero growth — while `μ_max = 41.6`.** Above `λ* = max_v μ(v)/C(v)` the L1 cost of growing outruns the biomass it returns and the cell's best move is to shut down; on this model `λ* = 1.89e-3`, so our default is **529× past the cliff** and the spec's own suggested `0.01` is **5.3×** past it. Nothing inside the LP can see this — optimal status, zero residual, `z = |v|` exactly — so `solve_sparse_objective` now always solves the biomass-only LP too and flags `is_sparsity_dominated`. The collapse needs a feasible origin: this model has **no forced-flux reaction at all** (no `ATPM` lower bound), which is also why the toy (`FIX = 2.0`) cannot reproduce it and the genome-scale model can. **Decision left OPEN** (raw λ vs scale-referenced λ), recorded in BUILD_PLAN §1.7 — it does not block M4/M5 (geometry is λ-independent; β=0 ignores `J`) but **must be settled before M6 tilts by `J`**. **Next: M4** (math-critical — `/collab` review required at the gate).
- 2026-07-13 — **M3 addendum: λ is now scale-referenced** (the open decision from the gate, settled by the user). The config takes a **dimensionless `λ̃`** (`objective.l1_penalty_scaled`, default 0.5) and `resolve_objective` computes the raw `λ = λ̃ · λ*` per model. **`λ*` is exact, from a single LP** — `λ* = max_v μ(v)/C(v)` is a linear-fractional program, and the Charnes–Cooper substitution (`y = v·t`, `t = 1/C(v)`) turns it into "maximize `μ(y)` subject to a unit cost budget `C(y) ≤ 1`", with the bounds homogenizing into rows `l·t ≤ y ≤ u·t` and the absolute value linearizing by the same `z ≥ ±y` trick as §12. It reproduces the 40-step bisection to 8 figures and hits the fork toy's hand-derived `λ* = 1/2` to 10 decimals, so it is checked against arithmetic rather than against itself. λ* is **exactly** the cliff, not merely near it: at `0.999·λ*` the model grows, at `1.001·λ*` the optimum is the origin — pinned as a test on the genome-scale model. The λ̃ ladder is now the selection-pressure dial the study wanted: **λ̃ = 0 → 100% of μ_max retained, 0.25 → 95%, 0.5 → 60%, 0.9 → 30%**. `λ̃ ≥ 1` is *refused* when the origin is feasible (a guaranteed collapse, nothing to sample) and *allowed* when it is not — a model with forced maintenance flux, like the toy, cannot answer a large λ by shutting down and so has no cliff at all. Why this matters beyond one model: λ̃ = 0.5 resolves to `λ = 9.4e-4` on Bifido and `λ = 0.25` on the toy — **a factor of 265** — so a shared *raw* λ would have meant wildly different selection pressures across a batch while looking, in the config file, like a controlled comparison. `λ̃`, `λ*`, the raw `λ` and `origin_is_feasible` all go to the manifest (spec §3.6: no hidden scaling). BUILD_PLAN §1.7 records the decision. **Left open for M7**: `λ*` is a function of `w` (doubling every weight halves it), so the reweighting loop must choose — and record — whether λ stays frozen at its base-weight value or is re-resolved from the frozen final weights. 392 tests green; ruff + mypy --strict clean.
- 2026-07-13 — **M4 gate PASSED (math-critical).** Built `affine_geometry` (scaled coords, orthonormal basis by support LPs, feasible centre, deterministic span certificate) and grew `native_csc`/`highs_backend`/`config`/`provenance` to serve it. 442 tests green; ruff + mypy --strict clean. **The milestone's central finding is scientific: the sampling dimension is 46, not 55.** 61 of the 260 free reactions cannot carry flux at all — the model file leaves `l < u`, but mass balance pins them — so `n_free − rank(S) = 55` is only an upper bound. `d = 46` is confirmed by an **independent FVA+rank oracle that shares no code with the geometry**, which is the gate's load-bearing test. **And those 61 reactions were not bookkeeping: they were a landmine under M5.** A blocked reaction is an *exact* structural zero of the direction space, so a basis row of ~1e-15 there is pure noise; divided by a centre sitting ~1e-13 *outside* its own bound (also noise), it produced **a chord limit of order 0.03–0.5** — inside the legitimate chord. Measured, the chord through the centre came out `[−0.54, −0.39]`, *excluding `t = 0`*, and `line_geometry` rightly refused to sample it: **M5 could not have started.** Now projected out exactly, and the geometry proves its own centre samplable before shipping it. **The `/collab` review ran 6 rounds and was worth every one — it found a crash, a silently dropped dimension, a dual-side blind spot, an accumulating basis error, an unsound bound formula, and two test bugs, one of which made a test unable to fail.** Every counterexample was reproduced before being fixed. Four things it changed that I would have shipped wrong: (a) **the certificate's resolution is `√k`, not its largest width** — width is subadditive, so a direction tilted across all 214 probes hides a factor of 14.6 from each one individually; (b) **flatness cannot be certified from the primal width**, which is a *lower* bound and the wrong end of the interval — a solve that stops short of optimality reports the width too *small* and certifies a real dimension as flat, so flatness now rests on a **weak-duality** upper bound that assumes nothing of the returned point, not even feasibility; (c) **the same bound must be applied to every FVA range**, since a range read off the primal is a lower bound and "this reaction cannot move" is exactly the conclusion a lower bound cannot support; (d) **error was accumulating along the Gram-Schmidt chain** (each column inherits `‖S·diag(s)·B‖·‖BᵀΔx‖` from its predecessors — the worst column hit 8e-10 against a 1e-9 tolerance), and an attempt to fix it by re-probing made it *worse*, so the chain is now cut by projecting every candidate into the mass-balanced subspace before it can pollute the basis. The honest consequence, reported rather than hidden: **the dual bound cancels terms of size ~5e3, so evaluating it in float64 costs ~1e-9 — that is the floor on what the certificate can resolve**, orders coarser than the ~1e-13 the arithmetic appears to produce. Certified resolution: **2.78e-11 scaled = 5.6e-8 flux units**, outward-rounded. What it licenses is stated precisely and narrowly: *every exact-polytope direction has its component orthogonal to `range(B)` bounded in width by that resolution* — **not** "cannot under-count a dimension". Also settled: three tolerances describe the same polytope and must not contradict each other (`scale_floor ≥ blocked_tol/span_tol`; `‖r_blocked/s_blocked‖₂ ≤ span_tol`; SVD rank cutoff ≥ the LP's `feasibility_tol`) — the first of those was a real crash. **Next: M5** (math-critical — `/collab` review required at the gate). M5 inherits a centre that is exactly bound-feasible, chords that all contain `t = 0` (min length 0.018), and support points that span all 46 directions, so its covariance ridge cannot conceal a singular covariance.
- 2026-07-14 — **M5 gate PASSED (math-critical).** Built `rounding` (support-covariance Cholesky → `T = diag(s)·B·L`, ridge + escalation, per-coordinate precompute), `maxent_sampler` (β=0 coordinate hit-and-run, multi-chain, dispersed starts) and `diagnostics` (split-R̂, Geyer ESS, autocorrelation, feasibility). 569 tests green; ruff + mypy --strict clean. **Zero HiGHS solves after sampling starts, zero bound violations across 12000 genome-scale samples, and every lifted sample feasible in the *full* 773-reaction polytope** — the one check only the lift can fail. The load-bearing statistical test is the **2-simplex**, whose uniform law has marginal density `2(1−x)`: it is *not* uniform, so a sampler that quietly returned uniform marginals — the failure a box cannot detect — dies there. It also has a control that must fail (a KS test against the uniform CDF, p < 1e-12), so a passing test proves something. **Rounding earns its keep, measured:** the shortest chord at the centre goes 0.018 → 0.744 (41×) and the spread across axes 77× → 3.8×. And the division of labour is now explicit: `diag(s)` absorbs *axis-aligned* stretch (a 1000:1 box gives unrounded chords differing by 1.41, not 1000), while `L` fixes the *correlations* no diagonal matrix can see. **The `/collab` review ran 3 rounds and found six defects that 553 tests passed over — none corrupting the β=0 distribution, but three corrupting what we *believed* about it, which on a sampler is nearly as bad.** (a) *The Geyer ESS paired from the wrong lag.* Γ_m = ρ_{2m}+ρ_{2m+1} starts at **lag zero**; pairing from lag 1 sums identically **only if nothing is truncated**, and truncation is the whole method. On an antithetic AR(1) (ρ = −0.5) the correct Γ₀ = +0.50 says *keep going* while the offset first pair is −0.25 and says *stop at once* — so it reported ESS = N for a chain whose true ESS is **3N**. (b) *`is_feasible` ignored mass balance entirely*, so a chain that had walked off the steady-state manifold — the half of `P`'s definition the entire affine geometry exists to enforce — reported `True`. (c) *The stored flux was the incremental cache*, so `to_flux(coordinates) != fluxes`, and the drift was measured **only at refresh instants** — not a bound on anything, since drift can peak and cancel *between* refreshes and a long `refresh_interval` would report a serene 0.0 having measured nothing. (d) *Nothing checked that `T` has full column rank*, the identity that licenses any ridge; **a `T` that quietly loses a column produces no bad numbers, only absent ones** — the chain samples a lower-dimensional slice with every sample feasible and every chord positive. (e) *`@dataclass(frozen=True)` freezes the binding, not the buffer*, and the precompute holds **copies** of `T`'s columns — so an in-place write would have the chord and the flux disagree silently; fixing it exposed a second latent bug, that `np.ascontiguousarray` returns its argument *unchanged*, so freezing the centre without copying would have made `ReducedGeometry.center` read-only underneath its owner. (f) *`np.min(x, initial=0.0)` is not an empty-array guard* — `initial` is a **candidate** for a min, so `min_ess` reported **0** for a sample whose every entry was 8000. I contested one Codex claim and it conceded: the relative-residual floor is correct **where the noise exists and absent where it cannot be** — a sampled *flux* carries solver noise at the blocked reactions (one row divides a noise value by itself and reports exactly 1.0), a *direction* does not (`T`'s blocked rows are exactly 0.0), so the transform's check is unfloored while the flux check is floored. Also settled honestly: **in float64 the chain is not Markov in `y` alone** (its state is `(y, cache error, refresh phase)`), exact Gibbs invariance is claimed *only* in exact arithmetic, and a measured drift is **not** a bound on stationary-law error. **🔴 The scientific finding: this model mixes slowly and the defaults do not cover it.** `step_scale_ratio = 0.008`, and at 4×(3000+3000) sweeps R̂ = 1.08 with min ESS 34/12000; R̂ reaches 1.04 only at 8000+8000. ESS grows linearly and R̂ falls monotonically, so the chain is correct-but-slow, not stuck (no coordinate has zero within-chain variance at any length). The remedy is **measured, not hypothetical**: spec §17.4 pilot rounding — re-rounding from a β=0 pilot chain's own covariance instead of the support-LP vertices — improves cond(C) 1.56e4 → 5.14e3 and **ESS by 2.5×**. It stays deferred to M10 (it is a separate DAG stage and does not touch the target), but it is the lever if wall-clock binds. **Next: M6** (math-critical — `/collab` review required at the gate). M6 inherits a kernel that already accepts `beta`/`objective`/`energy_scale` and *refuses* β>0 without them; it must supply the reduced-coordinate `L1Objective`, `s_J`, the ladder and the traces.
