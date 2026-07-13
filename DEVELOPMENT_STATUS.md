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
| **Active milestone** | **M2 — One-dimensional kernel (math oracle)** |
| **Status** | ⬜ NOT STARTED (M0 + M1 gates passed 2026-07-13; 164 tests green) |
| **Next action** | `line_geometry.py` — feasible chord `[t_lo, t_hi]`, keeping **all nonzero** direction components |
| **Blockers** | none |
| **Last updated** | 2026-07-13 |

> ⚠️ **M2 is math-critical**: its gate requires a `/collab` adversarial review (exactness of the 1D
> conditional; `expm1`/`log1p` stability across all κL; no tolerance-merging of distinct breakpoints).

### Platform facts established by M0 (build on these, don't re-derive)

- Python **3.11.15**, aarch64/Jetson. Wheel-only install verified with `uv pip install -e ".[dev]" --only-binary=:all:` — **no source builds**. Pinned in `uv.lock` (47 packages).
- Resolved versions: **cobra 0.31.1 · highspy 1.15.1 · numpy 2.4.6**.
- **SciPy is absent from the venv entirely** — stronger than the gate required (§4 anticipated cobra might pull it transitively; it does not). The no-scipy gate is enforced by `tests/unit/test_no_scipy.py` at both runtime and source level.
- HiGHS accepts native `int32` CSC index/start arrays + `float64` values via `passModel`. The biomass LP matches cobra's own FBA optimum to `rel=1e-6` — the CSC assembly is verified, not merely accepted.
- ⚠️ **highspy attribute reads return Python `list`, not NumPy views** (the pybind layer copies). M3's LP layer must keep its own float64 arrays and extract solutions in one `np.asarray` shot — never element-wise. Pinned by `test_highspy_returns_python_lists_not_arrays`.

### Facts established by M1

- **CSC index width is `int32`, not the spec's `int64`.** `highspy.kHighsIInf == 2**31 - 1`, i.e. HiGHS is built with a 32-bit `HighsInt`; int64 arrays are *accepted* but silently narrowed on every `passModel`. `native_csc.INDEX_DTYPE = np.int32`, and the 2³¹ nnz ceiling is a construction-time check. Pinned by `test_highs_index_width_is_what_native_csc_stores`.
- **cobra rejects NaN bounds, inverted bounds and duplicate IDs itself** — but happily accepts **infinite** bounds, which we must reject (an unbounded polytope has nothing to sample). Our own guards still matter for models built/mutated in memory rather than parsed.
- **New module `provenance.py`** (not in spec §6): L0/L1 content keys were needed long before M8's cache exists, and the core must hash arrays without importing cobra.
- The toy network (`examples/toy_network.json`) exists to supply the case the example model **cannot**: a reaction fixed at a **nonzero** value (`FIX = 2.0`), making the reduced mass balance genuinely affine. All 513 of the example model's fixed reactions sit at zero, so it would never catch a homogeneous-RHS bug.

## Verify current state

```bash
cd /home/mcpu/GitHub/gsmm_compiler_maxent_sampling
.venv/bin/python -V                                    # expect 3.11.15
.venv/bin/python -m pytest -q | tail -3                # expect 164 passed
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

### 🟨 M2 — One-dimensional kernel (math oracle, before any genome-scale MCMC)  *(ACTIVE)*
Gate: analytic + property tests across κL ∈ {0,±1e-16,±1e-12,±1e-8,±1,±100,±1000}; continuity; monotone slopes; edge cases.
- [ ] `line_geometry.py` — feasible chord `[t_lo,t_hi]`; **keep all nonzero direction components**; `nextafter` inward; redraw on zero-length
- [ ] `line_distribution.py` — breakpoints `τ_r=-v_r/d_r`; piecewise-linear concave `J(t)` via slope drops `2λw_r|d_r|`; segment log-masses (`expm1`/`log1p` + custom logsumexp); categorical segment choice; stable truncated-exp inverse-CDF
- [ ] unit tests: chord (±/zero comps, t=0, zero-length); piecewise J vs direct grid eval, continuity, nonincreasing slopes, duplicate/no breakpoints
- [ ] statistical tests: uniform (β=0), `e^{κt}`, `e^{-α|t|}` moments/quantiles (fixed seeds, non-flaky CIs)
- [ ] 🤝 **`/collab` adversarial review (required gate step)** — exactness of the conditional; expm1/log1p stability across all κL; no tolerance-merging of distinct breakpoints; chord keeps every nonzero component

### ⬜ M3 — Native HiGHS LP layer
Gate: solver objective == direct J · feasibility on degenerate toys · z=|v| within tol · no scipy.
- [ ] `highs_backend.py` — `HighsLinearProgram` adapter: build `HighsLp`, `passModel`, `set_objective`/`set_maximize`, `solve` w/ status+model-status checks, one-shot solution extraction, basis get/set, **solve counter**
- [ ] `sparse_objective.py` — `SparseFluxObjective`; (v,z) LP by **direct CSC assembly** (no scipy blocks); z=|v| checks; biomass-only diagnostic LP; recompute + compare J
- [ ] unit tests: analytic toy optimum, solver-obj==direct-J, z=|v|, feasibility, λ sensitivity, biomass excluded by default, custom weights

### ⬜ M4 — Affine geometry + span certificate
Gate: known toy dims recovered · truncated basis rejected · ‖S·diag(s)·B‖≈0 · singleton path.
- [ ] `affine_geometry.py` — scaled coords `s_i`; orthonormal basis (2-pass MGS, Fortran-contig, block-allocated); support-LP discovery w/ warm starts; memory guard (8nd bytes vs `max_geometry_memory_gb`)
- [ ] **deterministic span certificate** — probe orthonormal basis of range(B)ᗮ (pivoted QR), ordered by residual norm; random probes as cheap pre-pass; manifest flag if capped
- [ ] feasible center from support points (mass-balance/bound checks, no clipping); geometry diagnostics
- [ ] tests: known dims, orthonormality, mass-balanced directions, **intentionally truncated basis rejected**, dim-0 singleton returns constant sample
- [ ] 🤝 **`/collab` adversarial review (required gate step)** — completeness of the deterministic span certificate; scaling/tolerance coupling to LP feasibility tolerance

### ⬜ M5 — Rounding + β=0 sampler
Gate: uniform analytic targets reproduced · transform-invariance · ‖ST‖≈0 · **zero inner-loop HiGHS solves**.
- [ ] `rounding.py` — support-point covariance, ridge (rel. to trace/d) + geometric escalation, Cholesky `L`, `T=diag(s)BL`, ‖ST‖ check, per-coordinate precompute
- [ ] `maxent_sampler.py` (β=0) — coordinate hit-and-run, reduced state `y` + incremental `v`, periodic exact refresh, SeedSequence keyed on `(model_id,stage,β,chain)`
- [ ] multi-chain (≥4); `diagnostics.py` R̂/ESS/autocorr/feasibility
- [ ] tests: uniform targets, transform-invariance of moments, positive chords at start, solve-counter==0 after sampling begins
- [ ] 🤝 **`/collab` adversarial review (required gate step)** — Gibbs/coordinate-hit-and-run **stationarity** argument; transform frozen during production; uniform-target correctness

### ⬜ M6 — Positive-β maximum-entropy sampler
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
