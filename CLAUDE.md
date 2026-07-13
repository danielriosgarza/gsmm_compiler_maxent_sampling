# GSMM-Compiler — project instructions for Claude

This repo builds a sparse-objective **maximum-entropy flux sampler** for genome-scale metabolic
models. See [BUILD_PLAN.md](BUILD_PLAN.md) (design + milestones), [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)
(live progress), and [GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md](GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md) (math spec).

## ▶ "Continue package development"

When the user asks to **continue package development** (or "resume the build", "keep building the
package", or similar), follow this protocol exactly:

1. **Read [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)** — the single source of truth for progress.
   Note the ACTIVE milestone and its next unchecked task.
2. **Read that milestone's row in [BUILD_PLAN.md](BUILD_PLAN.md)** for full deliverables + the
   acceptance gate.
3. **Run the "Verify current state" commands** in DEVELOPMENT_STATUS.md. Trust observed results over
   checkboxes — reconcile the tracker if they disagree.
4. **Work the next unchecked task.** Build **one milestone at a time**, in order. Do not skip ahead;
   later milestones depend on earlier gates.
5. **Before closing a math-critical gate (M2, M4, M5, M6, M7), run `/collab`** for an adversarial
   review — see "Cross-model collaboration" below. This is a required gate step, not optional.
6. **Close out a milestone only when its acceptance gate passes** (tests green + gate criteria met +
   collab review clean where required). Then: tick its boxes, advance **Current state** to the next
   milestone, append a **Session log** line, and `git commit` (branch first if on `main`).
7. If you finish a milestone and time remains, continue to the next one — but re-run step 2's gate
   framing first.
8. Keep BUILD_PLAN.md authoritative for design; if a decision changes mid-build, update it there and
   note it in the Session log.

Report progress at each gate so the user stays in the loop without reviewing everything at once.

## Cross-model collaboration (`/collab` — Claude × Codex)

The design was produced by a Claude×Codex collaboration; Codex caught three defects that would have
silently corrupted the sampled distribution (see `.collab/specs/collab-outcome.md`). Keep it in the
loop **where a subtle error corrupts the target distribution and no ordinary test would catch it** —
not everywhere.

**Required — `/collab` think mode, as a gate step before closing these milestones:**

| Milestone | What Codex must adversarially review |
|---|---|
| **M2** 1D kernel | Exactness of the piecewise-exponential conditional; `expm1`/`log1p` stability across all κL regimes; breakpoint handling (no tolerance-merging of distinct cuts); chord keeps every nonzero component |
| **M4** Geometry | Completeness of the deterministic span certificate; scaling/tolerance coupling to LP feasibility tolerance |
| **M5** Rounding + β=0 | Gibbs/coordinate-hit-and-run **stationarity** argument; transform frozen; uniform-target correctness |
| **M6** Positive-β | That the sampled law is exactly π_β (no hidden approximation); `s_J`/`J*` handling; mean-J monotonicity |
| **M7** Reweighted-L1 | That weights are frozen before sampling and `J` never changes mid-chain (would invalidate stationarity) |

**Also use `/collab`:**
- **think** — for any design fork BUILD_PLAN.md does not already settle.
- **debug** — when an analytic/statistical test fails and the root cause is not obvious after one
  pass (independent competing hypotheses; a discriminating test decides).
- **build** — optional, for a well-specified isolated module with a clean interface + test command
  (write the spec to `.collab/specs/`). Review the diff and run the tests yourself; never let Codex
  and a Claude subagent touch the same files.

**Do not use `/collab`** for routine scaffolding, mechanical edits, or anything BUILD_PLAN already
settles — it costs latency and adds nothing.

When a collab round changes a decision, record it in `.collab/specs/collab-outcome.md`, update
BUILD_PLAN.md, and note it in the Session log.

## Conventions (non-negotiable — derived from BUILD_PLAN.md §1 + spec)

- **Environment**: Python **3.11**, `uv` venv, src layout, `uv pip install -e .`. Wheel-only installs.
- **No SciPy in the numerical path.** cobra is a parser/metadata layer only. Build LPs from native
  NumPy CSC arrays and `highspy.Highs.passModel` — never `scipy.optimize.linprog`, never
  `scipy.sparse`, never optlang for the computational LP.
- **No HiGHS solve in the MCMC inner loop.** Instrument a solve counter; assert it is unchanged after
  sampling starts.
- **float64 everywhere in computation.** float32 only as an explicit *storage* option.
- **Never snap small fluxes to zero** (thresholds apply only in analysis). Auxiliary `z=|v|` variables
  are LP-only and are never sampled. Keep **all nonzero** direction components in the chord.
- **Reproducibility**: `numpy.random.SeedSequence` keyed on `(model_id, stage, β_index, chain_index)`;
  store spawn keys. HiGHS threads=1 for geometry; OPENBLAS/OMP/MKL threads=1 in MCMC workers.
- **Fixed reactions** (l==u) are eliminated from the sampled state (reduced polytope IR) but every
  saved sample is a full-length flux vector with reaction IDs + fixed-status metadata.
- **Math-first**: the 1D kernel (M2) and geometry must pass analytic tests before any genome-scale MCMC.
- **Tests + `ruff` + type checks must pass** before a milestone is checked off. Prefer small typed
  dataclasses across the numerical core.

## Key files

| File | Role |
|---|---|
| `DEVELOPMENT_STATUS.md` | Live progress tracker — where we are / what's next (update every session). |
| `BUILD_PLAN.md` | Design, milestones, acceptance gates, cross-cutting decisions. |
| `GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md` | Original math spec (source of truth for the method). |
| `.collab/specs/collab-outcome.md` | Locked decisions from the Claude×Codex design collaboration. |
| `models/` | Example GSMMs from `metabolicSubcommunities` method_3_curated (see memory: model provenance). |

## Example model

`models/GCF_000010425_1_..._noO2.json` = *Bifidobacterium adolescentis* ATCC 15703 (anaerobe).
773 reactions, 894 metabolites, biomass `bio1`, 513 reactions fixed at 0 (FVA-blocked under the
anaerobic medium — provably unable to carry flux). Effective sampling dimension **d ≤ 55**.
