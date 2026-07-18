# M11.5(a) — a dimension-scaled sampling schedule — BUILD SPEC (handoff)

**Status:** spec only, not built. This is the build-ready handoff for a next session. Read it, run the
**MEASURE-FIRST** step, run **`/collab` think** on the design fork, *then* build.

## Objective

The MCMC schedule (`burn_in`, `n_samples`, in *sweeps*) is a flat default tuned on the d=46 anaerobe.
The M11.4 census (`benchmarks/M11_4_CENSUS.md`) measured that mixing efficiency degrades ~linearly with
dimension, so an aerobe at that flat default is under-sampled. Make the effective schedule scale so a
run reaches a target sampling quality regardless of `d` (and, ideally, `β`), instead of a fixed sweep
count that means "well-mixed" at d=46 and "not converged" at d=145.

## Measured premise (from M11.4) — and what is NOT yet measured

**Known (census):**
- Mixing is *slow-but-healthy*, not stuck: ESS scaled **3.19× for a 4× sweep increase** (Rahnella β=0,
  2k→8k) — near-linear, so ESS ∝ sweeps and a larger budget reaches any target.
- Median flux-ESS at the fixed 2000-sweep budget: **270 (d=46) → 200 (d=71) → 63 (d=145)** — ~4× drop.
- ESS also degrades with β (lactis: 239 at β=0 → 90 at β=16; Rahnella d=145: 63 → 45).
- **J-only ESS under-reports the problem** — size on flux/coordinate ESS, never on `ESS(J)` alone
  (a chain mixes in J while trapped along objective-neutral directions).

**NOT yet measured — the MEASURE-FIRST step must produce this before a rule is chosen:**
- The **integrated autocorrelation time τ as a function of (d, β, conditioning)** across more than 4
  strains. The census has 4 sentinels; a scaling *rule* fitted to 4 points is a guess.
- **Whether a β=0 (geometry-pilot) τ predicts the β>0 production τ.** The census shows β inflates τ, so
  a schedule sized only on the β=0 pilot will under-size the high-β rungs. Quantify the inflation.

`benchmarks/census_diag.py` already computes per-reaction ESS; extend it to emit
`τ = n_samples / ESS` per movable reaction (worst + high percentile), and run it across a wider strain
set at β ∈ {0, 1, 8, 16}. That measurement chooses the rule below.

## The design fork — run `/collab` think BEFORE building

BUILD_PLAN does not settle this; the `/collab` rules require **think mode for a design fork**. It is
**not** a distribution-corrupting change (a longer/shorter chain samples the *same* π_β with more/less
MC error), so it is **not** an M2/M4/M5/M6/M7 math gate — no closing `/collab` is required. The fork:

- **(A) Fixed heuristic** `N(d) = base · (d / d_ref)^p`. Cheap, deterministic, needs no extra measurement
  at runtime — but ignores β and conditioning, which the census shows both matter. A guess dressed as a
  rule.
- **(B) Target-ESS from pilot autocorrelation (recommended direction).** The pilot chains already run;
  measure the worst-movable-coordinate (or a high-percentile) τ from the pilot trace and size production
  to a declared target ESS: `n_samples ≈ target_ESS · τ`, `burn_in ≈ k · τ`. Principled, reuses existing
  infrastructure (`diagnostics.effective_sample_size`). Sub-forks for `/collab`: size on worst-coordinate
  vs a percentile vs J (census says not J); a single β=0 pilot vs per-rung pilots; and the β>0 inflation.
- **(C) Doubling-until-target** (the census's own method): run the base, measure flux-ESS, extend
  (2k→4k→8k…) until the target or a resource cap. Most robust, most compute. Report "mixing time exceeds
  budget" as a legitimate outcome, never widen the target to pass. ⚠️ **CORRECTED (M11.5 `/collab`):
  the restart guard does NOT support resumption of a changed schedule** — `batch._already_done` *raises*
  on a changed `recipe_key` (a larger `n_samples` changes it) and no per-chain RNG checkpoint is stored,
  so C is *re-run longer in a fresh directory, regenerating from seed*, not resume/extend. This is why C
  is deferred: it needs either a checkpoint mechanism or a fresh-dir re-run loop, both out of M11.5(a)'s
  scope.

A **B+C hybrid** (pilot-sized initial guess, doubling backstop to the target) is likely the honest
answer, but let the MEASURE-FIRST data + `/collab` decide.

## THE TRAP — the resolved schedule must be keyed (do not skip)

`batch.py`'s sample `content_key` **already hashes** `burn_in`, `n_samples`, `thin`, `refresh_interval`
("the schedule moves the draws"). So an adaptive schedule is safe **only if the *resolved* (effective)
values are what flow into the key and into `run_chains`** — never the raw config default. This is the
exact pattern `energy_scale_value` (the pilot's σ̂₀) already follows for `s_J`: the *mode* is config, the
*resolved number* is keyed. Requirements:

1. Resolve the schedule to effective `(burn_in, n_samples[, per-β])` **after** the pilot and **before**
   the content key is computed, so the key reflects what actually ran.
2. The resolution must be a **deterministic function of the pilot** (which is itself keyed and
   reproducible), so the effective schedule is reproducible — no fresh RNG in the sizing.
3. Record base config **and** resolved schedule **and** the pilot τ it was derived from in the manifest
   (M10.2e "visibility beside elimination"; recompute-don't-store per M10.2b — the resolved schedule is
   recomputed from the keyed pilot, so it is evidence, not a stored claim).

Get this wrong and two runs of the "same config" draw different numbers and collide in the cache — the
M10.2b hazard, one layer out.

## Hook point + files

- **`src/gsmm_compiler/config.py`** — add `SamplerConfig` knobs: `target_ess: int | None`,
  `schedule_mode: "fixed" | "target_ess"` (default `"fixed"` — the change is opt-in and back-compatible),
  and a cap for the doubling backstop. Counts stay in sweeps.
- **A new `resolve_schedule(sampler, geometry, pilot) -> SamplerConfig`** (likely in `calibration.py`
  beside the pilot, or a small `schedule.py`) — pure, deterministic, returns the effective config. When
  `schedule_mode == "fixed"` it is the identity (so every existing test and cache key is unchanged).
- **`src/gsmm_compiler/calibration.py` / `batch.py`** — call `resolve_schedule` after the pilot, before
  the production `run_chains` and before the content key; feed the resolved config to both.
- **DO NOT TOUCH** the transition kernel (`maxent_sampler.run_chains` inner loop), the target law, the
  reachability/rounding certificates, or `HighsLinearProgram`. This milestone changes *how many* draws,
  not *what* is drawn.

## Verification

- A resolved `target_ess` schedule at d=145 reaches the target median flux-ESS where the flat default did
  not; determinism: same inputs → bit-identical resolved schedule and draws.
- **Keying**: two configs with different `target_ess` MISS each other in the cache; the same config HITS;
  a resolved schedule never collides with the flat default it replaced. (A regression test that fails if
  the *raw* config, not the resolved one, reaches the key.)
- No correctness regression: feasibility / T₀-vs-T₁ invariance / mean-J monotonicity unchanged (re-run
  the census oracles). `schedule_mode="fixed"` is byte-identical to today.
- Full suite green + `ruff` + `mypy` clean; new tests for `resolve_schedule` (fixed = identity; target
  mode monotone in d; deterministic) and the keying regression.

## Ordering (do not reorder)

1. **MEASURE first** — extend `census_diag.py` to τ, sweep a wider strain set at several β; this chooses
   A/B/C and any exponent. A rule fitted to n=4 is the thing this repo keeps burning itself on.
2. **`/collab` think** on the fork with the measurement in hand.
3. Build `schedule_mode="fixed"`-default + `resolve_schedule`, keyed, tested.
4. Re-run the census oracles to confirm no correctness drift.
