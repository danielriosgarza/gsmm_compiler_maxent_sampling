# M11.4 — the end-to-end census: plan for /collab design-review

**Goal.** Geometry now builds on all 40 curated strains (M11.0–M11.3), but **rounding, the pilot DAG,
and the sampler have only ever run on the one anaerobe** (*B. adolescentis*, d=46). Every downstream
premise in the plan (§1.2 "at d≤55 sequential wins", the worker sweep, mean-J monotonicity, the
R̂/ESS bars, `s_J` stability) is a claim about that one organism. M11.4 **measures what breaks on an
aerobe** — it is a measurement, not a gate. "Make it pass" is explicitly not the objective.

## The subjects (chosen for a dimension spread, + the anaerobe as control)

| strain | d | note |
|---|---|---|
| *B. adolescentis* (control) | 46 | the only organism the pipeline has ever run on |
| *L. lactis* BIA2553 | 51 | just above the anaerobe; isolates "aerobe" from "large d" |
| *L. pentosus* | 71 | mid |
| *Rahnella aquatilis* | 145 | 3× the anaerobe — the stress case for cond/`s_J`/mixing |

## Config (production-shaped but bounded)

`--set sampler.energy_scale=pilot_sd --set sampler.pilot_reround=true`, a β-ladder (see Q3), a fixed
seed, `--workers 1` for determinism. Then `maxent diagnose` for the validity + convergence report.

## What to measure, per stage (the premise-measurement discipline)

1. **Rounding.** Does `build_transform` produce a `T` that certifies? `cond(C_q)` before/after
   re-round. Does the **re-rounded `T₁`** certify reachable mass balance? — M10.2e finding-4 (deferred)
   was that a `T₁` can be one `certify` cannot solve; **M11.3 may have already fixed that** by
   accepting the `kUnknown` duals, so the census should confirm whether re-rounded `T₁` on aerobes
   now certifies (and record `n_unknown_witnesses`).
2. **Pilots.** `run_pilot` completes; `s_J` finite and stable; `λ* = critical_l1_penalty` finite (an
   unbounded `J*` at a zero-cost growth path is legal but changes the axis); mean-J monotone across β.
3. **Sampler.** Chains complete; **no HiGHS solve in the inner loop** (solve counter unchanged after
   sampling starts); feasibility from `diagnose` — `max_bound_violation` and `max_mass_balance_residual`
   ≤ 1e-9; `max_r_hat_j` and `min_ess_j` at the chosen draw counts.
4. **Reproducibility.** `s_J` bit-identical cold vs warm cache (the L0/L3/pilot DAG).
5. **Performance.** Wall-time per stage; whether d=145 blows up memory / LP count / time relative to
   the d=46 control; whether the pilot still dominates `prepare_model`.

## Failure modes to watch (from this repo's own history)

- `cond(C_q)` growing with d → re-round fails or `T₁` won't certify (M10.2e finding-4, at d=145 unmeasured).
- `s_J` non-finite / unstable → the whole β axis breaks (spec §1.6.5); the pilot amplifies last-bit
  geometry differences at ~1e15 gain (M10.2e finding-3), and that was characterized only at d=46.
- **R̂/ESS bars sitting inside the distribution of valid runs** → false mixing failures. M10.2e's
  third lesson: the bars were a coin flip at 1500 draws on d=46, fixed by the *schedule* (R̂→1 is a
  theorem as draws grow). At d=145 the mixing time is unknown, so the draw count must be chosen so the
  bar is not a coin flip — this is Q2.

## My position (to be attacked)

Run the 4 strains (3 aerobes + control) at one production-shaped config, `diagnose` each, and read the
five stage measurements above. Expect the **most likely real break to be mixing at d=145** (ESS/wall
far worse than d=46) and the **second the re-rounded `T₁` certificate** — which M11.3 should now carry.

## Questions for Codex (design-review, before I run anything)

1. **Subject spread.** 3 aerobes + the control across d = 46/51/71/145 — is that the right axis, or
   is there a confound (three of four are Lactobacillales; only Rahnella is an enterobacterium)?
   Would a more phylogenetically diverse pick separate "aerobe" from "large d" better?
2. **Draw counts vs d.** What burn-in / n_samples makes R̂(J)/ESS(J) *meaningful* (not a coin flip)
   at d=145, given the mixing time is unknown? Is there a principled draws-vs-d rule, or must I
   measure the autocorrelation first and size the run to it?
3. **β-ladder.** Is `[0.0, 16.0]` enough to exercise `s_J` and mean-J monotonicity, or does the
   census need the full production ladder (8 rungs) to surface a monotonicity/`s_J` break?
4. **The M11.3 interaction.** Should the census explicitly assert re-rounded `T₁` certifies (the
   deferred M10.2e fragility), and treat a non-zero `n_unknown_witnesses` there as expected-and-fine
   rather than a warning?
5. **Blind spots.** What stage or metric could silently corrupt the sampled distribution on an aerobe
   but not on the anaerobe, that this plan does not measure?

---

## Refined protocol — adopted from /collab design-review (Codex DISAGREE, all 5 adopted)

M11.4 is a measurement, so Codex's DISAGREE was "measure more, don't declare success shallowly."
All adopted; the refined census is:

**1. Subjects are *sentinels* ("find the first downstream failure"), not an aerobe/d identification.**
The sampler sees `S, l, u, rhs`, not taxonomy — so before calling any subject "aerobic," record its
**O2-exchange bounds, FVA width, whether O2 flux is movable, and its β=0 distribution**. (Taxonomy
correction: B. adolescentis is Bifidobacteriales, L. lactis + L. pentosus Lactobacillales, R. aquatilis
Enterobacterales.) Four unpaired strains cannot *identify* an aerobe or a dimension effect — do not
claim they do; add a geometry-selected d≈65–80 strain later if a real design is wanted.

**2. Size runs from measured autocorrelation, not from d.** Time is in **sweeps** (1 sweep = d coordinate
updates), so equal `burn_in` already normalizes first-order — do **not** multiply by 145/46. Predeclared
doubling schedule: 4 chains × (2000 burn-in + 2000 retained), extend 4000 → 8000 → 16000 retained under
a fixed cap; **preserve and report the 2000-sweep result** (extension characterizes a mixing failure, it
does not erase it). Targets: pooled ESS(J) ≥ 400, R̂(J) < 1.01, successive-prefix stability within MCSE,
autocorr truncation comfortably inside the trace, `thin=1`. Cap reached → "mixing time exceeds budget"
is a *successful* measurement.

**3. Convergence for MORE than J.** `maxent diagnose` loads only `trace_j`; a chain can mix in J while
trapped along objective-neutral flux directions. Post-process the stored coordinate arrays: worst +
distribution of R̂/ESS over rounded coordinates, named fluxes (incl. O2 exchange), **chain-wise** means.

**4. Full ladder `(0, 0.25, 0.5, 1, 2, 4, 8, 16)`** on at least d=51 and d=145 (endpoints on all 4 is a
*partial* ladder, labelled so). Intermediate rungs expose localized numerical regimes the endpoints miss;
monotonicity is only meaningful after each rung has usable convergence.

**5. The blind spots (the part I most under-measured):**
  - **`max_refresh_drift`** per chain/rung (relative to chord length + bound slack) — the code states this
    perturbation has *no stationary-distribution bound*, and d=145 does more arithmetic between (sweep-based)
    refreshes. Plus degenerate-step fraction, `mean_chord_length`, `start_shrink`.
  - Rounding diagnostics that already exist: `covariance_rank`, `ridge_relative`, `step_scale_ratio`,
    `min_chord_at_center` — not merely `cond(C_q)`.
  - **Geometry-pilot quality** at d=145: per-coordinate pilot R̂/ESS, covariance rank, eigenvalue spectrum,
    T₁ sensitivity across pilot seeds.
  - **Parameterization invariance (a free oracle):** β=0, `pilot_reround` false vs true — T₀ and T₁ must
    target the *same* uniform flux law; compare named flux means/distributions within MCSE. Catches
    feasible-but-biased transforms that feasibility and J cannot.
  - **`s_J` stability ≠ cache bit-identity.** Cold-vs-warm measures cache correctness; for s_J stability use
    the scale impl's own measures (ESS of (J−J̄)², rel-SE of σ̂₀, R̂(J), between-chain SD ratio) and repeat
    control + d=145 under an **independent pilot seed** (the M10.2e amplification is about slightly-different
    geometry, which a warm replay deliberately avoids). Record transform hashes.
  - `critical_l1_penalty`: record finite / +∞ (legal zero-cost-growth) / error + resolved λ + regime —
    "finite" is not an acceptance premise.

**Execution order (cheapest, highest-signal first):** (i) d=145 Rahnella, β=0, initial 2000+2000 schedule
— does the largest case complete end-to-end at all, and what do coordinate-level R̂/ESS + refresh drift say?
(ii) β=0 T₀-vs-T₁ invariance on d=145. (iii) expand to the doubling schedule where mixing is marginal.
(iv) full ladder on d=51 + d=145. (v) the control + descriptors for all 4.
