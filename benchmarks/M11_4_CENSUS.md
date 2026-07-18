# M11.4 — the end-to-end census: does the pipeline work on an aerobe?

**Question.** M11.0–M11.3 got `build-geometry` (the *first* stage) working on all 40 curated strains.
But **rounding, the pilot DAG, and the sampler had only ever run on the one anaerobe** (*B. adolescentis*,
d=46). Every downstream premise (§1.2 "at d≤55 sequential wins", the worker sweep, `s_J` stability,
mean-J monotonicity, the R̂/ESS bars) was a claim about that one organism. M11.4 **measures what breaks
on an aerobe.** It is a measurement, not a gate.

**Design-reviewed by `/collab` before running** (Codex DISAGREE, all 5 adopted — see
`.collab/specs/m114-census-plan.md`). The load-bearing correction: `maxent diagnose` reports R̂/ESS for
**J only**, and a chain can mix in the objective while trapped along objective-neutral flux directions —
so a naive census would declare success on a badly-mixed distribution. This census post-processes the
stored **full-flux** arrays for per-reaction R̂/ESS (`scratchpad/census_diag.py`, mirroring the package's
own `convergence_report`), and reads the per-chain manifest diagnostics (`max_refresh_drift`, degenerate
steps, chord length) that `diagnose` never surfaces.

## Subjects (sentinels across a dimension spread — NOT an aerobe/d identification)

The sampler sees `S, l, u, rhs`, not taxonomy; four unpaired strains cannot *identify* an aerobe or a
dimension effect. They are sentinels for "find the first downstream failure".

| strain | order | d | n_free | movable rx (β=0) |
|---|---|---|---|---|
| *B. adolescentis* (control) | Bifidobacteriales | 46 | 260 | 199 |
| *L. lactis* BIA2553 | Lactobacillales | 51 | — | 262 |
| *L. pentosus* | Lactobacillales | 71 | — | 277 |
| *R. aquatilis* | Enterobacterales | 145 | 647 | 463 |

Config: `energy_scale=pilot_sd`, `pilot_reround=true`, 4 chains, seed 0, `--workers 1`. Base schedule
2000 burn-in + 2000 retained sweeps (1 sweep = d coordinate updates).

## Headline: the pipeline is **correct** on aerobes; the cost is **efficiency**

Nothing downstream produces an invalid or biased distribution on a d=145 aerobe — 3× the dimension of
anything the pipeline had seen. What degrades, predictably and measurably, is *mixing efficiency* and
*pilot precision*, both with dimension. The fixed budgets tuned on the d=46 anaerobe are under-powered
on aerobes by roughly the dimension ratio — a premise correction, not a bug.

### 1. Validity holds everywhere (feasibility, drift, degeneracy)

Across every strain and every β rung measured:

| metric | range over all runs | contract |
|---|---|---|
| max bound violation | **0** | 0 |
| max mass-balance residual | 3.4e-12 … 1.1e-11 | ≤ 1e-9 (≈100× inside) |
| max refresh drift | 2.0e-11 … 4.5e-11 | (no stationary bound; must be ~noise) |
| degenerate steps | **0** | — |

**Refresh drift** was Codex's top blind spot (the incremental flux cache has no stationary-distribution
bound, and d=145 does more arithmetic between sweep-based refreshes). Measured: it stays ~1e-11 across
*all* d and, on the d=51 ladder, is **flat across β** (2.5e-11 at β=0 → 2.0e-11 at β=16) — it does not
grow with the tilt. Closed.

### 2. Re-rounding preserves the target (parameterization invariance)

β=0, d=145, `pilot_reround` false (T₀) vs true (T₁) — the uniform target is known, so the two must agree
within MCSE. Over 463 movable reactions: **max |z| = 2.20, median 0.55, 0% beyond 3 MCSE** (z = mean
difference / combined MCSE; z ~ N(0,1) if the laws match). The largest-flux reactions agree to z ≤ 1.5.
The re-rounding change of coordinates does not bias the sampled law — first confirmation on an aerobe.

### 3. mean-J monotonicity holds on the aerobe ladders

Full 8-rung ladder `(0, 0.25, 0.5, 1, 2, 4, 8, 16)`, both aerobes: E[J] monotone across all adjacent
rungs — *L. lactis* d=51 (worst drop −2.1σ) and *R. aquatilis* d=145 (worst drop +1.17σ, i.e. strictly
increasing). Every rung's samples are valid (bound viol 0, mass-bal ≤ 1.8e-11, 0 degenerate steps). The
tilted target is exactly π_{β/s_J} at every rung on an aerobe.

### 4. The real finding — mixing efficiency degrades with dimension AND with β

Flux-level (not J-only) convergence at the fixed 2000-sweep budget:

| strain | d | median flux-ESS | worst R̂ | % reactions ESS<400 |
|---|---|---|---|---|
| control | 46 | **270** | 1.05 | 92% |
| pentosus | 71 | **200** | 1.06 | 95% |
| Rahnella | 145 | **63** | 1.48 | 98% |

A clean ~4× drop in effective samples per reaction from d=46 to d=145 at equal budget. Within a strain,
ESS also degrades with β — two full ladders measured:

| β | 0 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| *L. lactis* d=51, median flux-ESS | 239 | 227 | 240 | 200 | 177 | 120 | 90 |
| *R. aquatilis* d=145, median flux-ESS | 63 | 61 | 62 | 60 | 53 | 51 | 45 |
| *R. aquatilis* d=145, worst R̂ | 1.48 | 1.35 | 1.29 | 1.32 | 1.28 | 1.46 | 1.84 |

Two independent efficiency axes — dimension and β — both pointing the same way. At d=145 the chain is
already inefficient at β=0 and the tilt compounds it; at d=51 β must reach ~8–16 before ESS halves.

**J-only diagnostics hide this.** On Rahnella d=145 β=0, `diagnose` reports R̂(J) = 1.07 (looks marginal),
while the *flux* worst R̂ is 1.48 and median flux-ESS is 63 (not converged). Measuring only J — the pre-M11.4
default — would have declared success. This is exactly the failure the `/collab` review existed to catch.

**The under-mixing is a budget problem, not a pathology.** Doubling datapoint, Rahnella β=0:

| sweeps | median flux-ESS | min ESS | worst R̂ |
|---|---|---|---|
| 2000 | 63 | 12 | 1.48 |
| 8000 (4×) | **202** | 57 | **1.08** |

ESS scaled **3.19× for a 4× sweep increase (~linear)** and worst R̂ fell 1.48 → 1.08 toward 1. Near-linear
ESS-vs-sweeps is the signature of a chain genuinely *exploring* (stable autocorrelation time), not one
*trapped* (which plateaus). So ~4× the sweeps recovers the d=46 efficiency — matching the ~4× ESS
degradation from d=46→145. The sampler is correct *and* healthy on aerobes; it needs a dimension-scaled
schedule (a config default), not a code fix.

### 5. `s_J` pilot precision degrades with dimension

`se(σ̂₀)/σ̂₀` at the fixed 2000-sweep pilot: **2.6% (d=46) → 3.1% (d=51) → 5.3% (d=145)** (target 2.0%).
`s_J` remains a valid scale at every d (each rung is exactly π_{β/s_J}), but β's *label* carries this
error, so two strains compared at nominal β differ by it — the scale pilot needs a d-scaled length to
hold a fixed precision. (A warning by design, never a refusal — §1.4.2.)

### 6. Re-rounding's conditioning benefit is anaerobe-shaped

The re-round exists to improve `cond(C_q)`. On the anaerobe it does (1.54e4 → 5.97e3, 2.57× better). On
Rahnella d=145 it makes it **worse** (3.58e6 → 6.01e6, 0.60×), and 6e6 is ~1000× the anaerobe's ~6e3.
The re-rounded T₁ still certifies (‖Sv−b‖ 7.91e-11, 13× inside contract — M11.3's dual-witness path
carrying it), so this is an efficiency observation, not a correctness one, but the "re-rounding improves
conditioning" premise does not generalize off the anaerobe.

## What this means for the plan

- **§1.2's "at d≤55 sequential wins by default"** and the worker-sweep numbers are *efficiency* claims,
  and efficiency is exactly what's dimension-dependent — they must be re-derived per dimension, not
  inherited from d=46.
- **The base schedule (2000 sweeps, 2000-sweep pilot) is under-powered even at d=46** (92% of reactions
  below ESS=400) and needs to scale with d for a fixed ESS target. A production run at d=145 needs
  roughly the dimension-ratio more sweeps.
- **No correctness defect surfaced.** Feasibility, invariance, and monotonicity all hold on aerobes; the
  M11.3 reachability fix carries the re-rounded transform in production.

## Reproduce

```
benchmarks/census_diag.py <model_run_dir>   # flux-level R̂/ESS + per-chain diagnostics per β
```
Base config above; ladder = the 8-rung production ladder `(0, 0.25, 0.5, 1, 2, 4, 8, 16)`.

## Scope — what this census is and is not

This answers **"does the downstream pipeline work on an aerobe?"** on a **4-strain sentinel spread**
(d = 46/51/71/145) — enough to find the first downstream failure and to establish the efficiency
scaling. It is **not** a full production census: a 40-strain sweep through the sampler, a geometry-pilot
eigenvalue/covariance-rank study at d=145, and an `s_J`-under-independent-pilot-seed replication remain,
as does turning "scale the schedule with d" into a **config default** (M11.5). No correctness defect was
found, so those are production-hardening, not gates.
