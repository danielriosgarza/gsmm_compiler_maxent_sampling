# M11.5(a) MEASURE-FIRST — the autocorrelation time τ across dimension and β

**Question.** The M11.5(a) spec must choose a schedule rule (fixed d-power vs pilot-target-ESS vs
doubling). The M11.4 census measured the efficiency drop on **4 sentinels**; the spec forbids fitting
a rule to 4 points. This sweeps the **integrated autocorrelation time** τ across **9 strains spanning
d = 34…145** at **β ∈ {0, 1, 8, 16}**, under the production config, and asks: *how does τ scale with
d? with β? and can a β=0 pilot predict the production schedule a run needs?*

**τ, precisely.** `benchmarks/census_diag.py` (extended in M11.5) reports, per movable reaction,
`τ_int = n_chains·n_samples / ESS_pooled` — the integrated autocorrelation time in sweeps, the
frame- and chain-count-independent physical quantity (ESS_pooled = n_chains·n_samples / τ_int). It is
summarized as **worst** (max over movable reactions — the bottleneck coordinate that governs
convergence), **p90** (robust to a single noisy coordinate), and **median**.

**Method.** 9 strains (three of them the M11.4 census sentinels, so the numbers cross-check), each at
the census config — `energy_scale=pilot_sd`, `pilot_reround=true` (so the chain steps in **T₁**, the
frame a resolver's scale pilot measures in), 4 chains, seed 0, 2000 burn-in + 2000 retained sweeps.
Total wall-clock **1002 s** (`--workers 8`), 47 s (d=34) → 270 s (d=145) per strain.

**Cross-check against M11.4 (sanity).** β=0 median flux-ESS: **bifido d=46 → 270, pentosus d=71 →
200, Rahnella d=145 → 63** — *identical* to the census's 270/200/63. The measurement reproduces the
census on its own sentinels before extending past them.

## 1. τ vs d at β=0 — super-linear, statistic-dependent, and **scattered**

| statistic | fit τ_int ∝ d^p | τ@d=46 | τ@d=145 | d46→145 ratio |
|---|---|---|---|---|
| worst | d^**2.23** | 54 | 705 | 13.0× |
| **p90** | d^**1.63** | 39 | 255 | 6.5× |
| median | d^**1.18** | 29 | 112 | 3.9× |

The **median** exponent (~1.2, ratio 3.9×) reproduces the census's "clean ~4× ESS drop". But the
schedule must serve the *slow* coordinates, and on **p90** the scaling is **super-linear (~d^1.6,
6.5×)**; on worst it is d^2.2. **The exponent is not a constant of the method — it depends on which
quantile you protect**, which is the first reason a single fixed d-power rule is a guess.

The second reason is **scatter at fixed d**: p90-τ_int is 50.6 (bifido d=46) vs 40.5 (lactis d=51) —
the larger model mixes *better* — and 56.2 (gilvus d=70) vs 77.1 (pentosus d=71) at essentially equal
d. A global d-power rule carries this ±1.5–2× strain-to-strain error *even at β=0, even before β*. A
per-model pilot measures **this strain's** τ and removes it.

## 2. β inflates τ — hugely, and **not as a function of d** (the decisive finding)

p90-τ_int(β) / p90-τ_int(0), per strain:

| strain | d | β=1 | β=8 | β=16 |
|---|---|---|---|---|
| parvulus | 34 | 1.07 | 3.45 | 5.71 |
| ethanolidurans | 42 | 1.12 | 2.56 | 5.50 |
| bifido (ctrl) | 46 | 0.80 | 4.60 | **26.80** |
| lactis (S) | 51 | 1.04 | 3.32 | 4.65 |
| kefiri | 58 | 1.11 | 1.96 | 5.03 |
| gilvus | 70 | 1.37 | 2.01 | 8.28 |
| pentosus (S) | 71 | 0.64 | 1.78 | 4.49 |
| pseudomonas | 109 | 0.70 | 1.87 | 2.19 |
| Rahnella (S) | 145 | 0.84 | 1.47 | 4.02 |
| **mean** | | **0.97** | **2.56** | **7.41** |
| **max** | | 1.37 | 4.60 | **26.80** |

Two things, and both matter for the fork:

- **β=1 does not inflate τ at all** (mean 0.97× — some strains mix *better* tilted). Consistent with
  the census: at s_J calibration β=1 barely tilts. The inflation switches on at β≳8.
- **β=16 inflates p90-τ by a mean 7.4× and up to 27×**, and **it is not predictable from d**: the
  largest inflation is the *smallest* model (bifido d=46, 26.8×) and the smallest inflation the
  *largest* (Rahnella d=145 → 4.0×; pseudomonas d=109 → 2.2×). It is organism- and rung-specific.

**Consequence:** a schedule sized only on a **β=0** measurement — whether a fitted d-rule (A) or a
β=0 pilot (B) — under-sizes the high-β rungs by this factor. No β=0 quantity carries it. This is the
single most important number in the sweep: **the β axis needs its own evidence, measured on the tilted
chain, which only the production run (or a per-rung pilot) has.**

## 3. "worst" is a noise floor; p90 is the honest target

worst/p90 ratio: 1.5 (β=0) → 3.2 median, **6.2 max** (β=16). At β=16 the worst coordinate has minESS
**3–9** — split-R̂/ESS at the estimator's floor for a near-constant reaction, not a measurement of
mixing. Sizing on **worst** chases that noise (the sizing preview below sends bifido β=16 to 135 k
sweeps off one 26.8× outlier). **J-only hides the problem** (census: R̂(J)=1.07 while flux worst
R̂=1.48). So the target statistic is a **high percentile of the flux/coordinate τ** — p90 here — not
worst and not J.

## 4. Sizing preview — what a target-ESS schedule would actually ask for

Sweeps to reach **p90-ESS = 400** from a 2000-sweep β=0 pilot, `n_new = 2000·400/ESS_pilot,p90`:

| strain | d | ESS_p90(β0) | n_new(β0) | ×β16 infl | n_new(β16) |
|---|---|---|---|---|---|
| parvulus | 34 | 299 | 2 675 | 5.71 | 15 274 |
| bifido | 46 | 158 | 5 063 | 26.80 | 135 685 |
| lactis | 51 | 197 | 4 052 | 4.65 | 18 829 |
| gilvus | 70 | 142 | 5 616 | 8.28 | 46 477 |
| pentosus | 71 | 104 | 7 712 | 4.49 | 34 615 |
| pseudomonas | 109 | 46 | 17 295 | 2.19 | 37 845 |
| Rahnella | 145 | 27 | 29 554 | 4.02 | 118 920 |

The β=0 column is sane (1.3×–15× the current 2000 default, monotone in d). The β=16 column shows the
inflation is real budget, and the bifido outlier (135 k, driven by a single p90 spike to 1357) shows
why a hard target on an erratic β-rung needs a **cap** and an honest "mixing exceeds budget" outcome,
not an unbounded chase.

## What this means for the design fork (input to `/collab`)

- **(A) Fixed d-power `N(d)=base·(d/d_ref)^p` is refuted by §1 and §2.** The exponent is not a
  constant (1.2/1.6/2.2 by statistic; it *falls* with β as τ saturates into noise), there is ±1.5–2×
  per-strain scatter at fixed d, and it ignores a β-inflation of up to 27×. It is precisely "a guess
  dressed as a rule".
- **(B) Pilot target-ESS is the right *mechanism* for the d axis.** The β=0 pilot measures *this
  strain's* τ (no fitted exponent, no fixed-d scatter) and the census's linear ESS∝sweeps law makes
  `n ≈ target·(n_pilot/ESS_pilot)` valid. Sub-forks the data settles: **p90, not worst** (§3) **and
  not J** (§3); **the β=0 pilot alone cannot size β>0** (§2).
- **(C) Doubling-until-target is the only honest handle on the β-inflation** that no β=0 pilot can
  predict (§2) — measure the *production* rung's ESS and extend to target or a cap. The restart guard
  already resumes on a changed schedule.

**The measurement points at a B+C hybrid:** pilot-p90-τ sizes the initial per-(model) schedule
(captures d-scaling and per-strain variation for free), and a doubling backstop to a declared target
handles the β-rungs, capped, reporting "mixing exceeds budget" rather than widening the target. The
exact split between B and C — and whether β>0 rungs get a conservative multiplier, a per-rung pilot,
or only the doubling backstop — is the `/collab` question.

## Reproduce

```
scratchpad/schedule_sweep.py     # runs the 9 strains × 4 β, writes sweep/tau_sweep.json
scratchpad/analyze_tau.py        # the four tables above + the power-law fits
benchmarks/census_diag.py <run>  # per-run τ (worst/p90/median tau_int), the extended summarizer
```
