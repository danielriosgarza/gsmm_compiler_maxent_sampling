# M11.5(c) — the full production census: the pipeline on all 40 strains, under the adaptive schedule

**Question.** M11.4 ran the pipeline on **4 sentinels** and found no correctness defect — the cost was
efficiency, which M11.5(a) addressed with a dimension-scaled schedule (`schedule_mode="pilot_ess"`)
measured on **9 strains**. This is the census the tracker's M11.5(c) names: the pipeline through the
**production `pilot_ess` sampler on all 40 curated strains**, plus two focused studies M11.4 deferred.
It is a **measurement, not a gate** — M11.4 established the pipeline is correct on aerobes, so this is
production-hardening: it confirms the *schedule* and *calibration* behave across the whole batch, not
just the strains their rules were fitted on.

Three components, run at the M11.4 census config + the M11.5 schedule (`energy_scale=pilot_sd`,
`pilot_reround=true`, 4 chains, 2000+2000 pilot, the 8-rung production ladder
`(0, 0.25, 0.5, 1, 2, 4, 8, 16)`, seed 0, `schedule_mode=pilot_ess`, `target_ess=400`,
`schedule_ess_quantile=0.90`). Deterministic. Reproduce:

```
benchmarks/census_m115.py [OUT]           # component 1 — the 40-strain sweep
benchmarks/analyze_census.py [OUT]         # component 1 — the aggregate tables
benchmarks/covariance_study_d145.py [OUT]  # component 2 — the d=145 covariance/eigenvalue study
benchmarks/sj_seed_replication.py [OUT]    # component 3 — s_J under independent pilot seeds
```

---

## 1. The 40-strain sampler sweep at `pilot_ess`

**36 of 40 strains sampled; 4 refused, fail-closed** (campaign 199 min, seed 0). The headline: on
every strain that built geometry, the production pipeline is **correct and predictable**, and the
adaptive schedule delivers its β=0 target. See `plots/fig3_census_scaling.png` and
`fig4_meanj_monotonicity.png`.

### 1a. The 4 refusals are the basis-marginal strains — and they fail *closed*

| strain | d | refusal |
|---|---|---|
| *Hafnia alvei* BIA2718 | — | `blocked_reactions`: 2 of 620 reactions unresolvable as blocked/moving |
| *Hafnia alvei* BIA2828 | — | `blocked_reactions`: 2 of 639 reactions unresolvable as blocked/moving |
| *Bacillus pumilus* BIA2784 | — | span certificate does not resolve to `span_tol` (the √k floor) |
| *Liquorilactobacillus satsumensis* BIA2521 | — | span certificate does not resolve to `span_tol` |

These are **exactly** the four the tracker flagged as "pass on this machine but not durably closed
(basis-marginal, §1.6.10)". The M11.3 census reported build-geometry "40 of 40" at the **default**
config/seed; under the **production** census config (`pilot_sd` + reround + `pilot_ess`, seed 0) they
refuse. That is not a regression — it is the census settling the caveat: **these 4 are not durably
closed.** Every refusal is a `GeometryError` *before* any sampling — no wrong numbers were produced,
and the harness recorded a fail-closed row and continued. The owed work is unchanged: a tighter span
certificate (pumilus/Liquorilactobacillus) and blocked-floor guidance (Hafnia), each its own step.

### 1b. On the 36 that sampled, the schedule delivers and the pipeline is valid

| property | result across 36 strains × 8 rungs | contract |
|---|---|---|
| **schedule resolves** | 36/36, **0 cap hits**; `n` scales 2601 (d=34) → 35289 (d=145), τ_q 26 → 353 | — |
| **β=0 target met** (p90-ESS ≥ 360) | **34/36** (2 near-misses at 347 & 314 — within 13–21% of 400) | the β=0 *prediction* |
| **max bound violation** | **0** | 0 |
| **max mass-balance residual** | **3.98e-11** | ≤ 1e-9 (~250× inside) |
| **max refresh drift** | 5.33e-11 | ~noise |
| **degenerate steps** | **0** | — |
| **mean-J monotonicity** | **36/36** monotone within 3·MCSE | the §1.6.2 theorem |
| **T₁ reachability certified** | **36/36**, min margin **6.0×** inside contract | certified |

Two β=0 near-misses (*Leuconostoc pseudomesenteroides* d=52 at p90-ESS 347; a *Lentilactobacillus*
d=67 at 314) are the schedule being **slightly optimistic**: the pilot τ under-predicted the
production τ by 13–21%. The schedule *predicts* the β=0 target from a keyed pilot; it does not
*guarantee* it, and `run_diagnostics` reports the achieved ESS separately (which is how these two are
visible at all). Both drew valid, monotone, certified samples.

The **β-inflation** the schedule deliberately does *not* correct (it is a β=0 prediction) is reported
per rung: median **6.2×**, mean 7.6×, max **20.3×** at β=16 — matching the τ-sweep's independently
measured 7.4× mean (`M11_5_SCHEDULE_TAU.md §2`). `s_J` reproduces the calibration exactly (Bifido
control `s_J = 2.52`; Rahnella `s_J = 29.0`, se 5.3%), and se(σ̂₀) climbs 2.2% (d=34) → 5.3% (d=145) —
the pilot-precision-vs-d finding of §3, now across the batch.

**Cross-check against M11.4:** the census reproduces every sentinel it shares with the 4-strain
M11.4 census (Bifido/pentosus/Rahnella `s_J`, se, and reround), so the pipeline measured itself twice
and agreed — the verification standing in for a formal `/collab` gate this production-hardening
milestone does not require.

---

## 2. The d=145 covariance / eigenvalue-rank study

**Question M11.4 left open.** M11.4 reported one covariance summary off each end of the dimension
range: re-rounding improves `cond(C_q)` **2.57×** at d=46 but makes it **0.60× (worse)** at d=145.
That is two condition numbers; it does not say *why*, and "why" is a property of the **eigenvalue
spectrum**, which no manifest stores. This study captures the full spectrum of both covariances —
`T₀` (the M4 support-vertex rounding) and `T₁` (the pilot re-rounding, spec §17.4) — at d=145
(Rahnella) and d=46 (the Bifido anaerobe), by spying on the code's own `_transform_from_coordinates`
during a cold build. **Validated by reproduction:** the recomputed ridged condition numbers equal the
manifest's to all printed digits (3.58e6 / 6.01e6 / 1.54e4 / 5.97e3).

| | rank(C_q) | cond(C_q) | step-scale √(λ_min/λ_max) | λ log₁₀-range | reround |
|---|---|---|---|---|---|
| **Rahnella d=145** T₀ (support, K=291) | **145 / 145** | 3.58e6 | 5.28e-4 | 7.4 | — |
| **Rahnella d=145** T₁ (pilot, K=8000) | **145 / 145** | 6.01e6 | 4.08e-4 | 7.3 | **0.60× worse** |
| **Bifido d=46** T₀ (support, K=93) | **46 / 46** | 1.54e4 | 8.07e-3 | 4.2 | — |
| **Bifido d=46** T₁ (pilot, K=8000) | **46 / 46** | 5.97e3 | 1.29e-2 | 3.8 | **2.57× better** |

**Finding 1 — the d=145 ill-conditioning is spectral *spread*, not rank collapse.** `C_q` is **full
rank at d=145** (145 of 145, deficit 0) for both estimators: the support points genuinely span every
direction, and nothing is being held open by the ridge alone. The `cond ≈ 3.6e6` comes from a
full-rank spectrum **7.4 orders of magnitude wide**, against 4.2 orders at d=46. The
"covariance-rank study" answer is clean: **no rank deficiency at any dimension the batch reaches.**

**Finding 2 — one near-flat direction sets the conditioning.** The spectrum is not uniformly spread.
Rahnella `T₀`'s eigenvalue deciles (log₁₀) are `[-7.4, -1.3, -1.1, -1.0, -0.8, -0.7, -0.6, -0.5,
-0.4, -0.3, 0.0]`: a **tight bulk** — 102 of 145 eigenvalues within one decade of λ_max, 144 of 145
within six — plus a **single outlier at ~10⁻⁷·⁴**. That one nearly-flat direction (a polytope axis the
model is extremely thin along) is what the ridge holds open and what alone drives the condition number.

**Finding 3 — the reround paradox is the pilot's mixing, and it points back at the schedule.**
Re-rounding replaces the support-vertex covariance with the *pilot's*. At d=46 the pilot mixes well, so
its covariance captures correlations the vertices miss and **compresses the bulk toward λ_min** (deciles
shift down, cond 2.57× better). At d=145 the pilot mixes *poorly* — its τ is large, which is the very
thing `pilot_ess` exists to fix — so it **under-estimates the variance along the slow directions**,
lowering λ_min and spreading the bulk (10th-percentile decile −1.3 → −2.4), and cond degrades 0.60×.
**The reround is only as good as the pilot's exploration.** A dimension-scaled pilot would make it help
even at d=145 — so this is not a defect in re-rounding but another face of the efficiency scaling the
schedule addresses. (It never threatens correctness: the re-rounded `T₁` certifies at every d — M11.3's
dual-witness path carrying it — and a preconditioner cannot move the target, §1.6.1.)

---

## 3. s_J under an independent pilot seed

**Question.** `s_J = σ̂₀` is estimated from one β=0 pilot, and each run reports a *within-pilot*
precision `se(σ̂₀)/σ̂₀` (2.6%…5.3% over d, M11.4). §1.6.6 / M10.2e claim `s_J` is reproducible **in
distribution, not bit-for-bit** — a different pilot seed is "another honest draw from the same pilot
law". That reported se is a **prediction** of how far `s_J` moves if you rerun the pilot; it had never
been checked against reality. This replicates the whole pilot DAG at **5 independent seeds** on three
strains and measures the **actual between-seed spread**.

| strain | d | s_J mean ± SD (5 seeds) | between-seed rel SD | mean reported se |
|---|---|---|---|---|
| Bifidobacterium adolescentis | 46 | 2.508 ± 0.079 | **3.1%** | 2.7% |
| Lactococcus lactis 2553 | 51 | 6.939 ± 0.230 | **3.3%** | 3.1% |
| Rahnella aquatilis | 145 | 28.84 ± 1.82 | **6.3%** | **6.3%** |

**Finding — the reported precision is honest.** The between-seed relative SD matches the mean reported
within-pilot se at every dimension (6.3% ≈ 6.3% at d=145). So the single-run precision warning is
telling the truth: **β's cross-strain label carries exactly the calibration uncertainty the manifest
states — no more.** This is a calibration finding, never a correctness one — every individual run
samples exactly `π_{β/s_J}` for its own `s_J`; the seed-to-seed variation is *which* calibrated target,
not an invariance failure (§1.6.6's "mixture of calibrated targets"). The se *itself* varies seed to
seed (Rahnella's ranged 4.7%…9.2%), but its mean predicts the truth — consistent with se being an
estimate whose *expectation* is the estimand. Bifido seed 0 returns `s_J = 2.5200`, reproducing
M10.2b's recorded `2.520009578949248` to the digit.

**Implication for the batch.** The se grows with d (2.6% → 6.3% over d=46 → 145) because the fixed
2000-sweep pilot buys fewer effective `J` samples at high d — the same dimension-scaling the production
schedule addresses, one layer up. A d-scaled *scale pilot* would hold se fixed across the batch; today
it is reported per strain and honest about itself, which is the §1.4.2 contract (a precision warning,
never a refusal).

---

## What the census establishes

The production pipeline — adaptive schedule, calibration, and sampler — is **correct and predictable
on every strain that builds geometry**, at up to 3× the dimension the model v1 was tuned on, and
**fails closed on the four it cannot**:

1. **36 of 40 sampled; 4 refused, fail-closed.** The four are precisely the basis-marginal strains
   (§1.6.10): the census settles the tracker's open caveat — they are **not durably closed**. No
   wrong numbers were produced.
2. **On the 36, validity is absolute** (bound violation 0, mass balance ≤ 4e-11, 0 degenerate steps,
   36/36 mean-J monotone, 36/36 T₁ certified) — the M11.4 guarantees, now batch-wide.
3. **The `pilot_ess` schedule delivers its β=0 target** on 34/36 (the 2 misses within ~20%, and
   *reported* as such), with 0 cap hits and `n` scaling cleanly with each strain's own τ. The
   β-inflation it does not correct is reported per rung (median 6.2×, matching the τ-sweep).
4. **The two calibration artifacts behave understandably at d=145** (components 2–3): `T₁` is
   full-rank with one explained hard direction, and `s_J`'s reported precision matches its actual
   between-seed spread.

No correctness defect surfaced — as in M11.4, the open work is **efficiency and the four geometry
residues**, not the distribution. That makes M11.5(c) production-hardening complete: the package does
what it claims on the batch it exists for, and refuses — visibly, closed — where it cannot yet.

## Figures

Regenerate all: `plot_census.py`, `biomass_activity.py`, `biomass_distribution.py {<strain>|all|gallery}`.

| figure | shows |
|---|---|
| `fig1_covariance_spectra.png` | §2 — T₀/T₁ eigenvalue spectra at d=145 and d=46 (full rank, one hard direction) |
| `fig2_sJ_replication.png` | §3 — s_J across 5 seeds; between-seed spread ≈ reported se |
| `fig3_census_scaling.png` | §1 — schedule resolution, β=0 target attainment, β-inflation, se vs d |
| `fig4_meanj_monotonicity.png` | §1 — normalized mean-J trajectories (36/36 monotone) |
| `fig5_biomass_activity.png` | biomass vs active-reaction count across strains (cross-strain, size-driven) |
| `fig6_biomass_distribution.png` | one model zoom — biomass distribution per β + bits→range tradeoff |
| `fig7_biomass_gallery.png` | biomass distribution for all 36 strains, sorted by dimension |
| `per_strain/<model>.png` | the fig6 pair for every one of the 36 sampled strains |
