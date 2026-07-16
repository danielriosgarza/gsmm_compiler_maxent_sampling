# M9 benchmark report — GSMM-Compiler MaxEnt Sampler

Regenerate with:

```bash
gsmm-compiler maxent benchmark models/GCF_000010425_1_..._noO2.json --report benchmarks/bifido_benchmark.json
```

**Platform.** Jetson (aarch64), Linux 6.8.12-tegra, 14 cores · Python 3.11.15 · numpy 2.4.6 ·
cobra 0.31.1 · highspy 1.15.1. BLAS/OMP/MKL threads pinned to 1. Every stage is a **cold** cost —
the L3 cache is bypassed.

**Model.** *Bifidobacterium adolescentis* ATCC 15703, anaerobic. 773 reactions · 894 metabolites ·
260 free · **d = 46** · 61 FVA-blocked · `step_scale_ratio` 0.0081 · `cond(C_ε)` 1.5e4 · λ̃ = 0.5
(λ = 9.4e-4, λ\* = 1.9e-3) · `s_J` = 32.5.

---

## 1. Where the wall-clock goes

| stage | median | spread | per unit | unit |
|---|---|---|---|---|
| parse | 0.5377 s | 118% | 537.7 ms | call |
| csc_assembly | 0.0139 s | 170% | 13.9 ms | call |
| reduce | 0.0003 s | 145% | 282 µs | call |
| pass_model | 0.0004 s | 128% | 417 µs | call |
| first_lp (cold) | 0.0029 s | 15% | 2.9 ms | call |
| warm_start_lps | 0.0538 s | 2% | **1.08 ms** | LP |
| sparse_lp | 0.0086 s | 7% | 8.6 ms | call |
| **geometry** | **1.1701 s** | 0.2% | 1170 ms | call |
| rounding | 0.0040 s | 8% | 4.0 ms | call |
| kernel_uniform (β=0 draw) | — | 4% | **1.4 µs** | draw |
| kernel_tilted (β>0 draw) | — | 0.2% | **92.4 µs** | draw |
| kernel_breakpoints (`build_piecewise_j`) | — | 0.4% | **57.1 µs** | build |
| output float64 / float32 / reduced | — | ~14% | 49.4 / 42.6 / **34.4** µs | sample |

**Sweep rates** (slope of two schedules, so the per-chain fixed cost is removed rather than
amortized into the rate — see `benchmark.SweepRate`):

| β | sweeps/s | coordinate updates/s | fixed cost per chain |
|---|---|---|---|
| 0 | 1667.6 | 76 708 | 3.0 ms |
| 4 | 209.1 | 9 618 | 7.1 ms |

**A β>0 sweep costs 7.98× a β=0 sweep.**

### The headline: the tilted line kernel is the whole cost

One β>0 coordinate update costs ~104 µs, and it decomposes cleanly:

| component | µs | share of a β>0 update |
|---|---|---|
| `sample_line` total | 92.4 | **89%** |
|  └ `build_piecewise_j` | 57.1 | **55%** |
|  └ `log_segment_masses` + segment choice + inverse CDF | 35.3 | 34% |
| chord + incremental `v` update + RNG | ~11 | 11% |

The β=0 path corroborates it independently: 603 µs/sweep ÷ 46 = 13.1 µs/update, of which the uniform
draw is 1.4 µs — leaving ~11.7 µs of chord and incremental update, which is exactly the ~11 µs the
β>0 accounting leaves over.

### ⚠️ The breakpoint **sort** is not the hot spot — the M9 task's hypothesis was wrong

BUILD_PLAN asked for "breakpoint-sort profiling". Measured: the chords carry a **median of 2 interior
breakpoints, max 8**. `np.unique` costs ~12 µs of the 104 — real, but it is paying fixed NumPy call
overhead, not sorting anything. **You cannot optimize a sort of 2 elements.**

Where `build_piecewise_j`'s 57 µs actually goes:

| component | µs | note |
|---|---|---|
| `np.unique` (the sort) | ~12 | median 2 interior cuts — fixed overhead, not data |
| `PiecewiseLinearJ.validate()` | **11.8** | re-checks concavity + knot ordering on every construction |
| `baseline` via `evaluate_on_line` | **6.5** | absolute `J` — **`sample_line` never reads it** |
| gathers over `penalized_indices` | 0.8 | negligible |
| the rest (`diff`, `cumsum`, `concatenate`, `where`, `sign`, `sum`) | ~26 | ~15 tiny array ops |

**`validate()` + `baseline` = 18.3 µs = ~18% of every β>0 coordinate update is work the draw does not
use.** `baseline` is provably unused: M2 established that an additive constant of `J` cancels out of
`p(t)` exactly (which is why `J*` left the kernel's API entirely), and `sample_line` reads only
`knots`, `slopes` and `heights`. The remaining ~26 µs is NumPy per-call overhead across ~15
operations on arrays of ~259 elements — the cost is the *number of calls*, not the data.

> **Not acted on in M9, deliberately.** M9's deliverables are measure-and-assert; `line_distribution`
> is the math-critical M2 kernel, and removing a correctness check (`validate`) or changing a frozen
> dataclass's contract (`baseline`) is a design fork BUILD_PLAN does not settle. Recorded here as the
> measured lever if β>0 wall-clock ever binds. **M10** is the place.

### The other stages

- **`parse` (0.54 s) is the largest non-sampling cost** — larger than the sparse LP, the CSC assembly
  and rounding combined. It is cobra's JSON load, once per model, and it is not on any hot path.
- **Warm starts pay, quantified**: a cold build+solve is 2.9 ms; a warm re-solve off the retained
  basis is **1.08 ms** (2.7×). This is what makes the geometry's ~1100 LPs affordable.
- **`geometry` (1.17 s) is the expensive stage**, and it is the only one M8 caches. Reported cold.
- **`reduced` storage is the cheapest to write** (34.4 µs/sample vs 49.4 for float64) as well as the
  smallest — see §3.

---

## 2. Worker-count sweep — ESS(J) per wall-second

2 strains × 8 β × 4 chains = **64 units**; 200 burn-in + 200 sampling sweeps. `cache_dir=None`, so
every run does identical work. ESS is taken on **J** — the scalar the study reports — not summed over
the 46 coordinates, which would let a worker count look productive by mixing well in directions
nobody asked about.

| workers | wall (s) | ESS(J) | ESS/wall-s | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 114.2 | 217.3 | 1.90 | 1.00× | 100% |
| 2 | 60.7 | 217.3 | 3.58 | 1.88× | 94% |
| 4 | 33.8 | 217.3 | 6.42 | 3.38× | 84% |
| 7 | 23.0 | 217.3 | 9.46 | 4.97× | 71% |
| **14** | **14.7** | 217.3 | **14.77** | 7.76× | 55% |

**ESS(J) is identical to the last digit at every worker count — the spread is exactly 0.** That is
M8's determinism guarantee confirmed by an instrument that was not built to check it: every chain's
RNG is keyed on `(model_id, "sample", β_index, chain_index)`, never on a position in a dispatch
queue, so the *only* thing a worker count changes is wall-clock. If ESS had moved at all, this sweep
is what would have seen it.

**The efficiency decay is Amdahl, not contention.** The parent does all parsing, LP work and geometry
serially before the pool starts: 2.29 s per model, **4.0% of the 1-worker run**.

| workers | measured | Amdahl (f=0.040) | measured/predicted |
|---|---|---|---|
| 2 | 1.88× | 1.92× | 98% |
| 4 | 3.38× | 3.57× | 95% |
| 7 | 4.97× | 5.64× | 88% |
| 14 | 7.76× | 9.20× | 84% |

Amdahl's ceiling at this serial fraction is **24.9×**. The residual 16% at 14 workers is real
contention — memory bandwidth, and no core left for the parent — and it is the only part of the curve
the serial fraction does not explain.

**Recommendation: use all 14 workers.** Throughput rises monotonically and is best there
(14.77 ESS/wall-s); the 55% efficiency is the honest price and 7 workers is the knee if the machine is
shared. The lever on the ceiling is **M8's L3 cache**: `geometry` (1.17 s of the 2.29 s serial) is
cached and content-addressed, so a re-run of the same strains halves the serial fraction without
touching the sampler.

---

## 3. `reduced` storage mode — validated

| | full_flux f64 | full_flux f32 | reduced |
|---|---|---|---|
| write cost | 49.4 µs/sample | 42.6 µs/sample | **34.4 µs/sample** |
| array shape | (n, 773) | (n, 773) | **(n, 46)** |

- **Round-trip is correct to ~1e-13 relative, and not bit-identical.** `output.load_chain` lifts a
  stored `y` through `centre + T·y`, the same expression `_walk` stores — but `_walk` lifts **one row
  at a time** (a matrix–vector product) while `load_chain` lifts the whole `(n, 46)` block (a
  matrix–matrix product). Different BLAS kernels accumulate in different orders: 8642 of 26000 entries
  differ, by up to 4096 ULP (max **1.1e-13** relative).
- **This is not a defect and the tolerance is not a concession.** Neither rounding is more correct —
  they are two float64 evaluations of the same exact quantity — and 1.1e-13 is ~100× *below* the
  refresh drift (6e-12) the sampler already measures, and four orders below the 1e-9 feasibility
  tolerance. The test localizes it rather than hiding it behind a tolerance: applying the **per-row**
  lift reproduces the stored flux **bit-for-bit**, which pins the difference to the batching and rules
  out a mis-rebuilt transform.
- **The honest consequence, recorded**: byte-identity holds **within** a storage mode, not across
  them. M8's serial-vs-pool guarantee is untouched (same mode, same code path).
- The realised on-disk saving is below the 16.8× array ratio because the manifest and six trace arrays
  are written identically in both modes.

---

## 4. Performance invariants — all hold

`tests/performance/test_m9_invariants.py`. These are structural claims, not timings: a slower machine
does not change the answer.

| # | invariant | how it is asserted |
|---|---|---|
| 1 | **No HiGHS solve in the MCMC inner loop** | process-global solve counter unchanged across a genome-scale chain at β ∈ {0, 4}; `freeze()` turns the convention into an error |
| 2 | **No SciPy in the numerical path** | a *live sampling run* in a fresh interpreter imports no scipy (the module scan in `test_no_scipy.py` would miss a lazily-imported one in a rare branch) |
| 3 | **No Python loop in the chord** | AST scan of `feasible_chord`/`chord_on_support` + a per-**reaction** cost bar |
| 4 | **No element-wise highspy extraction** | AST scan: `solve()` has no loop, and every `getSolution()` field is the direct argument of one `np.asarray` |
| 5 | **No full reconstruction every step** | `to_flux` call count, asserted **differentially** |

Two are worth spelling out because the naive version of each cannot fail:

- **(3)** The bar is 50 ns/reaction, and both endpoints were measured *before* it was chosen: the
  vectorized chord costs **7.6 ns/reaction**, an equivalent interpreted loop **710 ns/reaction** — a
  **93× gap**. The bar sits 6.6× above the real cost and 14× below a loop's. The slope across two
  support sizes cancels the fixed call overhead, which at n=200 would otherwise make a perfectly
  vectorized chord look expensive per element.
- **(5)** Asserted differentially — two schedules differing only in `n_samples` must differ in
  reconstruction count by exactly the extra samples plus the extra refreshes (100 → 104). A raw count
  would fold in `dispersed_start`'s own reconstructions, whose number depends on how many times the
  start had to shrink. The absolute bound is asserted too, since the differential test alone would
  pass a walk that reconstructed every step *and* honoured the schedule on top.

---

## 5. What M9 changed beyond measuring

**The benchmark's first real act was to fail.** The worker sweep could not run: both strains died with
`RoundingError: ‖S·T‖ relative ... above span_tol`. The gate rejected a valid genome-scale geometry on
**8 of 24 RNG streams** — and since `model_id` keys the RNG, *a label decided whether a model could be
sampled*. See BUILD_PLAN §1.4.2 and `.collab/specs/collab-outcome.md` § M9. The mass-balance gate is
now a **reachable-state certificate** (334 LPs, ~0.5 s, `max_i R_i` = 3.6e-11…5.1e-11, a **1.41×**
spread where the old gate swung **373×**, 20–28× inside the contract).

That 0.5 s is included in the 4.0% serial fraction above — the certificate is a per-model,
parent-side, cached-with-the-geometry cost, and it does not touch the inner loop.
