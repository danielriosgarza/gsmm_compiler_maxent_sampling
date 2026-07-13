# Collab outcome log — locked decisions from Claude × Codex rounds

Four documents (`CLAUDE.md`, `BUILD_PLAN.md`, `DEVELOPMENT_STATUS.md`, and this one) point here for the
decisions that a cross-model review changed. Each entry records **what was believed, what the review
found, and what is now locked** — so a later session cannot quietly revert a fix whose reasoning it has
forgotten.

The rule these rounds exist to serve: *a defect that corrupts the target distribution is invisible to an
ordinary test.* The sampler will still run, still look converged, and still produce plausible fluxes. So
the math-critical milestones (M2, M4, M5, M6, M7) each buy an independent adversarial reading before their
gate closes.

---

## Round 0 — Design (pre-M0, 2026-07-13)

Two rounds, converged. Codex caught three defects that would have silently corrupted the sampled
distribution. They are recorded as **BUILD_PLAN §1.6 deltas 1–3** and are load-bearing:

1. **The chord must keep every nonzero direction component.** The spec's "ignore `|dᵢ|` below a direction
   tolerance" (§19) is a feasibility bug: a tiny `dᵢ` with `vᵢ` near its bound still binds a short, finite
   limit, and dropping it samples outside the bounds.
2. **Distinct breakpoints must never be tolerance-merged.** The spec's "merge breakpoints equal within
   tolerance" (§20.3) moves a bend of `J`, which changes `J`, which changes the target.
3. **`s_J` belongs to the objective layer, not geometry** (it is evaluated through `J`).

Plus **delta 4**: `J*` is a solver optimum, not a strict numeric upper bound on `J` — the log-density must
not assume the exponent is `≤ 0`.

Also locked: batch-aware v1; reweighted-L1 in v1 (M7); configurable sample storage; Python 3.11.

---

## M2 — the one-dimensional kernel (2026-07-13) — **4 rounds, converged**

| Round | Codex verdict | Outcome |
|---|---|---|
| 1 | DISAGREE — "do not close the acceptance gate", 5 points | all 5 conceded; **2 were real distribution-corrupting bugs in code that passed 264 tests** |
| 2 | DISAGREE, 5 points | 4 conceded (incl. one defect *introduced* by the round-1 fix); 1 severity claim rebutted with measurements |
| 3 | DISAGREE, 4 points | 3 conceded; 1 conceded in substance, rejected in remedy |
| 4 | **AGREE, contested: none** | gate closed |

The two rounds that mattered most were 1 and 2, and the lesson of round 2 is worth keeping: **the first
fix for a numerical bug is often itself buggy.** Both of round 2's real findings were defects in round 1's
repairs, not in the original code.

### Round 1

### 🔴 Locked: the absolute magnitude of `J` must never reach a probability

*The one that mattered.* `log_segment_masses` formed `h_a = β·(J(a) − J*)/s_J` from **absolute** knot
values, which were themselves propagated as `J(t_lo) + cumsum(...)`. Both steps destroy the only quantity
the target depends on — the knot heights *relative* to the chord — whenever `J` or `J*` is large.

Reproduced: with a biomass flux of `1e16`, the slopes came out exactly right (`[1, 0]`) while the true
segment probabilities `[0.387, 0.613]` came back as `[0.632, 0.368]`. **The favoured segment reversed.**
The chain would have converged, cleanly and confidently, to the wrong distribution.

Locked:
- `PiecewiseLinearJ` stores `heights` — knot heights **relative to `t_lo`**, `heights[0] == 0`, accumulated
  from the slopes and never routed through the absolute value — plus a separate `baseline` used only for
  reporting and `evaluate()`.
- **`log_segment_masses` takes no `J*` at all.** It provably cancels (it shifts every log-mass by one
  constant, which the log-sum-exp removes); carrying it bought nothing and invited the cancellation.
- Regression: `test_a_large_objective_baseline_does_not_corrupt_the_segment_probabilities`.

Making the cancellation structurally impossible beats testing for it — which is why `J*` left the API
rather than acquiring a guard.

### 🔴 Locked: a degenerate chord is a SELF-LOOP, never a redraw *(overrides spec §19)*

Spec §19 says to "reject and redraw a direction if the chord length is numerically zero", and BUILD_PLAN
§1.6 repeated it. **Redrawing a different coordinate breaks stationarity.** Random-scan Gibbs preserves
`π_β` because the coordinate is chosen *independently of the state*: the kernel is the uniform mixture
`(1/d)·Σ_k P_k`. Inspect the chord and pick a different coordinate when it comes back narrow, and the
mixture weights become functions of the current state — they can no longer be pulled out of the integral,
and the invariance argument collapses.

Locked:
- `sample_line` returns `0.0` on a chord with no positive width — a self-loop, which **is** the exact
  conditional when the feasible set on the line is the single point `t = 0`.
- **There is no minimum chord width.** `Chord.is_degenerate(min_length=1e-12)` is gone, replaced by
  `Chord.is_samplable = (length > 0)`. A 1e-13-wide chord has a well-defined conditional and is simply
  sampled. No tolerance is allowed to decide what belongs to the support.
- **M5 must not respond to a degenerate chord by choosing another coordinate.**

### Locked: the uniform branch is a representability limit, not an approximation

The first draft switched to a uniform draw below `|κL| < 1e-16`, which Codex correctly called a smuggled
approximation — the mass integral was still tilted while the draw was flat, so the two stages targeted
different laws. Now the threshold is `MIN_NORMAL = 2.2e-308` (the smallest normal double), where two facts
coincide: the tilt `e^{κL} = 1 + κL` is *bitwise* `1.0` in float64, and `−expm1(−|κL|)` would go denormal
and shed precision. The exact inversion therefore runs for **every `κ` whose tilt float64 can represent**.

### Locked: `κ = (β/s_J)·m`, grouped so it cannot overflow

`β·m` overflows to `inf` for `β = 1e308, m = 2` even when `s_J = 1e308` would return `κ = 2`. One `_tilt`
helper now computes the tilt for both `log_segment_masses` and `sample_line`, so the two can never weight
a segment with one `κ` and sample it with another.

### Locked: no unvalidated `support` parameter

`feasible_chord` briefly took a caller-supplied `support` array of nonzero indices, trusted without
checking. A truncated one silently reintroduces the exact tolerance bug of delta 1. Removed. **M5 builds
the per-coordinate precompute inside the library**, where the invariant holds by construction rather than
by caller discipline.

### Locked: the opening slope is fixed by side, never by midpoint *(overrides spec §20.3 step 6)*

Spec §20.3 step 6 reads the first segment's slope at its midpoint. When that segment is one ULP wide,
`0.5·(a₀ + a₁)` rounds onto the cut (round-half-to-even on `2·a₀`), the crossing flux there is exactly
`0.0`, `sgn(0) = 0`, and the slope returns as a subgradient — **2× off**. Measured: this fires in 10.5% of
thin-first-segment configurations (1260/12009). Each sign is instead fixed by which side of `τ_r` the
segment lies on, which is exact at any width. Codex verified the replacement case by case (`τ_r` at `t_lo`,
at `t_hi`, outside the chord, `d_r == 0`, empty penalized set).

---

### Round 2 — the fixes were buggy

#### 🔴 Locked: anchor each segment's mass integral at its HIGHER endpoint

Round 1's repair (heights relative to `t_lo`) fixed the *baseline* cancellation and missed the
*excursion* one. If `J` climbs 1e16 over the first segment and then 0.5 over the second, `heights[1]` and
`heights[2]` store the same float and the 0.5 — a factor of `e^{0.5}` in the weights — is gone. All three
segments came out equally likely against a true `[0.145, 0.377, 0.478]`.

The fix has two halves, and *both* are needed:

- **`heights` are anchored at the peak of `J`**, accumulated outward from it along the slopes. The peak is
  located from the **slopes alone** (`peak = count(slopes > 0)`, exact by concavity), so a corrupted height
  cannot mislocate it. Every height is `≤ 0`, the largest exactly `0`.
- **Each segment's mass is anchored at its higher endpoint**, using
  `M = e^{h_a}·L·φ(+κL) = e^{h_b}·L·φ(−κL)` (since `h_b = h_a + κL`). Anchoring at the peak alone is *not*
  enough: a long rising segment far below the peak has a huge negative `h_a` and a huge positive `κL` that
  cancel to something `O(1)`, and float64 cannot hold that cancellation in the low-order bits of a number
  of magnitude 1e16.

A structural bonus: the argument of `log φ` is then **always ≤ 0**, so no positive number is exponentiated
anywhere in the mass path — by construction, not by branch discipline.

#### 🔴 Locked: a degenerate chord moves to its feasible point, which is not always `0.0`

Round 1's self-loop returned a hard-coded `0.0`. But a state that has drifted a hair *outside* a bound
(within `feasibility_tol`) yields a singleton chord at some `t ≠ 0`, and returning `0.0` there pins the
state outside its bound **forever** — every later visit to that coordinate makes the same non-move.
`Chord.degenerate_point` is now the midpoint of the collapsed interval: `0.0` for an on-bounds state (the
true self-loop), and the one feasible `t` for a drifted one.

#### Locked: a non-finite chord raises

A direction component small enough to overflow its bound ratio (a denormal) makes the intersection
`[−inf, +inf]`; that reached `rng.uniform(-inf, inf)` and died with `OverflowError` inside the sampler.
`feasible_chord` now raises `UnboundedChordError`.

#### Rebutted: the `MIN_NORMAL` "collapse"

Codex reported that the round-1 threshold made the inverse CDF "return `x = 1.0` where the true quantile is
`1.1e-16`". **Measured over 200k draws, the law stayed uniform** (mean 0.500180, var 0.083197, zero draws
pinned at an endpoint); the error was ≤ 1 ULP, confined to quantiles of probability ~1e-15. The comparison
was against the *non-reflected* inverse — for `κ > 0` the draw inverts from the far end, so `x ≈ L(1−u)` is
correct. Codex accepted the correction. The threshold moved anyway (see below), but the docstring says what
actually happened rather than the dramatic version.

---

### Round 3

#### Locked: `UNIFORM_LIMIT = eps/4`, not `eps/2` — float spacing is asymmetric about 1.0

The elegant catch of the review. The uniform branch's exactness claim is "below this `|κL|`, `exp(κx)` is
bitwise `1.0` across the whole segment". Spacing is `eps` *above* 1.0 but `eps/2` *below* it, so the
low-side rounding midpoint is `1 − eps/4`: a **negative** tilt in `(−eps/2, −eps/4)` rounds `1 + x` down to
`nextafter(1.0, 0)`, not to `1.0`. An `eps/2` limit holds the claim for `κ > 0` and quietly breaks it for
`κ < 0` — exactly the half of an argument that is easy to forget to check.

#### Locked: `β/s_J` is validated, not trusted

`β·m` overflows for `β = 1e308, m = 2` even when `s_J = 1e308` returns `κ = 2`; grouping `β/s_J` first fixes
that but breaks the mirror case, where `β/s_J` **underflows to zero** while the true `κ` is an ordinary
2e-16 — and that one is *silent*, flattening the tilt with no error. No ordering is safe for every input,
so the ratio is computed **once**, in a validated `_inverse_temperature`, which raises on either pathology.
Computing it once also structurally guarantees the mass stage and the sampling stage use the *same* `κ`.

The hot-path arithmetic was deliberately **not** contorted to almost-survive these regimes: `λ` is a
penalty weight of order 0.1–10, `s_J` an energy scale of order 1–1e3, `β` a ladder value of order 0–1e3.
Trading a loud failure for a quiet approximation is the wrong direction.

#### Locked: an empty feasible set raises; it is not a degenerate chord

A **raw** crossed chord (`t_lo > t_hi` before the nudge) means no `t` satisfies every bound — `v` violates
opposing bounds and no move along the line repairs it. Round 2's `degenerate_point` happily returned its
midpoint and left the state out of bounds while reporting success. `feasible_chord` now raises
`InfeasibleChordError`. `degenerate_point` is reachable only from a true singleton or a nudge-inverted
one-ULP interval — both legitimate.
