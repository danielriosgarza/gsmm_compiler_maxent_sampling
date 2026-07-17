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

---

## M4 — Affine geometry + span certificate (6 rounds, converged on substance)

Codex reviewed `affine_geometry.py` adversarially. **It found a crash, a silently dropped dimension,
a dual-side blind spot, an unsound bound formula, and two test bugs** — one of which meant a test
could not fail. Every counterexample it gave was reproduced before being fixed. Its final position:
it does not contest `d = 46` or its independent corroboration, and it does not count "float64 numpy is
not certified interval arithmetic" as a defect. That is where this closes.

### Locked: the certificate is *resolution-bounded*, and the resolution is √k, not max-width

Width is a support-function difference — subadditive and positively homogeneous — so for a unit
`p = Σ aⱼpⱼ` in the complement (`‖a‖₂ = 1`, hence `‖a‖₁ ≤ √k`):

```
width(p) ≤ Σ |aⱼ|·width(pⱼ) ≤ √k · √(1+leakage) · maxⱼ width(pⱼ)  +  leakage · diameter
```

A direction tilted equally across all `k` probes hides a factor of `√k` from every one of them
individually. `SpanCertificate.resolution` reports this, and it is the number to quote. The
`√(1+leakage)` factor comes from `‖Qᵀp‖² = 1 − pᵀEp ≤ 1 + ‖E‖`; the `leakage·diameter` term exists
because float64 Gram–Schmidt does not *exactly* span `range(B)ᗮ`, and a direction hiding in that gap
would be probed by nothing.

### Locked: a width has two ends, and they need two different instruments

A **lower** bound (the objective difference of two returned endpoints) proves a direction *exists*.
It cannot certify one *flat* — that is the wrong end of the interval, and a solve that stops short of
optimality reports the width too **small**, certifying a real dimension as flat. `max_dual_infeasibility`
cannot catch that either: 1e-10 is dual-feasible anywhere, yet on a variable of range 1e10 it hides a
whole unit of width.

So flatness rests on **weak duality**, which assumes nothing about the returned point — not
optimality, not even primal feasibility:

```
max cᵀv ≤ rhsᵀy + Σⱼ max(dⱼlⱼ, dⱼuⱼ),    d = c − Sᵀy      (any y whatsoever)
```

A first attempt used complementary slackness (`Σⱼ dinfⱼ·(uⱼ−lⱼ)`); Codex showed it silently drops the
`yᵀ(rhs − Sv̂)` term and so assumes exact row-feasibility, which no LP delivers. Replaced wholesale.
`d` is recomputed as `c − Sᵀy` via our own `rmatvec`, so a stationarity residual in HiGHS's arrays
cannot leak in either. **The same bound is applied to every FVA range**: a range read off the primal
is a *lower* bound, and a lower bound is precisely the wrong end when the conclusion drawn is "this
reaction cannot move."

The consequence is worth stating plainly: `U` cancels terms of size ~5e3, so evaluating it in float64
costs ~1e-9 absolutely. **That is the floor on what this certificate can resolve** — orders coarser
than the ~1e-13 the arithmetic appears to produce. The measured certified resolution is 2.78e-11
scaled (5.6e-8 flux units), outward-rounded. The earlier "1.4e-11" was not licensed.

### Locked: what the certificate actually claims

Not "cannot under-count a dimension." A direction thinner than the certified resolution *can* be
missed, and `blocked_tol` will drop one narrower than itself. The licensed claim is:

> **Every feasible direction of the exact polytope has its component orthogonal to `range(B)` bounded
> in width by `SpanCertificate.resolution`.**

The asymmetry runs the safe way: the geometry may **over**-count (admit an ε-feasible direction) but
cannot omit a direction it had the resolving power to see. Over-counting is benign for a sampler — the
chain explores a slightly larger set and every sample is still checked. Omitting a wide direction
would silently delete part of the support, and no downstream test would ever see the samples that
were never drawn.

### Locked: FVA-blocked reactions are structural zeros of the direction space

**61 of the example model's 260 free reactions cannot carry flux at all** — the file leaves `l < u`,
but mass balance pins them. If `max vᵢ == min vᵢ` over `P`, every feasible direction has `dᵢ = 0`
identically, so a nonzero `B[i,:]` is numerical error. Left in, it is not harmless: a basis row of
~1e-15 in a coordinate whose centre sits ~1e-13 *outside* its own bound (both solver noise) divides
into **a chord limit of order 0.03–0.5**. The measured chord at the centre was `[−0.54, −0.39]` —
excluding `t = 0`. `line_geometry` rightly refuses to sample it, so **M5 could not have started.**

This is not the forbidden snapping of small fluxes: no flux is rounded, and a pinned reaction keeps
its value. What is zeroed is a component of the *direction space* that an LP measured as zero. But it
is *numerically fixed at resolution `blocked_tol`*, **not provably constant** — Codex's counterexample
(a true 5e-16-wide dimension, dropped, with a separation of 2e15× sailing through the guard) is
reproduced and pinned as a characterization test.

### Locked: the resolutions must not contradict each other

Three tolerances describe the same polytope and must agree, or the components fight:

- **`scale_floor ≥ blocked_tol / span_tol`** (default 1.0). Below it, a blocked reaction still shows a
  scaled width above `span_tol`: the sweep reports the very axis the projection removed, and cannot
  append it — which produced a real crash, with a misleading "already lies in the discovered span".
- **`‖r_blocked / s_blocked‖₂ ≤ span_tol`**, checked at runtime on the measured ranges. Bounding each
  blocked range individually lets the *combination* reach `√n_blocked · span_tol` (the same
  subadditivity as the certificate's `√k`).
- **SVD rank cutoff = max(machine, `feasibility_tol`)**. An equality the LP will not *enforce* must
  not be one the projector *insists* on. With `σ_min = 1e-14` — above machine epsilon, below the LP's
  1e-9 model — the support LPs return moving endpoints while the projector zeroes their difference: a
  contradiction, not a geometry.

### Locked: the mass-balance bar is relative, not absolute

`S·v` sums terms of size `‖S‖·‖v‖` ≈ 1e5 here, so *evaluating* it costs ~1e-10 of rounding before any
solver error. An absolute 1e-9 bar charges that to the solver — measured, it failed a perfectly good
certificate — and on a polytope with 1e10 bounds no basis could ever pass. `NativeCSC.cancellation_scale`
(`|S|·|x|`) supplies the scale to divide by; note it *cannot* be had from `matvec(abs(x))`, which
re-applies the signed `S` and cancels all over again.

### What M5 inherits (checked here, so it need not discover them)

Every basis direction's chord through the centre contains `t = 0` with positive length (min 0.018);
the centre is *exactly* bound-feasible after a clamp bounded by the LP tolerance (3.1e-13); and the
support points span all `d` directions (rank 46/46) — so M5's covariance ridge cannot conceal a
singular covariance instead of failing on it.

---

## M5 — Rounding + β=0 sampler (3 rounds, converged: AGREE, contested: none)

Codex's sandbox (bubblewrap) could not read the working tree at all, and it refused to review the
stale GitHub mirror rather than invent findings against code that was not there — the right call.
The code was pasted into the prompt instead, docstrings stripped.

**Six defects found, all in code that 553 tests passed over. None corrupted the β=0 distribution;
three would have corrupted what we *believed* about it, which on a sampler is nearly as bad.**

### Locked: Geyer's pairs start at lag ZERO — `Γ_m = ρ_{2m} + ρ_{2m+1}`

The ESS estimator paired from lag 1: `(ρ₁+ρ₂), (ρ₃+ρ₄), …`. That sums to the same value **only if
nothing is truncated**, and the truncation is the entire method. Geyer's theorem — that `Γ_m` is
positive and decreasing for a reversible chain — is about the pairs that *include lag zero*; applying
his stopping rule to an offset sequence applies it to something the theorem says nothing about.

Measured trigger: an antithetic AR(1) with `ρ = −0.5` has `ρ_t = (−0.5)ᵗ`, so the correct
`Γ₀ = ρ₀ + ρ₁ = +0.50` (keep going) while the offset first pair is `ρ₁ + ρ₂ = −0.25` (stop at once).
The estimator therefore truncated on its very first term and fell back to "ESS = N" for a chain whose
true ESS is **3N**. Now pairs from lag zero, truncates at the first nonpositive `Γ`, applies the
initial-monotone accumulate, and `τ = −1 + 2·ΣΓ`. Pinned against the analytic `τ = (1+ρ)/(1−ρ)` for
`ρ ∈ {−0.8, −0.5, −0.2, 0.5, 0.8, 0.9}`.

### Locked: "feasible" means *both* halves of the polytope's definition

`FeasibilityReport.is_feasible` tested only the bound violations. `P = {v : S·v = rhs, l ≤ v ≤ u}` —
so a chain that had walked clean off the steady-state manifold, the half the entire affine geometry
exists to enforce, reported `is_feasible = True` with an arbitrarily large residual. Nothing else in
the suite asked the question, so nothing else would have caught it. It now requires both.

### Locked: the stored flux is the exact function of the stored state

The sampler stored the *incremental cache* `v`, so `to_flux(coordinates) != fluxes` — two quantities
that are supposed to be the same and were not. Worse, `max_refresh_drift` was measured **only at
refresh instants**, which is not a bound on anything: drift can peak and partly cancel between two
refreshes, and a `refresh_interval` longer than the run would have reported a serene 0.0 having
measured nothing. Now the exact `centre + T·y` is recomputed at every stored sample, so the equality
is *bitwise*, and the drift is observed at every sample as well as every refresh.

### Locked: `range(T) = range(diag(s)·B)` is an assumption until something checks it

The identity that licenses *any* ridge holds in exact arithmetic (`diag(s)` invertible, `B` full
column rank, `L` invertible) — and nothing verified float64 had delivered it. **A `T` that quietly
lost a column produces no bad numbers, only absent ones**: the chain explores a lower-dimensional
slice of the polytope, every sample is feasible, every chord positive, mass balance exact, and part
of the support is simply never visited. Now an SVD rank check, compared against `T`'s *own* column
count (comparing against the `d` passed in would pass an `n×(d+1)` matrix of rank `d` — the very
deficiency it exists to find). Measured: rank 46/46, cond(T) = 165.

### Locked: `@dataclass(frozen=True)` freezes the binding, not the buffer

It stops `t.transform = X` and does nothing whatever about `t.transform[0,0] = X`. Not academic:
`CoordinatePrecompute` holds **copies** of `T`'s columns, validated against `T` once at construction,
so an in-place write afterwards makes the chord (from the stale precompute) and the flux (from the
mutated `T`) disagree — a chain sampling one polytope and reporting fluxes from another, with no
error raised anywhere. All arrays are now physically read-only.

That fix exposed a second, latent bug: **`np.ascontiguousarray` returns its argument unchanged** when
it is already contiguous and float64, so freezing the centre without copying would have reached back
through the alias and made `ReducedGeometry.center` read-only *underneath its owner* — a transform
silently mutating the object it was built from. Hence the explicit `.copy()`.

The guarantee is stated for what it is: **accident-proof, not adversary-proof.** A caller can flip
`writeable` back on an owning array. The alias hole is closed at the source instead — every frozen
array is built from a buffer nothing else holds.

### Locked: the residual floor belongs to fluxes and *must not* touch directions

The `scale_floor = 1.0` in `NativeCSC.relative_residual` exists because a sampled **flux** carries
solver noise at the FVA-blocked reactions (~1e-14, not 0): a metabolite row touched only by blocked
reactions then divides a noise value by *itself* and reports a relative residual of exactly **1.0**.
(Measured: `cpd02375_c0`, both its free reactions blocked, absolute residual 3.4e-14, unfloored ratio
1.0. 24 rows of this model touch a single free reaction and every one is identically 1.0.)

A **direction** carries no such noise — `T`'s rows at blocked reactions are *exactly* `0.0` — so such
a row's cancellation scale is exactly zero, its residual is a sum of no terms, and it is *excluded*
rather than divided. Flooring the transform's own check would therefore only weaken it, and measurably
does: it loosens the bar on 2049 of the 41124 (column, row) pairs. `_transform_mass_balance` is
unfloored (3.5e-10 against `span_tol` 1e-9, passing on its own merits); the floor is used only for
sampled fluxes. **The floor is correct where the noise exists and absent where it cannot be.**

### Locked: what per-column `‖S·T‖` does and does not certify

Codex: checking each column independently does not bound `S·T·y` over the support, because the
cancellation scale at `T·y` can shrink through inter-column cancellation while the column residuals
reinforce. **Conceded for the *relative* residual.** But the columns *do* bound every combination in
absolute terms, `‖S·T·y‖_∞ ≤ ‖y‖₁ · max_k ‖S·T_k‖_∞`, and both factors are now **measured** rather
than assumed: `‖y‖₁ ≤ 27.2` over 12000 sampled states and `max_k ‖S·T_k‖_∞ = 4.1e-12`, giving a
support-wide absolute bound of **1.1e-10** against a measured 1.7e-11. Consistent.

And the operative guarantee is stronger than any a-priori certificate, because it is empirical: the
mass balance of every **stored sample** is recomputed and checked, and `is_feasible` now fails on it.
The points the chain actually emits are *verified*, not bounded. Codex conceded this.

### Settled: the stationarity argument, and exactly how far it goes

Codex's standing objection, conceded in full: **in float64 the chain is not Markov in `y` alone** —
its state is `(y, cache error, refresh phase)` — and a *measured* per-step drift is **not** a bound on
the error induced in the stationary law, which would need a spectral-gap argument this package does
not have. The module docstring now says exactly that, and claims exact Gibbs invariance only in exact
arithmetic. What is claimed in float64 is that the perturbation is small, corrigible and observed.

Three sub-claims survived attack unchanged:
- **The batched per-sweep coordinate draw is fine.** `rng.integers(0, d, size=d)` is drawn before any
  chord is inspected, so the mixture weights are state-independent. A degenerate chord consumes no
  RNG draw while a samplable one consumes a uniform, so the *number* of draws is state-dependent —
  which is harmless: each new uniform is still independent of the past given the stream. Codex agreed.
- **Continuing from the rebuilt `v` is correct**, `y` being the state; the refresh is phase-dependent,
  never state-dependent, and is an identity in exact arithmetic.
- **The degenerate-chord move is a self-loop in the sense that matters.** §1.6.6 forbids redrawing a
  *different coordinate* (that is what makes selection state-dependent). Moving to the single feasible
  `t` on the *same* coordinate does not touch the coordinate-selection law, and staying put would
  strand a drifted state outside its bound forever. Codex conceded, asking only that the branch be
  described as numerical recovery rather than as an exact Gibbs transition. It is.

---

# M6 — Positive-β maximum-entropy sampler  *(gate review, 2026-07-14)*

Codex reviewed three claims: that the sampled law is exactly `π_β`; the `s_J`/`J*` handling; and the
mean-J monotonicity check. It returned **DISAGREE** with five contested points. **Four were real
defects and one was a real methodological error.** Every one was reproduced before it was fixed.

None of them corrupts the β=0 or β>0 *distribution* — the kernel is M2's, untouched. What they
corrupt is **the calibration of β** and **what the run believes about itself**, which on a sampler
whose whole output is a β-ladder is very nearly as bad.

## The five findings

### 1. The `s_J` floor was not invariant to an additive constant of `J` — the M2 bug, relocated

`s_J = J* − Q₀.₀₅(J(W))` is invariant under `J → J + c`. The floor it was compared against,
`1e-9·max(1, |J*|)`, is **not**. So a constant that provably cannot change any probability could
change `s_J`, and with it every rung of the ladder.

Codex's counterexample, reproduced exactly: shift `J` by `+1e16` (an exact additive constant, via
`mu_offset`). The true range is still 12. The floor becomes 1e7. `s_J` falls back to 1.0 and **every
positive rung becomes 12× hotter** — for a reason with no physical content whatsoever.

This is M2's delta 7 (*the absolute magnitude of `J` must never reach a probability*) wearing the
calibration layer's hat, and it is the fourth time in this project that a **magnitude** has been used
where a **resolution** was needed.

**Fixed** by replacing it with a *cancellation* floor. `J* − Q` is a difference of two numbers of
magnitude `~|J*|`, so its float64 resolution is `~eps·max(|J*|, |Q|)`; below a few ULPs of that the
subtraction has no significant digits and the "range" is its own rounding. `ENERGY_SCALE_ULP_MARGIN
= 64`. The floor now asks *the question that has an answer*.

The consequences run **both** ways, which is how you know it is the right criterion: at `|J*| = 1e5`
the old floor was `1e-4` and the new one is `9.3e-10`, so a genuine range of `1e-6` is now **kept**
where it used to be thrown away. And at a `1e16` baseline it still falls back — but now because the
arithmetic really cannot support it (ULP there is 2.0, so a range of 12 is six ULPs wide and each
`J(w)` in the quantile carries ~1 ULP of error, i.e. 17% of the "range" is noise).

### 2. The degenerate-range fallback was silent, and silence is the one thing it must not be

Spec §22.2 says to fall back on a "**declared** positive scale". **A library default is not a
declaration.** A silent `s_J = 1` makes this strain's `β = 2` name a different selection pressure
from every other strain's `β = 2` — the exact failure `s_J` exists to prevent (§1.1's cross-model
comparison *is* the point of the batch design) — and it would arrive as a warning in a log nobody
reads.

**Fixed, going further than Codex asked**: `energy_scale_fallback` is now `float | None = None`, and a
degenerate range **raises** `DegenerateEnergyScaleError`. A degenerate range means the objective
barely varies over this polytope, so *no* β would mean much; stopping is the honest response. A
caller who wants to proceed declares a scale, and the manifest records that they did.

### 3. The Monte-Carlo standard error used the wrong variance — and was anti-conservative exactly where it mattered

`effective_sample_size` estimates the autocorrelation against **`var⁺`**, the overdispersed variance
that counts between-chain disagreement, *precisely so that* a chain trapped in one mode cannot claim
a large ESS. Pairing that ESS with the **pooled sample** variance in the numerator throws half of
that conservatism away.

Measured, exactly as Codex predicted: two chains trapped at `±a` give `var⁺/var_pooled = 1.995`, so
`sd_pooled/√ESS` under-reports the error by `√2` — **when the chains disagree**, which is the only
time an error bar earns its keep.

**Fixed**: `diagnostics.posterior_variance` and `diagnostics.mcse` are new;
`BetaRung.standard_error_j` is now `√(var⁺/ESS)`. An ESS of 0 against a nonzero `var⁺` now reports an
**infinite** error rather than zero — *"we cannot tell"* is not the same claim as *"we know it
exactly"*.

### 4. `monotonicity()` would have read a NaN as the most monotone ladder imaginable

`max(-inf, nan)` is `-inf` in Python — `nan > -inf` is `False` — so a single NaN σ would have sailed
through the fold and been reported as a pass. **The same shape of trap as M5's
`np.min(x, initial=0.0)`**, which reported an ESS of 0 for a sample whose every entry was 8000: a
sentinel that silently wins a comparison it was never meant to enter.

**Contested and won, partially.** Codex's specific trigger (`n_samples = 1`) does not fire:
`diagnostics._as_draws` raises below 4 draws per chain and on any non-finite draw, so no NaN reaches
`monotonicity()` by that path today. Codex conceded the trigger. But the trap is real, latent, and
one refactor away from being reachable — **fixed** regardless: non-finite means and errors are now
refused explicitly, and checked *before* the errors are computed so the message names the quantity
that is actually broken.

### 5. Nothing bound the objective to the polytope — and a mismatched pair makes its own traces confirm it

**The best find of the review**, and the one with the nastiest failure mode.

Three artifacts meet in `run_ladder` — the L1 polytope, the L3 transform, the L2 objective — and
nothing checked that they had ever met before. They are all just arrays. Hand it an objective lowered
from a *different* model of the same size and:

- the chain tilts by the reactions **that objective** names;
- `ReducedObjective.evaluate_many` reports **those same reactions** as `μ` and `C`;
- so the trace of `J` **rises monotonically with β, exactly as the theorem demands** — because the
  chain really is maximizing the thing the trace is measuring.

Every diagnostic in the package agrees, and every one of them is describing the wrong model.
Feasibility, mass balance, the chords and R̂ cannot help: **none of them knows which reaction `J` is
supposed to be about.** This is not hypothetical once M8 exists — L2 and L3 are *separate cache
artifacts*, and a stale key is all it takes to load two that were never computed against each other.

**Fixed**: `ReducedPolytope.content_key()` is promoted to the public L1 key (the geometry's private
`_polytope_key` now delegates to it, so all three layers name the polytope the *same way*).
`ReducedObjective` carries `polytope_key`; `run_ladder` refuses a mismatched objective **or**
transform. One string comparison per run.

**And one Codex did not ask for, of the same class**: `ReducedObjective` now refuses
`line.lam != l1_penalty`. The kernel bends `J` by `line.lam` while the traces report
`J = μ − l1_penalty·C`; let those two drift apart and the chain samples one distribution while the run
describes another — and each half looks perfectly healthy on its own.

## Amendments Codex won on the reporting

- **Report both near-zero counts, not one.** Excluding the 61 FVA-blocked reactions from a
  *selection* statistic is right; deleting them from the record is not — structural blockage is a
  real biological fact about the model. `ObjectiveTrace` now carries `near_zero_counts` (movable),
  `near_zero_counts_all_free`, and `n_blocked`, and a test asserts they reconcile at every threshold.
- **R̂ of the `J` trace, not just its ESS.** An ESS says nothing about *retained initialization*, and
  this model is run below convergence. A high-β chain that merely kept its high-`J` start produces the
  same rising curve as one the tilt actually pulled there. R̂ is what tells them apart — chains
  launched far apart that nonetheless *agree* about `E[J]` are not each sitting in their own initial
  neighbourhood. `BetaRung.r_hat_j` is new; the gate asserts `max R̂(J) < 1.2` (measured 1.03–1.07).

## Settled: what M6 ships, stated without varnish

Codex's closing argument, and it is right: *"Shipping the engine is defensible; shipping this as a
calibrated ladder is not."*

**M6 ships a validated engine with an uncalibrated β scale.** The tilt is exact, its magnitude is
pinned against the linear-response identity, and mean-`J` rises monotonically — but on this model
`s_J = 31.3` against a chain that explores `sd(J) = 2.6`, so the ladder is a fine-tuning knob and not
a switch, and the top rung of spec §22.1's own ladder closes only 13% of the gap to `J*`.

That is a fact about the **calibration**, not the sampler, and it has a named remedy that spec §22.2
already gestures at ("support **or pilot** points"): **M10's pilot-based `s_J`**. It is *required*
before the ladder may be presented as spanning neutral-to-strongly-selected regimes. Until then, a run
reports what it measured and does not pretend the β axis is comparable to anything but itself.

## Claims that survived attack unchanged

- The fixed reactions' contribution to `J` is genuinely **additive with no cross-terms** (biomass is
  linear, the L1 cost is reaction-separable, and `reduced.offset` has disjoint fixed/free support), so
  `L1Objective` really is `J` up to a constant, and a constant provably cancels from `p(t)`.
- All four `biomass_index = None` consumers are correct: `evaluate`, `evaluate_on_line`,
  `biomass_slope`, and `build_piecewise_j`'s opening slope.
- The `λw > 0` (bending) / `w > 0` (cost) split is right everywhere, including at `λ = 0`.
- `run_ladder` re-derives **nothing** per rung: `T`, `w`, `λ` and `s_J` are all frozen before the
  first chain starts.
- `J*` never reaches the kernel — only the reported log-energy `(J − J*)/s_J`.
- The theorem is stated correctly: `dE_β[J]/dβ = Var_β(J)/s_J ≥ 0`.
- **"Exactly π_β" was my overstatement to Codex, not the code's.** The module docstring already said,
  at length, that in float64 the chain is not Markov in `y` alone, that exact Gibbs invariance is
  claimed *only* in exact arithmetic, and that a measured drift is explicitly **not** a bound on
  stationary-law error. Codex's `2**53` counterexample is unreachable in any case: `model_input`
  rejects infinite bounds and every bound on these models is ≤ 1000.

## Round 2 — Codex found three defects **in the fixes**, and one of them was a test that could not fail

Round 1's five findings were fixed. Codex re-reviewed the fixes and conceded seven of ten points
(including all three I had contested), but held on three. **All three were right**, and the first is
the one to remember.

### R2-1. The regression test for the `s_J` floor **could not fail on the bug it existed to catch**

The fix was correct. The test was not. It shifted `J` by `+1e6`, where the *old* magnitude floor is
`1e-3` — a thousand times **below** the range of 12 — so the buggy code would have sailed straight
through it. A regression test the bug passes is not a regression test.

This is **the M4 lesson repeating** (that review found "two test bugs, one of which made a test unable
to fail"), and it is now the second time in this project a green test has certified nothing. The
shift is `+1e12` (old floor `1e3` > 12 → the old code falls back; the new 64-ULP floor is `7.8e-3` ≪
12 → the new code keeps it), and **the test now asserts its own premise** — `plain.value <= old_floor`
— with the failure message *"this test cannot fail on the bug it exists to catch"*. It cannot go
toothless a second time without saying so.

One thing the fix revealed that neither of us anticipated: at a `1e12` baseline one ULP is `1.2e-4`,
so the shifted range does **not** round-trip to `rel = 1e-6` — it differs by `4.9e-5` in the low bits.
Demanding bit-equality would demand precision float64 does not have. The assertion is therefore
`|shifted − plain| < resolution`: *the range survives an additive shift to within the resolution of
the arithmetic that computed it*, which is exactly the claim the ULP floor licenses, and no more.

### R2-2. The new identity key **omitted the biomass reaction** — reopening the hole it was cut to close

`ReducedPolytope.content_key()` hashed the reaction IDs, the free indices, the fixed values, the
bounds, the CSC arrays and the RHS — and **not `biomass_index`**. So two polytopes differing *only* in
which reaction is biomass shared a key: an objective lowered from one would bind happily to the other,
tilt the wrong reaction, and have its own traces confirm it — **precisely the failure the key was
added to prevent**, walked back in through the front door.

The hole has an instructive origin: the field list was copied from `affine_geometry._polytope_key`,
which omits biomass **legitimately**, because the *geometry* does not depend on it (the affine hull is
the same whichever reaction you call biomass). The omission was correct in the module it came from and
wrong in the module it went to.

Fixed, with the trade-off recorded in the code: including it costs a **false cache miss** in M8
whenever `biomass_id` changes and the geometry could in principle have been reused. *A false miss
recomputes; a false hit corrupts.*

### R2-3. "Immovable" does not mean "at zero"

The near-zero reconciliation asserted `all_free − movable == n_blocked` as an **identity**. It is not
one. A zero row of `T` means the chain *cannot move* that reaction — not that the reaction is *at*
zero. Mass balance can pin a **free** reaction (`l < u`) at a nonzero constant, and Codex's minimal
case makes it plain: `{v₀ = 1, 0 ≤ v₀ ≤ 2}`. That reaction is immovable and nowhere near zero, so at a
threshold of 0.5 the gap between the two counts is **0** while `n_blocked` is **1**.

The genome-scale assertion passed only because all 61 of *that* model's immovable reactions happen to
sit at ~1e-13 — a measured property of one model, promoted by accident to a law.

Fixed: the docstrings say *immovable ≠ at-zero* explicitly; the integration test now asserts what is
actually true (the gap is a per-threshold **constant**, because immovable reactions never move) and
only *then* asserts the value 61, **after measuring its premise in the same test** (`max |blocked
flux| < min threshold`). A measurement of this model, standing next to the measurement that licenses
it. And `tests/conftest.py::pinned_nonzero_polytope` +
`test_maxent_sampler.py::TestImmovableIsNotTheSameAsZero` keep the counterexample permanently in the
suite.

## Round 3 — the identity fix was still incomplete, and the real defect was in the **IR**

Codex confirmed R2-1 and R2-3 fixed, and held on one point: **`biomass_index` is a coordinate, not an
identity.** He was right, and the fix was not to the hash.

`FluxPolytope.reduce()` sets `biomass_index = None` for **every** model whose biomass is a *fixed*
reaction. So round 2's fix — hashing the reduced coordinate — collapsed all of them onto the same
value. Two models differing only in **which** fixed reaction is the biomass still shared a key, still
cross-bound, and — since ``μ`` on such a model **is** that fixed constant — the *entire objective*
would have been wrong: one strain's growth reported as another's, every diagnostic green.

Codex's counterexample, reproduced exactly: reactions `a` (fixed at 1.0), `b` (fixed at 2.0), `c`
(free). Choose either as biomass; both reduce to `biomass_index = None`; the keys matched; and
`lower_objective` produced `mu_offset = 1.0` against `2.0` while `binds_to` returned `True`.

**The root cause was worse than the hash: `reduce()` was throwing the biomass identity away
entirely.** A `ReducedPolytope` whose biomass is fixed could not even *name* its own biomass reaction.
`sparse_objective.biomass_maximum` had been quietly working around that by borrowing the index from
the **objective** — the polytope asking the objective to tell it what it is, exactly backwards, and
precisely how the hole survived unnoticed.

So the fix is to the **IR**:

- `ReducedPolytope` now carries **`biomass_full_index`** (always known) beside `biomass_index`
  (a *coordinate*, `None` when biomass is eliminated), plus `biomass_id` and `biomass_is_fixed`.
- `validate()` enforces that the two agree — a reduced index not pointing at `biomass_full_index` is a
  `J` that rewards the wrong reaction, and nothing downstream could see it.
- `content_key()` hashes the **identity**, not the coordinate.
- `biomass_maximum` reads the polytope's own field instead of borrowing the objective's.
- `lower_objective()` **refuses a mismatch outright**, before any key comparison, with a message that
  names both reactions — because this is the fact the key exists to protect, and at the moment the two
  objects are first joined the error can still say what it is about.

Codex also noted that round 2's regression test used three *free* reactions and therefore could not
have caught the collapse. It now uses two different **fixed** biomass reactions, exactly as specified.

## Round 4 — the real mistake was *patching joins instead of having an invariant*

Codex held three more points. All three were right, but the third sentence of his reply is the one
that mattered: **"a shared compatibility guard is needed at every public objective/polytope join."**

I had been fixing the *places where the bug had been demonstrated* rather than establishing the
property. Each round he pointed at a different join and each time I patched that join.

### R4-1. Comparing biomass *indices* is not comparing *models*

`lower_objective` checked `objective.biomass_index == reduced.biomass_full_index` and nothing else.
Objective biomass `"a"` at index 0 and polytope biomass `"x"` at index 0 **agree numerically** while
naming different reactions of different models — and then *every index in the objective addresses the
wrong reaction*, not merely biomass. The reaction IDs **are** the coordinate system; they have to be
the same one.

### R4-2. The LP layer checked nothing at all

`build_sparse_objective_lp`, `solve_sparse_objective`, `biomass_maximum` and `critical_l1_penalty` had
**no** compatibility check. Reproduced on the round-3 model: `solve_sparse_objective(polytope_b,
objective_a)` succeeds and returns a bundle whose `mu_at_optimum = 1.0` (objective `a`'s biomass) sits
next to `biomass_maximum = 2.0` (polytope `b`'s biomass) — *two different reactions reported as
"biomass" in one result object*, with every §12 check passing.

`critical_l1_penalty` was also still borrowing `objective.biomass_index` for a fixed biomass — the
same backwards dependency that hid the round-3 hole.

### R4-3. The new invariant could be bypassed with unsorted `free_indices`

`validate()` used `searchsorted`, which **silently agrees with itself on an unsorted array**. With
`free_indices = [1, 0]`, `biomass_full_index = 0` and `biomass_index = 0`, validation passed although
reduced coordinate 0 addresses full reaction **1**.

### The fix: one guard, five joins, and an IR invariant

`sparse_objective.check_compatible(reduced, objective)` now checks **both** halves — same reaction set,
same biomass reaction — and is called from **every** public join: `lower_objective`,
`build_sparse_objective_lp`, `biomass_maximum`, `critical_l1_penalty`, `solve_sparse_objective`. It was
previously called from none of them.

`critical_l1_penalty` reads the polytope's own `biomass_full_index`. The polytope answers questions
about itself.

And `ReducedPolytope.validate()` now (a) *dereferences* — `free_indices[biomass_index] ==
biomass_full_index`, correct whatever the order — and (b) requires `free_indices` strictly increasing,
which `reduce()` always produces (`np.flatnonzero`) and which much of the class silently assumes.

The regression test is parametrized over all four LP entry points, with the docstring that records
what went wrong: *"Each of these was, at some stage of the review, the only one that checked — and
each of the others let the same mismatch straight through."*

## Rounds 5–6 — the sweep, and where the class actually ends

### R5. `resolve_objective` mixed one polytope's cliff decision with another's LP

It takes the **canonical** polytope and the **reduced** one as separate arguments and never checked
they were the same model. `origin_is_feasible` reads the *canonical* bounds; `critical_l1_penalty`
runs on the *reduced* LP. Reproduced: pass a forced-flux canonical polytope (origin infeasible → no
cliff → `λ̃ ≥ 1` permitted) together with the reduction of an origin-feasible variant, and `λ̃ = 1.5`
is **accepted and recorded as `origin_is_feasible = False`** — while the polytope that actually gets
sampled collapses to `v* = 0`. BUILD_PLAN §1.7's guard, inverted.

Fixed both ways Codex offered, because they are complementary: `reduced` is now **optional and derived
by default** (so the two *cannot* disagree), and when supplied it is validated by
`ReducedPolytope.is_reduction_of(polytope)` — content, not identity.

### R6. Sweeping for the class rather than the instance

By round 5 the pattern was unmistakable, and it was **not** "these five bugs". It was:

> **Two artifacts that were never computed against each other, silently joined.**

Each round Codex named one more join and each time I patched *that join*. So round 6 asked him to
sweep for the class. He found three more in M6's path, and the first is the one that mattered:

1. **`rounding.build_transform` was the worst of them.** It takes a geometry **and** a polytope,
   builds `T` from the geometry's basis while taking the *bounds* and the `CoordinatePrecompute` from
   the polytope — and records the **geometry's** `polytope_key`. So a mismatched pair produces a
   **hybrid that passes `run_ladder`'s binding check**, because the key it reports is the geometry's,
   while stepping against another model's bounds entirely. The guard I had added in round 2 was
   defeatable by the artifact it was guarding.
2. **`run_chain` / `run_chains` never bound transform to polytope.** `run_ladder` did; the low-level
   entry points did not.
3. **`EnergyScale` carried no key.** `s_J` is the range `J` spans over *one* objective on *one*
   polytope; borrowed from another, every β on the ladder silently names a different selection
   pressure — the entire failure `s_J` exists to prevent. It now carries `polytope_key`, and
   `run_ladder` refuses a scale that came from a different objective.

Also fixed: `trace_objective` now refuses a `movable` set that does not index its own flux vector.

### Contested, and why — the inner loop keeps its invariant *by construction*

Codex also flagged `chord_on_support`, `build_piecewise_j` and `sample_line` for accepting
"an independently produced chord, state, direction, and objective". **Held.** These are hot-path
primitives — 46 coordinates × 12 000 sweeps × 4 chains × 3 rungs of them — and BUILD_PLAN §1.3
forbids per-step overhead in exactly this code. More importantly, their invariant is already enforced
in the right place: M5 *removed* a caller-supplied `support` argument from `feasible_chord` precisely
because an unvalidated one reintroduced the §1.6.1 tolerance bug, and `CoordinatePrecompute.build`
now **derives** the support from `T` and `validate`s it against the column. The invariant holds by
construction, once, rather than by a check paid for on every step. Adding runtime key checks there
would buy nothing and cost the thing the module exists to protect.

### Recorded as an **M8 follow-up**, not fixed here

`model_input.build_canonical_model(model, source_path)` hashes `source_path` while canonicalizing a
separately supplied cobra `Model`, without verifying they correspond — so a model loaded or mutated
elsewhere can receive **another file's L0 cache identity**. `load_canonical_model` derives both
together and is safe, and it is the only production caller.

This is a genuine defect of the **L0 cache key** and it belongs to M8, where the cache is actually
built. It is out of M6's scope and is written into DEVELOPMENT_STATUS so it cannot be lost.

---

## M7 — Reweighted-L1 (frozen weights)  ·  Claude × Codex, 5 rounds, converged (AGREE)

**Setting.** M7 is the first milestone where **two objectives exist on one polytope** — the base
weights and the reweighted ones — which is precisely the shape of the M6 "two artifacts never
computed against each other" bug. Before writing M7 code, that hole was reproduced: on the toy, `s_J`
is **0.68** under the base objective and **0.0068** under the reweighted one, and M6's guard
(`energy_scale.polytope_key != objective.polytope_key`) could not tell them apart — the two objectives
share a `polytope_key` exactly. So the structural fixes came first, the loop second.

**The λ fork left open since M3 — settled by measurement, not debate.** BUILD_PLAN §1.7 left M7 to
choose whether the raw λ stays frozen at its base-weight value or is re-resolved from the current
weights. It is not a close call: `λ*` is a function of `w`, and one reweighting step moves it from
1.9e-3 to ~4e2 (default clip) or ~2.3e5 (wider), because `C_w` changes *units* (a sum of fluxes → very
nearly a count of active reactions). Freezing λ collapses the effective pressure `λ/λ*(w)` from 0.5 to
~4e-6 **and crashes M3's `z == |v|` LP gate by the second iteration** (deviation 25 at the default
clip). So **λ is re-resolved every iteration: `λ_k = λ̃·λ*(w_k)`**. This also buys the invariance that
makes the median renormalization a mathematical no-op (`w → cw ⇒ λ* → λ*/c ⇒ λw` unchanged), which is
why step-4 normalization is a *conditioning* step that cannot move the target.

**What Codex found (and what it did not).** Codex's first round ran in a **broken sandbox** (`bwrap`
failed; it saw the pre-M7 public branch), so its line-level claims were verified against the real code
rather than trusted. Points 1–2 (λ policy, median no-op) held. Point 3 (freeze completeness) — the
concrete `with_weights(view)` attack was **refuted** (all `_frozen` callers pass owned fancy-index /
`.copy()` buffers; verified at runtime: `owndata=True`, `writeable=False`, in-place mutation raises,
owner mutation does not propagate); the residual "adversary flips `writeable` back" is accepted as out
of scope per M5's "accident-proof, not adversary-proof" precedent. The rest were real:

| # | Codex finding | Disposition |
|---|---|---|
| 5 | **Convergence tested the fluxes, not the weights.** A *global* relative `max\|Δv\|` is dominated by a large flux (1e3) and blind to a sparsity-critical one (1e-3) whose weight is still halving — so the loop could freeze weights one stale step short of the fixed point. | **Fixed.** Converge on `max_r \|w_{k+1,r} − w_{k,r}\|/max(...)`, a **per-reaction relative** metric that sees the small coordinate. The frozen artifact is the weights, so convergence must be about the weights. |
| 6 | **`n_shed` is net cardinality only** — an active-set *replacement* (turn one off, one on) reports net 0 while the support changed, and the no-op warning fired spuriously on it. | **Fixed.** Report `n_turned_off`, `n_turned_on`, and `support_unchanged` (symmetric difference empty); the warning now keys on `support_unchanged`, not net count. |
| 4/D (r2) | **`LPOptimum` carried `objective_key` but no `polytope_key`.** `content_key` hashes the objective's params, *not* the polytope's bounds/stoichiometry, so two polytopes differing only in bounds hash identically — and `s_J = J*(polytope A) − Q(J_B(W))` passed every check. | **Fixed.** Added `polytope_key` to `LPOptimum`; `choose_energy_scale` checks it. Reproduced the cross-join (same `objective_key`, different `polytope_key`) and confirmed it is now refused. |
| B (r2) | The weight-change denominator was claimed `≥ clip_min`; after median normalization it is bounded by `clip_min/clip_max` (the ratio), and config allowed any finite ratio → underflow. | **Fixed.** Config caps the clip ratio at 1e9 (default [1e-3,1e3] is 1e6; LP breaks at 1e12); `_relative_weight_change` returns `inf` (loud non-convergence) on a non-finite/zero denominator. |
| 3/D (r3) | **`warmup_fluxes` was an unkeyed positional array** — the *third* input to `s_J = J* − Q_q(J(W))`. A same-shaped support set from the wrong polytope silently changes `s_J`. | **Fixed.** `choose_energy_scale` gains a **required** `warmup_polytope_key`, checked against the objective; production callers pass `geometry.polytope_key` (the `ReducedGeometry` is keyed). |

**The one point held to consensus — `optimum_coordinates` (r4→r5).** Codex flagged it as a fourth
unkeyed model-derived input to `run_ladder`. **Held, and Codex conceded (AGREE).** It is an
initialization *hint*: blended as one vertex of a Dirichlet convex combination and then made
bound-and-chord-feasible (or the run raises), it enters *only* the start state — never the kernel,
objective, `s_J`, or traces. So unlike the `s_J` joins it **cannot change the invariant target**; a
wrong hint only seeds a poorer start, which is *observable* via feasibility and R̂/ESS. Keying it would
imply it defines the distribution, which it does not. The decision: **document the boundary, do not key
it** — the docstring now states the exact limits Codex named (feasibility only to `feasibility_tol`; a
bad hint costs convergence time not correctness, and can raise rather than mislead silently; R̂/ESS are
evidence not proof).

**The invariant, restated for M7.** Every input to the `s_J` subtraction is now keyed on *both* the
objective and the polytope, and cross-checked before a single `J(W)` is formed: the `LPOptimum` carries
`objective_key` + `polytope_key`, the `ReducedObjective` carries both, the warm-up array's key is
passed explicitly, and `run_ladder` re-checks the `EnergyScale` and transform against the objective.
The frozen weight buffers are physically read-only; the reweighter structurally cannot import the
sampler (and vice-versa), so a weight cannot move mid-chain. λ, weights and `s_J` are all rebuilt from
the **frozen** final weights — the fixed point the converged loop actually solved, not a rebuild that
could differ in its last ulp.

---

# § M9 — the rounding gate rejected valid geometry 1 time in 3, and the fix I proposed was unsound

**2 rounds, converged.** Codex conceded 1 point, I conceded 4 — including my entire proposed fix.
This is the round where **being adversarially reviewed changed the answer, not just the wording**.

## What M9 found (measurement, before any argument)

The M9 benchmark's worker sweep would not run: both strains failed with
`RoundingError: ‖S·T‖ relative ... 1.544e-09, above span_tol 1.0e-09`. But the *same polytope* built
fine under a different `model_id`. `model_id` keys the RNG (`stream_seed`) → span-certificate probes
and support-LP directions → support points → covariance → `L` → `T`. So **a label decided whether a
genome-scale model could be sampled at all**. Across 24 streams: **8 raised, 16 passed.**

| measure | min | median | max | spread | fails 1e-9 |
|---|---|---|---|---|---|
| `‖S·T‖` absolute | 3.139e-12 | 3.851e-12 | 5.535e-12 | **1.8×** | — |
| `‖S·T‖` relative (the gate) | 9.588e-11 | 4.835e-10 | 3.038e-08 | **373×** | **8/24** |

Also found: **`rounding.py:156` documents `max_k ‖S·T_k‖_∞/‖|S|·|T_k|‖_∞` (per-column norms) while
`rounding.py:788` computes `max_k max_i (r_i/s_i)` (per-(col,row) ratios).** The code never
implemented its own documentation, and the two differ by five orders of magnitude here.

## The debate

**My position:** the residual is an inherited absolute floor, not a locally-generated one; dividing
it by a small per-row scale manufactures a meaningless number. **Fix: implement the documented
per-column formula.**

**Codex, round 1 — DISAGREE on 5 points.** The decisive one killed my fix outright:

> `S=[[1,-1,0],[0,0,1]]`, `|v₁|,|v₂| ≤ 1e12`, `|v₃| ≤ 1`, `T_k=(1,1,1e-10)`.
> `S·T_k=(0,1e-10)`, `|S|·|T_k|=(2,1e-10)`. **Per-column: `1e-10/2 = 5e-11` — PASSES.**
> But `v₃` binds the chord at `|y| ≲ 1e10`, so `v₃` reaches **1.0**: a mass-balance violation of
> order 1 with every reaction bound satisfied.

Dividing by the **largest** row scale lets two unrelated huge reactions hide a 100%-relative
violation on a tiny one. My fix would have shipped a gate that admits a grossly off-manifold
transform. **Withdrawn.** (Codex also killed candidate `‖S·T_k‖_∞/(‖S‖_∞‖T_k‖_∞)` as weaker still.)

**I ran Codex's discriminator** — log-log slope of residual `r` vs row scale `q`, 61009 (col,row)
pairs over 8 streams: **slope +0.165**, not the **+1** a locally-generated error requires. Across ≥4
decades of `q` the median `r` rises only **6.6×** (locally-generated ⇒ ~1e4×). At large `q`,
`r/q → 2.4·eps`; at small `q`, `r/q → 1.8e5·eps`. **Round 2: Codex conceded the error model.**

**Codex held one point, and it was right.** `r/q` is *not* meaningless: it is exactly the
Oettli–Prager componentwise backward error —
`min{η : |ΔS_ij| ≤ η|S_ij|, (S_i+ΔS_i)·T_k = 0} = |S_i·T_k|/q`. My slope experiment proves it does
not identify *how* `T_k` acquired its error; it does not make the ratio describe nothing. **Conceded.**
The honest statement: it is a *structural backward error*, not a bound on what the sampler can emit —
so it is the wrong thing to **gate** on, not a meaningless number.

**I also conceded:** worst *reachable* coordinate rather than typical chord length; `span_tol`
conflates three meanings; M5's "exactly-zero rows" argument only solves `0/0` and never licensed
"legitimate however small it is"; and leakage does **not** accumulate as a random walk
(`Sv−b = (Sc−b) + STy` is determined by the current bounded `y`, not by step count).

**Codex killed my cheap version too.** I proposed bounding `|y_k| ≤ ρ_k` and summing
`Σ_k |E_ik|·ρ_k`. Counterexample: `Y = {|y₁|≤1, |y₂|≤1, |y₁−y₂|≤δ}`, `E_i=(1,−1)` → box bound `2`,
truth `δ`. Unbounded looseness from coordinate coupling; `step_scale_ratio` does not control it.

## Settled — the consensus position

The gate is **`certify_reachable_mass_balance`**: 2 LPs per metabolite over the fixed reachable set
`Y`, against the **same `η=1e-9` contract `diagnostics.feasibility_report` already applies to emitted
samples**. One declared definition of "mass balanced", proved a priori and checked a posteriori.
`transform_mass_balance_error` is retained as a **reported diagnostic that never raises** — Codex
showed it catches a corruption the certificate deliberately misses (`S=[1]`, `T=[δ]`, true dimension
zero but `T` invents motion: diagnostic 1.0, certificate 1e-12 and passes).

**Codex's scope verdict, which the user adopted:** build it in M9, not M10. M9 explicitly *is* "GSMM
hardening"; M5's gate demands `‖S·T‖≈0` and enforced it with the wrong instrument; a proof that the
represented support meets v1's feasibility contract is not an "extension"; and disabling the gate
would leave rare reachable endpoints unvalidated, which Codex's own counterexample proves an
emitted-sample campaign cannot exclude.

## Two things the *implementation* then found, both self-inflicted

1. **I read the primal.** M4's recorded lesson says verbatim: *"Never certify flatness from a primal
   reading; M5/M6 will face the same temptation."* M9 walked straight in. `objective_value` is a
   **lower** bound on the max, so a solve stopping short reports the reachable residual too *small*
   and certifies a transform that reaches further — unsound in the dangerous direction. Now a
   weak-duality bound: `max e·y ≤ Σ_j max(π_j lo_j, π_j hi_j) + Σ_k |d_k|·Ω_k`, `d = e − Tᵀπ`, valid
   for **any** `π`, with an outward rounding allowance. `Ω` is unavoidable (`y` is free, so any
   `d ≠ 0` sends the sup to `+∞`) and comes from a provable outer box via a **freshly recomputed**
   `T⁺` — the stored inverse does not get to vouch for the transform it is stored beside.
2. **HiGHS silently drops a tiny row.** With a 1e-10 coefficient beside 1.0 coefficients it reports
   that row's activity as **0.0** where the truth is **133.3**, `max_primal_infeasibility = 0.0`,
   status optimal. The row does not survive matrix scaling. That relaxation happens to enlarge `Y`
   and stay conservative — but *relying on which way a solver's scaling errs is not a certificate*,
   which is precisely why the bound is dual. Each `E_i` is normalized to unit norm before it reaches
   the solver and rescaled after; raw, `E_i ≈ 1e-13` sits under the dual feasibility tolerance and
   every reduced cost reads as zero.

## Result

Certified on every RNG stream. `max_i R_i` = **3.6e-11 … 5.1e-11** — a **1.41×** spread where the old
gate swung **373×** — **20–28× inside** the contract, **334 LPs / ~0.5 s** (only 167 of 894 metabolite
rows have `E_i` structurally nonzero). It lands just above M5's independently measured **2.6e-11**
emitted-sample residual, exactly as an upper bound on a superset must: two calculations sharing no
code, agreeing.

**Still open, recorded for M10:** the span certificate is a *second* RNG-marginal gate —
`build_geometry` raises "not exhaustive (214/214 probes, 1 inconclusive)" on ~1–2 of 20 streams. Same
shape, not diagnosed, deliberately untouched by M9.

**The lesson, which is the M4 lesson at one more remove:** *never divide by a small number that is
noise* — and the corollary M9 adds, **a bar that a valid input clears only 2 times in 3 is not a
tolerance, it is a coin flip.** The signature that exposed it: the *absolute* quantity was constant
to 1.8× while the *relative* one swung 373×. When a ratio is unstable and its numerator is not, the
denominator is the thing that is wrong.

---

# M10 — the pilot DAG: what should `s_J` actually be?  ·  Claude × Codex, 4 rounds, converged (AGREE)

## The finding that opened it: **the recorded prerequisite was arithmetically false**

M6 deferred a "pilot-based `s_J`" to M10 and promoted it to a **prerequisite**, recording that spec
§22.2 "already names the remedy" with its phrase "support **or pilot** points", and that reading the
scale off a pilot chain "tilts ~12× harder".

Measured first, on the example model (d = 46, λ̃ = 0.5, `J*` = 9.4664, 4 chains × (3000+3000),
N = 12000 pilot draws) — **it does not**:

| candidate `s_J` | value | `dE/dβ|₀` | β to close the gap (linear response) |
|---|---|---|---|
| **A** `J* − Q₀₅(J(support))` — current (M6) | 32.5118 | 0.1831 | 116.9 |
| **B** `J* − Q₀₅(J(pilot))` — **spec §22.2 literal, pilot W** | 25.4075 | 0.2343 | 91.4 |
| **C** `J* − mean(J(pilot))` | 21.3998 | 0.2781 | 76.9 |
| **D** `Q₉₅ − Q₀₅(J(pilot))` | 8.0729 | 0.7373 | 29.0 |
| **E** `sd(J(pilot))` — M6's "12×" | 2.4397 | 2.4397 | **8.77** |

Spec §22.1's own ladder tops out at **β = 16**. Swapping W from support → pilot *inside the spec's
formula* (A → B) buys **1.28×**, not 12×. **M6 conflated an anchored range *to `J*`* with a
*spread*, and then cited the spec as authority for a formula the spec does not state.** The "12×" is
just `32.5 / 2.44` — a ratio between two different quantities.

A prerequisite was recorded, and a milestone gated on it, on arithmetic nobody had done.

## Settled — the consensus position

**`s_J = σ̂₀` = the sd of `J` over a frozen β=0 pilot, as a NEW additive mode
`energy_scale="pilot_sd"`.** `warmup_range` keeps its semantics and its label; existing results keep
their original scale method. Codex's opening verdict was DISAGREE — but against my *overclaims*, not
against E; its own recommendation was E.

**What may be claimed:** `I₀ = 1` and `KL(π_β‖π_0) = ½β² + O(β³)` — a **local Fisher-standardized
coordinate**, and *exact at the estimand level only*. With the frozen plug-in the implemented
coordinate has `I₀ = σ₀²/σ̂₀²`. **What may NOT be claimed:** a universal finite-β axis; Fisher–Rao
arc length at finite β; that the ladder spans.

### Five overclaims of mine that did not survive

1. **"β is Fisher–Rao arc length from the neutral ensemble."** False at finite β. Arc length is
   `ℓ(β) = ∫₀^β √(Var_t(J))/σ₀ dt`, which equals β **only infinitesimally**. A local property
   claimed globally.
2. **"The anchored coordinate governs no realized expectation."** Exactly backwards, and Codex's
   refutation is the best argument in the exchange. If the neutral deficit `X = J* − J` has a density
   of states `g(x) ~ C·x^{r−1}` as `x ↓ 0`, the tilted law is `e^{−κx}·g(x)`: **measure-zero is
   precisely what *produces* the `x^{r−1}` power**, and hence `r/κ` — it does not make `e^{−κx}`
   irrelevant. So `1 − q(κ) ~ r/(κΔ₀)`, and `κΔ₀/r` governs fractional gap closure. Entropy
   **modifies** the coordinate; it does not defeat it. Consequence: under C, `1 − q ~ r/β` — q depends
   only on β and r, so **C is the strain-comparable coordinate in the *sharp* regime while E is
   natural in the *weak* regime.** No scalar is universal.
3. **"sd has one input."** It removes the `J*` join only: **3 provenance-bearing artifacts → 2**, not
   1. The pilot fluxes still join an objective to a polytope.
4. **"Var_β(J) shrinks."** Not a theorem. `d/dβ Var_β(J) = E_β[(J−E_β J)³]/s_J` — the sign is the
   tilted **third central moment**. (It *is* measured to shrink here: M6's mean-J rise of 3.14 against
   a linear-response prediction of 3.44 is exactly that.)
5. **"β\* = 8.77 closes the gap."** Full closure needs **infinite** β when the argmax set has measure
   zero. `β*` is a *local scale*, never a predicted endpoint.

Codex conceded that **σ₀ is the better primary local coordinate**, and that its own spanning-by-
construction alternative `s_J = B·Var₀/Δ₀` is **circular** — it folds `G` into the treatment axis.
It also killed my candidate D by normalizing it: `(Q₉₅−Q₀₅)/3.289707 = 2.45399`, only **0.586% above
sd**. Raw D looked like a rival only because it was never normalized. **D and E are one estimand with
a robustness knob, not a fork.**

### The synthesis: **σ₀ sets the axis, Δ₀ is reported**

Because `1 − q ~ r/(κΔ₀)`, publishing `q(β)`, `Δ₀ = J* − E₀[J]`, `G = Δ₀/σ₀ = 8.77` ("the strain's
headroom in neutral standard deviations") and `βG` hands the reader the anchored coordinate's
information as a **derived observable** — instead of baking it into the x-axis, where it would hide
the very cross-strain quantity the study exists to compare (BUILD_PLAN §1.1).

### `r_eff(κ)` — Codex's falsifiable prediction, now a diagnostic

For a piecewise-linear objective near an optimal face of dimension `f`, with `c = d − f` the
transverse codimension, Laplace gives `Z(κ) ~ e^{κJ*}·C·κ^{−c}`, hence

```
J* − E_κ[J] ~ c/κ        r_eff(κ) := κ·[J* − E_κ J] → c        (corroborator: κ²·Var_κ(J) → c)
```

An **integer-ish plateau under regular local geometry** (not an unconditional expectation). At small
κ, `r_eff = κΔ₀ − κ²σ₀² + O(κ³)` starts at **zero**, so non-constancy *before* the asymptotic region
is expected; a **sustained plateau** is the signature of entering the sharp regime. A **linear drift**
instead of a plateau indicts `J*` itself: a `+δ` error adds `κδ`. Hence the `J*` solver tolerance is
recorded — at high κ a tiny optimum error masquerades as non-plateauing `r_eff`.

## The pipeline (Codex's sequential design, adopted)

```
1. geometry pilot at β=0 under T₀     (OBJECTIVE-INDEPENDENT)
2. freeze its covariance → build T₁
3. INDEPENDENT scale pilot at β=0 under T₁   (better mixing → better ESS for σ̂₀)
4. freeze σ̂₀ → production chains on independent streams
```

It **separates random *efficiency* calibration from random *target* calibration.** One shared pilot
would be valid — the transform cannot move the stationary law and both artifacts are frozen — but it
would make pilot-seed sensitivity **unattributable**, because geometry quality and the selected target
would move together. The scale pilot runs under `T₁` deliberately: a poor `T₀` cannot deform the
neutral *target*, only the *efficiency* of estimating σ̂₀.

**The β=0 law is objective-independent**, so **one neutral pilot serves many objectives on a
polytope** — which matters directly, because M7 puts a base *and* a reweighted objective on one
polytope. The pilot artifact is objective-**independent**; the derived scale artifacts are objective-
**keyed**. The pilot key covers polytope + input transform + schedule (chains/burn-in/samples/thin) +
impl version + semantic RNG stream — not merely polytope+objective+stream.

## Precision is a warning; validity is a refusal

`se(σ̂)/σ ≈ √(K−1) / (2·√ESS_{(J−μ)²})` with **Pearson** kurtosis `K` (so `√(K_ex+2)` when reporting
excess), and the ESS belongs to the **centered-square** influence series — not raw `J²`, and not my
original Gaussian `√(2·ESS)`. Target ~2%. Degraded precision ⇒ **WARNING, never a gate** — this repo
has been burned by exactly one noise-floor gate too many (§1.4.2).

**But Codex caught the gate trying to re-enter through the door I opened**: "the numerical-resolution
refusal needs a predeclared, implementation-level criterion. Otherwise it can quietly become the
noise-floor gate you explicitly rejected." So the refusal reuses M6's **existing predeclared**
mechanism (`ENERGY_SCALE_ULP_MARGIN = 64`), and refuses **only** nonpositive / nonfinite / below-
resolution — the cases where the target is **undefined**, which is a different failure from imprecise.

**The estimand is predeclared as SD and is never switched per-strain after seeing diagnostics** —
doing so would make β mean different things across the batch, the one thing `s_J` exists to prevent.
`R₉₀ = (Q₉₅−Q₀₅)/(3.289707·σ)` (= 1 for a Gaussian population; **1.00586** measured here), skewness
and excess kurtosis are reported *as diagnostics*, not as estimator selectors. Codex's check: the
central-90 midpoint sits `0.012σ` from the mean — **no evidence E is tail-driven on this model.**

## What the DAG does and does not guarantee (Codex, round 3 — all conceded)

Freezing `T₁` and `σ̂₀` before production gives a **time-homogeneous kernel with a fixed conditional
invariant law**. It does **not** give stationarity from iteration zero — burn-in provides
*convergence*, not stationarity, unless the initial state is drawn from the law. And conditional on
the pilot artifact the invariant target is `π_{β/σ̂₀}`, **not** the ideal `π_{β/σ₀}`; marginalizing
over pilot randomness yields a **mixture of calibrated targets**. That is *calibration uncertainty*,
not an MCMC invariance failure — and the distinction is stated rather than blurred.

**Range-invariance alone is not the clean condition.** The requirement is that `T₁` be a *nonsingular
affine coordinate change on the affine hull*, with constraints and Jacobian handled consistently. The
real implementation risks are **feasibility tolerances, rank loss, state carry-over, production-
dependent rerunning, and residual adaptation** — which is what the tests must target. The algebra was
never in doubt.

Freezing plus reported precision also does **not** exclude **finite-burn-in bias** in σ̂₀, which an
ESS-based precision estimate cannot see (ESS measures within-chain information, not retained
initialization). Defences: dispersed independent scale chains, **between-chain agreement on σ̂₀
itself**, R̂ on the scale pilot's `J` trace, and the transform rank/conditioning checks M5 already has.

**The lesson:** *a deferred remedy is a hypothesis, not a plan.* M6 recorded a prerequisite, a
mechanism and a magnitude for a fix it had not computed — and all three were wrong while the
*diagnosis* was exactly right. The β axis really was uncalibrated; the named cure simply was not one.

---

# M10.2 — wiring the pilot DAG into `batch`/CLI  ·  Claude × Codex, 4 rounds, converged (AGREE)

## The finding that opened it: **the recorded fork was not a fork**

M10.1's tracker recorded M10.2 as blocked on "a real fork BUILD_PLAN does not settle" — whether the
pilot and `T₁` join §1.1's 4-layer DAG as a new layer, since a cache hit returns a `RoundedTransform`
with no `ReducedGeometry`. Measured before arguing:

| stage | Bifido, d = 46, serial | cached? |
|---|---|---|
| `build_geometry` (~1100 LPs) | **1.168 s** | yes (as `T₀`'s bundle) |
| `build_transform` → `T₀` | 0.005 s | — |
| the two β=0 pilots | **19.202 s** | **no** |
| `reround_transform` → `T₁` | 0.009 s | **no** |

A layer for `T₁` would exist to avoid rebuilding a 1.17 s stage while costing 19.2 s to fill — 16.4×
upside-down. And the blocker itself was **plan/code drift**: §1.1 has always said L3 holds `B` and the
span certificate; `to_bundle` held neither and `ReducedGeometry` had no serializer at all. *The M10.1
lesson repeating one milestone later: a recorded remedy is a hypothesis until someone does the
arithmetic.*

## Where Codex was right and I was wrong (round 1–2)

- **"Repairing L3 dissolves the fork" was overreach.** Decisive counter: §1.1's **L2 was already not a
  strict layer** — `warmup_range`'s `s_J` is nominally L2 but reads L3's support points while the
  stated L2 key omits L3. `pilot_sd` only makes the hidden edge impossible to ignore. The numeric
  labels are outliving their usefulness; named immutable nodes would serve better. Conceded.
- **"L3 caches the transform, not the geometry" is too strong** — it caches a *non-reconstructible
  hybrid* (`batch` bolts support points + manifest + certificate onto a transform bundle). Conceded.
- **"The pilot key already exists" was wrong, and an *incomplete* key is worse than an absent one** —
  absent means no cache; incomplete is a false-hit generator, and §1.1's rule is that a false miss only
  recomputes while a false hit corrupts. `NeutralPilot.content_key` omitted `seed` (the `SeedSequence`
  entropy), `refresh_interval` (M5: the float64 state is `(y, cache error, refresh phase)`) and
  `SAMPLER_IMPL_VERSION`. All verified in code. Conceded.

## Codex's best catch: **the neutral pilot was objective-dependent and its docstring denied it**

`calibrate` fed both β=0 pilots `optimum_coordinates` — the objective's own LP optimum — while
`NeutralPilot` claimed "**objective-independent** … one neutral pilot serves every objective on a
polytope" and hashed no objective and no start. Measured, two pilots differing in nothing else:

| | |
|---|---|
| `content_key` | **identical** (`d9c3ff…` both) |
| draws | not bit-identical, max \|Δy\| = 2.79 |
| `T₁` | cond(C_q) 7198 vs 9663 |
| `s_J = σ̂₀` | 2.6287 vs 2.4995 |

**Not bias** — both are honest draws from one β=0 law and the spread is Monte Carlo noise, so claiming
bias would have been the overclaim. The real defect is sharper and unarguable: **the artifact was not a
function of its key.** M7's two-objectives-on-one-polytope case takes the first hit and never knows.

Codex's mechanism beat mine: I said "a different start is a different trajectory"; Codex said the hint
changes the **support hull's cardinality**, hence the **Dirichlet draw's dimension**, hence **RNG
consumption on every later transition** — the streams *desynchronise*. Fix (Codex's option 1, adopted):
**remove the parameter**, don't default it to `None`. A defaulted parameter can be forgotten; an absent
one cannot. Its true claim: "…every objective sharing this polytope, **transform, and pilot recipe**".
M10.1 had shipped that hint with **zero test coverage** — removing it broke no test.

## The v1 defects M10 merely made reachable

- **M9's mass-balance gate was bypassable through the package's own cache-warming path.** It lived only
  in the `compute()` closure of `_load_or_build_geometry`, which runs **only on a miss**; on a hit
  nothing read the certificate. Meanwhile `maxent build-geometry --cache-dir` wrote its own bundle
  under `batch`'s key, omitted the certificate, and **stored it after printing `REFUSED`**. Warm, then
  sample. That gate cost a 2-round review and Codex's counterexample killed its first fix.
- **A `COMPLETE` marker named a chain, not an experiment.** §1.1 specified the sample key from the
  start; nothing computed it. `store_chain` recorded only `polytope_key`. Two experiments in one tree,
  stacked into one cross-model table, every per-chain diagnostic green *because each chain is correct*.

## Round 3 — Codex reviewed the built diff and found a defect I had just introduced

It opened **DISAGREE** on the implementation, holding 1, 3, 6. All three conceded:

1. **`require_certified_transform` was not total.** `certificate=None` was safe *behind*
   `prepare_model` and a hole in the public API — a library caller could hand `calibrate` an
   uncertified `T₀`, and the pilots are chains that step in its frame. Fix: `bootstrap_certificate` is a
   **required argument** (the proof exists already; recomputing = 334 wasted LPs; a default can be
   forgotten). `CalibrationResult.certificate` is now never `None` — always the proof for the transform
   that result ships.
3. **`optimum_coordinates` had the same identity defect I had just fixed for pilots.** I excluded it
   from `sample_recipe_key` by importing §1.6.5's reasoning. Codex's refutation is decisive and general:
   **the recipe key already hashes `seed`, `chain_index`, `schedule`, `storage_mode` — none of which
   define the stationary law.** So "only initialization" cannot be the criterion. **An artifact key asks
   "are these bytes the same artifact?", not "is this the same distribution?"** Both keys are right;
   they answer different questions. Also missing: `model_id` (keys the RNG), `feasibility_tol` (start
   selection + chord validation), `near_zero_thresholds` (change the stored trace arrays). `movable` is
   the one exclusion that survives — an exact function of a transform already hashed.
6. **The manifest bug, committed by me, in the milestone about it.** `prepare_model` copied the
   cache-shaped `to_cache()` dict into the human-facing report (losing `as_dict`'s derived
   `reachable_is_certified`/`reachable_margin`) — and **under re-rounding it named `T₀` while production
   sampled `T₁`**. A manifest describing an artifact that was not used: this package's signature bug.

## Settled — the design

- **Cache what is expensive, derive what is cheap, key everything.** Geometry (1.17 s) and pilots
  (19.2 s) are cached; `T₀` (5 ms) and `T₁` (9 ms) are derived. `T₁` needs no layer — but it needs an
  **identity** and **certified provenance**.
- **`T₁` must be certified before the *scale pilot***, not before production: the scale pilot itself
  steps in `T₁`'s frame, so an uncertified `T₁` means σ̂₀ is read off off-manifold fluxes. The
  exact-arithmetic theorem does not transfer `T₀`'s certificate — `range(T₁) = range(T₀)` makes the
  *true* worst residual identical, but the certificate is a numerical bound over `E = S·T₁` and a fresh
  `T₁⁺`, and `fl(B·L₀)`/`fl(B·L₁)` need not share a floating-point column space. Measured: `T₁`
  certifies at **3.86e-11**, inside M9's independently measured `T₀` range 3.6e-11 … 5.1e-11 — two
  certificates, two matrices, no shared computation, agreeing where the theorem says they must.
- **`ReachabilityCertificate` gained a `transform_key`.** `T₀` and `T₁` share a `polytope_key`
  *exactly*, so nothing else could separate them — a pre-M10 world had one transform per polytope and
  never needed it.
- **`to_cache` stores evidence; `as_dict` stores the verdict.** `is_certified` is derived, so caching
  the fields makes a bundle asserting innocence beside contrary evidence *inexpressible* — the loader
  re-derives. (M9: never trust a reading, check the bound.)
- **Refuse, don't recompute, on a recipe mismatch.** A results tree is the user's output, not a cache.
  Codex noted `_already_done` runs over all jobs before `_execute`, so refusal precedes any write —
  neither destructive overwrite nor partial mixing.
- **The scope cut (Codex confirmed (f) does not depend on (e)):** **M10.2a** = correctness (L3
  serialization, pilot key, `T₁` certification, gate totality, recipe key). **M10.2b** = performance
  (cache the pilots per chain; two-phase pool dispatch — 19.2 s serial per model, Amdahl ceiling 24.9× →
  ~3.65×). Payload when built: geometry pilot → coordinates only (~2.9 MB); scale pilot → reduced fluxes
  only (~16.6 MB, preserving multi-objective reuse); **never** reconstruct stored fluxes from
  coordinates — M9 measured gemv-vs-gemm at 1.1e-13.

## Rounds 4–5 — each repair had a smaller hole behind it, and that is the shape of the thing

Codex confirmed the round-3 repairs introduced no new defect (branch provenance of
`CalibrationResult.certificate` correct in both branches; `content_key`'s ndarray/`None`
canonicalization can only cause conservative false *misses*; `CalibrationResult` is the right owner of
the report conversion). It then found four more, all conceded, and the sequence is the finding:

1. **`run_neutral_pilot` still bypassed the gate.** Requiring the proof on `calibrate` closed one door
   and left the public one beside it open — the same chain, no fabrication needed. It now takes a
   required `certificate` too. *Gating an orchestrator does not gate the primitive it calls.*
2. **`sample_recipe_key` had no *writer* identity.** `SAMPLER_IMPL_VERSION` is scoped to the
   transition kernel by its own docstring, while `store_chain` decides the arrays, names, casts and
   manifest fields — and carried no version at all. New `output.OUTPUT_IMPL_VERSION`. §1.1's own rule
   ("provenance in every key: parser + code + artifact-schema versions"), unapplied to samples.
3. **The evidence itself was never checked.** The sharpest catch, because it is a hole in *this
   milestone's own reasoning*: `to_cache` stores fields so `is_certified` can re-derive the verdict,
   defeating a bundle that *asserts* it passed — but `from_cache` only checked fields were **present**,
   so `worst_absolute = −1` sails through (`−1 <= 1e-9`). **"Trust the claim" was closed and "trust the
   evidence" left open.** `__post_init__` now refuses non-finite/negative residuals, non-positive
   contracts, negative counts, `n_rows_certified > n_rows`.
4. **The certificate chose its own bar.** One level up again: re-deriving the verdict is worthless if
   the artifact selects *what it is judged against*. `certify_reachable_mass_balance` accepts any
   positive `contract`, so `contract=1.0` yields a **truthful** `is_certified` and passes the gate — a
   proof of a different and useless proposition, no corruption involved. M9 settled that there is **one**
   declared definition of mass-balanced (η = 1e-9, the same bar emitted samples meet), so
   `require_certified_transform` now tests `worst_absolute` against **the policy it was given**, and the
   certificate's own `contract` is demoted to provenance. A *stricter* certificate still passes, since a
   smaller residual clears a larger bar.

**Deferred by agreement:** a **metadata digest** in `cache.ArtifactCache`, which hashes every array and
trusts the meta. Domain validation closes *malformed* evidence; integrity of cached metadata is a
generic property of every layer, not an L3 concern, and is recorded rather than bolted on here. Neither
this nor the required-argument guards are adversary-proof — a caller who hand-builds a plausible
certificate defeats any Python-level proof object — and the docstrings say so rather than overclaiming.

**The lesson:** *an artifact must be a function of its key* — and the question a key answers is
**artifact identity**, never target identity. Confusing the two is how a correct-looking cache serves
the wrong bytes, and it is subtle enough that this milestone made the mistake **while fixing it**.

**The second lesson, from rounds 3–5:** *a guard is only as total as its weakest entrance, and each fix
reveals the next one.* Don't trust the claim → check the evidence. Don't trust the evidence → validate
it. Don't trust the validation → own the bar it is judged against. Four rounds, each finding the hole
behind the previous repair — which is exactly what M2 recorded ("the first fix for a numerical bug is
often itself buggy") generalized from arithmetic to authority.

---

## M10.2b — pilot caching (3 rounds, converged after Codex refuted my design and my repair)

**Round 1 killed my position outright, and it was the good kind of kill.** I opposed the recorded
payload split (geometry pilot → coordinates, scale pilot → fluxes) because it destroys both tests
that "prove the pilots are independent streams". Codex:

1. **`not np.allclose(a, b)` proves NON-IDENTITY, not independence.** The tests' names overclaimed
   what they checked.
2. **The property they name is not even true.** `T₁` is *derived from* the geometry pilot and the
   scale pilot runs under `T₁`, so the two pilots are **causally dependent**. What is independent is
   each pilot's **RNG stream given its inputs**.
3. The direct evidence is the **spawn key**, which `run_chain` already records and `NeutralPilot`
   **discarded wholesale** — so BUILD_PLAN §1.2's "…and store the spawn keys" was unmet.

I conceded all four contested points. **Settled:** stage-specific artifact types (`GeometryPilot` /
`ScalePilot`), `STAGE` as a `ClassVar`, separate builders with **no `stage` parameter**, public
`run_neutral_pilot` retired, spawn keys stored as a **cache-hashed array** and **recomputed** on every
construction.

**Round 2: Codex refused its own round-1 proposal.** It had suggested storing a SHA-256 flux
fingerprint; it then argued against it — for `GeometryPilot` the source fluxes are *deliberately
discarded*, so the digest would be an **unrederivable assertion in trusted metadata**, and comparing
two digests re-runs the very proxy round 1 rejected. **Evidence you recompute is evidence; evidence
you store and read back is a claim.** Also settled: the gate goes **before** the cache dispatch (not
in `compute()`, which runs only on a miss); no certificate inside the pilot artifact (it certifies the
`(polytope, transform, policy)` edge, not the pilot); and the `feasibility_tol` fix is **in scope**,
because M10.2b creates the cache identity that would otherwise preserve the inconsistency
indefinitely.

**Round 3 — review the BUILT diff — found the hole inside the repair.** Opened DISAGREE. Hoisting the
*certificate* gate out of `compute()` left the **polytope relation** in `_run_pilot_chains`, on the
miss path only. My own probe for this passed, because an *honest* wrong polytope changes the keys and
is caught three other ways. Codex's attack: `dataclasses.replace(transform, polytope_key=<a lie>)` —
`RoundedTransform.content_key` hashes `geometry_key`/`transform`/`center`/`ridge` and **not the
transform's own `polytope_key`** — so the lie keys identically and a **hit serves what a miss
refuses**. Executed and confirmed. Fixed via `require_pilot_inputs`, the one place both paths ask.
Three overclaims of mine also refused and narrowed: the spawn-key guard proves the four *semantic
coordinates* only (`seed` lives in the `SeedSequence` **entropy**, absent from `spawn_key`);
`_frozen` was **not** the only array constructor (plain dataclass construction produced a mutable
pilot, so freezing was a convention, not a class invariant); and the `IMPL_VERSION = 3` rationale was
**wrong** (a named `feasibility_tol` component already separates v2 from v3 keys, and no v2 pilot was
ever stored) — kept as bookkeeping with an honest reason.

**The lesson:** *hoisting one of two checks is how a fix closes an asymmetry while claiming to have
closed the asymmetry.* A precondition a cache lookup can skip is not a precondition — and two call
sites checking **different subsets** is how one of them ends up checking none. Sibling to M10.2a's
"a guard is only as total as its weakest entrance", now applied to the guard's own repair.

**And a second, unrelated lesson from the same milestone:** *a recorded **measurement** is a claim
with a premise, and it expires silently when the premise moves.* §1.6.6 recorded `cond(C_q)` 5.36e3
(2.87×); the shipped code gives 5.97e3 (2.57×) and has since M10.2a removed the pilots' start hint —
whose own version-bump note **says** it changes every draw. The bell was rung and not heard. See
BUILD_PLAN §1.6.6b.
