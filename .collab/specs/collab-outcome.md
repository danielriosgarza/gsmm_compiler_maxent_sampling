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
