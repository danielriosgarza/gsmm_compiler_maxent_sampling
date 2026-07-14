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
