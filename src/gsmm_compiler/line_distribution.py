"""The exact 1D conditional of the maximum-entropy target along a chord (spec §20).

Along the line ``v + t·d`` the sparse objective

    J(t) = (v_b + t·d_b) − λ · Σ_{r∈R_p} w_r · |v_r + t·d_r|

is concave and piecewise linear: each penalized reaction ``r`` bends it exactly once, where its flux
crosses zero, at ``τ_r = −v_r/d_r``, and the slope drops there by ``2·λ·w_r·|d_r|``. The conditional

    p(t) ∝ exp[ β·(J(t) − J*) / s_J ] · 1_{[t_lo, t_hi]}(t)

is therefore piecewise *exponential*: on each linear segment it is ``e^{κx}`` with ``κ = β·m/s_J``.
We sample it exactly — pick a segment from its integrated mass, then invert the truncated
exponential CDF within it. There is no Metropolis correction because there is nothing to correct.

Four things here are deliberately not what spec §20 says. The first three come from BUILD_PLAN §1.6;
the fourth was found by the M2 collab review and is the one that actually bites.

1. **Distinct cuts are never merged** (§1.6.2). Spec §20.3 step 3 merges breakpoints "equal within
   tolerance". Two breakpoints differing by 1e-13 are two genuine bends of ``J``; collapsing them
   moves a slope change to the wrong place, which changes ``J`` and so changes the distribution
   being sampled — silently, and by an amount no feasibility check would flag. We group only
   **exactly equal** cuts (`np.unique`), summing their slope drops. Distinct cuts stay distinct
   however close they are; the sliver between them carries proportionally little mass, which is the
   right answer rather than a problem to tolerate.

2. **The first slope is not read off a midpoint.** Spec §20.3 step 6 evaluates the opening slope at
   the midpoint of the first segment. When that segment is one ULP wide, ``0.5·(a₀ + a₁)`` rounds to
   ``a₀`` or to ``a₁`` depending on the round-half-to-even parity of ``2·a₀`` — and when it lands on
   ``a₁``, that *is* the cut: the crossing reaction's flux there is exactly ``0.0``, ``sgn(0) = 0``,
   and the slope comes back as the subgradient midway between the left and right slopes rather than
   as either one. Measured over 12009 thin-first-segment configurations, the midpoint lands on the
   cut in 10.5% of them, each time reporting a slope 2× off. Instead we fix each sign from the side
   of ``τ_r`` on which the segment lies, which is exact at any segment width.

3. **``J*`` is not assumed to bound ``J``** (§1.6.4). Solver tolerance lets ``J − J*`` come out
   slightly positive, so nothing may assume the exponent is ``≤ 0``.

4. **The absolute value of ``J`` never enters the draw — only heights relative to its peak do.**
   ``J*`` and any constant offset cancel out of ``p(t)`` algebraically, but they do *not* cancel
   numerically if you form them first: ``h_a = β·(J(a) − J*)/s_J`` catastrophically cancels when
   ``J`` and ``J*`` are large and their difference is small, which is the normal case. Worse,
   propagating the knot values as ``J(t_lo) + cumsum(...)`` destroys the relative heights *before*
   ``J*`` is even subtracted, whenever the baseline is large.

   Concretely, with a biomass flux of 1e16 the true segment probabilities ``[0.387, 0.613]`` came
   back as ``[0.632, 0.368]`` — *the favoured segment reversed*, from slopes that were themselves
   exactly right. And anchoring the heights at ``t_lo`` fixes only half of it: a large excursion
   *within* the chord then swamps every later increment, so a rise of 1e16 followed by one of 0.5
   stores both knots as the same float and loses a factor of ``e^{0.5}`` in the weights.

   So `PiecewiseLinearJ` stores `heights` **relative to the peak of ``J``** — accumulated outward
   from it, along the slopes, never through the absolute value — and `log_segment_masses` takes no
   ``J*`` at all. Concavity makes the peak the right anchor: the segments near it carry
   essentially all the mass and keep their increments exact, while the ones far below lose low-order
   bits they do not need, their mass being ``e^{−large} ≈ 0`` regardless. The absolute baseline is
   kept separately, for reporting only. Making the cancellation structurally impossible beats
   testing for it.

Stability across ``κL``: the segment mass is ``M = e^{h_a}·L·φ(κL)`` with ``φ(x) = expm1(x)/x``, and
only its log is ever taken — see `_log_phi`, accurate for ``|κL|`` from 1e-16 to 1000 and beyond
without ever exponentiating a large positive number.

Implemented in **M2** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.line_geometry import Chord

SERIES_LIMIT = 1e-2
"""Below this ``|x|``, ``log φ(x)`` is summed as a series instead of by its closed form.

Not an approximation — the series is the *more accurate* branch here, not the cheaper one. Both
closed forms in `_log_phi` difference two logarithms that collide as ``x → 0``, so they shed digits
in exactly the regime the series nails: at ``x = 1e-3`` the closed form carries ~5e-14 of absolute
error, while the series truncates at ``x⁶/181440`` — below 1e-17 anywhere under this limit.

The crossover was measured, not guessed. At this limit the worst absolute error of `_log_phi` over
``|x| ∈ [1e-18, 1e3]`` is 5.7e-14, at ``x ≈ 318`` — which is **one ULP of the result** (``log φ ≈
312.6``) and so the float64 floor rather than a defect of the formula. Below ``|x| = 1`` the error
stays under 6e-15. Dropping the limit to 1e-3 leaves a cancellation band just above the boundary
where the error reaches 4.8e-14 on a result of order 5e-4 — eight orders of magnitude worse in
relative terms, for no gain.

Absolute error is the criterion, not relative: ``log φ`` is one addend of a segment's log-mass, and
`choose_segment` only ever looks at *differences* of log-masses.
"""

UNIFORM_LIMIT = float(np.finfo(np.float64).eps / 4.0)
"""``|κL|`` below which the tilt is *bitwise* absent and `sample_on_segment` draws uniformly.

A representability limit, not a modelling shortcut (BUILD_PLAN §1.4). Below ``eps/4 ≈ 5.55e-17``,
``exp(κx)`` rounds to exactly ``1.0`` for **every** ``x`` in the segment and for **either sign** of
``κ``: the tilted law and the uniform law are not merely close, they are the same float64 law, so
drawing uniformly is exact rather than approximate.

``eps/4``, not ``eps/2``, because float spacing is asymmetric about 1.0. Above it the spacing is
``eps``; below it, ``eps/2``. So the rounding midpoint on the *low* side is ``1 − eps/4``, and a
negative tilt of ``−eps/2 < x < −eps/4`` rounds ``1 + x`` down to ``nextafter(1.0, 0)`` — not to
``1.0``. A threshold of ``eps/2`` would hold the bitwise claim for positive ``κ`` and quietly break
it for negative ``κ``, which is exactly the half of the argument it is easy to forget to check.

This constant is wrong in *both* directions, and both bounds were found the hard way:

* **Too high** and it really is an approximation — the mass integral stays tilted while the draw
  goes flat, so the two stages target different laws.
* **Too low** and the inverse CDF is pushed into a regime where it sheds precision for nothing. The
  obvious repair — drop the limit to the smallest normal double, so that ``A = −expm1(−|κL|)`` never
  goes denormal — is a trap, and was briefly shipped here. Keeping ``A`` normal does not keep
  ``u·A`` normal: at ``κ = 2.2e-308``, ``u·A`` underflows for the smallest quantiles and the draw
  loses its low-order bits. Measured, the damage is ≤ 1 ULP of ``x`` and confined to ``u ≲ 1e-15``,
  so the *law* stayed uniform — a blemish, not the corruption it first looked like. But it is
  precision spent to resolve a tilt float64 cannot represent in the first place.

``eps/2`` satisfies both: it clears the denormal band by ~292 orders of magnitude, so ``u·A`` is
always normal and the exact inversion runs everywhere it is meaningful, while still being small
enough that the branch it guards changes nothing. The mass side needs no special case — ``log φ(κL)
≈ κL/2`` there is below the ULP of the ``O(1)`` terms it is added to.
"""


class InvalidObjectiveError(ValueError):
    """The objective's arrays are inconsistent, or a parameter is out of range."""


class DegenerateChordError(ValueError):
    """The chord has no positive width, so it carries no piecewise structure to build."""


@dataclass(frozen=True)
class L1Objective:
    """The sparse flux objective ``J(v) = v_b − λ·Σ_{r∈R_p} w_r·|v_r|`` over the active reactions.

    This is the objective as the *line kernel* needs it — the pure function of the flux vector. M3's
    `sparse_objective` builds the ``(v, z)`` LP that maximizes the same ``J`` and hands one of these
    to the sampler; M7 rebuilds one from its frozen reweighted ``w``. The weights are frozen before
    sampling starts and never depend on chain state (BUILD_PLAN M7): were they to move mid-chain,
    the conditionals below would target a different distribution on every step, and stationarity
    would be lost.
    """

    biomass_index: int
    penalized_indices: NDArray[np.intp]
    weights: NDArray[np.float64]
    lam: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.penalized_indices.ndim != 1 or self.weights.ndim != 1:
            raise InvalidObjectiveError("penalized_indices and weights must be 1-D")
        if self.penalized_indices.size != self.weights.size:
            raise InvalidObjectiveError(
                f"penalized_indices ({self.penalized_indices.size}) and weights "
                f"({self.weights.size}) differ in length"
            )
        if self.weights.size and not np.all(np.isfinite(self.weights)):
            raise InvalidObjectiveError("weights contain NaN or inf")
        if self.weights.size and np.any(self.weights < 0.0):
            raise InvalidObjectiveError("weights must be nonnegative (J must stay concave)")
        if not np.isfinite(self.lam) or self.lam < 0.0:
            raise InvalidObjectiveError(f"lam must be finite and nonnegative, got {self.lam}")
        if self.biomass_index < 0:
            raise InvalidObjectiveError(
                f"biomass_index must be nonnegative, got {self.biomass_index}"
            )

    def evaluate(self, v: NDArray[np.float64]) -> float:
        """``J(v)`` at a single flux vector."""
        penalty = float(np.sum(self.weights * np.abs(v[self.penalized_indices])))
        return float(v[self.biomass_index]) - self.lam * penalty

    def evaluate_on_line(
        self,
        v: NDArray[np.float64],
        direction: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """``J(v + t·d)`` for every ``t``, evaluated straight from the definition.

        The reference path: `build_piecewise_j` reconstructs the same function from breakpoints and
        slope drops instead, and the M2 gate holds the two against each other on a dense grid.
        """
        steps = np.asarray(t, dtype=np.float64)
        biomass = v[self.biomass_index] + steps * direction[self.biomass_index]
        if self.penalized_indices.size == 0:
            return np.asarray(biomass, dtype=np.float64)

        v_p = v[self.penalized_indices]
        d_p = direction[self.penalized_indices]
        # (n_t, n_penalized): |v_r + t·d_r| for every (t, r).
        fluxes = np.abs(v_p[np.newaxis, :] + steps[:, np.newaxis] * d_p[np.newaxis, :])
        return np.asarray(biomass - self.lam * (fluxes @ self.weights), dtype=np.float64)


@dataclass(frozen=True)
class PiecewiseLinearJ:
    """``J`` restricted to a chord: ``K`` linear segments between ``K+1`` knots.

    Segment ``k`` spans ``[knots[k], knots[k+1]]`` with slope ``slopes[k]``. Concavity shows up as
    ``slopes`` being nonincreasing.

    The knot values are split in two, and the split is what keeps the sampler exact (see the module
    docstring, delta 4):

    * ``heights[k]`` is ``J(knots[k]) − max J`` — the height **relative to the peak** of ``J`` on
      the chord, so every entry is ``≤ 0`` and the largest is exactly ``0``. It is accumulated from
      the slopes and never routed through the absolute value, so it survives no matter how large
      ``J`` itself is. This is the only thing `log_segment_masses` is allowed to look at, because it
      is the only thing the target depends on: a constant offset of ``J`` cancels out of ``p(t)``.
    * ``baseline`` is the absolute ``max J``, carried for reporting and for `evaluate`.

    Anchoring at the **peak** rather than at ``t_lo`` is not cosmetic. ``J`` is concave, so the
    segments that carry essentially all the mass are those near the peak, and accumulating outward
    from it keeps *their* increments exact. Anchor at ``t_lo`` instead and a single large excursion
    early in the chord swamps every later increment: a rise of 1e16 followed by one of 0.5 stores
    both knots as the same float, and the 0.5 — a factor of ``e^{0.5}`` in the segment weights — is
    gone. The far-from-peak segments do lose low-order bits under this scheme, and that is harmless
    by construction: their mass is ``e^{−large} ≈ 0`` either way.

    Add the two back together and you get the absolute ``J`` — with whatever cancellation a large
    baseline implies, which is fine for a diagnostic and fatal for a probability.
    """

    knots: NDArray[np.float64]
    heights: NDArray[np.float64]
    slopes: NDArray[np.float64]
    baseline: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.knots.size < 2:
            raise DegenerateChordError("a chord needs at least two knots (one segment)")
        if self.heights.size != self.knots.size:
            raise InvalidObjectiveError("heights must hold one entry per knot")
        if self.slopes.size != self.knots.size - 1:
            raise InvalidObjectiveError("slopes must hold one entry per segment")
        if not (np.all(np.isfinite(self.heights)) and np.all(np.isfinite(self.slopes))):
            raise InvalidObjectiveError("piecewise J is not finite")
        if self.heights.max() != 0.0 or np.any(self.heights > 0.0):
            raise InvalidObjectiveError(
                "heights are relative to the peak of J, so they must be ≤ 0 with a maximum of 0"
            )
        if np.any(np.diff(self.knots) <= 0.0):
            raise InvalidObjectiveError("knots must be strictly increasing")

    @property
    def n_segments(self) -> int:
        return int(self.slopes.size)

    @property
    def t_lo(self) -> float:
        return float(self.knots[0])

    @property
    def t_hi(self) -> float:
        return float(self.knots[-1])

    @property
    def values(self) -> NDArray[np.float64]:
        """Absolute ``J`` at each knot. Reporting only — never form a probability from these."""
        return np.asarray(self.baseline + self.heights, dtype=np.float64)

    def evaluate(self, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Absolute ``J(t)`` from the piecewise representation, for ``t`` anywhere on the chord."""
        steps = np.atleast_1d(np.asarray(t, dtype=np.float64))
        # side="right" places a t landing exactly on a knot in the segment to its right; the two
        # segments agree there (they share the knot), so the choice only matters at t_hi, which the
        # clip pulls back into the last segment.
        segment = np.clip(
            np.searchsorted(self.knots, steps, side="right") - 1, 0, self.n_segments - 1
        )
        relative = self.heights[segment] + self.slopes[segment] * (steps - self.knots[segment])
        return np.asarray(self.baseline + relative, dtype=np.float64)


def build_piecewise_j(
    v: NDArray[np.float64],
    direction: NDArray[np.float64],
    chord: Chord,
    objective: L1Objective,
) -> PiecewiseLinearJ:
    """Build ``J`` on the chord from its breakpoints and slope drops (spec §20.3).

    Only penalized reactions with ``d_r != 0`` bend ``J``, and only where their bend ``τ_r =
    −v_r/d_r`` falls strictly inside the chord. Cuts that are *exactly* equal are one bend and their
    drops add; cuts that merely round close stay separate (module docstring, delta 1).
    """
    if not chord.is_samplable:
        raise DegenerateChordError(
            f"chord [{chord.t_lo:.6e}, {chord.t_hi:.6e}] has no positive width"
        )
    t_lo, t_hi = chord.t_lo, chord.t_hi

    d_p = direction[objective.penalized_indices]
    v_p = v[objective.penalized_indices]

    crossing = d_p != 0.0  # exact: a zero component never crosses zero
    d_crossing = d_p[crossing]
    w_crossing = objective.weights[crossing]
    tau = -v_p[crossing] / d_crossing

    interior = (tau > t_lo) & (tau < t_hi)
    cuts, grouped = np.unique(tau[interior], return_inverse=True)
    # λ·w first: the product is the physical weight and stays finite even where λ alone would
    # overflow the leading 2·λ of the textbook ordering.
    drops = 2.0 * (objective.lam * w_crossing[interior] * np.abs(d_crossing[interior]))
    drop_at_cut = np.bincount(grouped, weights=drops, minlength=cuts.size)

    knots = np.concatenate(([t_lo], cuts, [t_hi])).astype(np.float64)

    # Opening slope: m = d_b − λ·Σ w_r·sgn(v_r + t·d_r)·d_r on the first open segment. No interior
    # cut lies inside that segment, so each crossing reaction's sign there is fixed by which side of
    # τ_r the segment sits on — exact even for a segment one ULP wide, where a midpoint evaluation
    # would land on the cut and read sgn(0) = 0. Reactions with d_r == 0 contribute
    # w_r·sgn(v_r)·0 = 0, so they are absent from the sum entirely.
    opens_after_cut = tau <= t_lo
    sign_on_first = np.where(opens_after_cut, np.sign(d_crossing), -np.sign(d_crossing))
    first_slope = float(direction[objective.biomass_index]) - objective.lam * float(
        np.sum(w_crossing * sign_on_first * d_crossing)
    )

    # Concavity, by construction: every drop is nonnegative, so the slopes only decrease (§20.2).
    slopes = first_slope - np.concatenate(([0.0], np.cumsum(drop_at_cut)))

    # Heights relative to the PEAK of J, accumulated outward from it (module docstring, delta 4).
    # Concavity puts the peak where the slopes change sign, which the slopes alone locate exactly —
    # no need to consult any (possibly already-corrupted) value of J.
    increments = slopes * np.diff(knots)  # J(knot k+1) − J(knot k)
    peak = int(np.count_nonzero(slopes > 0.0))  # J rises while the slope is positive
    heights = np.empty(knots.size, dtype=np.float64)
    heights[peak] = 0.0
    if peak < slopes.size:  # to the right of the peak every slope is ≤ 0, so J only falls
        heights[peak + 1 :] = np.cumsum(increments[peak:])
    if peak > 0:  # to the left every slope is > 0, so walk backwards subtracting
        heights[:peak] = -np.cumsum(increments[peak - 1 :: -1])[::-1]

    baseline = float(objective.evaluate_on_line(v, direction, knots[peak : peak + 1])[0])

    return PiecewiseLinearJ(
        knots=knots,
        heights=heights,
        slopes=np.asarray(slopes, dtype=np.float64),
        baseline=baseline,
    )


def _log_phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log(expm1(x)/x)``, with ``φ(0) = 1``, stable for every finite ``x``.

    ``φ(x) = (e^x − 1)/x > 0`` for all real ``x`` (both factors flip sign together), so the log is
    always defined. Three regimes:

    * ``|x| < SERIES_LIMIT``: ``x/2 + x²/24 − x⁴/2880``, the Maclaurin series (Bernoulli
      coefficients) — the next term is ``x⁶/181440``, so truncation stays below 1e-17 across the
      whole branch. Both closed forms below difference two logs that collide as ``x → 0``, so this
      is the accurate branch there, not merely the fast one.
    * ``x > 0``: ``log(e^x − 1) = x + log1p(−e^{−x})``, hence ``log φ = x + log1p(−e^{−x}) −
      log(x)``. Nothing large is exponentiated: at ``x = 1000`` the ``e^{−x}`` underflows to 0 and
      the result is exactly ``1000 − log(1000)``, where a direct ``expm1(1000)`` would be ``inf``.
    * ``x < 0``: ``expm1(x) ∈ (−1, 0)``, so ``log φ = log(−expm1(x)) − log(−x)``, both arguments
      safely positive, with ``−expm1(x) → 1`` as ``x → −∞``.
    """
    out = np.empty_like(x)

    small = np.abs(x) < SERIES_LIMIT
    positive = ~small & (x > 0.0)
    negative = ~small & (x < 0.0)

    xs = x[small]
    out[small] = xs / 2.0 + xs * xs / 24.0 - xs**4 / 2880.0

    xp = x[positive]
    out[positive] = xp + np.log1p(-np.exp(-xp)) - np.log(xp)

    xn = x[negative]
    out[negative] = np.log(-np.expm1(xn)) - np.log(-xn)

    return out


def _inverse_temperature(beta: float, energy_scale: float) -> float:
    """``β/s_J``, validated — the one place the two scaling parameters ever meet.

    Grouping ``β/s_J`` before multiplying by a slope is not cosmetic: ``β·m`` overflows to ``inf``
    for ``β = 1e308, m = 2`` even though ``s_J = 1e308`` would bring it straight back to ``κ = 2``.

    But no ordering is safe for *every* finite input, so the ratio is checked rather than trusted.
    It can overflow (``β = 1e308, s_J = 1e-100``), and — the case that actually matters — it can
    **underflow to zero** (``β = 2e-200, s_J = 1e124``) while the true ``κ = β·m/s_J`` is an
    ordinary 2e-16. That one is dangerous precisely because it is *quiet*: ``κ`` silently becomes 0,
    the segment is sampled uniformly, and nothing complains. Every other pathology in this module
    ends in a raised error; this one would not, so it is made to.

    None of these regimes is physical — ``λ`` is a penalty weight of order 0.1–10, ``s_J`` an energy
    scale of order 1–1e3, ``β`` a ladder value of order 0–1e3. They are rejected rather than
    accommodated: contorting the arithmetic of the hot path to *almost* survive inputs that cannot
    occur would trade a loud failure for a quiet approximation, which is the wrong direction.
    """
    ratio = beta / energy_scale
    if not np.isfinite(ratio):
        raise InvalidObjectiveError(
            f"beta/energy_scale = {beta}/{energy_scale} overflows; both are out of any sane range"
        )
    if beta > 0.0 and ratio == 0.0:
        raise InvalidObjectiveError(
            f"beta/energy_scale = {beta}/{energy_scale} underflows to zero, which would silently "
            "flatten the tilt; both are out of any sane range"
        )
    return float(ratio)


def _tilt(
    inverse_temperature: float, slopes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``κ = (β/s_J)·m`` per segment."""
    return np.asarray(inverse_temperature * slopes, dtype=np.float64)


def log_segment_masses(
    piecewise: PiecewiseLinearJ,
    beta: float,
    energy_scale: float,
) -> NDArray[np.float64]:
    """Log of each segment's unnormalized mass ``∫ exp[β·J(t)/s_J] dt``, up to a common constant.

    With ``κ = β·m/s_J``, ``L = b − a`` and ``h_a = β·(J(a) − max J)/s_J``, the integral is
    ``M = e^{h_a}·L·φ(κL)``, so ``log M = h_a + log L + log φ(κL)`` — never exponentiated.

    **There is no ``J*`` parameter, on purpose.** ``J*`` shifts every entry by the same constant, so
    it cannot change a segment probability; carrying it would only invite the catastrophic
    cancellation that ``β(J − J*)/s_J`` suffers when ``J`` and ``J*`` are both large. For the same
    reason ``h_a`` is built from `PiecewiseLinearJ.heights` — anchored at the peak of ``J``, hence
    ``≤ 0`` — rather than from the absolute knot values. The returned masses are therefore
    unnormalized *and* offset by an unspecified constant, which is all `choose_segment` needs.
    """
    if not np.isfinite(beta) or beta < 0.0:
        raise InvalidObjectiveError(f"beta must be finite and nonnegative, got {beta}")
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise InvalidObjectiveError(f"energy_scale must be finite and positive, got {energy_scale}")

    lengths = np.diff(piecewise.knots)
    inverse_temperature = _inverse_temperature(beta, energy_scale)
    kappa = _tilt(inverse_temperature, piecewise.slopes)
    scaled = inverse_temperature * piecewise.heights

    # Anchor each segment's integral at its HIGHER endpoint. Both forms are algebraically the same —
    #     M = e^{h_a}·L·φ(+κL) = e^{h_b}·L·φ(−κL),   since h_b = h_a + κL
    # — but only the second is numerically safe on a rising segment, and vice versa. A long rising
    # segment far below the peak has a huge negative h_a and a huge positive κL that cancel to
    # something O(1); anchoring at h_a asks float64 to hold that cancellation in the low-order bits
    # of a number of magnitude 1e16, and it cannot. Anchoring at the peak-facing end keeps the
    # exponent small and the cancellation never arises.
    #
    # A pleasant consequence: the argument of log φ is then **always ≤ 0**, so no positive number is
    # exponentiated anywhere in the mass path — that safety is now structural rather than a property
    # of `_log_phi`'s branches.
    rising = kappa > 0.0
    anchor = np.where(rising, scaled[1:], scaled[:-1])
    tilt = np.where(rising, -kappa * lengths, kappa * lengths)

    log_masses = anchor + np.log(lengths) + _log_phi(tilt)
    if not np.all(np.isfinite(log_masses)):
        raise InvalidObjectiveError(
            "segment log-masses are not finite; check beta, energy_scale and the objective"
        )
    return np.asarray(log_masses, dtype=np.float64)


def choose_segment(log_masses: NDArray[np.float64], rng: np.random.Generator) -> int:
    """Draw a segment index with probability proportional to ``exp(log_masses)`` (spec §20.4).

    The largest log-mass is subtracted before exponentiating (the custom log-sum-exp the spec
    asks for), so a well-separated ladder — segment log-masses hundreds apart at large ``β`` —
    resolves to probabilities 1 and 0 rather than to ``inf/inf``.
    """
    shifted = np.exp(log_masses - log_masses.max())
    total = shifted.sum()
    if not total > 0.0:
        raise InvalidObjectiveError("segment masses sum to zero")

    cumulative = np.cumsum(shifted / total)
    index = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return min(index, log_masses.size - 1)  # guards against u landing past a rounded cumulative[-1]


def sample_on_segment(length: float, kappa: float, rng: np.random.Generator) -> float:
    """Draw ``x ∈ [0, L]`` with density ``∝ e^{κx}`` by inverse CDF (spec §20.5).

    With ``A = 1 − e^{−|κ|L} = −expm1(−|κ|L) ∈ (0, 1]``, the decreasing case ``κ < 0`` inverts to
    ``x = −log1p(−U·A)/|κ|``. The increasing case ``κ > 0`` is that same draw reflected: sample the
    distance ``y`` down from the far endpoint and return ``L − y``. Neither branch exponentiates a
    positive number, so ``κL = +1000`` is as safe as ``κL = −1000``.

    The reflection means the map ``U ↦ x`` *decreases* for ``κ > 0``: it inverts the CDF from the
    far end, giving ``F(x) = 1 − U`` there against ``F(x) = U`` for ``κ < 0``. Both draw the same
    law — ``U`` and ``1 − U`` are equally uniform — and orientation is nothing the sampler uses.

    The exact inversion runs for every ``κ`` whose tilt float64 can represent; see `UNIFORM_LIMIT`
    for the one case it cannot, where the density is bitwise flat and the uniform draw is exact.
    """
    kappa_length = kappa * length
    u = rng.random()

    if abs(kappa_length) < UNIFORM_LIMIT:
        return float(u * length)

    magnitude = abs(kappa)
    a = -np.expm1(-magnitude * length)
    x = -float(np.log1p(-u * a)) / magnitude
    if kappa > 0.0:
        x = length - x

    # Exact arithmetic keeps x in [0, L]; one ULP of rounding in the inversion need not, and the
    # chord's endpoints are hard bounds.
    return float(min(max(x, 0.0), length))


def sample_line(
    v: NDArray[np.float64],
    direction: NDArray[np.float64],
    chord: Chord,
    objective: L1Objective,
    beta: float,
    energy_scale: float,
    rng: np.random.Generator,
) -> float:
    """Draw ``t`` from the exact conditional ``p(t) ∝ exp[β·J(v + t·d)/s_J]`` on the chord.

    The whole line step (spec §20.7). It is exact, so the caller applies no Metropolis correction:
    this is a Gibbs update of one reduced coordinate and leaves ``π_β`` invariant.

    A chord with no positive width returns its `Chord.degenerate_point` — the single feasible ``t``,
    whose point mass is the exact conditional there. That is ``0.0`` (a **self-loop**) for a state
    on its bounds, and the nearby feasible ``t`` for one that has drifted a hair outside. The caller
    must not respond by redrawing a different coordinate: that would make the coordinate-selection
    law depend on the state and break stationarity (see `line_geometry`).
    """
    if not chord.is_samplable:
        return chord.degenerate_point

    if beta == 0.0:
        # The target is flat: uniform on the chord, with no segment structure built at all
        # (spec §18.2). This is the entire β=0 inner loop.
        return float(rng.uniform(chord.t_lo, chord.t_hi))

    piecewise = build_piecewise_j(v, direction, chord, objective)
    log_masses = log_segment_masses(piecewise, beta, energy_scale)
    k = choose_segment(log_masses, rng)

    a = float(piecewise.knots[k])
    b = float(piecewise.knots[k + 1])
    # The same tilt the mass was weighted with — weighting a segment with one κ and sampling it
    # with another is exactly the class of bug that makes a chain converge to the wrong law.
    inverse_temperature = _inverse_temperature(beta, energy_scale)
    kappa = float(_tilt(inverse_temperature, piecewise.slopes[k : k + 1])[0])
    t = a + sample_on_segment(b - a, kappa, rng)

    return float(min(max(t, chord.t_lo), chord.t_hi))
