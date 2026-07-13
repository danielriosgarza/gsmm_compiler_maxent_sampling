"""The feasible chord ``[t_lo, t_hi]`` of a line ``v + t·d`` inside the box ``l ≤ v ≤ u``.

**Delta from spec §19 (BUILD_PLAN §1.6.1).** The spec says to "ignore components with ``|dᵢ|`` below
a direction tolerance" when intersecting the per-reaction intervals. That is a feasibility bug, not
an optimization. The limit a reaction imposes is ``(uᵢ − vᵢ)/dᵢ``, and a *small* ``dᵢ`` does not
make that limit loose — it does so only if ``uᵢ − vᵢ`` is not small as well. A reaction sitting one
part in 10¹² from its bound with ``dᵢ = 1e-14`` still binds ``t`` at ~100, and dropping it lets the
chain step straight through the bound. So we keep **every structurally nonzero component**
(``dᵢ != 0``, an exact test) and no tolerance enters the intersection at all.

**Delta from spec §19 (M2 collab).** The spec also says to "reject and redraw a direction if the
chord length is numerically zero". *Redrawing another coordinate breaks stationarity.* Random-scan
Gibbs preserves ``π_β`` because the coordinate is chosen independently of the state: the kernel is
the uniform mixture ``(1/d)·Σ_k P_k``, and each ``P_k`` preserves ``π_β``. If instead the chord is
inspected and a *different* coordinate is chosen when it comes back narrow, the mixture weights
become functions of the current state, and the invariance argument collapses — the weights can no
longer be pulled out of the integral.

The exact alternative costs nothing. When the feasible set along the line is the single point
``t = 0``, the conditional distribution on it **is** the point mass at ``t = 0``, so the correct
Gibbs update is to stay put: a self-loop. `line_distribution.sample_line` returns ``0.0`` rather
than raising or redrawing. And a chord that is merely *narrow* — width 1e-13, say — has a perfectly
well-defined conditional and is simply sampled; nothing here refuses it, so no tolerance is allowed
to decide what belongs to the support.

Tolerances therefore appear here for exactly one job, and it is about floating point rather than
about which constraints exist: ``feasibility_tol`` guards the assertion ``t_lo ≤ 0 ≤ t_hi``
(spec §19). The current point lies on the line at ``t = 0``, so a feasible ``v`` must produce a
chord containing the origin. We *assert* this rather than clamping the chord around it: if ``v`` has
drifted a hair outside a bound, the honest chord is the one that excludes ``t = 0``, and the next
draw then pulls the state back inside. Clamping would instead widen the sampled support past the
true bound.

The one ULP of ``nextafter`` inward is not a safety margin — it cannot be, since ``v + t·d`` is
itself rounded. It removes the case that actually produces an out-of-bounds flux in practice: an
endpoint landing *exactly* on the bound, where the rounding of the division alone decides the sign
of the residual. Excursions of order ``eps·|u|`` remain, are unavoidable in float64, and are what
the feasibility tolerance in the diagnostics is for.

Implemented in **M2** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_FEASIBILITY_TOL = 1e-9
"""How far outside its bounds the current point may sit before the chord is called invalid."""


class ZeroDirectionError(ValueError):
    """The direction has no structurally nonzero component, so the line is a single point.

    It means the transform has a zero column — a geometry defect (M4/M5), not an unlucky draw.
    """


class UnboundedChordError(ValueError):
    """The chord came out non-finite, so there is nothing to sample.

    All bounds are finite (`model_input` rejects infinite ones), so a *mathematically* unbounded
    chord is impossible. A **numerically** unbounded one is not: a direction component small enough
    (a denormal, say) overflows its bound ratio ``(uᵢ − vᵢ)/dᵢ`` to ``inf``, and if every nonzero
    component does so, the intersection is ``[−inf, +inf]``. That is caught here rather than left to
    reach ``rng.uniform(-inf, inf)``, which raises `OverflowError` from deep inside the sampler.
    """


class InfeasibleChordError(ValueError):
    """The current point ``v`` violates its bounds by more than the feasibility tolerance."""


@dataclass(frozen=True)
class Chord:
    """The interval of step sizes ``t`` for which ``v + t·d`` stays inside the box."""

    t_lo: float
    t_hi: float

    @property
    def length(self) -> float:
        """Chord width. **Negative** if the endpoints crossed — see `is_samplable`."""
        return self.t_hi - self.t_lo

    @property
    def is_samplable(self) -> bool:
        """True when the chord has positive width, so a conditional can be drawn on it.

        Positive width is the *only* requirement — no minimum. A chord 1e-13 wide has a perfectly
        well-defined conditional and gets sampled like any other; letting a magnitude threshold
        refuse it would be a tolerance deciding what belongs to the support.

        When this is false the feasible set on the line has collapsed to a single point, and
        `sample_line` moves to it — see `degenerate_point`. That covers an exact singleton chord and
        one the inward ``nextafter`` has inverted from a one-ULP interval. It does **not** cover an
        empty feasible set: `feasible_chord` raises on a raw crossed chord rather than returning
        one, because no step along that line can restore feasibility.
        """
        return self.length > 0.0

    @property
    def degenerate_point(self) -> float:
        """The single feasible ``t`` when the chord has no width. Only meaningful if not samplable.

        **Not** hard-coded to ``0.0``. It is ``0.0`` for a state sitting exactly on its bounds — the
        self-loop — but a state that has drifted a hair *outside* a bound (within `feasibility_tol`,
        so `feasible_chord` accepts it) produces a singleton chord at some small ``t ≠ 0``, and that
        ``t`` is the exact conditional: it is the one feasible point on the line. Returning ``0.0``
        there would pin the state outside its bound permanently, since every later visit to this
        coordinate would make the same non-move.

        Stationarity is untouched either way. On-support, the singleton at ``t = 0`` is the identity
        kernel. Off-support, the target assigns the state probability zero, so nothing done there
        can perturb the invariant measure — but moving to the feasible point is what gets the chain
        back onto the support, and staying is what would strand it.
        """
        return 0.5 * (self.t_lo + self.t_hi)

    def contains(self, t: float) -> bool:
        return self.t_lo <= t <= self.t_hi


def feasible_chord(
    v: NDArray[np.float64],
    direction: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
    nudge: bool = True,
) -> Chord:
    """Return the chord ``[t_lo, t_hi]`` of ``{t : l ≤ v + t·d ≤ u}``.

    Every structurally nonzero component of ``direction`` constrains the result; see the module
    docstring for why no magnitude tolerance is applied.

    Raises `ZeroDirectionError` on a zero direction and `InfeasibleChordError` when ``v`` sits
    further outside its bounds than ``feasibility_tol``. A narrow or empty chord is **not** an
    error: it comes back with ``is_samplable`` false, and the caller stays put.
    """
    nonzero = np.flatnonzero(direction)
    if nonzero.size == 0:
        raise ZeroDirectionError("direction has no nonzero component; the chord is unbounded")

    d = direction[nonzero]
    # Both ratios per reaction. For d > 0 the lower bound gives the left limit; for d < 0 the
    # division flips their order, so take the elementwise min/max rather than branching on sign.
    # A component small enough to overflow its ratio yields ±inf here, which is the honest answer —
    # it constrains nothing that float64 can express — and is caught below if every one does so.
    with np.errstate(over="ignore"):
        to_lower = (lower[nonzero] - v[nonzero]) / d
        to_upper = (upper[nonzero] - v[nonzero]) / d

    t_lo = float(np.minimum(to_lower, to_upper).max())
    t_hi = float(np.maximum(to_lower, to_upper).min())

    if not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        raise UnboundedChordError(
            f"chord [{t_lo}, {t_hi}] is not finite: every nonzero direction component is too small "
            "for its bound ratio to be representable"
        )

    # A *raw* crossed chord means no t satisfies every bound — the feasible set on this line is
    # empty, so v is infeasible however small the violation looks. It is not a degenerate chord to
    # be stepped through: no move along this line repairs it, and pretending otherwise would leave
    # the state outside its bounds while reporting success. (The inward nudge below can also cross
    # the endpoints, but that one is *ours* and does denote a real, if one-ULP, feasible set.)
    if t_lo > t_hi:
        raise InfeasibleChordError(
            f"no feasible step exists: chord [{t_lo:.3e}, {t_hi:.3e}] is empty, so v violates "
            "opposing bounds along this direction"
        )

    # v lies on the line at t = 0, so a feasible v must admit it (spec §19).
    if t_lo > feasibility_tol or t_hi < -feasibility_tol:
        raise InfeasibleChordError(
            f"current point is outside its bounds: chord [{t_lo:.3e}, {t_hi:.3e}] excludes t=0 "
            f"by more than {feasibility_tol:.1e}"
        )

    if nudge:
        t_lo, t_hi = float(np.nextafter(t_lo, t_hi)), float(np.nextafter(t_hi, t_lo))

    return Chord(t_lo=t_lo, t_hi=t_hi)
