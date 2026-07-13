"""M2 — the feasible chord.

The load-bearing test here is `test_tiny_direction_component_still_binds`: it is the one that fails
if anyone reintroduces the spec's "ignore |dᵢ| below tolerance" (BUILD_PLAN §1.6.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.line_geometry import (
    Chord,
    InfeasibleChordError,
    UnboundedChordError,
    ZeroDirectionError,
    feasible_chord,
)


def test_chord_intersects_per_reaction_intervals() -> None:
    v = np.zeros(2)
    direction = np.array([1.0, -1.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([2.0, 2.0])

    # Reaction 0 (d > 0) allows t ∈ [-1, 2]; reaction 1 (d < 0) flips to t ∈ [-2, 1].
    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    assert chord.t_lo == pytest.approx(-1.0)
    assert chord.t_hi == pytest.approx(1.0)
    assert chord.contains(0.0)


def test_negative_direction_swaps_the_ratio_order() -> None:
    v = np.array([1.0])
    direction = np.array([-2.0])
    lower = np.array([0.0])
    upper = np.array([5.0])

    # (l - v)/d = 0.5 and (u - v)/d = -2.0 — the *upper* bound gives the left limit here.
    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    assert chord.t_lo == pytest.approx(-2.0)
    assert chord.t_hi == pytest.approx(0.5)


def test_zero_components_do_not_constrain() -> None:
    v = np.array([0.0, 0.5])
    direction = np.array([1.0, 0.0])
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 1.0])

    # Reaction 1 never moves, so it imposes no limit however close to a bound it sits.
    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    assert chord.t_lo == pytest.approx(-1.0)
    assert chord.t_hi == pytest.approx(1.0)


def test_tiny_direction_component_still_binds() -> None:
    """BUILD_PLAN §1.6.1 — the reason the chord uses ``d != 0`` and not ``|d| > tol``.

    Reaction 1 has a direction component of 1e-12, far below any plausible "direction tolerance",
    but it sits only 1e-9 from its upper bound. It therefore caps ``t`` at 1000 — three orders of
    magnitude tighter than the other reaction's 1e+06. Dropping it as negligible would let the
    sampler take t ≈ 1e6 and push reaction 1 to 1e-6 *past* its bound.
    """
    v = np.array([0.0, 1.0 - 1e-9])
    direction = np.array([1e-6, 1e-12])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    # ~1e-9 / 1e-12 = 1000, not the 1e6 that reaction 0 alone would allow. (The gap is 1000 only to
    # within the rounding of 1 − 1e-9, hence the loose relative tolerance; the point is the three
    # orders of magnitude, not the digits.)
    assert chord.t_hi == pytest.approx(1000.0, rel=1e-6)
    assert v[1] + chord.t_hi * direction[1] <= upper[1] + 1e-15


def test_endpoints_stay_within_bounds_on_random_boxes() -> None:
    rng = np.random.default_rng(20260713)
    n = 40

    for _ in range(200):
        lower = rng.uniform(-10.0, 0.0, size=n)
        upper = lower + rng.uniform(1e-6, 20.0, size=n)
        v = rng.uniform(lower, upper)
        direction = rng.normal(size=n)
        direction[rng.random(n) < 0.3] = 0.0  # structural zeros are common in a real transform
        if not np.any(direction):
            continue

        chord = feasible_chord(v, direction, lower, upper)
        assert chord.is_samplable

        for t in (chord.t_lo, chord.t_hi):
            moved = v + t * direction
            # eps·|bound| excursions are unavoidable in float64 — the nudge removes the exactly-on-
            # the-bound case, not the rounding of the product itself.
            assert np.all(moved >= lower - 1e-12)
            assert np.all(moved <= upper + 1e-12)


def test_current_point_is_on_the_chord() -> None:
    rng = np.random.default_rng(7)
    n = 25

    for _ in range(100):
        lower = rng.uniform(-5.0, 0.0, size=n)
        upper = lower + rng.uniform(0.5, 10.0, size=n)
        v = rng.uniform(lower, upper)
        direction = rng.normal(size=n)

        chord = feasible_chord(v, direction, lower, upper)

        assert chord.t_lo <= 0.0 <= chord.t_hi


def test_point_at_a_bound_gives_a_one_sided_chord() -> None:
    v = np.array([1.0])  # sitting exactly on the upper bound
    direction = np.array([1.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])

    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    assert chord.t_lo == pytest.approx(-2.0)
    assert chord.t_hi == 0.0


def test_zero_length_chord_is_not_samplable_and_is_not_an_error() -> None:
    v = np.array([0.0])
    direction = np.array([1.0])
    lower = np.array([0.0])
    upper = np.array([0.0])  # pinned: the only feasible step is t = 0

    chord = feasible_chord(v, direction, lower, upper)

    # Not an exception, and not a signal to pick a different coordinate: the caller stays put.
    assert not chord.is_samplable


def test_zero_direction_raises() -> None:
    with pytest.raises(ZeroDirectionError):
        feasible_chord(
            np.zeros(3), np.zeros(3), np.full(3, -1.0), np.full(3, 1.0)
        )


def test_infeasible_current_point_raises() -> None:
    v = np.array([2.0])  # a full unit outside the upper bound
    direction = np.array([1.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])

    with pytest.raises(InfeasibleChordError):
        feasible_chord(v, direction, lower, upper)


def test_drift_within_tolerance_is_tolerated_and_self_correcting() -> None:
    v = np.array([1.0 + 1e-12])  # a hair outside the upper bound
    direction = np.array([1.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])

    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    # The chord excludes t = 0 rather than being clamped around it, so the next draw pulls the
    # state back inside the bound instead of widening the sampled support past it.
    assert chord.t_hi < 0.0
    assert not chord.contains(0.0)
    assert v[0] + chord.t_hi * direction[0] <= upper[0]


def test_nudge_pulls_the_endpoints_inward_by_one_ulp() -> None:
    v = np.zeros(1)
    direction = np.array([1.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])

    raw = feasible_chord(v, direction, lower, upper, nudge=False)
    nudged = feasible_chord(v, direction, lower, upper, nudge=True)

    assert nudged.t_lo == np.nextafter(raw.t_lo, raw.t_hi)
    assert nudged.t_hi == np.nextafter(raw.t_hi, raw.t_lo)
    assert nudged.t_lo > raw.t_lo
    assert nudged.t_hi < raw.t_hi
    assert nudged.contains(0.0)


def test_a_very_narrow_but_positive_chord_is_still_samplable() -> None:
    """No minimum width. A tolerance must never decide what belongs to the support.

    An earlier draft refused any chord under 1e-12, which is both an approximation (the conditional
    on a 1e-13-wide chord is perfectly well defined) and — if the caller answers by picking another
    coordinate — a *stationarity* bug, since coordinate selection would then depend on the state.
    """
    width = np.nextafter(1.0, 2.0) - 1.0  # one ULP at 1.0 ≈ 2.2e-16
    v = np.array([1.0])
    direction = np.array([1.0])
    lower = np.array([1.0])
    upper = np.array([1.0 + width])

    chord = feasible_chord(v, direction, lower, upper, nudge=False)

    assert 0.0 < chord.length < 1e-15
    assert chord.is_samplable


def test_nudging_an_adjacent_float_chord_inverts_it_and_it_reads_as_degenerate() -> None:
    # t_lo and t_hi land on *adjacent* doubles, so pulling each one ULP inward crosses them over.
    denormal = float(np.nextafter(0.0, 1.0))
    v = np.array([0.0])
    direction = np.array([1.0])
    lower = np.array([0.0])
    upper = np.array([denormal])

    raw = feasible_chord(v, direction, lower, upper, nudge=False)
    assert (raw.t_lo, raw.t_hi) == (0.0, denormal)

    chord = feasible_chord(v, direction, lower, upper, nudge=True)

    assert chord.length < 0.0  # the endpoints crossed
    assert not chord.is_samplable


def test_chord_length_and_samplability() -> None:
    assert Chord(-1.0, 1.0).length == pytest.approx(2.0)
    assert Chord(-1.0, 1.0).is_samplable
    assert Chord(0.0, 1e-13).is_samplable  # narrow, but a real interval
    assert Chord(0.0, 5e-324).is_samplable  # one denormal step is still positive width
    assert not Chord(0.0, 0.0).is_samplable  # a single point
    assert not Chord(1.0, -1.0).is_samplable  # crossed


# --- round 3 of the M2 collab review --------------------------------------------------------------


def test_an_empty_feasible_set_raises_rather_than_degenerating() -> None:
    """A *raw* crossed chord means no ``t`` satisfies every bound — not a singleton to step onto.

    ``v`` violates two opposing upper bounds at once, each by 1e-12, so both endpoints slip past the
    feasibility tolerance while the intersection itself is empty. Treating that as a degenerate
    chord returned its midpoint (0.0) and left both fluxes outside their bounds while reporting
    success. No move along this line can repair it, so it is an infeasible point and says so.
    """
    v = np.array([1e-12, 1e-12])
    direction = np.array([1.0, -1.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([0.0, 0.0])

    with pytest.raises(InfeasibleChordError, match="empty"):
        feasible_chord(v, direction, lower, upper)


def test_a_denormal_direction_component_raises_instead_of_producing_nan() -> None:
    """Every nonzero component's bound ratio overflows, so the chord is numerically unbounded.

    Left alone this reached ``rng.uniform(-inf, inf)``, which raises `OverflowError` from inside the
    sampler — or, on other paths, would hand back ``nan`` fluxes.
    """
    v = np.array([0.0])
    direction = np.array([5e-324])  # the smallest denormal: (±1 − 0)/5e-324 overflows
    lower = np.array([-1.0])
    upper = np.array([1.0])

    with pytest.raises(UnboundedChordError):
        feasible_chord(v, direction, lower, upper)
