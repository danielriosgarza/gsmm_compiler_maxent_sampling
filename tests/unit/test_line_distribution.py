"""M2 — the exact 1D conditional.

The gate's κL grid is exercised against a 60-digit `decimal` oracle rather than against a
restatement of the same float64 formulas, so a stability regression cannot pass by agreeing with
itself. The three BUILD_PLAN §1.6 deltas each have a named regression test:

* §1.6.1 chord keeps every nonzero component → `tests/unit/test_line_geometry.py`
* §1.6.2 distinct cuts are never merged → `test_near_coincident_cuts_are_not_merged`
* §1.6.4 ``J*`` need not bound ``J`` → `test_the_draw_is_invariant_to_any_constant_offset_of_j`

plus `test_opening_slope_survives_a_razor_thin_first_segment` for the midpoint-slope trap, and
`test_a_large_objective_baseline_does_not_corrupt_the_segment_probabilities` for the cancellation
the M2 collab review caught — the one that reversed which segment the sampler favoured.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from gsmm_compiler.line_distribution import (
    SERIES_LIMIT,
    UNIFORM_LIMIT,
    DegenerateChordError,
    InvalidObjectiveError,
    L1Objective,
    _log_phi,
    build_piecewise_j,
    choose_segment,
    log_segment_masses,
    sample_line,
    sample_on_segment,
)
from gsmm_compiler.line_geometry import Chord, feasible_chord

getcontext().prec = 60

KAPPA_L_GRID = [
    0.0,
    1e-16,
    -1e-16,
    1e-12,
    -1e-12,
    1e-8,
    -1e-8,
    1.0,
    -1.0,
    100.0,
    -100.0,
    1000.0,
    -1000.0,
]
"""The regimes the M2 gate names (BUILD_PLAN §2)."""


def reference_log_phi(x: float) -> float:
    """``log((e^x − 1)/x)`` at 60 significant digits — an oracle independent of the float64 path."""
    if x == 0.0:
        return 0.0
    xd = Decimal(x)
    return float(((xd.exp() - Decimal(1)) / xd).ln())


def reference_truncated_exponential_mean(length: float, kappa: float) -> float:
    """``E[x]`` for ``x ∝ e^{κx}`` on ``[0, L]``, at 60 digits.

    The float64 closed form ``L/(1 − e^{−κL}) − 1/κ`` overflows for ``κL = −1000``; Decimal's
    exponent range does not.
    """
    k, ell = Decimal(kappa), Decimal(length)
    return float(ell / (Decimal(1) - (-k * ell).exp()) - Decimal(1) / k)


def reference_truncated_exponential_cdf(x: float, length: float, kappa: float) -> float:
    """``F(x) = (e^{κx} − 1)/(e^{κL} − 1)`` at 60 digits."""
    k, ell, xd = Decimal(kappa), Decimal(length), Decimal(x)
    return float(((k * xd).exp() - Decimal(1)) / ((k * ell).exp() - Decimal(1)))


def simpson(y: np.ndarray, dx: float) -> float:
    """Composite Simpson over an odd number of samples (no scipy in this repo)."""
    assert y.size % 2 == 1 and y.size >= 3
    return float(dx / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum()))


def objective(
    biomass_index: int | None = 0,
    penalized: list[int] | None = None,
    weights: list[float] | None = None,
    lam: float = 1.0,
) -> L1Objective:
    idx = np.asarray([] if penalized is None else penalized, dtype=np.intp)
    w = np.ones(idx.size) if weights is None else np.asarray(weights, dtype=np.float64)
    return L1Objective(biomass_index=biomass_index, penalized_indices=idx, weights=w, lam=lam)


# --- log φ: the numerical core of every segment mass ---------------------------------------------


@pytest.mark.parametrize("x", KAPPA_L_GRID)
def test_log_phi_matches_a_60_digit_oracle(x: float) -> None:
    computed = float(_log_phi(np.array([x]))[0])
    expected = reference_log_phi(x)

    assert computed == pytest.approx(expected, rel=1e-14, abs=1e-15)


def test_log_phi_is_accurate_across_a_dense_sweep() -> None:
    """Absolute error stays at the float64 floor across every regime and branch boundary.

    *Absolute*, not relative, is the criterion: ``log φ`` is one addend of a segment's log-mass, and
    `choose_segment` only ever looks at differences of log-masses. Relative error cannot be bounded
    tightly anyway — in the cancellation band just above `SERIES_LIMIT` a result of order 5e-3
    carries ~1e-12 of relative error while being absolutely accurate to 6e-15, which is what counts.

    The global worst case (~5.7e-14, near ``x ≈ 318``) is one ULP of the result itself and so
    irreducible. The sharp assertion is the small-``|x|`` one: it is what fails if `SERIES_LIMIT`
    slips back down to 1e-3, where the closed form's cancellation reaches 4.8e-14 on a result of
    order 5e-4.
    """
    magnitudes = np.concatenate(
        [
            np.logspace(-18, 3, 3000),
            [SERIES_LIMIT, np.nextafter(SERIES_LIMIT, 0.0), np.nextafter(SERIES_LIMIT, 1.0)],
        ]
    )
    xs = np.concatenate([magnitudes, -magnitudes])

    computed = _log_phi(xs)
    expected = np.array([reference_log_phi(float(x)) for x in xs])
    absolute_error = np.abs(computed - expected)

    worst = int(absolute_error.argmax())
    assert absolute_error[worst] < 1e-13, f"worst at x={xs[worst]:+.4e}"

    # Sharp where the branch choice actually decides the accuracy.
    small = np.abs(xs) <= 1.0
    worst_small = int(np.argmax(np.where(small, absolute_error, -1.0)))
    assert absolute_error[worst_small] < 1e-14, f"worst small-x at x={xs[worst_small]:+.4e}"

    # At large |x| the result is O(|x|), so the floor shows up as ~1 ULP of the result.
    assert np.all(absolute_error <= 4.0 * np.spacing(np.abs(expected)) + 1e-14)


def test_log_phi_never_overflows_for_large_positive_x() -> None:
    # expm1(710) is inf in float64; log φ must still be finite and equal to x − log(x).
    xs = np.array([500.0, 710.0, 1000.0, 1e4])

    computed = _log_phi(xs)

    assert np.all(np.isfinite(computed))
    assert computed == pytest.approx(xs - np.log(xs), rel=1e-15)


def test_log_phi_is_continuous_at_the_series_boundary() -> None:
    below = float(_log_phi(np.array([np.nextafter(SERIES_LIMIT, 0.0)]))[0])
    above = float(_log_phi(np.array([np.nextafter(SERIES_LIMIT, 1.0)]))[0])

    assert below == pytest.approx(above, rel=1e-13)


# --- the piecewise-linear objective ---------------------------------------------------------------


def test_piecewise_j_matches_direct_evaluation_on_a_dense_grid() -> None:
    rng = np.random.default_rng(2026)
    n = 12

    for _ in range(50):
        lower = rng.uniform(-5.0, -0.5, size=n)
        upper = rng.uniform(0.5, 5.0, size=n)
        v = rng.uniform(lower, upper)
        direction = rng.normal(size=n)
        obj = objective(
            biomass_index=0,
            penalized=list(range(1, n)),
            weights=list(rng.uniform(0.1, 3.0, size=n - 1)),
            lam=float(rng.uniform(0.1, 2.0)),
        )
        chord = feasible_chord(v, direction, lower, upper)

        piecewise = build_piecewise_j(v, direction, chord, obj)

        grid = np.linspace(chord.t_lo, chord.t_hi, 501)
        assert piecewise.evaluate(grid) == pytest.approx(
            obj.evaluate_on_line(v, direction, grid), rel=1e-11, abs=1e-11
        )


def test_piecewise_j_is_continuous_at_every_breakpoint() -> None:
    rng = np.random.default_rng(99)
    n = 10
    lower = np.full(n, -3.0)
    upper = np.full(n, 3.0)
    v = rng.uniform(-1.0, 1.0, size=n)
    direction = rng.normal(size=n)
    obj = objective(biomass_index=0, penalized=list(range(1, n)), lam=1.0)
    chord = feasible_chord(v, direction, lower, upper)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.n_segments > 1  # there are breakpoints to be continuous at
    for k in range(piecewise.n_segments - 1):
        knot = piecewise.knots[k + 1]
        from_left = piecewise.values[k] + piecewise.slopes[k] * (knot - piecewise.knots[k])
        from_right = piecewise.values[k + 1]
        assert from_left == pytest.approx(from_right, rel=1e-12, abs=1e-12)


def test_slopes_are_nonincreasing_so_j_is_concave() -> None:
    rng = np.random.default_rng(5)
    n = 15

    for _ in range(50):
        lower = np.full(n, -4.0)
        upper = np.full(n, 4.0)
        v = rng.uniform(-2.0, 2.0, size=n)
        direction = rng.normal(size=n)
        obj = objective(
            biomass_index=0,
            penalized=list(range(1, n)),
            weights=list(rng.uniform(0.1, 2.0, size=n - 1)),
            lam=float(rng.uniform(0.5, 3.0)),
        )
        chord = feasible_chord(v, direction, lower, upper)

        piecewise = build_piecewise_j(v, direction, chord, obj)

        assert np.all(np.diff(piecewise.slopes) <= 1e-12)


def test_line_with_no_breakpoints_is_a_single_segment() -> None:
    # The penalized reaction stays strictly positive across the whole chord: no zero crossing.
    v = np.array([0.0, 10.0])
    direction = np.array([1.0, 1.0])
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 20.0])
    obj = objective(biomass_index=0, penalized=[1], lam=0.5)
    chord = feasible_chord(v, direction, lower, upper)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.n_segments == 1
    assert piecewise.slopes[0] == pytest.approx(1.0 - 0.5 * 1.0)  # d_b − λ·w·sgn(+)·d_r


def test_exactly_coincident_cuts_are_one_bend_with_summed_drops() -> None:
    # τ = 1 for both penalized reactions, from different (v, d) pairs.
    v = np.array([0.0, 1.0, 2.0])
    direction = np.array([1.0, -1.0, -2.0])
    lower = np.array([-5.0, -5.0, -5.0])
    upper = np.array([5.0, 5.0, 5.0])
    obj = objective(biomass_index=0, penalized=[1, 2], weights=[1.0, 1.0], lam=1.0)
    chord = feasible_chord(v, direction, lower, upper)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    cuts = piecewise.knots[1:-1]
    assert cuts.size == 1
    assert cuts[0] == pytest.approx(1.0)
    # Drops add: 2·λ·w₁·|d₁| + 2·λ·w₂·|d₂| = 2·1·1·1 + 2·1·1·2 = 6.
    assert piecewise.slopes[1] - piecewise.slopes[0] == pytest.approx(-6.0)


def test_near_coincident_cuts_are_not_merged() -> None:
    """BUILD_PLAN §1.6.2 — the spec's tolerance-merge of close breakpoints changes the target.

    Two bends 1e-13 apart are two bends. Merging them would move a slope change and so alter ``J``,
    silently sampling a different distribution.
    """
    delta = 1e-13
    v = np.array([0.0, 1.0, 1.0 + delta])
    direction = np.array([1.0, -1.0, -1.0])
    lower = np.full(3, -5.0)
    upper = np.full(3, 5.0)
    obj = objective(biomass_index=0, penalized=[1, 2], weights=[1.0, 1.0], lam=1.0)
    chord = feasible_chord(v, direction, lower, upper)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    cuts = piecewise.knots[1:-1]
    assert cuts.size == 2  # NOT merged
    assert cuts[1] - cuts[0] == pytest.approx(delta, rel=1e-6)
    assert piecewise.n_segments == 3

    # Each bend drops the slope by its own 2·λ·w·|d| = 2, one at a time.
    assert np.diff(piecewise.slopes) == pytest.approx([-2.0, -2.0])

    # And the reconstruction still tracks the true J through the sliver between them.
    grid = np.linspace(chord.t_lo, chord.t_hi, 1001)
    assert piecewise.evaluate(grid) == pytest.approx(
        obj.evaluate_on_line(v, direction, grid), rel=1e-12, abs=1e-12
    )


def test_opening_slope_survives_a_razor_thin_first_segment() -> None:
    """The trap in spec §20.3 step 6: a midpoint that rounds onto the cut reads ``sgn(0) = 0``.

    ``t_lo`` here is chosen so that ``0.5·(t_lo + τ)`` rounds *up* onto ``τ`` (round-half-to-even on
    ``2·t_lo``) rather than back down onto ``t_lo``. The penalized flux at that midpoint is then
    exactly ``0.0``, so a midpoint-derived slope reads ``sgn(0) = 0`` and returns the subgradient
    1.0 instead of the true left slope 2.0 — a 2× error, in the very segment whose probability mass
    is computed from that slope. Sweeping ``t_lo``, this parity is hit in 10.5% of one-ULP first
    segments, so it is a live hazard rather than a curiosity.
    """
    t_lo = -1.1528111067677826  # a parity where the midpoint rounds up onto the cut
    tau = float(np.nextafter(t_lo, 0.0))  # one ULP inside the chord's left endpoint
    v = np.array([0.0, -tau])  # penalized flux vanishes exactly at tau
    direction = np.array([1.0, 1.0])
    obj = objective(biomass_index=0, penalized=[1], weights=[1.0], lam=1.0)
    chord = Chord(t_lo=t_lo, t_hi=1.0)

    # The trap is live, not hypothetical: the midpoint collapses onto the cut, where the flux is
    # exactly zero and the sign is undefined.
    midpoint = 0.5 * (t_lo + tau)
    assert midpoint == tau
    assert v[1] + midpoint * direction[1] == 0.0
    assert 1.0 - np.sign(v[1] + midpoint * direction[1]) * 1.0 == 1.0  # what a midpoint would say

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.knots[1] == tau
    assert piecewise.slopes[0] == pytest.approx(2.0)  # d_b − λ·w·(−1)·d_r = 1 + 1, the true slope
    assert piecewise.slopes[1] == pytest.approx(0.0)  # d_b − λ·w·(+1)·d_r = 1 − 1

    grid = np.linspace(chord.t_lo, chord.t_hi, 2001)
    assert piecewise.evaluate(grid) == pytest.approx(
        obj.evaluate_on_line(v, direction, grid), rel=1e-12, abs=1e-12
    )


def test_opening_slope_is_exact_across_thin_first_segment_parities() -> None:
    """Both round-half-to-even parities, over many ULP offsets: the left slope is always exact."""
    rng = np.random.default_rng(4242)

    for t_lo in rng.uniform(-3.0, -0.1, size=400):
        for offset in (1, 2, 3):
            tau = float(t_lo)
            for _ in range(offset):
                tau = float(np.nextafter(tau, 0.0))
            v = np.array([0.0, -tau])
            direction = np.array([1.0, 1.0])
            obj = objective(biomass_index=0, penalized=[1], weights=[1.0], lam=1.0)

            piecewise = build_piecewise_j(v, direction, Chord(float(t_lo), 1.0), obj)

            assert piecewise.slopes[0] == pytest.approx(2.0)
            assert piecewise.slopes[-1] == pytest.approx(0.0)


def test_cuts_on_the_chord_endpoints_are_not_interior_knots() -> None:
    # τ = 0 for reaction 1 would be interior, but we place the chord to start exactly there.
    v = np.array([0.0, 0.0])
    direction = np.array([1.0, 1.0])
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    chord = Chord(t_lo=0.0, t_hi=1.0)  # v[0] sits at its lower bound, so the chord opens at 0

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.n_segments == 1  # the bend sits *on* t_lo, not inside the chord
    assert piecewise.slopes[0] == pytest.approx(0.0)  # sign is +1 on (0, 1): 1 − 1·1·1


def test_a_zero_direction_component_never_creates_a_cut() -> None:
    v = np.array([0.0, 0.0])  # penalized flux sits at zero...
    direction = np.array([1.0, 0.0])  # ...but never moves
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    chord = feasible_chord(v, direction, lower, upper)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.n_segments == 1
    assert piecewise.slopes[0] == pytest.approx(1.0)  # the |0| term is flat in t


def test_degenerate_chord_is_refused() -> None:
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    with pytest.raises(DegenerateChordError):
        build_piecewise_j(np.zeros(2), np.ones(2), Chord(0.0, 0.0), obj)


# --- segment masses -------------------------------------------------------------------------------


def test_segment_probabilities_match_quadrature_of_the_directly_evaluated_j() -> None:
    """The masses, the breakpoints and the slopes, checked together against Simpson on raw ``J``.

    The quadrature never touches the piecewise representation — it exponentiates ``J`` evaluated
    from its definition — so this fails if a knot is misplaced or a slope is wrong, not only if the
    closed-form mass is.
    """
    rng = np.random.default_rng(31415)
    n = 8
    beta, energy_scale = 4.0, 2.0

    for _ in range(20):
        lower = np.full(n, -3.0)
        upper = np.full(n, 3.0)
        v = rng.uniform(-1.5, 1.5, size=n)
        direction = rng.normal(size=n)
        obj = objective(
            biomass_index=0,
            penalized=list(range(1, n)),
            weights=list(rng.uniform(0.2, 1.5, size=n - 1)),
            lam=0.8,
        )
        chord = feasible_chord(v, direction, lower, upper)
        piecewise = build_piecewise_j(v, direction, chord, obj)

        log_masses = log_segment_masses(piecewise, beta, energy_scale)
        probabilities = np.exp(log_masses - log_masses.max())
        probabilities /= probabilities.sum()

        # Reference: Simpson on exp[β·J_direct(t)/s_J], per segment, on the raw objective.
        offset = float((beta * piecewise.values / energy_scale).max())  # keeps the exp in range
        reference = np.empty(piecewise.n_segments)
        for k in range(piecewise.n_segments):
            a, b = float(piecewise.knots[k]), float(piecewise.knots[k + 1])
            grid = np.linspace(a, b, 201)
            j_direct = obj.evaluate_on_line(v, direction, grid)
            integrand = np.exp(beta * j_direct / energy_scale - offset)
            reference[k] = simpson(integrand, (b - a) / 200.0)
        reference /= reference.sum()

        assert probabilities == pytest.approx(reference, rel=1e-6, abs=1e-9)


def test_a_large_objective_baseline_does_not_corrupt_the_segment_probabilities() -> None:
    """Found by the M2 collab review — the bug that reversed which segment was favoured.

    A biomass flux of 1e16 makes ``J ≈ 1e16`` while the knot heights that decide the distribution
    are O(1). Propagating the knots as ``J(t_lo) + cumsum(...)`` rounds every one of them to the
    same 1e16 and the heights are gone — the slopes stay exactly right, and the probabilities come
    out [0.632, 0.368] against a true [0.387, 0.613]. Hence `PiecewiseLinearJ.heights`, accumulated
    from the slopes and never routed through the absolute value.
    """
    obj = objective(biomass_index=0, penalized=[1], weights=[1.0], lam=0.5)
    v = np.array([1e16, -1.0])
    direction = np.array([0.5, 1.0])
    chord = Chord(0.0, 2.0)

    piecewise = build_piecewise_j(v, direction, chord, obj)

    assert piecewise.slopes == pytest.approx([1.0, 0.0])
    assert piecewise.knots == pytest.approx([0.0, 1.0, 2.0])
    # The heights survive the baseline intact — this is the whole point. They are anchored at the
    # peak of J (knots 1 and 2, a plateau), so the rising first segment sits one unit below it.
    assert piecewise.heights == pytest.approx([-1.0, 0.0, 0.0])
    assert piecewise.baseline == pytest.approx(1e16, rel=1e-15)

    log_masses = log_segment_masses(piecewise, beta=1.0, energy_scale=1.0)
    probabilities = np.exp(log_masses - log_masses.max())
    probabilities /= probabilities.sum()

    # ∫₀¹ e^t dt = e−1 and ∫₁² e¹ dt = e.
    expected = np.array([np.e - 1.0, np.e])
    assert probabilities == pytest.approx(expected / expected.sum(), rel=1e-12)


def test_the_draw_is_invariant_to_any_constant_offset_of_j() -> None:
    """``J*`` and any baseline cancel out of ``p(t)``, so shifting ``J`` must not move the sampler.

    The offset is applied through the biomass flux, which shifts ``J`` by a constant on the whole
    chord. A 1e12 shift changes nothing — because the mass path never sees an absolute ``J``.
    """
    obj = objective(biomass_index=0, penalized=[1, 2], weights=[1.0, 1.5], lam=0.8)
    direction = np.array([0.0, 1.0, -0.7])  # d_b = 0, so the offset stays constant along the line
    chord = Chord(-1.5, 2.0)
    base = np.array([0.0, 0.3, -0.4])

    reference = None
    for offset in (0.0, 1e3, 1e8, 1e12):
        v = base + np.array([offset, 0.0, 0.0])
        piecewise = build_piecewise_j(v, direction, chord, obj)
        log_masses = log_segment_masses(piecewise, beta=3.0, energy_scale=1.0)
        probabilities = np.exp(log_masses - log_masses.max())
        probabilities /= probabilities.sum()

        draws = [
            sample_line(v, direction, chord, obj, 3.0, 1.0, np.random.default_rng(s))
            for s in range(50)
        ]
        if reference is None:
            reference = (probabilities, draws)
        else:
            assert probabilities == pytest.approx(reference[0], rel=1e-12)
            assert draws == reference[1]  # bit-identical draws, not merely the same law


def test_kappa_survives_an_overflowing_beta_times_slope() -> None:
    """β·m overflows to inf while β/s_J·m is finite — so the quotient is formed first."""
    obj = objective(biomass_index=0)  # no penalized reactions: a single segment of slope d_b
    v = np.array([0.0, 0.5])
    direction = np.array([2.0, 1.0])
    piecewise = build_piecewise_j(v, direction, Chord(-1.0, 1.0), obj)

    assert piecewise.slopes == pytest.approx([2.0])
    with np.errstate(over="ignore"):  # the naive ordering really does overflow — that is the point
        assert np.isinf(1e308 * piecewise.slopes[0])

    log_masses = log_segment_masses(piecewise, beta=1e308, energy_scale=1e308)

    assert np.all(np.isfinite(log_masses))  # κ = 2, not inf/1e308


def test_extreme_beta_does_not_overflow_the_masses() -> None:
    v = np.array([0.0, 0.5])
    direction = np.array([1.0, 1.0])
    lower = np.array([-10.0, -10.0])
    upper = np.array([10.0, 10.0])
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    chord = feasible_chord(v, direction, lower, upper)
    piecewise = build_piecewise_j(v, direction, chord, obj)

    log_masses = log_segment_masses(piecewise, beta=1e5, energy_scale=1.0)

    assert np.all(np.isfinite(log_masses))
    rng = np.random.default_rng(0)
    assert 0 <= choose_segment(log_masses, rng) < piecewise.n_segments


def test_invalid_parameters_are_rejected() -> None:
    piecewise = build_piecewise_j(
        np.zeros(2), np.array([1.0, 1.0]), Chord(-1.0, 1.0), objective(0, [1], lam=1.0)
    )
    with pytest.raises(InvalidObjectiveError):
        log_segment_masses(piecewise, beta=-1.0, energy_scale=1.0)
    with pytest.raises(InvalidObjectiveError):
        log_segment_masses(piecewise, beta=1.0, energy_scale=0.0)
    with pytest.raises(InvalidObjectiveError):
        L1Objective(
            biomass_index=0,
            penalized_indices=np.array([1], dtype=np.intp),
            weights=np.array([-1.0]),  # a negative weight would make J convex
            lam=1.0,
        )


# --- the within-segment inverse CDF ---------------------------------------------------------------


@pytest.mark.parametrize("kappa_length", KAPPA_L_GRID)
def test_sample_on_segment_stays_inside_the_segment(kappa_length: float) -> None:
    length = 2.5
    kappa = kappa_length / length
    rng = np.random.default_rng(123)

    draws = np.array([sample_on_segment(length, kappa, rng) for _ in range(2000)])

    assert np.all(draws >= 0.0)
    assert np.all(draws <= length)


@pytest.mark.parametrize("kappa_length", [-1000.0, -100.0, -1.0, 1.0, 100.0, 1000.0])
def test_sample_on_segment_mean_matches_the_analytic_truncated_exponential(
    kappa_length: float,
) -> None:
    length = 3.0
    kappa = kappa_length / length
    rng = np.random.default_rng(777)

    draws = np.array([sample_on_segment(length, kappa, rng) for _ in range(200_000)])

    expected = reference_truncated_exponential_mean(length, kappa)
    standard_error = draws.std() / np.sqrt(draws.size)
    assert abs(draws.mean() - expected) < 5.0 * standard_error + 1e-12


class _FixedUniform:
    """An RNG stub returning a prescribed ``U``, to probe the inversion pointwise."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


@pytest.mark.parametrize("kappa_length", [-1000.0, -100.0, -1.0, -1e-8, 1e-8, 1.0, 100.0, 1000.0])
def test_sample_on_segment_inverts_the_cdf_exactly(kappa_length: float) -> None:
    """``F(x(U))`` recovers ``U`` — the inversion is exact, not merely distributionally close.

    For ``κ > 0`` the draw is reflected off the far endpoint (see `sample_on_segment`), so it
    inverts to ``1 − U`` there. Both are exact inversions of the same law; asserting *increasing*
    monotonicity in ``U`` would be asserting an orientation the sampler never promised.
    """
    length = 4.0
    kappa = kappa_length / length
    reflected = kappa > 0.0

    for u in (1e-9, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-9):
        x = sample_on_segment(length, kappa, _FixedUniform(u))  # type: ignore[arg-type]

        assert 0.0 <= x <= length
        recovered = reference_truncated_exponential_cdf(x, length, kappa)
        assert recovered == pytest.approx(1.0 - u if reflected else u, rel=1e-9, abs=1e-9)


def test_zero_slope_segment_is_uniform() -> None:
    rng = np.random.default_rng(3)
    draws = np.array([sample_on_segment(2.0, 0.0, rng) for _ in range(50_000)])

    assert draws.mean() == pytest.approx(1.0, abs=0.02)
    assert draws.var() == pytest.approx(4.0 / 12.0, rel=0.05)


def test_beta_zero_short_circuits_to_a_uniform_draw() -> None:
    v = np.zeros(2)
    direction = np.array([1.0, 1.0])
    chord = Chord(-1.0, 2.0)
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)

    draws = np.array(
        [
            sample_line(v, direction, chord, obj, 0.0, 1.0, np.random.default_rng(s))
            for s in range(20_000)
        ]
    )

    assert np.all((draws >= chord.t_lo) & (draws <= chord.t_hi))
    assert draws.mean() == pytest.approx(0.5, abs=0.02)  # midpoint of [-1, 2]


def test_sample_line_never_leaves_the_chord() -> None:
    rng = np.random.default_rng(2718)
    n = 10

    for _ in range(300):
        lower = np.full(n, -3.0)
        upper = np.full(n, 3.0)
        v = rng.uniform(-1.0, 1.0, size=n)
        direction = rng.normal(size=n)
        obj = objective(
            biomass_index=0,
            penalized=list(range(1, n)),
            weights=list(rng.uniform(0.1, 2.0, size=n - 1)),
            lam=1.0,
        )
        chord = feasible_chord(v, direction, lower, upper)
        beta = float(rng.choice([0.0, 0.5, 5.0, 500.0]))

        t = sample_line(v, direction, chord, obj, beta, 1.0, rng)

        assert chord.contains(t)
        moved = v + t * direction
        assert np.all(moved >= lower - 1e-9) and np.all(moved <= upper + 1e-9)


def test_sample_line_is_reproducible_for_a_fixed_seed() -> None:
    v = np.array([0.0, 0.3, -0.2])
    direction = np.array([1.0, -0.5, 0.7])
    chord = Chord(-1.0, 1.0)
    obj = objective(biomass_index=0, penalized=[1, 2], lam=1.0)

    first = [
        sample_line(v, direction, chord, obj, 2.0, 1.0, np.random.default_rng(42))
        for _ in range(5)
    ]
    second = [
        sample_line(v, direction, chord, obj, 2.0, 1.0, np.random.default_rng(42))
        for _ in range(5)
    ]

    assert first == second


# --- degenerate chords: self-loop, never a redraw -------------------------------------------------


def test_a_chord_with_no_width_returns_a_self_loop() -> None:
    """M2 collab review — the exact Gibbs update on a single-point feasible set is to stay put.

    Spec §19 says to "reject and redraw a direction" here. Redrawing a *different coordinate* makes
    the coordinate-selection law depend on the current state, which breaks the random-scan Gibbs
    stationarity argument (the mixture weights can no longer be pulled out of the integral). The
    conditional on a degenerate chord is the point mass at t = 0, so returning 0.0 is both exact and
    stationarity-preserving.
    """
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    v = np.array([0.0, 0.5])
    direction = np.array([1.0, 1.0])

    for chord in (Chord(0.0, 0.0), Chord(1e-16, -1e-16)):  # zero-width and crossed
        assert not chord.is_samplable
        for beta in (0.0, 5.0):
            t = sample_line(v, direction, chord, obj, beta, 1.0, np.random.default_rng(0))
            assert t == 0.0


def test_a_narrow_positive_chord_is_sampled_rather_than_refused() -> None:
    """No minimum width: a 1e-13-wide chord has a well-defined conditional and gets drawn from."""
    obj = objective(biomass_index=0, penalized=[1], lam=1.0)
    v = np.array([0.0, 0.5])
    direction = np.array([1.0, 1.0])
    chord = Chord(-5e-14, 5e-14)

    rng = np.random.default_rng(3)
    draws = np.array([sample_line(v, direction, chord, obj, 4.0, 1.0, rng) for _ in range(500)])

    assert np.all((draws >= chord.t_lo) & (draws <= chord.t_hi))
    assert draws.std() > 0.0  # genuinely sampled, not collapsed onto a point


# --- round 2 of the M2 collab review --------------------------------------------------------------


def test_a_large_height_excursion_inside_the_chord_does_not_erase_later_increments() -> None:
    """Round 2 — anchoring the heights at ``t_lo`` fixed the baseline but not this.

    ``J`` climbs by 1e16 over the first segment and then by 0.5 over the second. A ``t_lo``-anchored
    ``cumsum`` stores both later knots as the same float and the 0.5 is gone — worth a factor of
    ``e^{0.5}`` in the weights — so all three segments came out equally likely. Anchoring at the
    peak of ``J`` and accumulating outward keeps the increments that carry the mass exact.
    """
    obj = objective(biomass_index=0, penalized=[1, 2], weights=[0.375, 0.125], lam=1.0)
    v = np.array([0.0, -1e16, -(1e16 + 2.0)])
    direction = np.array([0.5, 1.0, 1.0])
    chord = Chord(0.0, 1e16 + 4.0)

    piecewise = build_piecewise_j(v, direction, chord, obj)
    log_masses = log_segment_masses(piecewise, beta=1.0, energy_scale=1.0)
    probabilities = np.exp(log_masses - log_masses.max())
    probabilities /= probabilities.sum()

    assert piecewise.slopes == pytest.approx([1.0, 0.25, 0.0])
    expected = np.array([1.0, 4.0 * (np.exp(0.5) - 1.0), 2.0 * np.exp(0.5)])
    assert probabilities == pytest.approx(expected / expected.sum(), rel=1e-9)


def test_heights_are_anchored_at_the_peak_and_never_positive() -> None:
    rng = np.random.default_rng(606)
    n = 12

    for _ in range(50):
        lower, upper = np.full(n, -3.0), np.full(n, 3.0)
        v = rng.uniform(-2.0, 2.0, size=n)
        direction = rng.normal(size=n)
        obj = objective(
            biomass_index=0,
            penalized=list(range(1, n)),
            weights=list(rng.uniform(0.1, 2.0, size=n - 1)),
            lam=float(rng.uniform(0.2, 2.0)),
        )
        chord = feasible_chord(v, direction, lower, upper)

        piecewise = build_piecewise_j(v, direction, chord, obj)

        assert piecewise.heights.max() == 0.0
        assert np.all(piecewise.heights <= 0.0)

        # The anchor really is the maximizer of J on the chord. A concave piecewise-linear function
        # attains its maximum at a knot, so the knots are the exact reference; a dense grid can only
        # under-report it, and must never exceed it.
        at_knots = obj.evaluate_on_line(v, direction, piecewise.knots)
        assert piecewise.baseline == pytest.approx(float(at_knots.max()), rel=1e-9, abs=1e-9)
        grid = np.linspace(chord.t_lo, chord.t_hi, 4001)
        assert float(obj.evaluate_on_line(v, direction, grid).max()) <= piecewise.baseline + 1e-9


@pytest.mark.parametrize(
    "kappa",
    [
        np.finfo(np.float64).tiny,  # the value the MIN_NORMAL threshold made catastrophic
        float(np.nextafter(np.finfo(np.float64).tiny, 1.0)),
        1e-307,
        1e-100,
        -1e-100,
        1e-17,
        -1e-17,
        UNIFORM_LIMIT,
        -UNIFORM_LIMIT,
        1e-15,
        -1e-15,
    ],
)
def test_a_negligible_tilt_still_draws_uniformly(kappa: float) -> None:
    """Round 2 — a normal ``A = −expm1(−|κ|L)`` does not make ``u·A`` normal.

    With the threshold pushed down to the smallest normal double, ``κ = 2.2e-308`` sent the exact
    inversion quantiles whose ``u·A`` underflows, costing the draw its low-order bits. This asserts
    the **law**, which is what has to be right: at a tilt this weak the density is bitwise flat, so
    the draw must be uniform however the inversion is arranged internally.

    Asserting the law rather than ``x(u)`` is deliberate, and it is also what keeps the test honest.
    The pointwise story is a red herring twice over: the exact branch inverts from the far end for
    ``κ > 0`` (`sample_on_segment`) while the uniform branch does not, so the two disagree on which
    uniform quantile a given ``u`` maps to while agreeing on the law — and the underflow above turns
    out to cost ≤ 1 ULP, on a set of ``u`` of probability ~1e-15. The threshold moved because the
    exact inversion should not be spending precision on a tilt float64 cannot represent, not because
    the old one was corrupting the distribution. It was not.
    """
    length = 2.0
    rng = np.random.default_rng(9090)

    draws = np.array([sample_on_segment(length, kappa, rng) for _ in range(20_000)])

    assert np.all((draws >= 0.0) & (draws <= length))
    assert draws.mean() == pytest.approx(length / 2.0, abs=0.03)
    assert draws.var() == pytest.approx(length**2 / 12.0, rel=0.05)
    # The collapse this guards against would pile every draw onto one endpoint.
    assert 0.4 < float(np.mean(draws < length / 2.0)) < 0.6


def test_an_off_origin_singleton_chord_moves_to_its_feasible_point() -> None:
    """Round 2 — the self-loop is exact only when the singleton sits at ``t = 0``.

    A state that has drifted a hair outside a bound (within `feasibility_tol`, so the chord is
    accepted) collapses to a singleton at some ``t ≠ 0``. Returning ``0.0`` there would leave the
    state infeasible *forever*: every later visit to this coordinate would make the same non-move.
    The exact conditional is the point mass at the one feasible ``t``, so we go there.
    """
    obj = objective(biomass_index=0)
    v = np.array([1e-12])  # 1e-12 outside a bound pinned at zero
    direction = np.array([1.0])
    chord = feasible_chord(v, direction, np.array([0.0]), np.array([0.0]))

    assert not chord.is_samplable
    assert chord.t_lo == chord.t_hi == pytest.approx(-1e-12)

    t = sample_line(v, direction, chord, obj, 1.0, 1.0, np.random.default_rng(0))

    assert t == pytest.approx(-1e-12)  # not 0.0
    assert v[0] + t * direction[0] == pytest.approx(0.0, abs=1e-18)  # feasibility restored


def test_a_true_self_loop_still_returns_zero() -> None:
    obj = objective(biomass_index=0)
    v = np.array([0.0])
    direction = np.array([1.0])
    chord = feasible_chord(v, direction, np.array([0.0]), np.array([0.0]))

    assert not chord.is_samplable
    assert sample_line(v, direction, chord, obj, 3.0, 1.0, np.random.default_rng(0)) == 0.0


def test_slope_drops_survive_an_overflowing_lambda() -> None:
    """``2·λ`` overflows for λ=1e308 even though the physical drop ``2·λ·w·|d|`` is ~2."""
    obj = objective(biomass_index=0, penalized=[1], weights=[1e-308], lam=1e308)
    v = np.array([0.0, 0.5])
    direction = np.array([1.0, -1.0])

    piecewise = build_piecewise_j(v, direction, Chord(-1.0, 1.0), obj)

    assert np.all(np.isfinite(piecewise.slopes))
    assert piecewise.slopes[0] - piecewise.slopes[1] == pytest.approx(2.0, rel=1e-6)


def test_a_negative_tilt_just_under_eps_over_two_is_not_bitwise_flat() -> None:
    """Round 3 — the ``eps/4`` threshold, and why ``eps/2`` was the wrong constant.

    Float spacing is asymmetric about 1.0: ``eps`` above, ``eps/2`` below. So the rounding midpoint
    on the low side is ``1 − eps/4``, and a negative tilt between ``−eps/2`` and ``−eps/4`` rounds
    ``exp(x)`` down to ``nextafter(1.0, 0)`` rather than to ``1.0``. An ``eps/2`` limit would hold
    the bitwise-flat claim for ``κ > 0`` and quietly break it for ``κ < 0``.
    """
    eps = float(np.finfo(np.float64).eps)
    assert eps / 4.0 == UNIFORM_LIMIT

    just_under_half = -float(np.nextafter(eps / 2.0, 0.0))
    assert np.exp(just_under_half) != 1.0  # NOT flat — an eps/2 limit would have called it flat
    assert abs(just_under_half) >= UNIFORM_LIMIT  # ...and our limit correctly sends it to inversion

    # At the limit itself, both signs really are bitwise flat.
    for sign in (1.0, -1.0):
        x = sign * float(np.nextafter(UNIFORM_LIMIT, 0.0))
        assert np.exp(x) == 1.0
        assert 1.0 + x == 1.0


def test_an_underflowing_inverse_temperature_is_loud_not_silent() -> None:
    """Round 3 — ``β/s_J`` can underflow to zero while the true ``κ`` is perfectly ordinary.

    ``β = 2e-200``, ``s_J = 1e124``, slope 1e308 has a true ``κ = β·m/s_J ≈ 2e-16``, but the grouped
    ``β/s_J`` underflows first and ``κ`` silently becomes 0 — a flat segment where a tilted one was
    meant. Every other pathology in this module raises; this one would not have, so it is made to.
    """
    obj = objective(biomass_index=0)
    v = np.array([0.0, 0.0])
    direction = np.array([1e308, 0.0])
    piecewise = build_piecewise_j(v, direction, Chord(0.0, 1.0), obj)

    assert piecewise.slopes == pytest.approx([1e308])
    assert 2e-200 / 1e124 == 0.0  # the underflow is real

    with pytest.raises(InvalidObjectiveError, match="underflow"):
        log_segment_masses(piecewise, beta=2e-200, energy_scale=1e124)

    with pytest.raises(InvalidObjectiveError, match="overflow"):
        log_segment_masses(piecewise, beta=1e308, energy_scale=1e-100)


# --- M6: a FIXED biomass reaction, which the reduced polytope cannot hold as a variable -----------


class TestBiomassCanBeFixed:
    """``biomass_index = None`` — biomass is an ``l == u`` reaction, so ``μ(v)`` is a constant.

    This is not a hypothetical shape. `ReducedPolytope.biomass_index` has always been ``int |
    None``,
    and `sparse_objective.biomass_maximum` has always had a branch for it: a model whose biomass is
    pinned (a chemostat at a fixed growth rate, say) eliminates it along with every other fixed
    reaction, and the sampler's ``v`` then has no biomass component at all.

    The dangerous non-fix is to point `biomass_index` at *some* in-range reaction instead. ``J``
    would then reward that reaction's flux as though it were growth, the chain would tilt toward it,
    and nothing — not feasibility, not mass balance, not R̂ — would say a word. So the ``None`` is
    load-bearing, and these tests pin what it must mean: a constant contributes **zero slope**, and
    a
    constant cancels out of ``p(t)``.
    """

    def test_a_fixed_biomass_contributes_no_slope(self) -> None:
        v = np.array([0.5, 0.25])
        direction = np.array([1.0, 1.0])
        chord = Chord(t_lo=-0.2, t_hi=0.2)

        fixed = objective(biomass_index=None, penalized=[1], weights=[1.0], lam=0.5)
        free = objective(biomass_index=0, penalized=[1], weights=[1.0], lam=0.5)

        # The only difference is d_b = direction[0] = 1.0, so every slope must differ by exactly 1.
        assert np.allclose(
            build_piecewise_j(v, direction, chord, free).slopes
            - build_piecewise_j(v, direction, chord, fixed).slopes,
            1.0,
        )

    def test_it_evaluates_to_the_penalty_alone(self) -> None:
        v = np.array([7.0, -2.0])
        fixed = objective(biomass_index=None, penalized=[1], weights=[3.0], lam=0.5)

        assert fixed.evaluate(v) == pytest.approx(-0.5 * 3.0 * 2.0)
        assert fixed.biomass_slope(np.array([1.0, 1.0])) == 0.0

    def test_evaluate_on_line_agrees_with_the_piecewise_reconstruction(self) -> None:
        """The same reference the M2 gate uses, re-run on the branch M2 never had.

        `evaluate_on_line` computes ``J`` straight from the definition; `build_piecewise_j` rebuilds
        it from breakpoints and slope drops. A ``None`` handled in one and forgotten in the other
        would show up here as a constant offset — which is exactly the failure that *cannot* be seen
        in the sampled distribution, because a constant cancels.
        """
        v = np.array([1.0, -0.4, 0.3])
        direction = np.array([0.5, 1.0, -1.0])
        chord = Chord(t_lo=-0.9, t_hi=0.9)
        fixed = objective(biomass_index=None, penalized=[1, 2], weights=[1.0, 2.0], lam=0.75)

        piecewise = build_piecewise_j(v, direction, chord, fixed)
        grid = np.linspace(chord.t_lo, chord.t_hi, 401)

        assert np.allclose(piecewise.evaluate(grid), fixed.evaluate_on_line(v, direction, grid))

    def test_the_draw_is_a_valid_sample_of_the_penalty_only_target(self) -> None:
        """``J = −λw|v₁|`` on the line: a symmetric Laplace about the crossing, sampled exactly."""
        v = np.array([9.0, 0.0])
        direction = np.array([0.0, 1.0])
        chord = Chord(t_lo=-1.0, t_hi=1.0)
        fixed = objective(biomass_index=None, penalized=[1], weights=[1.0], lam=1.0)
        rng = np.random.default_rng(11)

        draws = np.array(
            [sample_line(v, direction, chord, fixed, 2.0, 1.0, rng) for _ in range(20000)]
        )

        assert np.all(np.abs(draws) <= 1.0)
        assert abs(float(draws.mean())) < 0.02  # symmetric about the bend
        # E|t| for density ∝ e^{−2|t|} on [−1,1]: ∫₀¹ t e^{−2t}dt / ∫₀¹ e^{−2t}dt
        grid = np.linspace(0.0, 1.0, 200001)
        density = np.exp(-2.0 * grid)
        expected = float(np.trapezoid(grid * density, grid) / np.trapezoid(density, grid))
        assert abs(float(np.abs(draws).mean()) - expected) < 0.01

    def test_a_negative_biomass_index_is_still_refused(self) -> None:
        with pytest.raises(InvalidObjectiveError, match="nonnegative or None"):
            objective(biomass_index=-1)
