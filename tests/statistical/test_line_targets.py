"""M2 — does the line kernel actually draw from the distribution it claims?

The unit tests check the pieces (chord, breakpoints, slopes, masses, inversion). These check the
composition: draw many ``t`` and compare the empirical law against the target.

Every test uses a **fixed seed** and a Kolmogorov–Smirnov p-value floor of 1e-3, so a pass is
deterministic rather than lucky. Two kinds of reference are used, and neither is the code under test
restated:

* **exact analytic CDFs**, where the target has one (uniform at β=0; the truncated Laplace);
* **a fine-grid quadrature of the directly evaluated ``J``**, where it does not. That reference
  exponentiates the raw objective and integrates it — it never touches `build_piecewise_j` — so a
  misplaced breakpoint or a wrong slope surfaces as a distribution mismatch instead of cancelling
  out against itself.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from gsmm_compiler.line_distribution import L1Objective, sample_line
from gsmm_compiler.line_geometry import Chord

getcontext().prec = 60

KS_ALPHA = 1e-3
"""With fixed seeds a pass is deterministic; this floor leaves no room for flakiness."""

N_DRAWS = 20_000


def kolmogorov_sf(c: float) -> float:
    """``P(√n·D > c)`` — the Kolmogorov survival function (this repo has no scipy)."""
    k = np.arange(1, 101)
    return float(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * c**2)))


def ks_pvalue(sample: np.ndarray, cdf_at_sample: np.ndarray) -> float:
    """Two-sided one-sample KS p-value. ``cdf_at_sample[i]`` is ``F(sample[i])``, unsorted."""
    n = sample.size
    theoretical = np.sort(cdf_at_sample)
    d = float(
        max(
            np.max(np.arange(1, n + 1) / n - theoretical),
            np.max(theoretical - np.arange(0, n) / n),
        )
    )
    return kolmogorov_sf(np.sqrt(n) * d)


def draw(
    v: np.ndarray,
    direction: np.ndarray,
    chord: Chord,
    objective: L1Objective,
    beta: float,
    energy_scale: float,
    seed: int,
    n: int = N_DRAWS,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            sample_line(v, direction, chord, objective, beta, energy_scale, rng)
            for _ in range(n)
        ]
    )


def quadrature_cdf(
    v: np.ndarray,
    direction: np.ndarray,
    chord: Chord,
    objective: L1Objective,
    beta: float,
    energy_scale: float,
    points: int = 200_001,
) -> tuple[np.ndarray, np.ndarray]:
    """The target CDF on a fine grid, built from ``J`` evaluated straight from its definition."""
    grid = np.linspace(chord.t_lo, chord.t_hi, points)
    exponent = beta * objective.evaluate_on_line(v, direction, grid) / energy_scale
    density = np.exp(exponent - exponent.max())  # the shift cancels in the normalization

    mass = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))))
    return grid, mass / mass[-1]


def penalized(indices: list[int], weights: list[float], lam: float) -> L1Objective:
    return L1Objective(
        biomass_index=0,
        penalized_indices=np.asarray(indices, dtype=np.intp),
        weights=np.asarray(weights, dtype=np.float64),
        lam=lam,
    )


# --- β = 0: the uniform target --------------------------------------------------------------------


def test_beta_zero_is_uniform_on_the_chord() -> None:
    v = np.array([0.2, -0.4, 0.7])
    direction = np.array([1.0, -0.6, 0.3])
    chord = Chord(-1.3, 2.1)
    objective = penalized([1, 2], [1.0, 2.0], lam=1.0)

    draws = draw(v, direction, chord, objective, beta=0.0, energy_scale=1.0, seed=101)

    assert ks_pvalue(draws, (draws - chord.t_lo) / chord.length) > KS_ALPHA
    assert draws.mean() == pytest.approx(0.5 * (chord.t_lo + chord.t_hi), abs=0.02)
    assert draws.var() == pytest.approx(chord.length**2 / 12.0, rel=0.05)


# --- a single segment: the truncated exponential e^{κt} -------------------------------------------


@pytest.mark.parametrize("kappa_length", [-1000.0, -100.0, -1.0, -1e-8, 1e-8, 1.0, 100.0, 1000.0])
def test_single_segment_reproduces_a_truncated_exponential(kappa_length: float) -> None:
    """No penalized reaction crosses zero, so ``J`` is linear and the target is ``e^{κt}``.

    The gate's whole κL ladder, both signs. ``β ≥ 0`` always, so a negative tilt is produced by a
    negative *slope* (drop the biomass component and let the L1 penalty set it), never by a
    negative β.
    """
    length = 4.0
    chord = Chord(-1.0, -1.0 + length)

    # Reaction 1 holds v₁ + t·d₁ = 10 + t ∈ [9, 13] > 0 across the chord: linear J, no breakpoint.
    v = np.array([0.0, 10.0])
    biomass_component = 1.0 if kappa_length > 0 else 0.0
    direction = np.array([biomass_component, 1.0])
    objective = penalized([1], [1.0], lam=0.5)

    # J(t) = d_b·t − 0.5·(10 + t)  ⇒  slope = d_b − 0.5 = +0.5 or −0.5.
    slope = biomass_component - 0.5
    energy_scale = 1.0
    beta = abs(kappa_length) / (length * abs(slope))
    assert np.isclose(beta * slope / energy_scale * length, kappa_length)

    draws = draw(v, direction, chord, objective, beta, energy_scale, seed=202)

    grid, reference = quadrature_cdf(v, direction, chord, objective, beta, energy_scale)
    assert ks_pvalue(draws, np.interp(draws, grid, reference)) > KS_ALPHA


@pytest.mark.parametrize("kappa_length", [-100.0, -1.0, 1.0, 100.0])
def test_single_segment_mean_matches_the_analytic_moment(kappa_length: float) -> None:
    """The first moment against its closed form, at 60 digits (float64 overflows at κL = −1000)."""
    length = 4.0
    chord = Chord(0.0, length)
    v = np.array([0.0, 10.0])
    biomass_component = 1.0 if kappa_length > 0 else 0.0
    direction = np.array([biomass_component, 1.0])
    objective = penalized([1], [1.0], lam=0.5)

    slope = biomass_component - 0.5
    beta = abs(kappa_length) / (length * abs(slope))
    kappa = beta * slope

    draws = draw(v, direction, chord, objective, beta, 1.0, seed=606)

    # E[x] for x ∝ e^{κx} on [0, L]:  L/(1 − e^{−κL}) − 1/κ.
    k, ell = Decimal(kappa), Decimal(length)
    expected = float(ell / (Decimal(1) - (-k * ell).exp()) - Decimal(1) / k)

    standard_error = draws.std() / np.sqrt(draws.size)
    assert abs(draws.mean() - expected) < 5.0 * standard_error


# --- one breakpoint: the truncated Laplace e^{−α|t|} ----------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 1.0, 5.0, 50.0])
def test_symmetric_kink_reproduces_a_truncated_laplace(alpha: float) -> None:
    """``d_b = 0`` and a penalized flux crossing zero mid-chord ⇒ ``J(t) = −λ|t|``.

    The target is a Laplace truncated to ``[−L, L]``, symmetric about the kink — the two-segment
    case the piecewise machinery exists for, with the mass split between the arms. Its CDF is exact,
    so nothing approximate stands between the sampler and the assertion.
    """
    length = 3.0
    chord = Chord(-length, length)
    v = np.array([0.0, 0.0])  # the penalized flux vanishes exactly at t = 0
    direction = np.array([0.0, 1.0])  # d_b = 0 ⇒ biomass adds no slope
    objective = penalized([1], [1.0], lam=1.0)

    # J(t) = −|t| ⇒ density ∝ exp(−β|t|/s_J) ≡ e^{−α|t|}.
    draws = draw(v, direction, chord, objective, beta=alpha, energy_scale=1.0, seed=303)

    # Exact truncated-Laplace CDF on [−L, L], normalizer 2(1 − e^{−αL})/α.
    half_mass = (1.0 - np.exp(-alpha * np.abs(draws))) / alpha
    total = 2.0 * (1.0 - np.exp(-alpha * length)) / alpha
    cdf = np.where(draws < 0.0, total / 2.0 - half_mass, total / 2.0 + half_mass) / total

    assert ks_pvalue(draws, cdf) > KS_ALPHA
    assert draws.mean() == pytest.approx(0.0, abs=6.0 * draws.std() / np.sqrt(draws.size))
    # Symmetric law: the two arms must carry equal mass.
    assert float(np.mean(draws < 0.0)) == pytest.approx(0.5, abs=0.02)


# --- many breakpoints: the general piecewise-exponential target -----------------------------------


@pytest.mark.parametrize("beta", [0.5, 2.0, 20.0, 200.0])
def test_many_breakpoints_match_a_quadrature_of_the_raw_objective(beta: float) -> None:
    """The full kernel against an independent integration of ``exp(β·J/s_J)``, across β.

    Eight penalized reactions crossing zero at eight distinct points of the chord: the segment
    ladder, its log-masses and the within-segment inversion all have to be right *together*, and the
    reference knows nothing about any of them.
    """
    rng = np.random.default_rng(9)
    n = 9
    v = np.concatenate(([0.0], rng.uniform(-1.0, 1.0, size=n - 1)))
    direction = np.concatenate(([1.0], rng.normal(size=n - 1)))
    chord = Chord(-2.0, 2.5)
    objective = penalized(
        list(range(1, n)), list(rng.uniform(0.3, 2.0, size=n - 1)), lam=0.7
    )
    energy_scale = 3.0

    piecewise_cuts = np.sum(
        [
            chord.t_lo < -v[r] / direction[r] < chord.t_hi
            for r in range(1, n)
            if direction[r] != 0.0
        ]
    )
    assert piecewise_cuts >= 5, "the fixture must actually exercise multiple segments"

    draws = draw(v, direction, chord, objective, beta, energy_scale, seed=404)

    grid, reference = quadrature_cdf(v, direction, chord, objective, beta, energy_scale)
    assert ks_pvalue(draws, np.interp(draws, grid, reference)) > KS_ALPHA


def test_large_beta_concentrates_on_the_objective_maximizer() -> None:
    """As ``β`` grows the draw collapses onto ``argmax J`` — the maxent ladder's endpoint."""
    rng = np.random.default_rng(9)
    n = 9
    v = np.concatenate(([0.0], rng.uniform(-1.0, 1.0, size=n - 1)))
    direction = np.concatenate(([1.0], rng.normal(size=n - 1)))
    chord = Chord(-2.0, 2.5)
    objective = penalized(
        list(range(1, n)), list(rng.uniform(0.3, 2.0, size=n - 1)), lam=0.7
    )

    grid = np.linspace(chord.t_lo, chord.t_hi, 100_001)
    t_star = float(grid[np.argmax(objective.evaluate_on_line(v, direction, grid))])

    spreads = [
        float(
            np.mean(
                np.abs(draw(v, direction, chord, objective, beta, 1.0, 505, n=4000) - t_star)
            )
        )
        for beta in (1.0, 10.0, 100.0, 1000.0)
    ]

    assert spreads == sorted(spreads, reverse=True), f"not concentrating with β: {spreads}"
    assert spreads[-1] < 0.02  # at β=1000 the draws sit essentially on the maximizer
