"""Per-model flux features — thresholds live here, never in the chain (M8)."""

from __future__ import annotations

import numpy as np

from gsmm_compiler.features import active_fraction, mean_abs_flux, mean_flux


def test_active_fraction_counts_per_reaction_over_a_threshold() -> None:
    # reaction 0 always on, reaction 1 half the time, reaction 2 always off.
    fluxes = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 0.0], [1.0, 3.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(active_fraction(fluxes, threshold=0.5), [1.0, 0.5, 0.0])


def test_active_fraction_counts_magnitude_so_a_negative_flux_is_active() -> None:
    fluxes = np.array([[-2.0], [-2.0]])
    np.testing.assert_allclose(active_fraction(fluxes, threshold=1.0), [1.0])


def test_mean_abs_and_signed_flux_differ_on_a_reversible_reaction() -> None:
    fluxes = np.array([[3.0], [-3.0]])  # cancels in the mean, not in the mean-abs
    np.testing.assert_allclose(mean_flux(fluxes), [0.0])
    np.testing.assert_allclose(mean_abs_flux(fluxes), [3.0])
