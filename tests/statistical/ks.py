"""Kolmogorov–Smirnov, by hand. This repo has no scipy (CLAUDE.md), including in its tests.

Shared by the M2 line-kernel targets and the M5 uniform-polytope targets, so that a sampler and the
1-D oracle it is built on are held to the *same* instrument, not to two implementations of it.
"""

from __future__ import annotations

import numpy as np


def kolmogorov_sf(c: float) -> float:
    """``P(√n·D > c)`` — the Kolmogorov survival function."""
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
