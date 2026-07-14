"""Convergence and validity diagnostics: R̂, ESS, autocorrelation, feasibility.

Nothing in this package can prove a finite chain has converged. What these functions do is make the
*failure* legible — and the failure mode that matters here is specific. A rounded polytope with a
`rounding.RoundingDiagnostics.step_scale_ratio` of 1e-3 has an axis the chain crawls along, and a
chain too short to have crossed it produces beautiful histograms, a bound violation of exactly zero,
and an R̂ of 1.00 — because every chain is stuck in the same place for the same reason. **R̂ is only
as informative as the starts are dispersed**, which is why `maxent_sampler.dispersed_start` draws
from a sub-1 Dirichlet over the support vertices rather than jittering around the centre.

Two choices worth stating:

* **Split-R̂, not plain R̂.** Splitting each chain in half turns a *drifting* chain — the classic
  under-burned trajectory — into two halves with different means. Plain R̂ cannot see that, because
  it only compares chains to each other: a single slowly-drifting chain has R̂ = 1 by construction.

* **ESS by Geyer's initial monotone positive sequence.** The naive ``1 + 2Σρ_t``, truncated "when
  ρ_t first goes negative", is not an estimator of anything: the noise in the tail of ρ̂ is the same
  size as the signal there, so summing it adds variance without adding information. Geyer's pairing
  (``P_t = ρ_{2t} + ρ_{2t+1}``, provably positive and decreasing for a reversible chain) is what
  gives the truncation point a reason to exist.

Implemented in **M5**; the objective/geometry/solver JSON reports are **M8**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.flux_polytope import ReducedPolytope

VALUE_DTYPE = np.float64

DEFAULT_MASS_BALANCE_TOL = 1e-9
"""Relative mass-balance residual above which a sample is *not* in the polytope.

Relative to a floored ``|S|·|v|`` (`NativeCSC.relative_residual`), and matched to the geometry's
``span_tol``: below this the residual is the arithmetic that measured it, not a real excursion.
"""


class DiagnosticsError(ValueError):
    """The draws handed in are not a shape a diagnostic can be computed from."""


@dataclass(frozen=True)
class FeasibilityReport:
    """Does every stored sample actually lie in the polytope? (spec §22)"""

    n_samples: int

    max_bound_violation: float
    """``max(l − v, v − u, 0)`` over every sample and reaction. Exactly 0 is the expectation: the
    chord's inward ``nextafter`` buys it, and nothing else in the loop can push a component out."""

    max_mass_balance_residual: float
    """``‖S·v − rhs‖_∞`` **relative** to a floored ``|S|·|v|`` — `NativeCSC.relative_residual`.

    Relative because an absolute bar charges the sampler for the float64 rounding of evaluating a
    sum of terms of size ~1e5 (the M4 lesson). Floored because on this model 24 metabolite rows are
    touched by a single free reaction, where an unfloored ratio is identically 1 and measures
    nothing at all."""

    max_mass_balance_absolute: float

    n_bound_violations: int
    """How many (sample, reaction) pairs are outside, at any magnitude. A count of zero is a
    stronger claim than a small maximum, and it is the one that should hold."""

    mass_balance_tol: float = DEFAULT_MASS_BALANCE_TOL
    """The bar `is_feasible` holds the relative residual to."""

    @property
    def is_feasible(self) -> bool:
        """Feasible means **both** halves of the polytope's definition, not just the box.

        ``P = {v : S·v = rhs, l ≤ v ≤ u}``. An earlier version of this property tested only the
        bound violations, so a chain that had walked off the steady-state manifold entirely — the
        *other* half of the definition, and the one the whole affine geometry exists to enforce —
        would report ``is_feasible = True`` with an arbitrarily large mass-balance residual. Nothing
        else in the suite asks the question, so nothing else would have caught it.
        """
        return (
            self.n_bound_violations == 0
            and self.max_mass_balance_residual <= self.mass_balance_tol
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "max_bound_violation": self.max_bound_violation,
            "max_mass_balance_residual": self.max_mass_balance_residual,
            "max_mass_balance_absolute": self.max_mass_balance_absolute,
            "mass_balance_tol": self.mass_balance_tol,
            "n_bound_violations": self.n_bound_violations,
            "is_feasible": self.is_feasible,
        }


@dataclass(frozen=True)
class ConvergenceReport:
    """Split-R̂ and ESS, per parameter and summarized (spec §22)."""

    n_chains: int
    n_draws: int
    n_parameters: int

    r_hat: NDArray[np.float64]
    """``(n_parameters,)`` — split-R̂."""

    ess: NDArray[np.float64]
    """``(n_parameters,)`` — effective sample size, pooled across chains."""

    @property
    def max_r_hat(self) -> float:
        # NOT `np.max(..., initial=1.0)`. For a reduction, `initial` is a *candidate* — it would
        # floor the answer at 1.0 and hide every R̂ below it. The mirror of that mistake in
        # `min_ess` reported an ESS of 0 for a sample whose every entry was 8000.
        return float(self.r_hat.max()) if self.r_hat.size else 1.0

    @property
    def min_ess(self) -> float:
        return float(self.ess.min()) if self.ess.size else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "n_parameters": self.n_parameters,
            "max_r_hat": self.max_r_hat,
            "min_ess": self.min_ess,
            "mean_ess": float(np.mean(self.ess)) if self.ess.size else 0.0,
        }


def _as_draws(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coerce to ``(n_chains, n_draws, n_parameters)`` and refuse anything degenerate."""
    array = np.asarray(draws, dtype=VALUE_DTYPE)
    if array.ndim == 2:  # (n_chains, n_draws) — a single parameter
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise DiagnosticsError(
            "draws must be (n_chains, n_draws) or (n_chains, n_draws, n_parameters), got shape "
            f"{array.shape}"
        )
    if array.shape[1] < 4:
        raise DiagnosticsError(
            f"split-R̂ halves each chain, so it needs at least 4 draws per chain, got "
            f"{array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise DiagnosticsError("draws contain NaN or inf")
    return array


def _autocovariance(chain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Biased (``1/n``) autocovariance by FFT — the estimator Geyer's argument is stated for.

    Zero-padded past ``2n`` before the transform, because the FFT computes a *circular* correlation:
    without the pad, lag ``t`` silently wraps the end of the chain onto its beginning, and a chain
    with any trend at all then reports correlations that are an artifact of the wrap rather than a
    property of the process.
    """
    x = np.asarray(chain, dtype=VALUE_DTYPE)
    n = x.size
    centered = x - x.mean()
    size = int(2 ** np.ceil(np.log2(2 * n)))  # ≥ 2n, so no wraparound
    spectrum = np.fft.rfft(centered, n=size)
    power = spectrum * np.conjugate(spectrum)
    return np.asarray(np.fft.irfft(power, n=size)[:n] / n, dtype=VALUE_DTYPE)


def autocorrelation(chain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized autocorrelation of one 1-D chain at lags ``0 … n−1``."""
    x = np.asarray(chain, dtype=VALUE_DTYPE)
    if x.ndim != 1:
        raise DiagnosticsError(f"autocorrelation takes one chain at a time, got shape {x.shape}")
    if x.size < 2:
        raise DiagnosticsError("autocorrelation needs at least 2 draws")

    acov = _autocovariance(x)
    if acov[0] <= 0.0:  # a constant chain: no variance, so no correlation structure to report
        return np.zeros(x.size, dtype=VALUE_DTYPE)
    return np.asarray(acov / acov[0], dtype=VALUE_DTYPE)


def split_r_hat(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Split-R̂ per parameter (Gelman–Rubin, with each chain halved).

    Halving is what catches a *single drifting chain*. Plain R̂ compares chains to one another and is
    identically 1 for one chain; split-R̂ compares the first half of each chain to its second half,
    so a trajectory still sliding toward the bulk of the target shows up as between-half variance
    whether or not there is another chain to compare it with.
    """
    array = _as_draws(draws)
    _, n_draws, n_parameters = array.shape

    half = n_draws // 2
    if half < 2:
        raise DiagnosticsError(f"each half needs at least 2 draws, got {half}")
    # First `half` and LAST `half`. With an odd n_draws that drops the middle draw rather than
    # letting the halves overlap on it, which would correlate them and bias R̂ toward 1.
    split = np.concatenate([array[:, :half, :], array[:, n_draws - half :, :]], axis=0)

    chain_means = split.mean(axis=1)  # (2·n_chains, p)
    chain_vars = split.var(axis=1, ddof=1)

    within = chain_vars.mean(axis=0)  # W
    between = half * chain_means.var(axis=0, ddof=1)  # B
    var_plus = ((half - 1) * within + between) / half

    # W = 0 means every split-chain is constant. If they are all the *same* constant the chain has
    # converged trivially and R̂ = 1; if they differ, they have not mixed at all and R̂ is infinite.
    # A bare 0/0 would report NaN for both, so the two cases are separated explicitly.
    r_hat = np.ones(n_parameters, dtype=VALUE_DTYPE)
    moving = within > 0.0
    r_hat[moving] = np.sqrt(var_plus[moving] / within[moving])
    r_hat[~moving & (between > 0.0)] = np.inf
    return r_hat


def effective_sample_size(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """ESS per parameter, by Geyer's initial monotone positive sequence.

    The autocorrelation is built from a **multi-chain** variance estimate,

        ρ_t = 1 − (W − mean_c ĉ_{c,t}) / var⁺

    with ``ĉ_{c,t}`` chain ``c``'s autocovariance at lag ``t``, ``W`` the within-chain variance and
    ``var⁺`` the overdispersed estimate R̂ also uses. Using ``var⁺`` rather than each chain's own
    variance is what makes the estimate *conservative* when the chains disagree: a chain trapped in
    one mode has a small within-chain autocovariance, and would otherwise report a large ESS for a
    sample that carries almost no information about the target.
    """
    array = _as_draws(draws)
    n_chains, n_draws, n_parameters = array.shape

    ess = np.empty(n_parameters, dtype=VALUE_DTYPE)
    for p in range(n_parameters):
        ess[p] = _ess_one(array[:, :, p], n_chains, n_draws)
    return ess


def _ess_one(chains: NDArray[np.float64], n_chains: int, n_draws: int) -> float:
    """ESS for one parameter, given its ``(n_chains, n_draws)`` draws."""
    total = float(n_chains * n_draws)

    within = float(chains.var(axis=1, ddof=1).mean())
    if within <= 0.0:
        return 0.0  # every chain is constant: the sample says nothing about the target's spread

    between = n_draws * float(chains.mean(axis=1).var(ddof=1)) if n_chains > 1 else 0.0
    var_plus = ((n_draws - 1) * within + between) / n_draws
    if var_plus <= 0.0:
        return 0.0

    acov = np.mean([_autocovariance(chains[c]) for c in range(n_chains)], axis=0)
    rho = 1.0 - (within - acov) / var_plus
    rho[0] = 1.0

    # Geyer's pairs are Γ_m = ρ_{2m} + ρ_{2m+1}, so the FIRST one is ``ρ₀ + ρ₁`` — it includes lag
    # zero. That grouping is the whole theorem: Γ_m is provably positive and decreasing for a
    # reversible chain, which is what licenses truncating at the first nonpositive Γ.
    #
    # Pairing from lag 1 instead — (ρ₁+ρ₂), (ρ₃+ρ₄), … — sums to the same thing *only if nothing is
    # truncated*, and the truncation is the entire point. It applies Geyer's stopping rule to a
    # sequence Geyer's theorem says nothing about. Measured: an antithetic chain with ρ_t = (−0.5)ᵗ
    # has Γ₀ = 1 + ρ₁ = +0.50 (keep), while ρ₁ + ρ₂ = −0.25 (stop at once) — so the offset pairing
    # truncates immediately and reports ESS = N for a chain whose true ESS is 3N.
    n_pairs = n_draws // 2
    if n_pairs < 1:
        return total
    gamma = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)

    nonpositive = np.flatnonzero(gamma <= 0.0)
    keep = int(nonpositive[0]) if nonpositive.size else n_pairs
    if keep == 0:
        # Γ₀ = 1 + ρ₁ ≤ 0 needs ρ₁ ≤ −1, which no autocorrelation can be. Reachable only through a
        # degenerate variance estimate, and there is nothing to integrate if it is.
        return total

    kept = np.minimum.accumulate(gamma[:keep])  # the initial *monotone* sequence
    tau = -1.0 + 2.0 * float(kept.sum())

    if tau <= 0.0:
        # Antithetic past the point of an interpretable integrated time: the sample is *better* than
        # iid. There is no honest way to claim more draws than we actually took.
        return total
    return total / tau


def posterior_variance(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """``var⁺`` per parameter — the **overdispersed** variance estimate R̂ and ESS are both built on.

    ``var⁺ = ((n−1)·W + B)/n``: the within-chain variance, plus the between-chain variance the
    chains' *disagreement* contributes. It is not the same thing as the variance of the pooled
    draws,
    and the difference is exactly the point. Two chains stuck at ``−a`` and ``+a`` have a pooled
    variance of ``a²`` and a ``var⁺`` of ``2a²`` — so ``var⁺`` says the spread is larger *because
    the
    chains do not agree*, which is the honest thing for an estimate that has not converged to say.
    """
    array = _as_draws(draws)
    _, n_draws, n_parameters = array.shape

    within = array.var(axis=1, ddof=1).mean(axis=0)  # (p,)
    between = (
        n_draws * array.mean(axis=1).var(axis=0, ddof=1)
        if array.shape[0] > 1
        else np.zeros(n_parameters, dtype=VALUE_DTYPE)
    )
    return np.asarray(((n_draws - 1) * within + between) / n_draws, dtype=VALUE_DTYPE)


def mcse(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Monte-Carlo standard error of the mean, per parameter: ``√(var⁺ / ESS)``.

    **The variance must be the one ESS was built from.** `effective_sample_size` estimates the
    autocorrelation against ``var⁺`` precisely so that a chain trapped in one mode cannot claim a
    large ESS; pairing that ESS with the *pooled sample* variance in the numerator throws half of
    that conservatism away. Measured on two chains trapped at ``±a``: ``var⁺ = 2a²`` while the
    pooled
    variance is ``a²``, so the naive ``sd/√ESS`` under-reports the error by ``√2`` — and it does so
    **exactly when the chains disagree**, which is when an honest error bar matters most.

    ESS = 0 (every chain constant, but at different values) means the sample carries no information
    about the target's spread; the error is then reported as infinite rather than as zero, because
    "we cannot tell" is not the same claim as "we know it exactly".
    """
    array = _as_draws(draws)
    variance = posterior_variance(array)
    ess = effective_sample_size(array)

    error = np.full(variance.size, np.inf, dtype=VALUE_DTYPE)
    exact = variance <= 0.0  # a genuinely constant statistic: the mean has no MC error at all
    error[exact] = 0.0

    usable = ~exact & (ess > 0.0)
    error[usable] = np.sqrt(variance[usable] / ess[usable])
    return error


def convergence_report(draws: NDArray[np.float64]) -> ConvergenceReport:
    """Split-R̂ and ESS for every parameter of a ``(n_chains, n_draws, n_parameters)`` array."""
    array = _as_draws(draws)
    n_chains, n_draws, n_parameters = array.shape

    return ConvergenceReport(
        n_chains=n_chains,
        n_draws=n_draws,
        n_parameters=n_parameters,
        r_hat=split_r_hat(array),
        ess=effective_sample_size(array),
    )


def feasibility_report(
    fluxes: NDArray[np.float64],
    reduced: ReducedPolytope,
    *,
    mass_balance_tol: float = DEFAULT_MASS_BALANCE_TOL,
) -> FeasibilityReport:
    """Check every stored sample against the polytope it is supposed to have come from.

    Takes **reduced** fluxes, ``(n_samples, n_free)`` or ``(n_chains, n_samples, n_free)``. This is
    the check that no amount of clean-looking convergence can substitute for: a chain that has left
    the polytope converges beautifully to the wrong thing.
    """
    v = np.asarray(fluxes, dtype=VALUE_DTYPE)
    if v.ndim == 3:
        v = v.reshape(-1, v.shape[-1])
    if v.ndim != 2 or v.shape[1] != reduced.n_free:
        raise DiagnosticsError(
            f"fluxes must have trailing dimension n_free={reduced.n_free}, got shape {v.shape}"
        )

    below = reduced.lower_bounds - v
    above = v - reduced.upper_bounds
    max_violation = max(float(np.max(below, initial=0.0)), float(np.max(above, initial=0.0)))
    n_violations = int(np.count_nonzero(below > 0.0) + np.count_nonzero(above > 0.0))

    worst_relative = 0.0
    worst_absolute = 0.0
    for sample in v:
        contiguous = np.ascontiguousarray(sample)
        residual = np.abs(reduced.stoichiometry.matvec(contiguous) - reduced.rhs)
        relative = reduced.stoichiometry.relative_residual(contiguous, reduced.rhs)
        worst_absolute = max(worst_absolute, float(residual.max(initial=0.0)))
        worst_relative = max(worst_relative, float(relative.max(initial=0.0)))

    return FeasibilityReport(
        n_samples=int(v.shape[0]),
        max_bound_violation=max_violation,
        max_mass_balance_residual=worst_relative,
        max_mass_balance_absolute=worst_absolute,
        n_bound_violations=n_violations,
        mass_balance_tol=mass_balance_tol,
    )
