"""R̂, ESS, autocorrelation and feasibility (M5).

A diagnostic that is only ever run on the sampler it is meant to judge is untested by construction:
if both are wrong in the same direction, everything looks fine. So every estimator here is driven by
a process whose answer is known in closed form — iid noise, an AR(1) with a derivable integrated
autocorrelation time, chains deliberately offset from one another — and checked against *that*.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.diagnostics import (
    DiagnosticsError,
    autocorrelation,
    convergence_report,
    effective_sample_size,
    feasibility_report,
    split_r_hat,
)
from gsmm_compiler.flux_polytope import ReducedPolytope


def ar1(rho: float, n_draws: int, n_chains: int, rng: np.random.Generator) -> np.ndarray:
    """An AR(1): ``x_t = ρ·x_{t−1} + ε``, started **in** its stationary law.

    Its integrated autocorrelation time is ``τ = (1 + ρ)/(1 − ρ)`` exactly, so it is the one process
    that can tell an ESS estimator whether it is right. Starting from the stationary distribution
    (variance ``1/(1 − ρ²)``) rather than from 0 keeps the answer τ rather than τ-plus-a-transient.
    """
    x = np.empty((n_chains, n_draws))
    x[:, 0] = rng.normal(scale=np.sqrt(1.0 / (1.0 - rho**2)), size=n_chains)
    noise = rng.normal(size=(n_chains, n_draws))
    for t in range(1, n_draws):
        x[:, t] = rho * x[:, t - 1] + noise[:, t]
    return x


class TestAutocorrelation:
    def test_an_ar1_recovers_rho_to_the_power_of_the_lag(self) -> None:
        """``ρ_t = ρ^t`` for an AR(1) — the closed form, not a restatement of the estimator."""
        rng = np.random.default_rng(0)
        chain = ar1(0.8, 200_000, 1, rng)[0]

        rho = autocorrelation(chain)
        lags = np.arange(1, 11)
        assert np.abs(rho[lags] - 0.8**lags).max() < 0.02

    def test_iid_noise_has_no_correlation_beyond_lag_zero(self) -> None:
        rng = np.random.default_rng(1)
        rho = autocorrelation(rng.normal(size=50_000))

        assert rho[0] == pytest.approx(1.0)
        assert np.abs(rho[1:50]).max() < 0.02

    def test_the_fft_matches_a_direct_sum_so_it_cannot_be_wrapping(self) -> None:
        """The FFT computes a *circular* correlation; without the zero-pad, lag ``t`` would fold the
        end of the chain onto its beginning. Checked against the definition itself — a naive O(n²)
        sum — rather than against a property of some particular process, so the reference cannot be
        wrong in the same way the implementation might be.

        Drop the pad (use ``n=n`` instead of ``n≥2n`` in the transform) and this test fails at every
        nonzero lag.
        """
        rng = np.random.default_rng(12)
        chain = rng.normal(size=257) + np.linspace(0.0, 5.0, 257)  # a trend, to expose wraparound
        n = chain.size

        centered = chain - chain.mean()
        direct = np.array(
            [float(np.dot(centered[: n - t], centered[t:])) / n for t in range(n)]
        )

        rho = autocorrelation(chain)
        assert np.abs(rho - direct / direct[0]).max() < 1e-12

    def test_a_constant_chain_reports_no_structure_rather_than_dividing_by_zero(self) -> None:
        assert np.all(autocorrelation(np.full(100, 3.0)) == 0.0)


class TestSplitRHat:
    def test_iid_chains_from_one_law_give_one(self) -> None:
        rng = np.random.default_rng(2)
        draws = rng.normal(size=(4, 4000))

        assert split_r_hat(draws) == pytest.approx(1.0, abs=0.02)

    def test_chains_stuck_in_different_places_are_caught(self) -> None:
        """The failure R̂ exists for: four chains, each converged beautifully, to four different
        answers."""
        rng = np.random.default_rng(3)
        draws = rng.normal(size=(4, 2000)) + np.array([-5.0, 0.0, 5.0, 10.0])[:, None]

        assert float(split_r_hat(draws)[0]) > 3.0

    def test_a_single_drifting_chain_is_caught_by_the_split(self) -> None:
        """Plain R̂ is *identically 1* for one chain, so it cannot see a trajectory still sliding
        toward the target. Splitting the chain in half is exactly what makes that visible — and this
        is the test that justifies the word "split"."""
        rng = np.random.default_rng(4)
        drifting = rng.normal(size=(1, 4000)) + np.linspace(0.0, 20.0, 4000)

        assert float(split_r_hat(drifting)[0]) > 2.0

    def test_it_is_computed_per_parameter(self) -> None:
        rng = np.random.default_rng(5)
        good = rng.normal(size=(4, 2000))
        bad = rng.normal(size=(4, 2000)) + np.array([-8.0, 0.0, 8.0, 16.0])[:, None]
        draws = np.stack([good, bad], axis=-1)  # (4, 2000, 2)

        r_hat = split_r_hat(draws)
        assert r_hat[0] == pytest.approx(1.0, abs=0.05)
        assert r_hat[1] > 3.0

    def test_identical_constant_chains_are_converged_not_infinite(self) -> None:
        """W = 0 and B = 0. The chain has trivially converged; a bare 0/0 would say NaN."""
        assert split_r_hat(np.full((4, 100), 2.0)) == pytest.approx(1.0)

    def test_differing_constant_chains_have_not_mixed_at_all(self) -> None:
        """W = 0 but B > 0 — four frozen chains in four different places. R̂ is infinite, and
        reporting 1.0 here (which the same 0/0 would) would be the worst possible answer."""
        draws = np.repeat(np.array([[1.0], [2.0], [3.0], [4.0]]), 100, axis=1)

        assert np.isinf(split_r_hat(draws)[0])


class TestEffectiveSampleSize:
    def test_iid_draws_have_ess_equal_to_their_count(self) -> None:
        rng = np.random.default_rng(6)
        draws = rng.normal(size=(4, 5000))

        ess = float(effective_sample_size(draws)[0])
        assert ess == pytest.approx(20_000, rel=0.1)

    @pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
    def test_an_ar1_recovers_its_analytic_integrated_time(self, rho: float) -> None:
        """``ESS = m·n·(1 − ρ)/(1 + ρ)``, derived on paper. The estimator is being checked against
        arithmetic, not against another implementation of itself."""
        rng = np.random.default_rng(7)
        n_chains, n_draws = 4, 20_000
        draws = ar1(rho, n_draws, n_chains, rng)

        expected = n_chains * n_draws * (1.0 - rho) / (1.0 + rho)
        assert float(effective_sample_size(draws)[0]) == pytest.approx(expected, rel=0.15)

    def test_chains_trapped_apart_report_a_small_ess_not_a_large_one(self) -> None:
        """The reason ESS is built on ``var⁺`` rather than each chain's own variance. Four chains
        each mixing perfectly *within* its own mode carry almost no information about a target that
        spans all four; a within-chain estimate would happily report ESS ≈ n."""
        rng = np.random.default_rng(8)
        draws = rng.normal(scale=0.1, size=(4, 4000)) + np.array([-5.0, 0.0, 5.0, 10.0])[:, None]

        assert float(effective_sample_size(draws)[0]) < 0.02 * draws.size

    def test_a_constant_parameter_has_no_effective_samples(self) -> None:
        assert float(effective_sample_size(np.full((4, 100), 1.0))[0]) == 0.0


class TestInputValidation:
    def test_too_few_draws_to_split(self) -> None:
        with pytest.raises(DiagnosticsError, match="at least 4 draws"):
            split_r_hat(np.zeros((4, 3)))

    def test_nan_draws_are_refused(self) -> None:
        draws = np.zeros((4, 100))
        draws[0, 0] = np.nan

        with pytest.raises(DiagnosticsError, match="NaN"):
            convergence_report(draws)

    def test_a_wrong_rank_is_refused(self) -> None:
        with pytest.raises(DiagnosticsError, match="must be"):
            convergence_report(np.zeros((2, 3, 4, 5)))


class TestFeasibilityReport:
    def test_the_center_of_a_simplex_is_feasible(self, simplex_polytope: ReducedPolytope) -> None:
        center = np.full(3, 1.0 / 3.0)

        report = feasibility_report(center[None, :], simplex_polytope)
        assert report.is_feasible
        assert report.n_bound_violations == 0
        assert report.max_bound_violation == 0.0
        assert report.max_mass_balance_residual < 1e-15

    def test_a_bound_violation_is_counted_not_just_measured(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        """A count of zero is a stronger statement than a small maximum, and it is the one that has
        to hold: one sample outside by 1e-12 is a bug, not a tolerance."""
        outside = np.array([[-1e-12, 0.5, 0.5]])

        report = feasibility_report(outside, simplex_polytope)
        assert not report.is_feasible
        assert report.n_bound_violations == 1
        assert report.max_bound_violation == pytest.approx(1e-12)

    def test_a_broken_mass_balance_is_caught(self, simplex_polytope: ReducedPolytope) -> None:
        off_manifold = np.array([[0.5, 0.5, 0.5]])  # sums to 1.5, not 1

        report = feasibility_report(off_manifold, simplex_polytope)
        assert report.max_mass_balance_absolute == pytest.approx(0.5)
        assert report.max_mass_balance_residual > 0.1

    def test_the_relative_residual_is_floored_so_it_cannot_divide_by_noise(self) -> None:
        """The measured M5 defect, reproduced in miniature. A metabolite touched by a single
        reaction has ``residual = |S_ij·v_j|`` and ``scale = |S_ij|·|v_j|`` — *the same number* — so
        an unfloored ratio is exactly 1.0 for any nonzero flux, however tiny. On the example model
        24 rows are like this, and one of them (``cpd02375_c0``, both its reactions FVA-blocked)
        made the sampler report a relative mass-balance error of 1.0 from an absolute one of
        3.4e-14."""
        from tests.conftest import dense_polytope

        # r0 is the only reaction touching m0, so m0 forces r0 = 0. r1/r2 keep the polytope alive.
        polytope = dense_polytope(
            stoichiometry=[[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
            lower=[-1.0, 0.0, 0.0],
            upper=[1.0, 1.0, 1.0],
        )
        noise = np.array([[3.4e-14, 0.5, 0.5]])  # r0 at solver noise, exactly the measured case

        report = feasibility_report(noise, polytope)
        assert report.max_mass_balance_absolute == pytest.approx(3.4e-14)
        # Unfloored this row would read exactly 1.0. Floored, it reads the truth.
        assert report.max_mass_balance_residual == pytest.approx(3.4e-14, rel=1e-6)

    def test_it_accepts_a_stack_of_chains(self, simplex_polytope: ReducedPolytope) -> None:
        chains = np.full((4, 10, 3), 1.0 / 3.0)

        assert feasibility_report(chains, simplex_polytope).n_samples == 40

    def test_a_wrong_width_is_refused(self, simplex_polytope: ReducedPolytope) -> None:
        with pytest.raises(DiagnosticsError, match="n_free"):
            feasibility_report(np.zeros((5, 7)), simplex_polytope)


class TestConvergenceReport:
    def test_it_summarizes_every_parameter(self) -> None:
        rng = np.random.default_rng(9)
        draws = rng.normal(size=(4, 2000, 3))

        report = convergence_report(draws)
        assert report.n_chains == 4
        assert report.n_draws == 2000
        assert report.n_parameters == 3
        assert report.max_r_hat == pytest.approx(1.0, abs=0.05)
        assert report.min_ess > 0.5 * 8000
        assert set(report.as_dict()) >= {"max_r_hat", "min_ess", "mean_ess"}


class TestTheGeyerPairingIsTheRealOne:
    """Found by the M5 `/collab` review. The initial-positive-sequence theorem is about the pairs

        Γ_m = ρ_{2m} + ρ_{2m+1},   so Γ₀ = ρ₀ + ρ₁  —  it *includes lag zero*.

    Pairing from lag 1 instead, (ρ₁+ρ₂), (ρ₃+ρ₄), …, sums to the same value **only if nothing is
    truncated** — and the truncation is the entire point of the method. It applies Geyer's stopping
    rule to a sequence his theorem says nothing about, so it can stop in the wrong place.
    """

    def test_an_antithetic_chain_is_not_reported_as_merely_iid(self) -> None:
        """The measured trigger. An AR(1) with ρ = −0.5 has ρ_t = (−0.5)ᵗ, so

            correct   Γ₀ = ρ₀ + ρ₁ = 1 − 0.5 = +0.50   → keep going
            offset    ρ₁ + ρ₂     = −0.5 + 0.25 = −0.25 → truncate immediately

        The offset pairing therefore truncates on its very first term and falls back to "ESS = N".
        The true integrated time is τ = (1+ρ)/(1−ρ) = 1/3, so the chain is *antithetic* and carries
        **three times** the information of iid draws. Reporting N is not conservative here in any
        useful sense — it is simply the wrong number, produced by the wrong stopping rule.
        """
        rng = np.random.default_rng(20)
        n_chains, n_draws = 4, 20_000
        draws = ar1(-0.5, n_draws, n_chains, rng)

        expected = n_chains * n_draws * (1.0 - -0.5) / (1.0 + -0.5)  # 3·N
        ess = float(effective_sample_size(draws)[0])

        assert ess == pytest.approx(expected, rel=0.15)
        assert ess > 2.0 * n_chains * n_draws  # the offset pairing returned exactly N here

    @pytest.mark.parametrize("rho", [-0.8, -0.5, -0.2])
    def test_negatively_correlated_chains_recover_their_analytic_time(self, rho: float) -> None:
        rng = np.random.default_rng(21)
        n_chains, n_draws = 4, 20_000
        draws = ar1(rho, n_draws, n_chains, rng)

        expected = n_chains * n_draws * (1.0 - rho) / (1.0 + rho)
        assert float(effective_sample_size(draws)[0]) == pytest.approx(expected, rel=0.2)


class TestIsFeasibleMeansBothHalvesOfThePolytope:
    """Also found by the M5 review: `is_feasible` used to test only the bound violations, so a chain
    that had walked clean off the steady-state manifold — the other half of ``P``'s definition, and
    the half the entire affine geometry exists to enforce — reported ``True``."""

    def test_a_mass_imbalanced_sample_is_not_feasible(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        # Inside every bound of [0,1]³, and nowhere near the plane x+y+z = 1.
        inside_the_box_off_the_plane = np.array([[0.9, 0.9, 0.9]])

        report = feasibility_report(inside_the_box_off_the_plane, simplex_polytope)

        assert report.n_bound_violations == 0  # the box is satisfied...
        assert report.max_mass_balance_absolute == pytest.approx(1.7)  # ...and S·v − rhs = 1.7
        assert not report.is_feasible

    def test_a_genuinely_feasible_sample_is_still_feasible(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        report = feasibility_report(np.full((1, 3), 1.0 / 3.0), simplex_polytope)

        assert report.is_feasible
