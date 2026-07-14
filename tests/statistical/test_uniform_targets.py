"""The β=0 sampler against distributions known on paper (M5 gate).

Every other M5 test can pass while the chain samples the wrong law: feasibility, R̂, ESS, the solve
counter and the refresh drift are all properties a *wrong* stationary distribution has too. These
are the tests that are not.

The load-bearing one is the **simplex**. Uniform on ``{x + y + z = 1, x,y,z ≥ 0}`` has marginal
density ``f(x) = 2(1 − x)`` — pointedly *not* uniform — so a sampler that quietly returned uniform
marginals, which a box cannot distinguish, fails here. The boxes then check the converse: that a law
which *is* uniform comes back uniform, on an axis-aligned polytope and on one stretched 1000:1.

Fixed seeds, and a p-value floor of 1e-3 (as in M2): a pass is deterministic, and the floor
leaves no room for flakiness.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.statistical.ks import ks_pvalue

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import GeometryConfig, SamplerConfig
from gsmm_compiler.diagnostics import convergence_report
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.maxent_sampler import run_chains
from gsmm_compiler.rounding import build_transform

pytestmark = pytest.mark.slow

P_FLOOR = 1e-3
"""Below this the sample is called inconsistent with the target. Seeds are fixed, so it never fires
by luck — it fires because the distribution is wrong."""

LONG = SamplerConfig(n_chains=4, n_samples=4000, burn_in=2000, thin=2, refresh_interval=500)
"""Thinned, because KS assumes independent draws and an MCMC chain does not deliver them. Thinning
does not make them independent either — it makes the residual correlation small enough that the KS
p-value is not systematically deflated by it, which is the honest reason to do it."""


def sample_fluxes(polytope: ReducedPolytope, model_id: str, config: SamplerConfig) -> np.ndarray:
    geometry = build_geometry(polytope, model_id=model_id)
    transform = build_transform(geometry, polytope)
    result = run_chains(transform, polytope, config=config, model_id=model_id, beta=0.0)
    return np.asarray(result.fluxes.reshape(-1, polytope.n_free))


class TestTheSimplex:
    """``{x + y + z = 1, 0 ≤ x,y,z ≤ 1}``. The uniform law here is *not* uniform in its marginals,
    which is precisely why it can catch a sampler that is."""

    @pytest.fixture(scope="class")
    def fluxes(self, simplex_polytope: ReducedPolytope) -> np.ndarray:
        return sample_fluxes(simplex_polytope, "simplex", LONG)

    @pytest.mark.parametrize("coordinate", [0, 1, 2])
    def test_each_marginal_matches_its_exact_cdf(
        self, fluxes: np.ndarray, coordinate: int
    ) -> None:
        """``F(x) = 1 − (1 − x)²``, from the length of the segment ``{y + z = 1 − x, y,z ≥ 0}``.

        Derived on paper, not from a reference implementation — so it cannot be wrong in the same
        way the sampler might be.
        """
        x = fluxes[:, coordinate]

        assert ks_pvalue(x, 1.0 - (1.0 - x) ** 2) > P_FLOOR

    def test_the_marginals_are_not_uniform(self, fluxes: np.ndarray) -> None:
        """The control for the test above. If the simplex's marginal *were* uniform, passing the KS
        test against ``1 − (1 − x)²`` would prove nothing about the sampler — so it must fail a KS
        test against the uniform CDF, and does, overwhelmingly."""
        x = fluxes[:, 0]

        assert ks_pvalue(x, x) < 1e-12

    def test_the_mean_is_one_third(self, fluxes: np.ndarray) -> None:
        """``E[x] = ∫₀¹ x·2(1−x) dx = 1/3``, and by symmetry the same for all three."""
        assert np.abs(fluxes.mean(axis=0) - 1.0 / 3.0).max() < 0.01

    def test_the_variance_matches(self, fluxes: np.ndarray) -> None:
        """``Var[x] = 1/18``. A mean can be right while the spread is badly wrong — a chain stuck
        near the centre would pass the mean test and fail this one."""
        assert np.abs(fluxes.var(axis=0) - 1.0 / 18.0).max() < 0.005

    def test_every_sample_is_on_the_simplex(self, fluxes: np.ndarray) -> None:
        assert np.abs(fluxes.sum(axis=1) - 1.0).max() < 1e-9
        assert fluxes.min() >= 0.0


class TestTheCoupledBox:
    """``{v0 = v1 ∈ [0, 2], v2 ∈ [−1, 3]}`` — uniform, behind a real equality constraint."""

    @pytest.fixture(scope="class")
    def fluxes(self, coupled_box_polytope: ReducedPolytope) -> np.ndarray:
        return sample_fluxes(coupled_box_polytope, "box", LONG)

    def test_the_first_marginal_is_uniform(self, fluxes: np.ndarray) -> None:
        v0 = fluxes[:, 0]

        assert ks_pvalue(v0, v0 / 2.0) > P_FLOOR

    def test_the_third_marginal_is_uniform(self, fluxes: np.ndarray) -> None:
        v2 = fluxes[:, 2]

        assert ks_pvalue(v2, (v2 + 1.0) / 4.0) > P_FLOOR

    def test_the_equality_constraint_holds_exactly(self, fluxes: np.ndarray) -> None:
        assert np.abs(fluxes[:, 0] - fluxes[:, 1]).max() < 1e-9

    def test_the_two_free_directions_are_independent(self, fluxes: np.ndarray) -> None:
        """Uniform on a box means the coordinates are independent. A sampler that mixed them —
        through a transform applied on one side and not the other, say — would show correlation here
        while both marginals still looked perfect."""
        correlation = np.corrcoef(fluxes[:, 0], fluxes[:, 2])[0, 1]

        assert abs(correlation) < 0.05


class TestTheAnisotropicBox:
    """The same uniform law on a polytope whose bounds span 1000:1.

    A scale stress test, and *not* a test of rounding — which is worth stating, because it is the
    obvious thing to mistake it for. M4's scaled coordinates already divide each reaction by its own
    range ``s_i = u_i − l_i``, so an **axis-aligned** stretch is absorbed before rounding is ever
    reached: measured here, the unrounded chords through the centre differ by a factor of 1.41, not
    1000. Scaling and rounding do different jobs — ``diag(s)`` fixes the axes' *units*, and ``L``
    fixes their *correlations*, which no diagonal matrix can see.

    Rounding's value shows up where the polytope is big enough to be genuinely ill-conditioned; on
    the genome-scale model it takes the shortest chord at the centre from 0.018 to 0.744, and that
    is asserted in `tests/integration/test_m5_sampler.py`, where it is true.
    """

    @pytest.fixture(scope="class")
    def fluxes(self, anisotropic_polytope: ReducedPolytope) -> np.ndarray:
        return sample_fluxes(anisotropic_polytope, "aniso", LONG)

    def test_the_long_marginal_is_still_uniform(self, fluxes: np.ndarray) -> None:
        long = fluxes[:, 0]

        assert ks_pvalue(long, long / 1000.0) > P_FLOOR

    def test_the_short_marginal_is_still_uniform(self, fluxes: np.ndarray) -> None:
        short = fluxes[:, 2]

        assert ks_pvalue(short, short) > P_FLOOR

    def test_the_scaling_already_absorbed_the_axis_aligned_stretch(
        self, anisotropic_polytope: ReducedPolytope
    ) -> None:
        """The division of labour, measured rather than asserted. If this ever starts failing, the
        geometry has stopped scaling by ``s_i`` and rounding is being asked to do two jobs."""
        from gsmm_compiler.line_geometry import feasible_chord

        geometry = build_geometry(anisotropic_polytope, model_id="aniso")
        assert np.allclose(np.sort(geometry.scaling), [1.0, 1000.0, 1000.0])

        unrounded = np.array(
            [
                feasible_chord(
                    geometry.center,
                    np.ascontiguousarray((geometry.basis * geometry.scaling[:, np.newaxis])[:, k]),
                    anisotropic_polytope.lower_bounds,
                    anisotropic_polytope.upper_bounds,
                ).length
                for k in range(geometry.dimension)
            ]
        )

        assert unrounded.max() / unrounded.min() < 2.0  # not 1000


class TestTransformInvariance:
    """The M5 gate's *transform-invariance of moments* — and the claim that lets the ridge exist.

    ``range(diag(s)·B·L) = range(diag(s)·B)`` for **any** invertible ``L``, so the sampled law does
    not depend on ``L`` at all. Two transforms whose ridges differ by seven orders of magnitude give
    genuinely different matrices; if they gave different *distributions*, the rounding would be
    silently retargeting the sampler and no feasibility check would ever notice.
    """

    def test_two_very_different_ridges_sample_the_same_law(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        config = SamplerConfig(n_chains=4, n_samples=3000, burn_in=1500, thin=2)

        small = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-8)
        )
        large = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-1)
        )
        assert np.abs(small.transform - large.transform).max() > 1e-3, "the two T are the same"

        first = run_chains(
            small, simplex_polytope, config=config, model_id="ridge-small", beta=0.0
        ).fluxes.reshape(-1, 3)
        second = run_chains(
            large, simplex_polytope, config=config, model_id="ridge-large", beta=0.0
        ).fluxes.reshape(-1, 3)

        # Both against the analytic truth, which is the only referee that cannot itself be wrong.
        for fluxes in (first, second):
            assert np.abs(fluxes.mean(axis=0) - 1.0 / 3.0).max() < 0.01
            assert np.abs(fluxes.var(axis=0) - 1.0 / 18.0).max() < 0.005

        # And against each other, more tightly than either is against the truth.
        assert np.abs(first.mean(axis=0) - second.mean(axis=0)).max() < 0.01
        assert np.abs(first.var(axis=0) - second.var(axis=0)).max() < 0.005


class TestConvergenceOnAKnownTarget:
    def test_r_hat_approaches_one_and_ess_is_substantial(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        """On a polytope this benign the chains should mix well. If they do not, the diagnostics are
        the thing that is broken, not the sampler — which is why this is asserted on a target we
        already know the sampler reproduces."""
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        transform = build_transform(geometry, simplex_polytope)
        result = run_chains(transform, simplex_polytope, config=LONG, model_id="simplex")

        report = convergence_report(result.coordinates)
        assert report.max_r_hat < 1.01
        assert report.min_ess > 0.05 * 4 * LONG.n_samples
