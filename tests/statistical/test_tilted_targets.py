"""The β>0 sampler against distributions known on paper (the M6 gate).

M5 established that the chain explores the right *set*. These tests are about whether it puts the
right *weight* on it — the half a uniform target cannot examine at all, because at β = 0 the whole
objective apparatus (`build_piecewise_j`, the segment log-masses, the categorical choice, the
truncated-exponential inverse CDF) is bypassed by a single `rng.uniform` call.

Three targets, chosen so that each can fail in a way the others cannot:

* **`TestTruncatedExponential`** — a 1-D chord, ``λ = 0``, so ``J = v₀`` and the law is a plain
  truncated exponential with **no breakpoints**. It isolates the exponential draw from the piecewise
  machinery: if this fails, `sample_on_segment` is wrong, and nothing else needs debugging first.
* **`TestAsymmetricTruncatedLaplace`** — a 2-D box whose ``J`` bends at ``v_r = 0``, strictly inside
  almost every chord. The load-bearing test. Its two slopes differ by 3×, so a sampler that
  symmetrized the bend, or weighted the two segments by the wrong masses, is caught — and being a
  *product* law, its coordinates' independence is checkable too.
* **`TestTiltedSimplex`** — a coupled polytope where the marginal is ``(1 − x)·e^{γx}``: one factor
  from the geometry, one from the objective. A sampler that gets the tilt right on the wrong shape
  (or the shape right with the wrong tilt) reproduces neither.

Every CDF here is derived on paper in the docstring that uses it, so the reference cannot be wrong
in the same way the sampler might be. `TestAgainstQuadrature` then adds one that shares even less:
it integrates ``exp(κ·J)`` over the polytope on a grid, evaluating ``J`` **straight from its
definition**, and never touches the piecewise representation at all — the same discipline that made
M2's gate mean something.

Fixed seeds, p-floor 1e-3 (as in M2 and M5): a pass is deterministic, and the floor leaves no room
for flakiness.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.statistical.ks import ks_pvalue

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.diagnostics import feasibility_report
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.line_distribution import L1Objective
from gsmm_compiler.maxent_sampler import BetaRung, run_ladder
from gsmm_compiler.rounding import RoundedTransform, build_transform
from gsmm_compiler.sparse_objective import (
    SparseFluxObjective,
    choose_energy_scale,
    lower_objective,
    solve_sparse_objective,
)

pytestmark = pytest.mark.slow

P_FLOOR = 1e-3

LONG = SamplerConfig(n_chains=4, n_samples=4000, burn_in=2000, thin=2, refresh_interval=500)
"""Thinned for the same reason M5's is: KS assumes independent draws and a chain does not supply
them. Thinning does not make them independent either — it makes the residual correlation small
enough not to deflate the p-value systematically, which is the honest reason to do it."""


# ---- the production path, once, so every test below exercises the same one ----------------------


class Tilted:
    """One polytope sampled at one ``β``, through the real pipeline and nothing else.

    Deliberately built from `SparseFluxObjective.from_polytope` → `lower_objective` → `run_ladder`,
    rather than by handing the sampler an `L1Objective` the test wrote out itself. A hand-built
    objective would encode the *test's* belief about which reduced index means which reaction, so a
    lowering bug would cancel against itself and the whole suite would pass on a broken package.
    """

    def __init__(
        self, full: FluxPolytope, model_id: str, lam: float, beta: float, energy_scale: float
    ) -> None:
        self.reduced: ReducedPolytope = full.reduce()
        self.objective = SparseFluxObjective.from_polytope(full, l1_penalty=lam)
        self.reduced_objective = lower_objective(self.reduced, self.objective)

        geometry = build_geometry(self.reduced, model_id=model_id)
        self.transform: RoundedTransform = build_transform(geometry, self.reduced)

        solution = solve_sparse_objective(self.reduced, self.objective)
        self.j_star = solution.optimum.j_star
        v_star = self.reduced.to_reduced(solution.optimum.v_full)

        # A *declared* s_J, so κ = β/s_J is exactly the number the analytic CDFs below are written
        # for. `warmup_range` gets its own tests; mixing the two here would leave a failing KS test
        # ambiguous between "the sampler is wrong" and "s_J is not what I assumed".
        self.energy_scale = choose_energy_scale(
            self.reduced_objective,
            geometry.support_points,
            optimum=solution.optimum,
            warmup_polytope_key=geometry.polytope_key,
            mode=energy_scale,
        )
        self.kappa = beta / energy_scale

        self.ladder = run_ladder(
            self.transform,
            self.reduced,
            config=replace(LONG, betas=(beta,)),
            model_id=model_id,
            objective=self.reduced_objective,
            energy_scale=self.energy_scale,
            optimum_coordinates=self.transform.to_coordinates(v_star),
        )

    @property
    def rung(self) -> BetaRung:
        return self.ladder.rungs[0]

    @property
    def fluxes(self) -> np.ndarray:
        """``(n_chains·n_samples, n_free)`` — every draw, pooled."""
        return np.asarray(self.rung.result.fluxes.reshape(-1, self.reduced.n_free))

    @property
    def coordinates(self) -> np.ndarray:
        return np.asarray(self.rung.result.coordinates.reshape(-1, self.transform.dimension))

    @property
    def line(self) -> L1Objective:
        return self.reduced_objective.line


# ---- 1-D: the truncated exponential, with no breakpoints at all ---------------------------------


def truncated_exponential_cdf(x: np.ndarray, kappa: float, width: float = 1.0) -> np.ndarray:
    """``F(x) = (e^{κx} − 1)/(e^{κW} − 1)`` for density ``∝ e^{κx}`` on ``[0, W]``.

    Two branches, and the reference needs them as much as the package does. The ``expm1`` form is
    exact as ``κ → 0`` (tending to the uniform ``x/W``) but **overflows to inf/inf = NaN** at
    ``κ = 1000`` — which is the large-β stress case, so the naive reference would have failed on the
    one input it exists to check. Dividing through by ``e^{κW}`` first,

        F(x) = (e^{κ(x−W)} − e^{−κW}) / (1 − e^{−κW}),

    exponentiates nothing positive; it merely cancels badly as ``κW → 0``, which is where the other
    branch is exact. (The sampler solves the same problem the same way — see `_log_phi`. That is a
    coincidence of arithmetic, not shared code: this function was derived from the CDF, not from
    it.)
    """
    tilt = kappa * width
    if tilt > 1.0:
        return np.asarray((np.exp(kappa * (x - width)) - np.exp(-tilt)) / -np.expm1(-tilt))
    return np.asarray(np.expm1(kappa * x) / np.expm1(tilt))


class TestTruncatedExponential:
    """``{v₀ = v₁ ∈ [0,1]}``, ``λ = 0`` ⇒ ``J = v₀`` ⇒ ``π ∝ e^{κ v₀}``. One segment, no bends."""

    @pytest.fixture(scope="class")
    def tilted(self, line_flux_polytope: FluxPolytope) -> Tilted:
        return Tilted(line_flux_polytope, "line", lam=0.0, beta=2.0, energy_scale=1.0)

    def test_the_marginal_matches_the_exact_truncated_exponential(self, tilted: Tilted) -> None:
        v0 = tilted.fluxes[:, 0]

        assert ks_pvalue(v0, truncated_exponential_cdf(v0, tilted.kappa)) > P_FLOOR

    def test_it_is_not_uniform(self, tilted: Tilted) -> None:
        """The control. Without it, passing the test above would prove only that the sampler stayed
        inside ``[0, 1]`` — which the β=0 sampler already did."""
        v0 = tilted.fluxes[:, 0]

        assert ks_pvalue(v0, v0) < 1e-12

    def test_the_mean_matches_the_exact_moment(self, tilted: Tilted) -> None:
        """``E[x] = 1/(1 − e^{−κ}) − 1/κ``, from ``∫₀¹ x e^{κx}dx / ∫₀¹ e^{κx}dx``."""
        kappa = tilted.kappa
        expected = 1.0 / (1.0 - np.exp(-kappa)) - 1.0 / kappa

        assert abs(float(tilted.fluxes[:, 0].mean()) - expected) < 0.01

    def test_lambda_zero_still_reports_a_real_cost(self, tilted: Tilted) -> None:
        """``C(v) = |v₁|`` even though **nothing bends ``J``**, and this is the only polytope here
        that can tell the difference.

        At λ = 0 the two penalty sets diverge: no reaction has ``λw > 0``, so `line` has no
        breakpoints — but ``w₁ = 1``, so the *cost* is still ``|v₁|``, and a run must report it.
        A `ReducedObjective` that reused one index set for both would report ``C = 0`` here, and no
        other test in this suite would notice.
        """
        assert tilted.line.penalized_indices.size == 0
        trace = tilted.rung.traces[0]

        assert np.allclose(trace.cost, np.abs(tilted.rung.result.chains[0].fluxes[:, 1]))
        assert float(trace.cost.mean()) > 0.5  # the tilt pushes v₁ = v₀ toward 1
        assert np.allclose(trace.j, trace.mu)  # J = μ − 0·C


class TestLargeBetaStress:
    """β = 1000 on the same chord: the mass sits within ``1/κ`` of the far endpoint.

    The M6 gate's large-β case. It is where a naive implementation exponentiates ``e^{1000}`` and
    returns ``inf/inf``; `_log_phi` and the reflected inverse CDF are built so that nothing positive
    is ever exponentiated, and this is the test that holds them to it.
    """

    @pytest.fixture(scope="class")
    def tilted(self, line_flux_polytope: FluxPolytope) -> Tilted:
        return Tilted(line_flux_polytope, "line-hot", lam=0.0, beta=1000.0, energy_scale=1.0)

    def test_every_draw_is_finite_and_inside_the_chord(self, tilted: Tilted) -> None:
        v0 = tilted.fluxes[:, 0]

        assert np.all(np.isfinite(v0))
        assert v0.min() >= 0.0
        assert v0.max() <= 1.0

    def test_the_marginal_still_matches_its_exact_cdf(self, tilted: Tilted) -> None:
        v0 = tilted.fluxes[:, 0]

        assert ks_pvalue(v0, truncated_exponential_cdf(v0, tilted.kappa)) > P_FLOOR

    def test_the_mean_concentrates_one_over_kappa_from_the_endpoint(
        self, tilted: Tilted
    ) -> None:
        """``E[x] → 1 − 1/κ = 0.999``. A sampler that had silently flattened the tilt would report
        0.5, and one that had clipped to the endpoint would report exactly 1.0."""
        assert abs(float(tilted.fluxes[:, 0].mean()) - (1.0 - 1.0 / 1000.0)) < 5e-4

    def test_the_chain_never_left_the_polytope(self, tilted: Tilted) -> None:
        report = feasibility_report(tilted.rung.result.fluxes, tilted.reduced)

        assert report.is_feasible


# ---- 2-D: the asymmetric truncated Laplace, with a bend inside every chord -----------------------


def asymmetric_laplace_cdf(x: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    """CDF of ``π ∝ e^{κ·g(x)}`` on ``[−1, 1]`` with ``g(x) = x − λ|x|``.

    ``g`` rises at ``a = κ(1 + λ)`` for ``x < 0`` and falls at ``b = κ(1 − λ) < 0`` for ``x ≥ 0``
    (given ``λ > 1``), so the density peaks at the bend. Integrating each piece,

        left  = ∫_{−1}^{0} e^{a x} dx = −expm1(−a)/a
        right = ∫_{0}^{1}  e^{b x} dx =  expm1(b)/b

    and ``F(X) = [∫_{−1}^{X}] / (left + right)``, taken piecewise about the bend. Derived here, on
    paper; nothing in the package was consulted.
    """
    a = kappa * (1.0 + lam)
    b = kappa * (1.0 - lam)

    left = -np.expm1(-a) / a
    right = np.expm1(b) / b
    total = left + right

    below = (np.exp(a * np.minimum(x, 0.0)) - np.exp(-a)) / a
    above = np.where(x > 0.0, left + np.expm1(b * np.maximum(x, 0.0)) / b, 0.0)
    return np.asarray(np.where(x > 0.0, above, below) / total)


def symmetric_laplace_cdf(x: np.ndarray, alpha: float) -> np.ndarray:
    """CDF of ``π ∝ e^{−α|x|}`` on ``[−1, 1]`` — the *wrong* law, kept as a control.

    ``Z = 2·(1 − e^{−α})/α``, and by symmetry ``F(x) = ½ ± (1 − e^{−α|x|}) / (2(1 − e^{−α}))``.
    """
    tail = -np.expm1(-alpha * np.abs(x)) / -np.expm1(-alpha)
    return np.asarray(0.5 + 0.5 * np.sign(x) * tail)


class TestAsymmetricTruncatedLaplace:
    """``{v₂ = v₀ + v₁, v₀,v₁ ∈ [−1,1]}`` with ``J = v₂ − 2(|v₀| + |v₁|)``.

    The M6 gate's load-bearing target. The bend at ``v_r = 0`` sits strictly inside almost every
    chord, so this is the first test in the package where `build_piecewise_j`, `log_segment_masses`
    and `choose_segment` all have to be right at once — and the 3:1 slope asymmetry means a sampler
    that weighted the two segments by the wrong masses cannot hide behind symmetry.
    """

    LAM = 2.0
    BETA = 1.5

    @pytest.fixture(scope="class")
    def tilted(self, laplace_box_flux_polytope: FluxPolytope) -> Tilted:
        return Tilted(
            laplace_box_flux_polytope, "laplace", lam=self.LAM, beta=self.BETA, energy_scale=1.0
        )

    @pytest.mark.parametrize("coordinate", [0, 1])
    def test_each_marginal_matches_the_exact_asymmetric_laplace(
        self, tilted: Tilted, coordinate: int
    ) -> None:
        x = tilted.fluxes[:, coordinate]

        assert ks_pvalue(x, asymmetric_laplace_cdf(x, tilted.kappa, self.LAM)) > P_FLOOR

    @pytest.mark.parametrize("coordinate", [0, 1])
    def test_the_marginal_is_not_the_symmetric_laplace(
        self, tilted: Tilted, coordinate: int
    ) -> None:
        """The control that gives the test above its teeth, and it is a *real* bug mode.

        Drop the biomass term ``d_b`` from the slope — leave ``J = −λΣw|v|`` — and the law becomes
        the **symmetric** Laplace ``∝ e^{−κλ|x|}``: right support, right mode, and a mean of exactly
        zero instead of −0.28. Nothing about it looks wrong. The same shape comes out of reading the
        opening slope off a midpoint that lands on the cut, where ``sgn(0) = 0`` gives the
        subgradient
        rather than either true slope (BUILD_PLAN §1.6 delta 8).

        So the asymmetric CDF above only proves something because this symmetric one is rejected.
        """
        x = tilted.fluxes[:, coordinate]

        assert ks_pvalue(x, symmetric_laplace_cdf(x, tilted.kappa * self.LAM)) < 1e-12

    def test_the_mean_leans_positive_and_matches_the_exact_moment(self, tilted: Tilted) -> None:
        """The asymmetry's signature — and it points the way the biology says it should.

        Away from the bend the density decays at ``κ(1+λ) = 4.5`` on the negative side and only at
        ``κ(λ−1) = 1.5`` on the positive side: the biomass reward ``+v`` partly pays for the L1 cost
        when the flux runs forward and *adds* to it when it runs backward. So the mass leans
        **positive**, ``E[v] = +0.20``, where a symmetric Laplace would sit at exactly 0.

        ``∫ x·e^{cx} dx = (cx − 1)e^{cx}/c²``, evaluated on each piece and divided by the total
        mass.
        """
        kappa = tilted.kappa
        a, b = kappa * (1.0 + self.LAM), kappa * (1.0 - self.LAM)

        def moment(c: float, p: float, q: float) -> float:
            return float(((c * q - 1.0) * np.exp(c * q) - (c * p - 1.0) * np.exp(c * p)) / (c * c))

        def mass(c: float, p: float, q: float) -> float:
            return float((np.exp(c * q) - np.exp(c * p)) / c)

        expected = (moment(a, -1.0, 0.0) + moment(b, 0.0, 1.0)) / (
            mass(a, -1.0, 0.0) + mass(b, 0.0, 1.0)
        )
        assert expected > 0.05, "the target itself must lean, or this test proves nothing"

        for coordinate in (0, 1):
            assert abs(float(tilted.fluxes[:, coordinate].mean()) - expected) < 0.02

    def test_the_two_coordinates_are_independent(self, tilted: Tilted) -> None:
        """``J`` separates, so ``π_β`` is a product. A sampler leaking one rounded axis into the
        other — a transform applied on one side and not the other — shows up here while both
        marginals still look perfect."""
        correlation = np.corrcoef(tilted.fluxes[:, 0], tilted.fluxes[:, 1])[0, 1]

        assert abs(correlation) < 0.05

    def test_the_equality_constraint_holds_on_every_draw(self, tilted: Tilted) -> None:
        residual = tilted.fluxes[:, 2] - (tilted.fluxes[:, 0] + tilted.fluxes[:, 1])

        assert np.abs(residual).max() < 1e-9


# ---- 2-D: a coupled tilt, where geometry and objective each supply a factor ----------------------


def tilted_simplex_cdf(x: np.ndarray, gamma: float) -> np.ndarray:
    """CDF of ``f(x) ∝ (1 − x)·e^{γx}`` on ``[0, 1]``.

    ``∫₀^X (1−x)e^{γx} dx = [(1−X)e^{γX} − 1]/γ + (e^{γX} − 1)/γ²`` — integrate by parts once, then
    check by differentiating: ``−e^{γX}/γ + (1−X)e^{γX} + e^{γX}/γ = (1−X)e^{γX}``. ✓
    """

    def primitive(t: np.ndarray | float) -> np.ndarray:
        e = np.exp(gamma * np.asarray(t, dtype=float))
        return np.asarray(((1.0 - t) * e - 1.0) / gamma + (e - 1.0) / (gamma * gamma))

    return np.asarray(primitive(x) / primitive(1.0))


class TestTiltedSimplex:
    """``{x+y+z = 1, v ≥ 0}`` with biomass ``x``: the marginal is ``(1 − x)·e^{γx}``.

    Two factors from two different places — ``(1 − x)`` is the *width of the polytope* at ``x``, and
    ``e^{γx}`` is the *objective*. M5's simplex test checked the first with the second switched off;
    M6's line test checks the second on a shape with no width to speak of. Only here do both have to
    be right simultaneously, and a sampler can be wrong about either one alone.
    """

    LAM = 1.0
    BETA = 1.0

    @pytest.fixture(scope="class")
    def tilted(self, simplex_flux_polytope: FluxPolytope) -> Tilted:
        return Tilted(
            simplex_flux_polytope, "tilted-simplex", lam=self.LAM, beta=self.BETA, energy_scale=1.0
        )

    @property
    def gamma(self) -> float:
        """``γ = κ(1 + λ)``: ``J = (1+λ)x − 2λ`` on the simplex, and the constant cancels."""
        return self.BETA * (1.0 + self.LAM)

    def test_the_biomass_marginal_matches_the_exact_cdf(self, tilted: Tilted) -> None:
        x = tilted.fluxes[:, 0]

        assert ks_pvalue(x, tilted_simplex_cdf(x, self.gamma)) > P_FLOOR

    def test_it_is_neither_the_untilted_simplex_nor_a_bare_exponential(
        self, tilted: Tilted
    ) -> None:
        """Both controls at once, and both must fail — otherwise the test above is satisfied by a
        sampler that dropped the tilt, or by one that dropped the geometry."""
        x = tilted.fluxes[:, 0]

        untilted = 1.0 - (1.0 - x) ** 2  # M5's uniform simplex marginal: the tilt is gone
        bare = truncated_exponential_cdf(x, self.gamma)  # the (1 − x) width factor is gone

        assert ks_pvalue(x, untilted) < 1e-12
        assert ks_pvalue(x, bare) < 1e-12

    def test_the_fixed_reactions_cost_reaches_the_trace(self, tilted: Tilted) -> None:
        """``SRC`` is pinned at 1.0 and *penalized*, so ``C`` carries a constant 1.0 that the
        reduced
        state cannot see. `lower_objective` must put it back, or every reported ``J`` sits a
        constant
        away from the ``J*`` it is compared with — and ``(J − J*)/s_J`` is wrong by that constant.

        Biomass is excluded from the penalty set, so ``C = (y + z) + |SRC| = (1 − x) + 1 = 2 − x``,
        and ``J = x − λ(2 − x) = (1 + λ)x − 2λ``. Only the ``+1`` is the fixed reaction's; drop it
        and ``J`` moves by ``λ`` while every marginal above still passes, the constant having
        cancelled out of the target. That is why this is checked against arithmetic and not against
        a
        distribution.
        """
        assert tilted.reduced_objective.cost_offset == pytest.approx(1.0)

        trace = tilted.rung.traces[0]
        x = tilted.rung.result.chains[0].fluxes[:, 0]

        assert np.allclose(trace.cost, 2.0 - x, atol=1e-9)
        assert np.allclose(trace.j, x - self.LAM * (2.0 - x), atol=1e-9)
        assert np.allclose(trace.j, (1.0 + self.LAM) * x - 2.0 * self.LAM, atol=1e-9)


# ---- the reference that shares the least: quadrature in the reduced coordinates ------------------


def _feasible_interval(
    v: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float] | None:
    """``{t : l ≤ v + t·d ≤ u}``, written out longhand. Returns ``None`` if empty.

    Deliberately **not** `line_geometry.feasible_chord`. This is a reference implementation, and a
    reference that calls the code under test can only confirm that code agrees with itself. It is
    also allowed to be slow and to have no opinion about ``t = 0`` — it is asked about points the
    chain never visits.
    """
    t_lo, t_hi = -np.inf, np.inf
    for i in range(v.size):
        if direction[i] > 0.0:
            t_lo = max(t_lo, (lower[i] - v[i]) / direction[i])
            t_hi = min(t_hi, (upper[i] - v[i]) / direction[i])
        elif direction[i] < 0.0:
            t_lo = max(t_lo, (upper[i] - v[i]) / direction[i])
            t_hi = min(t_hi, (lower[i] - v[i]) / direction[i])
        elif not lower[i] - 1e-12 <= v[i] <= upper[i] + 1e-12:
            return None
    return (t_lo, t_hi) if t_hi > t_lo else None


def reduced_marginal_cdf(
    tilted: Tilted, coordinate: int, n_outer: int = 1201, n_inner: int = 1201
) -> tuple[np.ndarray, np.ndarray]:
    """The marginal CDF of **reduced coordinate** ``y_k``, by 2-D quadrature. ``d == 2`` only.

    ``π_β(y) ∝ exp(κ·J(centre + T·y))`` with a *constant* Jacobian, so the marginal of ``y_k`` is
    ``∫ exp(κ·J) dy_j`` over the feasible slice — computed here on a grid, with ``J`` evaluated
    **straight from its definition** (`L1Objective.evaluate_on_line`, which is a sum of absolute
    values) rather than from the piecewise representation the sampler builds.

    That is the whole point. This reference touches no breakpoint, no slope drop, no segment mass
    and
    no inverse CDF, so a misplaced cut in `build_piecewise_j` cannot cancel out against itself here
    the way it would against a reimplementation of the same idea. It is the M2 gate's discipline —
    check the machinery against the definition, never against a restatement of the machinery —
    carried into the sampler's own coordinates.

    The exponent is shifted by a **global** ``max J`` before exponentiating, not a per-slice one: a
    per-slice shift would silently renormalize every slice to the same peak height and flatten the
    marginal into uniformity — the exact failure this is here to detect.
    """
    other = 1 - coordinate
    span = tilted.transform.support_coordinates[:, coordinate]
    pad = 0.05 * float(span.max() - span.min()) + 1e-9
    grid = np.linspace(float(span.min()) - pad, float(span.max()) + pad, n_outer)

    direction = np.ascontiguousarray(tilted.transform.transform[:, other])
    lower, upper = tilted.reduced.lower_bounds, tilted.reduced.upper_bounds

    slices: list[tuple[np.ndarray, np.ndarray] | None] = []
    peak = -np.inf
    for y_k in grid:
        y = np.zeros(tilted.transform.dimension)
        y[coordinate] = y_k
        v = np.ascontiguousarray(tilted.transform.to_flux(y))

        interval = _feasible_interval(v, direction, lower, upper)
        if interval is None:
            slices.append(None)
            continue
        t = np.linspace(interval[0], interval[1], n_inner)
        j = tilted.line.evaluate_on_line(v, direction, t)
        slices.append((t, j))
        peak = max(peak, float(j.max()))

    density = np.zeros(n_outer)
    for i, entry in enumerate(slices):
        if entry is None:
            continue
        t, j = entry
        density[i] = float(np.trapezoid(np.exp(tilted.kappa * (j - peak)), t))

    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))]
    )
    return grid, cumulative / cumulative[-1]


class TestAgainstQuadrature:
    """The M6 gate's *1-D quadrature cross-check in the reduced coordinate*.

    Every other test in this file checks a **flux** marginal against a CDF derived on paper. This
    one
    checks the sampler in the coordinates it actually walks in — where the target is a rotated,
    rescaled thing nobody derived by hand — against a numerical integral of the objective's own
    definition. A bug in the transform, in the chord, or in the piecewise reconstruction moves the
    sampled ``y`` away from that integral, and a bug in the *derivation* of any CDF above cannot
    make
    it agree.
    """

    @pytest.fixture(scope="class")
    def tilted(self, laplace_box_flux_polytope: FluxPolytope) -> Tilted:
        return Tilted(laplace_box_flux_polytope, "laplace-quad", lam=2.0, beta=1.5,
        energy_scale=1.0)

    @pytest.mark.parametrize("coordinate", [0, 1])
    def test_the_rounded_coordinate_matches_the_quadrature(
        self, tilted: Tilted, coordinate: int
    ) -> None:
        grid, cdf = reduced_marginal_cdf(tilted, coordinate)
        y = tilted.coordinates[:, coordinate]

        assert ks_pvalue(y, np.interp(y, grid, cdf)) > P_FLOOR

    def test_the_quadrature_reference_is_not_trivially_uniform(self, tilted: Tilted) -> None:
        """If the quadrature returned a near-uniform CDF, the test above would pass for a sampler
        that ignored ``β`` entirely. It does not: the tilt bends it visibly away from the
        diagonal."""
        grid, cdf = reduced_marginal_cdf(tilted, 0)
        uniform = (grid - grid[0]) / (grid[-1] - grid[0])

        assert np.abs(cdf - uniform).max() > 0.05
