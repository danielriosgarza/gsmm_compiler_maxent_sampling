"""The rounding transform ``T = diag(s)·B·L`` (M5, spec §17).

The central claim under test is the one that licenses the ridge to be a free parameter:
**``range(T) = range(diag(s)·B)`` for every invertible ``L``**, so no choice made here can move the
sampled distribution. Everything else is machinery around it — that the support is exact, that the
blocked reactions stay exactly zero, that ``S·T`` stays on the manifold.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import GeometryConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.line_geometry import chord_on_support, feasible_chord
from gsmm_compiler.rounding import (
    CoordinatePrecompute,
    RoundingError,
    build_transform,
)


@pytest.fixture(scope="module")
def simplex_transform(simplex_polytope: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(simplex_polytope, model_id="simplex")
    return build_transform(geometry, simplex_polytope), geometry


class TestTheTransformPreservesTheDirectionSpace:
    """The claim the whole module rests on. If it fails, every sample is from the wrong support."""

    def test_range_of_T_equals_range_of_scaled_basis(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """``L`` is invertible, so ``T`` spans exactly what ``diag(s)·B`` spans — no more, no less.

        Checked by projecting each matrix's columns onto the other's column space and demanding the
        residual vanish *both ways*. One direction alone would miss a ``T`` that spans a strict
        subspace, which is exactly what a singular ``L`` would produce.
        """
        transform, geometry = simplex_transform
        scaled_basis = geometry.basis * geometry.scaling[:, np.newaxis]
        t = transform.transform

        for source, target in ((t, scaled_basis), (scaled_basis, t)):
            basis_of_target, *_ = np.linalg.svd(target, full_matrices=False)
            residual = source - basis_of_target @ (basis_of_target.T @ source)
            assert np.abs(residual).max() < 1e-10

    def test_the_cholesky_factor_is_invertible(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        d = transform.dimension
        identity = transform.cholesky @ np.linalg.inv(transform.cholesky)
        assert np.abs(identity - np.eye(d)).max() < 1e-10

    def test_T_equals_scaling_times_basis_times_cholesky(self, simplex_transform) -> None:
        """The definition itself (spec §17.3), spelled out rather than assumed."""
        transform, geometry = simplex_transform
        expected = (geometry.basis @ transform.cholesky) * geometry.scaling[:, np.newaxis]
        assert np.abs(transform.transform - expected).max() == 0.0

    def test_a_much_larger_ridge_changes_T_but_not_its_span(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        """The ridge is a preconditioner, not a parameter of the target — the testable form."""
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        small = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-8)
        )
        large = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-1)
        )

        # Genuinely different matrices...
        assert np.abs(small.transform - large.transform).max() > 1e-3

        # ...spanning exactly the same subspace.
        basis_small, *_ = np.linalg.svd(small.transform, full_matrices=False)
        residual = large.transform - basis_small @ (basis_small.T @ large.transform)
        assert np.abs(residual).max() < 1e-10


class TestRoundTrip:
    def test_to_coordinates_inverts_to_flux(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        rng = np.random.default_rng(0)
        y = rng.normal(size=(20, transform.dimension))

        assert np.abs(transform.to_coordinates(transform.to_flux(y)) - y).max() < 1e-10

    def test_the_center_is_the_origin_of_rounded_coordinates(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        at_origin = transform.to_flux(np.zeros(transform.dimension))
        assert np.abs(at_origin - transform.center).max() == 0.0

    def test_support_coordinates_map_back_to_the_support_points(self, simplex_transform) -> None:
        """The starts are convex combinations of these; if they do not reconstruct, the starts are
        not in the polytope."""
        transform, geometry = simplex_transform
        reconstructed = transform.to_flux(transform.support_coordinates)
        assert np.abs(reconstructed - geometry.support_points).max() < 1e-9


class TestTheCovariance:
    def test_it_is_taken_about_the_support_mean_not_the_origin(self, simplex_transform) -> None:
        """The geometry's centre *is* the support mean, so ``‖q̄‖`` should be ~0 — and if it is
        not, the covariance would have picked up ``q̄·q̄ᵀ`` and inflated one direction."""
        transform, _ = simplex_transform
        assert transform.diagnostics.coordinate_mean_norm < 1e-9

    def test_a_full_rank_covariance_needs_no_escalation(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        assert transform.diagnostics.n_escalations == 0
        assert transform.diagnostics.covariance_rank == transform.dimension

    def test_the_ridge_is_relative_to_the_mean_coordinate_variance(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        diagnostics = transform.diagnostics
        expected = diagnostics.ridge * diagnostics.dimension / diagnostics.covariance_trace
        assert diagnostics.ridge_relative == pytest.approx(expected, rel=1e-12)
        assert diagnostics.ridge_relative == pytest.approx(1e-6, rel=1e-9)

    def test_the_ridged_eigenvalues_bound_the_step_scale_ratio(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        diagnostics = transform.diagnostics
        ratio = np.sqrt(diagnostics.min_eigenvalue / diagnostics.max_eigenvalue)
        assert diagnostics.step_scale_ratio == pytest.approx(ratio, rel=1e-12)
        assert 0.0 < diagnostics.step_scale_ratio <= 1.0


class TestThePrecomputeCannotLie:
    """M2 removed a caller-supplied ``support`` from `feasible_chord` because an unvalidated one
    silently reintroduces the §1.6.1 tolerance bug. The precompute is the thing that gives it back —
    so it has to be underivable from anything but ``T`` itself."""

    def test_the_support_is_the_exact_structural_support_of_each_column(
        self, simplex_transform
    ) -> None:
        transform, _ = simplex_transform
        for k in range(transform.dimension):
            expected = np.flatnonzero(transform.transform[:, k])
            assert np.array_equal(transform.precompute.support[k], expected)

    def test_the_chord_agrees_with_the_oracle_bit_for_bit(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """The hot path and M2's oracle must be the same arithmetic on the same values — not merely
        close. A one-ULP difference is a difference in which side of a bound an endpoint lands on,
        and the chord's inward ``nextafter`` is exactly one ULP wide.

        The points are convex combinations of the support vertices, so they are feasible by
        convexity — which is what a chord is defined at. Infeasible ``v`` is covered separately.
        """
        transform, _ = simplex_transform
        rng = np.random.default_rng(7)
        precompute = transform.precompute
        n_support = transform.support_coordinates.shape[0]

        for _ in range(200):
            weights = rng.dirichlet(np.full(n_support, 0.5))
            v = transform.to_flux(weights @ transform.support_coordinates)
            for k in range(transform.dimension):
                support = precompute.support[k]
                fast = chord_on_support(
                    v[support], precompute.direction[k], precompute.lower[k], precompute.upper[k]
                )
                oracle = feasible_chord(
                    v,
                    np.ascontiguousarray(transform.transform[:, k]),
                    simplex_polytope.lower_bounds,
                    simplex_polytope.upper_bounds,
                )
                assert fast.t_lo == oracle.t_lo
                assert fast.t_hi == oracle.t_hi

    def test_the_two_chords_also_agree_on_infeasible_points(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """Equivalence has to hold on the *rejections* too, or the fast path could quietly sample a
        point the oracle would have refused. Compared as outcomes: same value, or same exception."""
        transform, _ = simplex_transform
        rng = np.random.default_rng(11)
        precompute = transform.precompute

        def outcome(call):  # type: ignore[no-untyped-def]
            try:
                chord = call()
            except ValueError as error:
                return type(error)
            return chord.t_lo, chord.t_hi

        n_raised = 0
        for _ in range(300):
            y = rng.normal(scale=1.5, size=transform.dimension)  # far outside, on purpose
            v = transform.to_flux(y)
            for k in range(transform.dimension):
                support = precompute.support[k]
                fast = outcome(
                    lambda k=k, v=v, support=support: chord_on_support(
                        v[support],
                        precompute.direction[k],
                        precompute.lower[k],
                        precompute.upper[k],
                    )
                )
                oracle = outcome(
                    lambda k=k, v=v: feasible_chord(
                        v,
                        np.ascontiguousarray(transform.transform[:, k]),
                        simplex_polytope.lower_bounds,
                        simplex_polytope.upper_bounds,
                    )
                )
                assert fast == oracle
                n_raised += isinstance(fast, type)

        assert n_raised > 0, "the infeasible points were not infeasible; the test proved nothing"

    def test_a_truncated_support_is_rejected(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """The bug the validation exists to catch: drop one binding component and the chain walks
        through that bound."""
        transform, _ = simplex_transform
        precompute = transform.precompute

        truncated = CoordinatePrecompute(
            support=(precompute.support[0][:-1], *precompute.support[1:]),
            direction=(precompute.direction[0][:-1], *precompute.direction[1:]),
            lower=(precompute.lower[0][:-1], *precompute.lower[1:]),
            upper=(precompute.upper[0][:-1], *precompute.upper[1:]),
        )

        with pytest.raises(RoundingError, match="truncated"):
            truncated.validate(
                transform.transform,
                simplex_polytope.lower_bounds,
                simplex_polytope.upper_bounds,
            )

    def test_a_tampered_bound_is_rejected(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        transform, _ = simplex_transform
        precompute = transform.precompute
        wrong_upper = precompute.upper[0].copy()
        wrong_upper[0] += 1.0

        tampered = CoordinatePrecompute(
            support=precompute.support,
            direction=precompute.direction,
            lower=precompute.lower,
            upper=(wrong_upper, *precompute.upper[1:]),
        )

        with pytest.raises(RoundingError, match="upper bounds"):
            tampered.validate(
                transform.transform,
                simplex_polytope.lower_bounds,
                simplex_polytope.upper_bounds,
            )


class TestTheGuards:
    def test_the_chords_at_the_center_are_all_positive(self, simplex_transform) -> None:
        """The sampler's start condition, re-established for ``T``'s columns rather than inherited
        from M4's check on ``B``'s — they are different directions."""
        transform, _ = simplex_transform
        assert transform.diagnostics.min_chord_at_center > 0.0

    def test_S_times_T_stays_on_the_mass_balance_manifold(self, simplex_transform) -> None:
        transform, _ = simplex_transform
        assert transform.diagnostics.transform_mass_balance_error < 1e-9

    def test_the_transform_really_is_mass_balanced(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """Recomputed here from ``S`` directly, rather than trusting the reported diagnostic."""
        transform, _ = simplex_transform
        for k in range(transform.dimension):
            column = np.ascontiguousarray(transform.transform[:, k])
            assert np.abs(simplex_polytope.stoichiometry.matvec(column)).max() < 1e-9

    def test_a_singleton_polytope_yields_a_constant(
        self, singleton_polytope: ReducedPolytope
    ) -> None:
        """``d = 0``: nothing to round, and the transform must say so rather than divide by zero."""
        geometry = build_geometry(singleton_polytope, model_id="singleton")
        transform = build_transform(geometry, singleton_polytope)

        assert transform.is_singleton
        assert transform.dimension == 0
        assert transform.transform.shape == (singleton_polytope.n_free, 0)
        assert np.abs(transform.to_flux(np.zeros(0)) - transform.center).max() == 0.0

    def test_the_memory_guard_refuses_an_oversized_transform(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        config = GeometryConfig(max_geometry_memory_gb=1e-12)

        with pytest.raises(RoundingError, match="above the"):
            build_transform(geometry, simplex_polytope, config=config)


class TestTheContentKey:
    def test_the_same_geometry_gives_the_same_key(self, simplex_polytope: ReducedPolytope) -> None:
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        first = build_transform(geometry, simplex_polytope)
        second = build_transform(geometry, simplex_polytope)

        assert first.content_key() == second.content_key()

    def test_a_different_ridge_gives_a_different_key(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        """The ridge cannot change the distribution, but it *does* change ``T`` — so a run cached
        under one ridge must not be served to a run asking for another."""
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        small = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-8)
        )
        large = build_transform(
            geometry, simplex_polytope, config=GeometryConfig(ridge_relative=1e-1)
        )

        assert small.content_key() != large.content_key()


class TestTheClaimsFoundUncheckedByTheM5Review:
    """Six findings from the `/collab` adversarial review. Each of these is a property the module
    *asserted* about itself and did not actually verify."""

    def test_T_is_checked_for_full_column_rank(self, simplex_transform) -> None:
        """``range(T) = range(diag(s)·B)`` is the identity that licenses any ridge. It holds in
        exact arithmetic (``diag(s)`` invertible, ``B`` full column rank, ``L`` invertible) — and
        until the review, nothing checked float64 had delivered it. A ``T`` that quietly lost a
        column would make the chain explore a *lower-dimensional slice* of the polytope: every
        sample feasible, every chord positive, mass balance exact, and part of the support never
        visited. A missing dimension produces no bad numbers, only absent ones."""
        transform, _ = simplex_transform
        singular = np.linalg.svd(transform.transform, compute_uv=False)

        assert int(np.count_nonzero(singular > singular.max() * 1e-9)) == transform.dimension

    def test_a_rank_deficient_T_is_rejected(self, simplex_polytope: ReducedPolytope) -> None:
        """Drive the check directly: hand it a matrix with a duplicated column."""
        from gsmm_compiler.rounding import _check_full_column_rank

        rank_deficient = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        with pytest.raises(RoundingError, match="numerical rank"):
            _check_full_column_rank(rank_deficient, 2, rank_tol=1e-9)

    def test_the_transform_arrays_are_physically_read_only(self, simplex_transform) -> None:
        """``@dataclass(frozen=True)`` freezes the *binding*, not the buffer — it stops
        ``t.transform = X`` and does nothing about ``t.transform[0, 0] = X``. That gap is not
        academic: `CoordinatePrecompute` holds **copies** of ``T``'s columns, validated once at
        construction, so an in-place write to ``T`` afterwards makes the chord (from the stale
        precompute) and the flux (from the mutated ``T``) disagree — a chain sampling one polytope
        and reporting fluxes from another, with no error raised anywhere."""
        transform, _ = simplex_transform

        for array in (
            transform.transform,
            transform.inverse_transform,
            transform.cholesky,
            transform.center,
            transform.support_coordinates,
        ):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array[0] = 0.0

        for k in range(transform.dimension):
            assert not transform.precompute.support[k].flags.writeable
            assert not transform.precompute.direction[k].flags.writeable
            assert not transform.precompute.lower[k].flags.writeable
            assert not transform.precompute.upper[k].flags.writeable

    def test_the_transform_mass_balance_bar_is_unfloored(
        self, simplex_transform, simplex_polytope: ReducedPolytope
    ) -> None:
        """The floor in `NativeCSC.relative_residual` exists because a sampled *flux* carries solver
        noise at the FVA-blocked reactions. A *direction* carries none — ``T``'s blocked rows are
        exactly ``0.0`` — so flooring the transform's own check would only weaken it, and on the
        genome-scale model it weakens 2049 of 41124 (column, row) pairs. So this check divides by
        the true cancellation scale, and the report must match that, not the floored one."""
        from gsmm_compiler.rounding import _transform_mass_balance

        transform, _ = simplex_transform
        relative, absolute = _transform_mass_balance(transform.transform, simplex_polytope)

        assert relative == pytest.approx(transform.diagnostics.transform_mass_balance_error)
        assert absolute == pytest.approx(transform.diagnostics.transform_mass_balance_absolute)

        # Recomputed here from S directly, dividing by the row's own scale with no floor at all.
        worst = 0.0
        for k in range(transform.dimension):
            column = np.ascontiguousarray(transform.transform[:, k])
            residual = np.abs(simplex_polytope.stoichiometry.matvec(column))
            scale = simplex_polytope.stoichiometry.cancellation_scale(column)
            active = scale > 0.0
            worst = max(worst, float((residual[active] / scale[active]).max()))

        assert relative == pytest.approx(worst)

    def test_the_step_scale_ratio_is_never_a_nan(self, simplex_transform) -> None:
        """``eigvalsh`` can return a slightly negative eigenvalue for a PSD matrix while Cholesky
        still succeeds on the ridged one — and then ``√(λ_min/λ_max)`` is ``√(negative)``, a NaN
        entering the manifest in precisely the regime where the diagnostic was the only warning."""
        transform, _ = simplex_transform

        assert transform.diagnostics.min_eigenvalue > 0.0
        assert np.isfinite(transform.diagnostics.step_scale_ratio)
        assert np.isfinite(transform.diagnostics.condition_number)


class TestTheRoundTwoHardening:
    """Second `/collab` round: the round-one fixes were right, but two of the *checks* did not prove
    what their names implied."""

    def test_the_rank_check_rejects_an_extra_dependent_column(self) -> None:
        """Comparing the rank against the ``d`` handed in — rather than against ``T``'s own column
        count — passes an ``n × (d+1)`` matrix of rank ``d``: the extra dependent column is simply
        never counted, and the check misses the exact deficiency it exists to find."""
        from gsmm_compiler.rounding import _check_full_column_rank

        # 3 columns, rank 2 — the third is the sum of the first two.
        deficient = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

        with pytest.raises(RoundingError, match="columns"):
            _check_full_column_rank(deficient, 2, rank_tol=1e-9)  # claiming d=2 must not pass
        with pytest.raises(RoundingError, match="numerical rank"):
            _check_full_column_rank(deficient, 3, rank_tol=1e-9)

    def test_the_rank_check_validates_its_own_tolerance(self) -> None:
        """A negative tolerance makes an exactly-zero singular value count as nonzero, so the check
        would certify a singular matrix as full rank."""
        from gsmm_compiler.rounding import _check_full_column_rank

        for bad in (-1e-9, 0.0, np.nan, np.inf):
            with pytest.raises(RoundingError, match="rank_tol"):
                _check_full_column_rank(np.eye(2), 2, rank_tol=float(bad))

    def test_the_rank_check_accepts_a_zero_dimensional_transform(self) -> None:
        """The singleton polytope: no axes, so no SVD to take a max over."""
        from gsmm_compiler.rounding import _check_full_column_rank

        _check_full_column_rank(np.zeros((5, 0)), 0, rank_tol=1e-9)  # must not raise

    def test_freezing_the_transform_does_not_freeze_the_geometrys_own_center(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        """`np.ascontiguousarray` returns its argument *unchanged* when it is already contiguous and
        of the right dtype. So freezing the centre without copying would reach back through the
        alias and make `ReducedGeometry.center` read-only underneath its owner — a transform
        silently mutating the object it was built from. The `.copy()` is what prevents it."""
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        transform = build_transform(geometry, simplex_polytope)

        assert not transform.center.flags.writeable
        assert geometry.center.flags.writeable  # the geometry is untouched
        assert transform.center is not geometry.center

    def test_two_transforms_from_one_geometry_do_not_share_a_buffer(
        self, simplex_polytope: ReducedPolytope
    ) -> None:
        geometry = build_geometry(simplex_polytope, model_id="simplex")
        first = build_transform(geometry, simplex_polytope)
        second = build_transform(geometry, simplex_polytope)

        assert first.center is not second.center
        assert first.transform is not second.transform
