"""M4 acceptance gate — the affine geometry of the genome-scale model.

The gate (BUILD_PLAN M4): known dimensions recovered · a truncated basis rejected ·
``‖S·diag(s)·B‖ ≈ 0`` · the scale-sensitive narrow case classified right · the dim-0 singleton path.

The load-bearing test is `test_dimension_matches_an_independent_fva_oracle`. `build_geometry` finds
``d`` by probing the polytope with support LPs; the oracle finds it by an entirely separate route —
flux variability analysis to remove the reactions that cannot move, then a dense rank of what is
left. The two share no code beyond the LP adapter, so agreement is evidence rather than tautology.
It also matters that the two *disagree* with the naive count: ``n_free − rank(S) = 55`` is only an
upper bound, because 61 of the 260 free reactions turn out to be pinned by mass balance despite
``l < u`` in the model file. The true sampling dimension is **46**.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import (
    OrthonormalBasis,
    blocked_reactions,
    build_geometry,
    direction_space,
    sweep_complement,
)
from gsmm_compiler.config import GeometryConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import total_solve_count
from gsmm_compiler.line_geometry import feasible_chord
from gsmm_compiler.model_input import CanonicalModel
from gsmm_compiler.sparse_objective import build_flux_lp

pytestmark = pytest.mark.slow

MEMORY = 1 << 30


@pytest.fixture(scope="module")
def reduced(example_canonical: CanonicalModel) -> ReducedPolytope:
    return example_canonical.polytope.reduce()


@pytest.fixture(scope="module")
def geometry(reduced: ReducedPolytope):  # type: ignore[no-untyped-def]
    return build_geometry(reduced, model_id="bifido")


def fva_dimension_oracle(reduced: ReducedPolytope) -> tuple[int, int]:
    """The affine dimension, by a route that shares no code with `affine_geometry`.

    A reaction whose FVA range is a point cannot appear in any feasible direction, so drop it; the
    dimension of what remains is ``n_columns − rank``, computed densely by NumPy. Returns
    ``(dimension, n_blocked)``.
    """
    program = build_flux_lp(reduced)
    n = reduced.n_free
    cost = np.zeros(n)
    spans = np.empty(n)
    for i in range(n):
        cost[i] = 1.0
        high = float(program.maximize(cost).primal[i])
        low = float(program.maximize(-cost).primal[i])
        cost[i] = 0.0
        spans[i] = high - low

    moving = np.flatnonzero(spans > 1e-9)
    dense = reduced.stoichiometry.to_dense()[:, moving]
    return int(moving.size - np.linalg.matrix_rank(dense)), int(n - moving.size)


# ---- the dimension -------------------------------------------------------------------------------


def test_dimension_matches_an_independent_fva_oracle(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    dimension, n_blocked = fva_dimension_oracle(reduced)

    assert geometry.dimension == dimension == 46
    assert geometry.diagnostics.n_blocked == n_blocked == 61

    # The naive null-space count is *not* the answer: bounds flatten the polytope inside it.
    naive = reduced.n_free - np.linalg.matrix_rank(reduced.stoichiometry.to_dense())
    assert naive == 55
    assert geometry.dimension < naive


def test_the_blocked_split_is_not_a_judgement_call(geometry) -> None:  # type: ignore[no-untyped-def]
    """The widest blocked range is 8e-10 and the narrowest moving one is 0.30 — eight orders apart.

    So no tolerance in that gap changes the dimension. (The blocked ranges are *weak-duality upper
    bounds*, not primal readings, which is why they sit at 8e-10 rather than the 2e-12 the LP
    reports — the bound is the honest number, and the split survives it with room to spare.) If this
    ratio ever collapsed on some other strain, `blocked_reactions` would refuse rather than pick a
    dimension by tolerance.
    """
    assert geometry.diagnostics.blocked_separation > 1e7


# ---- the certificate -----------------------------------------------------------------------------


def test_the_span_certificate_is_exhaustive(geometry) -> None:  # type: ignore[no-untyped-def]
    certificate = geometry.certificate
    assert certificate.exhaustive
    assert certificate.complement_is_complete
    assert certificate.n_inconclusive == 0
    assert certificate.n_probes == certificate.n_complement == 260 - 46

    # `max_width` is a weak-duality UPPER bound, carrying an outward rounding allowance so float64
    # cannot round it below the truth — not the ~1e-13 the arithmetic appears to produce.
    assert certificate.max_width < 1e-11
    assert certificate.max_width < certificate.max_width_floor

    # And this is what the certificate actually licenses: √k·√(1+ε)·max_width + leakage·diameter.
    # No feasible direction orthogonal to B is wider than this — in the *exact* polytope, since weak
    # duality assumes nothing about the returned point, not even that it is feasible.
    assert certificate.resolution < 1e-9  # comfortably under the width that would be a dimension
    assert certificate.resolution > certificate.max_width  # the √k factor is not free
    assert certificate.leakage < 1e-12


def test_a_noise_swamped_probe_no_longer_refuses_a_certified_geometry(reduced) -> None:  # type: ignore[no-untyped-def]
    """M11.2 (§1.6.10, §1.6.11): the span gate is `resolution ≤ span_tol`, not `n_inconclusive==0`.

    `model_id="strain_1"` is the tracker's own #1 open defect: on this **unchanged** model it
    raised "the span certificate is not exhaustive (…, 2 inconclusive)" for ten milestones, while
    `strain_2` certified — and `model_id` varies nothing but the RNG stream that orders the solves.
    The audit measured why: an inconclusive probe reports that its *primal* discovery signal was
    noise-swamped, which is evidence about the solver's warm-start path, not about the span (46/46
    constructed truncations were detected on *conclusive* probes; the two signals are disjoint by
    ~5e4×). The certificate's rigorous `resolution` is 2.7e-11 here — 37× under the width that would
    be a real dimension — so the geometry **is** certified; v3 simply refused to say so.

    Non-vacuous: on the v3 gate (`n_inconclusive == 0`) this call raises. The assertion below —
    that a build with `n_inconclusive > 0` succeeds and reports `exhaustive` — cannot pass there.
    """
    geometry = build_geometry(reduced, model_id="strain_1")
    certificate = geometry.certificate

    assert geometry.dimension == 46  # the same geometry every seed finds
    assert certificate.exhaustive  # v3 refused this exact case
    assert certificate.n_inconclusive > 0  # …and it refused it *because* of this, now a diagnostic
    assert certificate.resolution <= GeometryConfig().span_tol  # the bound that licenses it
    assert certificate.complement_is_complete


def test_a_truncated_basis_is_rejected(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """Hide any one of the 46 dimensions and the sweep must find it. This is the gate's core claim.

    Not a sampled subset — *every* column, one at a time. A certificate that missed even one hidden
    direction would be a certificate that could miss a real one.
    """
    config = GeometryConfig()
    scales = geometry.scaling
    program = build_flux_lp(reduced)
    blocked = blocked_reactions(reduced, program, tol=config.blocked_tol)
    space = direction_space(reduced, scales, blocked.mask, memory_limit_bytes=MEMORY)
    inverse_scale_norm = float(np.linalg.norm(1.0 / scales))

    for dropped in range(geometry.dimension):
        truncated = OrthonormalBasis(reduced.n_free, memory_limit_bytes=MEMORY)
        for column in range(geometry.dimension):
            if column != dropped:
                truncated.append_normalized(geometry.basis[:, column], rank_tol=config.rank_tol)
        assert truncated.n_columns == geometry.dimension - 1

        certificate, failing = sweep_complement(
            program, reduced, truncated, scales, inverse_scale_norm, config, MEMORY, space
        )
        assert failing is not None, f"a basis missing column {dropped} was certified complete"
        assert not certificate.exhaustive


# ---- the basis -----------------------------------------------------------------------------------


def test_every_basis_direction_preserves_the_steady_state(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """``‖S·diag(s)·B‖ ≈ 0``. A direction failing this breaks mass balance, and the chord — which
    only ever looks at bounds — would never notice."""
    transform = geometry.scaling[:, None] * geometry.basis
    errors = np.array(
        [
            np.max(np.abs(reduced.stoichiometry.matvec(transform[:, k])))
            for k in range(geometry.dimension)
        ]
    )
    assert errors.max() < 1e-10
    # No column is an outlier: the DirectionSpace projection stops error accumulating along the
    # Gram-Schmidt chain, which before it existed grew the last column to 130× the median.
    assert errors.max() / np.median(errors) < 20.0


def test_the_basis_is_orthonormal(geometry) -> None:  # type: ignore[no-untyped-def]
    gram = geometry.basis.T @ geometry.basis
    assert np.max(np.abs(gram - np.eye(geometry.dimension))) < 1e-12
    assert geometry.basis.flags.f_contiguous


def test_the_basis_is_exactly_zero_on_blocked_reactions(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """Exactly, not approximately — an FVA-blocked reaction is a structural zero of the direction
    space, and a 1e-15 there is the numerator of the divide-by-noise that corrupts the chord."""
    blocked = blocked_reactions(reduced, build_flux_lp(reduced), tol=1e-9)
    assert np.all(geometry.basis[blocked.mask, :] == 0.0)
    assert np.abs(geometry.basis[~blocked.mask, :]).max() > 0.1  # the moving ones do carry weight


# ---- the centre ----------------------------------------------------------------------------------


def test_the_centre_is_feasible_in_the_full_model(example_canonical, reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """Not merely in the reduced polytope the geometry was built from — in all 773 reactions."""
    assert reduced.contains(geometry.center, tol=1e-9)

    full = reduced.to_full(geometry.center)
    assert full.shape == (example_canonical.polytope.n_reactions,)
    assert example_canonical.polytope.contains(full, tol=1e-9)

    # The clamp that makes the centre exactly bound-feasible must stay at solver-noise scale.
    assert geometry.diagnostics.center_clamp < 1e-11
    assert geometry.diagnostics.center_bound_slack >= 0.0


def test_every_chord_through_the_centre_is_samplable(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """M5 starts here, and `feasible_chord` raises on a point outside its bounds. Before the blocked
    reactions were projected out, this produced a chord of [-0.54, -0.39] — excluding t = 0."""
    transform = geometry.scaling[:, None] * geometry.basis
    for k in range(geometry.dimension):
        chord = feasible_chord(
            geometry.center, transform[:, k], reduced.lower_bounds, reduced.upper_bounds
        )
        assert chord.contains(0.0), f"coordinate {k} cannot move from the centre"
        assert chord.length > 0.0

    assert geometry.diagnostics.min_chord_at_center > 1e-3


def test_a_walk_along_the_geometry_stays_inside_the_full_polytope(  # type: ignore[no-untyped-def]
    example_canonical, reduced, geometry
) -> None:
    """500 coordinate hit-and-run steps — the M5 inner loop, in miniature. Every point must remain a
    feasible flux vector of the *original* model, and no HiGHS solve may happen along the way."""
    transform = geometry.scaling[:, None] * geometry.basis
    rng = np.random.default_rng(11)
    flux = geometry.center.copy()

    solves_before = total_solve_count()
    for _ in range(500):
        k = int(rng.integers(geometry.dimension))
        chord = feasible_chord(flux, transform[:, k], reduced.lower_bounds, reduced.upper_bounds)
        step = rng.uniform(chord.t_lo, chord.t_hi) if chord.is_samplable else 0.0
        flux = flux + step * transform[:, k]
    assert total_solve_count() == solves_before  # geometry is frozen; sampling never calls a solver

    assert reduced.contains(flux, tol=1e-8)
    assert example_canonical.polytope.contains(reduced.to_full(flux), tol=1e-8)


# ---- reproducibility -----------------------------------------------------------------------------


def test_the_geometry_is_deterministic(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    again = build_geometry(reduced, model_id="bifido")
    assert np.array_equal(again.basis, geometry.basis)
    assert np.array_equal(again.center, geometry.center)
    assert again.content_key() == geometry.content_key()


def test_a_different_seed_finds_the_same_dimension(reduced, geometry) -> None:  # type: ignore[no-untyped-def]
    """The basis is one of infinitely many; the span it certifies is a property of the polytope."""
    other = build_geometry(reduced, config=GeometryConfig(seed=7), model_id="bifido")

    assert other.dimension == geometry.dimension
    assert not np.array_equal(other.basis, geometry.basis)
    assert other.certificate.exhaustive

    # Same subspace: each basis's projector must reproduce the other's columns.
    projector = geometry.basis @ geometry.basis.T
    assert np.max(np.abs(projector @ other.basis - other.basis)) < 1e-9
