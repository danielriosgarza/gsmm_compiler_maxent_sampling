"""M4 — affine geometry: scales, basis, complement, direction space, and known toy dimensions.

The polytopes here are small enough that their affine dimension can be worked out on paper, which is
the point: `build_geometry` has to *recover* a number we already know, not merely produce one that
looks plausible.
"""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import gsmm_compiler.affine_geometry as ag
from gsmm_compiler.affine_geometry import (
    BlockedReactions,
    GeometryError,
    OrthonormalBasis,
    ReducedGeometry,
    SpanCertificate,
    SupportProbe,
    _append,
    _check_blocked_span,
    _reject_contradictory_bracket,
    _SolveSession,
    blocked_reactions,
    build_geometry,
    complement_basis,
    direction_space,
    dual_upper_bound,
    reaction_scales,
    sweep_complement,
)
from gsmm_compiler.config import ConfigError, GeometryConfig
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.highs_backend import HighsLinearProgram, LPNotOptimalError
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.sparse_objective import build_flux_lp


def make_polytope(
    stoichiometry: list[list[float]],
    lower: list[float],
    upper: list[float],
    *,
    biomass_index: int = 0,
) -> ReducedPolytope:
    """A reduced polytope from a dense S and bounds — ``l == u`` columns eliminate themselves."""
    matrix = np.asarray(stoichiometry, dtype=np.float64)
    full = FluxPolytope(
        reaction_ids=tuple(f"R{i}" for i in range(matrix.shape[1])),
        metabolite_ids=tuple(f"M{i}" for i in range(matrix.shape[0])),
        stoichiometry=NativeCSC.from_dense(matrix),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        biomass_index=biomass_index,
    )
    return full.reduce()


# ---- the toys, with their dimensions derived on paper --------------------------------------------


def triangle() -> ReducedPolytope:
    """``v0 = v1 + v2``, all in [0, 10] — the triangle ``{v1, v2 ≥ 0, v1 + v2 ≤ 10}``. **d = 2.**"""
    return make_polytope([[1.0, -1.0, -1.0]], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])


def pinned_by_mass_balance() -> ReducedPolytope:
    """``v0`` is free in [0, 10] but mass balance pins it to 0. **d = 0**, and ``v0`` is blocked.

    The fixed ``v1 == 0`` is eliminated by M1; what remains is ``v0 = 0`` with bounds that permit
    more. This is the case the sampler must answer with a constant, not a chain.
    """
    return make_polytope([[1.0, -1.0]], [0.0, 0.0], [10.0, 0.0])


def narrow_but_real() -> ReducedPolytope:
    """``v0 = v1`` with ``v0 ∈ [0, 1e-7]`` and ``v1 ∈ [0, 1000]``. **d = 1**, and it is 1e-7 wide.

    The gate's "scale-sensitive narrow example": a real dimension whose flux extent is ten million
    times smaller than the polytope's other axis. Scaling is what makes it visible — in raw flux
    units a width tolerance able to see it would be swamped by noise on the wide axis.
    """
    return make_polytope([[1.0, -1.0]], [0.0, 0.0], [1e-7, 1000.0])


def two_free_dimensions() -> ReducedPolytope:
    """``v0 = v1 + v2`` and ``v3 = v1``, all in [0, 10]. **d = 2**, with a 2-wide complement.

    Two independent constraints rather than one, so ``n_free − d = 2`` and a probe cap of 1 actually
    bites — which the single-constraint toys cannot test, their complement being 1 column wide.
    """
    return make_polytope(
        [[1.0, -1.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    )


def two_narrow_scales() -> ReducedPolytope:
    """Two independent narrow branches, 1e-7 and 5e-7 wide. **d = 2.**

    Their ranges are only 5× apart, so a `blocked_tol` landing between them would split the
    reactions on a distinction the polytope does not support. `blocked_reactions` must refuse.
    """
    return make_polytope(
        [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
        [0.0, 0.0, 0.0, 0.0],
        [1e-7, 1000.0, 5e-7, 1000.0],
    )


# ---- scaled coordinates --------------------------------------------------------------------------


def test_scales_are_the_bound_ranges_with_a_floor() -> None:
    reduced = narrow_but_real()
    scales = reaction_scales(reduced, floor=1e-6)
    assert scales[0] == pytest.approx(1e-6)  # floored: the raw range is 1e-7
    assert scales[1] == pytest.approx(1000.0)  # untouched


def test_scales_reject_a_corrupt_reduced_polytope() -> None:
    reduced = triangle()
    corrupt = ReducedPolytope(**{**reduced.__dict__, "upper_bounds": reduced.lower_bounds.copy()})
    with pytest.raises(GeometryError, match="l < u"):
        reaction_scales(corrupt, floor=1e-6)


# ---- the orthonormal basis -----------------------------------------------------------------------


def test_basis_stays_orthonormal_through_block_growth() -> None:
    rng = np.random.default_rng(0)
    basis = OrthonormalBasis(40, memory_limit_bytes=1 << 30, block=4)
    for _ in range(37):  # forces several block reallocations
        basis.append_normalized(basis.remove_components(rng.normal(size=40)), rank_tol=1e-10)
    gram = basis.matrix.T @ basis.matrix
    assert np.max(np.abs(gram - np.eye(37))) < 1e-13
    assert basis.matrix.flags.f_contiguous


def test_basis_refuses_a_direction_it_already_spans() -> None:
    basis = OrthonormalBasis(3, memory_limit_bytes=1 << 30)
    basis.append_normalized(np.array([1.0, 0.0, 0.0]), rank_tol=1e-10)
    with pytest.raises(GeometryError, match="already lies in the discovered span"):
        basis.append_normalized(np.array([2.0, 0.0, 0.0]), rank_tol=1e-10)


def test_basis_allocation_respects_the_memory_guard() -> None:
    with pytest.raises(GeometryError, match="max_geometry_memory_gb"):
        OrthonormalBasis(10_000, memory_limit_bytes=1024)


# ---- the complement ------------------------------------------------------------------------------


def test_complement_is_an_orthonormal_basis_of_the_orthogonal_complement() -> None:
    rng = np.random.default_rng(1)
    n, d = 12, 5
    basis, _ = np.linalg.qr(rng.normal(size=(n, d)))

    complement = complement_basis(
        basis, n, rank_tol=1e-10, max_columns=None, memory_limit_bytes=1 << 30
    )

    assert complement.shape == (n, n - d)
    assert np.max(np.abs(complement.T @ complement - np.eye(n - d))) < 1e-12  # orthonormal
    assert np.max(np.abs(basis.T @ complement)) < 1e-12  # orthogonal to B
    # Together they span ℝⁿ: the two projectors must sum to the identity.
    whole = basis @ basis.T + complement @ complement.T
    assert np.max(np.abs(whole - np.eye(n))) < 1e-12


def test_complement_is_capped_when_asked() -> None:
    basis = np.zeros((8, 2))
    basis[0, 0] = basis[1, 1] = 1.0
    capped = complement_basis(basis, 8, rank_tol=1e-10, max_columns=3, memory_limit_bytes=1 << 30)
    assert capped.shape == (8, 3)


def test_complement_of_a_full_basis_is_empty() -> None:
    basis = np.eye(4)
    empty = complement_basis(basis, 4, rank_tol=1e-10, max_columns=None, memory_limit_bytes=1 << 30)
    assert empty.shape == (4, 0)


# ---- the direction space -------------------------------------------------------------------------


def test_direction_space_projects_onto_mass_balanced_and_unblocked() -> None:
    reduced = triangle()
    scales = reaction_scales(reduced, floor=1e-6)
    blocked = np.array([False, False, False])
    space = direction_space(reduced, scales, blocked, memory_limit_bytes=1 << 30)

    # rank(S) = 1 over 3 columns, so the direction space is 2-dimensional.
    assert space.n_null == 2

    rng = np.random.default_rng(2)
    projected = space.project(rng.normal(size=3))
    residual = reduced.stoichiometry.matvec(scales * projected)
    assert np.max(np.abs(residual)) < 1e-13  # S·diag(s)·x = 0, to machine precision

    # Idempotent: projecting a direction that is already legal changes nothing.
    again = space.project(projected)
    assert np.allclose(again, projected, atol=1e-15)


def test_direction_space_zeroes_blocked_components_exactly() -> None:
    reduced = triangle()
    scales = reaction_scales(reduced, floor=1e-6)
    space = direction_space(
        reduced, scales, np.array([False, True, False]), memory_limit_bytes=1 << 30
    )
    projected = space.project(np.array([1.0, 1.0, 1.0]))
    assert projected[1] == 0.0  # exactly, not approximately


def test_direction_space_respects_the_memory_guard() -> None:
    reduced = triangle()
    scales = reaction_scales(reduced, floor=1e-6)
    with pytest.raises(GeometryError, match="max_geometry_memory_gb"):
        direction_space(reduced, scales, np.zeros(3, dtype=bool), memory_limit_bytes=1)


# ---- known dimensions ----------------------------------------------------------------------------


def test_triangle_has_dimension_two() -> None:
    geometry = build_geometry(triangle(), model_id="triangle")
    assert geometry.dimension == 2
    assert geometry.certificate.exhaustive
    assert geometry.diagnostics.n_blocked == 0


def test_narrow_dimension_is_found_not_rounded_away() -> None:
    """A real direction 1e-7 wide in flux units is a dimension, and must survive."""
    geometry = build_geometry(narrow_but_real(), model_id="narrow")
    assert geometry.dimension == 1
    assert geometry.certificate.exhaustive
    assert geometry.diagnostics.n_blocked == 0

    # And the polytope really is that thin: walking the chord cannot leave [0, 1e-7].
    reduced = narrow_but_real()
    transform = geometry.scaling[:, None] * geometry.basis
    for t in np.linspace(-1.0, 1.0, 11):
        flux = geometry.center + t * transform[:, 0]
        if reduced.contains(flux, tol=1e-12):
            assert -1e-12 <= flux[0] <= 1e-7 + 1e-12


def test_a_polytope_pinned_to_a_point_has_dimension_zero() -> None:
    reduced = pinned_by_mass_balance()
    geometry = build_geometry(reduced, model_id="pinned")

    assert geometry.dimension == 0
    assert geometry.is_singleton
    assert geometry.diagnostics.n_blocked == 1  # v0 is free in its bounds but cannot move
    assert geometry.center == pytest.approx([0.0])
    # Dimension zero means every sample is the centre — `to_flux` of the empty coordinate says so.
    assert geometry.to_flux(np.zeros(0)) == pytest.approx(geometry.center)


def test_a_fully_fixed_polytope_returns_the_singleton_geometry() -> None:
    reduced = make_polytope([[1.0, -1.0]], [2.0, 2.0], [2.0, 2.0])
    assert reduced.is_singleton

    geometry = build_geometry(reduced, model_id="fixed")
    assert geometry.dimension == 0
    assert geometry.is_singleton
    assert geometry.diagnostics.n_lp_solves == 0  # nothing to solve
    assert geometry.certificate.exhaustive


def test_a_tolerance_between_two_real_widths_classifies_rather_than_refusing() -> None:
    """M11.1 retired the separation gate: a `tol` between two real widths is no longer "ambiguous".

    The predecessor refused this exact polytope — the 1e-7 branch and the 5e-7 branch are only 5×
    apart, and `tol = 2e-7` lands between them, which the old ``min_separation = 100`` guard called
    a choice about `tol` rather than a fact. But each reaction *is* individually classifiable: the
    1e-7 branch has `U ≤ 2e-7` (certified blocked), the 5e-7 branch has `L > 2e-7` (moving). Under
    the declared resolution contract (§1.6.11) that is the honest answer — a dimension narrower than
    the tolerance the user set is declared absent — not a refusal. Nothing is *unresolved*.
    """
    reduced = two_narrow_scales()
    blocked = blocked_reactions(reduced, build_flux_lp(reduced), tol=2e-7)
    assert blocked.n_unresolved == 0
    assert blocked.n_blocked == 2  # the 1e-7 branch's two reactions, declared below-resolution
    assert int(np.sum(blocked.lower > 2e-7)) == 2  # the 5e-7 branch, resolution-qualified moving


def test_blocked_reactions_refuse_a_reaction_it_cannot_resolve() -> None:
    """The new gate (M11.1): refuse when a width straddles the bar even after a cold re-solve.

    A reaction bounded to ``[0, blocked_tol]`` has a true width sitting exactly on the tolerance, so
    its bracket brackets the bar: `U` is a hair above (not certified blocked) and `L` a hair below
    (not resolution-qualified moving). This is the *Hafnia* case in miniature, and the whole point
    of the third state — it is neither, and no cold re-solve moves a width that is genuinely at the
    floor. The message names the reactions and reports the bracket, and does **not** advise
    disabling a check.
    """
    reduced = make_polytope([[1.0, -1.0]], [0.0, 0.0], [1e-9, 1000.0])
    with pytest.raises(GeometryError, match="cannot be resolved as blocked or moving") as caught:
        blocked_reactions(reduced, build_flux_lp(reduced), tol=1e-9)
    assert "cold re-solve" in str(caught.value)


def test_a_kunknown_warm_solve_is_recovered_by_cold_escalation() -> None:
    """The heart of M11.1: a reaction whose *warm* solve gives no witness is re-solved cold.

    This is the mechanism that turns 9 of the 40 real strains from a `kUnknown` crash into a clean
    build. A toy cannot degrade a warm start, so the degradation is injected: a program wrapper that
    raises `LPNotOptimalError` on the solves for one reaction and delegates everything else. The
    reaction must still be classified — matching an all-clean run — and `n_escalated` must count it.

    Non-vacuous: without the escalation the `LPNotOptimalError` propagates and the call raises;
    without the *catch* being caller-specific it would loosen `solve()` for everyone.
    """
    reduced = triangle()  # every reaction wide open; nothing genuinely unresolved
    clean = blocked_reactions(reduced, build_flux_lp(reduced), tol=1e-9)
    assert clean.n_escalated == 0  # the control takes the warm fast path, no cold re-solves

    class _FailsWarmOnReaction:
        """Delegates to a real flux LP but refuses the two solves that probe `victim`."""

        def __init__(self, victim: int) -> None:
            self._inner = build_flux_lp(reduced)
            self._victim = victim

        def maximize(self, costs: NDArray[np.float64]) -> object:
            if float(costs[self._victim]) != 0.0:
                raise LPNotOptimalError("kUnknown", "flux_only")
            return self._inner.maximize(costs)

    degraded = blocked_reactions(
        reduced, cast(HighsLinearProgram, _FailsWarmOnReaction(victim=1)), tol=1e-9
    )
    assert degraded.n_escalated >= 1  # reaction 1 could not be solved warm
    assert degraded.n_unresolved == 0  # but cold resolved it
    assert bool(degraded.mask[1]) == bool(clean.mask[1])  # to the same verdict as a clean run


def test_the_solve_session_goes_cold_after_a_kunknown_and_stays_cold(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The build-wide session (M11.2): warm until a `kUnknown`, then cold-only for the rest.

    This is the abstraction that fixes the leak Codex found — M11.1 abandoned a degraded instance
    only inside `blocked_reactions`, while `build_geometry` kept warm-starting the later stages off
    the same instance. A toy cannot degrade a warm start, so the persistent instance is wrapped to
    raise `kUnknown` on its second solve; the fresh instances the cold retries build are real, so
    the session recovers and every later solve is cold.
    """
    reduced = triangle()
    real_build = ag.build_flux_lp
    builds: list[int] = []

    class _FailsSecondWarm:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._n = 0

        def maximize(self, costs: NDArray[np.float64]) -> object:
            self._n += 1
            if self._n == 2:
                raise LPNotOptimalError("HighsModelStatus.kUnknown", "flux_only")
            return self._inner.maximize(costs)  # type: ignore[attr-defined]

    def fake_build(r: object, threads: int = 1) -> object:
        builds.append(1)
        prog = real_build(r, threads=threads)  # type: ignore[arg-type]
        return _FailsSecondWarm(prog) if len(builds) == 1 else prog  # only the persistent instance

    monkeypatch.setattr(ag, "build_flux_lp", fake_build)
    session = _SolveSession(reduced)
    n = reduced.n_free
    cost = np.zeros(n)
    cost[0] = 1.0

    session.maximize(np.zeros(n))  # first solve: warm, fine
    assert session.n_cold_solves == 0 and session.degraded is False
    session.maximize(cost)  # second solve on the persistent instance: kUnknown -> cold retry
    assert session.degraded is True and session.n_cold_solves == 1
    session.maximize(-cost)  # cold-only now: no warm attempt at all
    assert session.n_cold_solves == 2


def test_the_solve_session_escalates_only_kunknown(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Infeasible / unbounded / limit keep their hard-failure meaning; only `kUnknown` escalates.

    A `kUnknown` is "the solver could not decide", which a fresh basis can fix; the others are
    verdicts about the model, which it cannot (Codex, M11.2). So a non-`kUnknown` status must
    propagate, not silently trigger a cold retry that would mask a genuinely infeasible model.
    """
    reduced = triangle()
    real_build = ag.build_flux_lp
    builds: list[int] = []

    class _FailsInfeasible:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def maximize(self, costs: NDArray[np.float64]) -> object:
            raise LPNotOptimalError("HighsModelStatus.kInfeasible", "flux_only")

    def fake_build(r: object, threads: int = 1) -> object:
        builds.append(1)
        prog = real_build(r, threads=threads)  # type: ignore[arg-type]
        return _FailsInfeasible(prog) if len(builds) == 1 else prog

    monkeypatch.setattr(ag, "build_flux_lp", fake_build)
    session = _SolveSession(reduced)
    with pytest.raises(LPNotOptimalError, match="kInfeasible"):
        session.maximize(np.zeros(reduced.n_free))
    assert session.n_cold_solves == 0  # it did not escalate


def test_a_cold_only_session_never_warm_starts() -> None:
    """`cold_only=True` — the mode the span sweep's fully-cold re-confirmation uses (M11.2).

    Every solve is a fresh instance, so `n_cold_solves` counts them all and `degraded` is True from
    the start. On the triangle each solve still returns the right answer; the point is the counting
    and that nothing warm-starts.
    """
    reduced = triangle()
    session = _SolveSession(reduced, cold_only=True)
    assert session.degraded is True
    n = reduced.n_free
    cost = np.zeros(n)
    cost[0] = 1.0
    session.maximize(cost)
    session.maximize(-cost)
    assert session.n_cold_solves == 2


def test_blocked_reactions_reraises_a_non_kunknown_status(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`blocked_reactions` escalates only ``kUnknown``; a model verdict is not silently retried.

    Codex's M11.2 closing catch: the warm bracket used to turn *every* `LPNotOptimalError` into
    "unresolved" and cold-escalate, so an infeasible or unbounded status (a fact about the model a
    fresh basis cannot change) would be swallowed. `_warm_width_bracket` now re-raises anything but
    ``kUnknown``. Injected via a program that raises ``kInfeasible`` on its first solve.
    """
    reduced = triangle()

    class _Infeasible:
        def maximize(self, costs: NDArray[np.float64]) -> object:
            raise LPNotOptimalError("HighsModelStatus.kInfeasible", "flux_only")

    with pytest.raises(LPNotOptimalError, match="kInfeasible"):
        blocked_reactions(reduced, cast(HighsLinearProgram, _Infeasible()), tol=1e-9)


def test_a_warm_resolution_failure_is_re_confirmed_by_a_fully_cold_sweep(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """End-to-end (Codex, M11.2): a `resolution > span_tol` refusal re-sweeps fully cold first.

    `width_upper` is a weak-duality bound that a warm-started basis can leave *loose*, so a warm
    sweep can report `resolution > span_tol` on a geometry a cold sweep certifies. Before refusing,
    `build_geometry` re-sweeps fully cold and gates on *that*. Here `sweep_complement` is monkey-
    patched to return a certificate that fails the gate **once** (the warm sweep) and passes it on
    the fully-cold re-sweep — so the build must consult the second, not the first, and succeed.

    Non-vacuous: without the re-sweep the first (failing) certificate reaches the raise and the
    build aborts; the assertion that it *succeeds* cannot pass.
    """
    reduced = triangle()
    real_sweep = ag.sweep_complement
    calls: list[int] = []

    def flaky_sweep(program, *args, **kwargs):  # type: ignore[no-untyped-def]
        certificate, failing = real_sweep(program, *args, **kwargs)
        calls.append(1)
        if len(calls) == 1 and failing is None:
            # First call is the warm sweep: force a resolution just over span_tol so the gate fires.
            certificate = dataclasses.replace(certificate, exhaustive=False)
        return certificate, failing

    monkeypatch.setattr(ag, "sweep_complement", flaky_sweep)
    geometry = build_geometry(reduced, model_id="triangle")  # must not raise
    assert geometry.certificate.exhaustive  # it consulted the cold re-sweep, which passed
    assert len(calls) >= 2  # warm sweep + the fully-cold re-confirmation


def test_a_contradictory_bracket_is_loud_not_repaired() -> None:
    """`L_i > U_i` breaks the arithmetic contract, so it raises — it is never maxed away (M11.1).

    The two bracket the same width from opposite sides. The predecessor's ``max(U, W)`` could not
    even observe this, because it resolved it. Driven through the helper on hand-built arrays so the
    guard is tested in isolation from any solver.
    """
    upper = np.array([1e-12, 5e-13])
    lower = np.array([1e-6, 0.0])  # reaction 0: a lower witness far above its rigorous upper bound
    with pytest.raises(GeometryError, match="above its .*upper bound"):
        _reject_contradictory_bracket(upper, lower)


def test_blocked_reactions_accept_the_example_models_wide_separation() -> None:
    """Nothing is blocked when every reaction moves — the split is then vacuously unambiguous."""
    reduced = two_narrow_scales()
    blocked = blocked_reactions(reduced, build_flux_lp(reduced), tol=1e-9)
    assert blocked.n_blocked == 0
    assert blocked.separation == np.inf


# ---- the certificate as a detector ---------------------------------------------------------------


def test_a_truncated_basis_is_rejected_by_the_sweep() -> None:
    """The gate's core claim: hide a dimension and the certificate finds it. Every time."""
    reduced = triangle()
    config = GeometryConfig()
    geometry = build_geometry(reduced, config=config, model_id="triangle")
    assert geometry.dimension == 2

    scales = geometry.scaling
    program = build_flux_lp(reduced)
    blocked = blocked_reactions(reduced, program, tol=config.blocked_tol)
    space = direction_space(reduced, scales, blocked.mask, memory_limit_bytes=1 << 30)
    inverse_scale_norm = float(np.linalg.norm(1.0 / scales))

    for dropped in range(geometry.dimension):
        kept = [c for c in range(geometry.dimension) if c != dropped]
        truncated = OrthonormalBasis(reduced.n_free, memory_limit_bytes=1 << 30)
        for column in kept:
            truncated.append_normalized(geometry.basis[:, column], rank_tol=config.rank_tol)

        certificate, failing = sweep_complement(
            program, reduced, truncated, scales, inverse_scale_norm, config, 1 << 30, space
        )
        assert failing is not None, f"dropping column {dropped} went undetected"
        assert not certificate.exhaustive
        assert failing.width > failing.width_floor or failing.residual_norm > failing.rank_floor


def test_the_certificate_passes_on_the_basis_it_was_built_from() -> None:
    reduced = triangle()
    config = GeometryConfig()
    geometry = build_geometry(reduced, config=config, model_id="triangle")

    scales = geometry.scaling
    program = build_flux_lp(reduced)
    blocked = blocked_reactions(reduced, program, tol=config.blocked_tol)
    space = direction_space(reduced, scales, blocked.mask, memory_limit_bytes=1 << 30)

    basis = OrthonormalBasis(reduced.n_free, memory_limit_bytes=1 << 30)
    for column in range(geometry.dimension):
        basis.append_normalized(geometry.basis[:, column], rank_tol=config.rank_tol)

    certificate, failing = sweep_complement(
        program,
        reduced,
        basis,
        scales,
        float(np.linalg.norm(1.0 / scales)),
        config,
        1 << 30,
        space,
    )
    assert failing is None
    assert certificate.exhaustive
    assert certificate.n_probes == certificate.n_complement


def test_a_capped_certificate_is_never_called_exhaustive() -> None:
    reduced = two_free_dimensions()  # complement is 2 wide, so a cap of 1 truly truncates it
    config = GeometryConfig(exhaustive_span_certificate=False, max_span_probes=1)
    geometry = build_geometry(reduced, config=config, model_id="capped")

    assert geometry.dimension == 2
    assert geometry.certificate.n_complement == 2
    assert geometry.certificate.n_probes == 1
    assert not geometry.certificate.exhaustive
    assert geometry.manifest()["span_certificate_exhaustive"] is False


def test_an_uncertifiable_geometry_is_refused_by_default() -> None:
    """`exhaustive_span_certificate` defaults to true, and then a partial check is a hard failure —
    spec §15.4 forbids silently sampling a lower-dimensional subset.

    Here the cause is a **cap** (`max_span_probes=1` on a 2-wide complement), which leaves a
    direction genuinely unprobed — so the resolution gate (M11.2) refuses via `leakage·diameter`,
    not via `n_inconclusive`. That the *capped* path still refuses is the point: M11.2 relaxed the
    noise-swamped-probe case, not the genuinely-incomplete one."""
    with pytest.raises(GeometryError, match="does not resolve the polytope to span_tol"):
        build_geometry(
            two_free_dimensions(),
            config=GeometryConfig(max_span_probes=1),
            model_id="capped",
        )


# ---- the artifact --------------------------------------------------------------------------------


def test_coordinates_round_trip_through_flux() -> None:
    geometry = build_geometry(triangle(), model_id="triangle")
    rng = np.random.default_rng(3)
    coordinates = rng.normal(size=(5, geometry.dimension)) * 0.1

    flux = geometry.to_flux(coordinates)
    assert flux.shape == (5, geometry.n_free)
    assert geometry.to_coordinates(flux) == pytest.approx(coordinates, abs=1e-12)


def test_to_flux_rejects_the_wrong_shape() -> None:
    geometry = build_geometry(triangle(), model_id="triangle")
    with pytest.raises(ValueError, match="trailing dimension"):
        geometry.to_flux(np.zeros(geometry.dimension + 1))
    with pytest.raises(ValueError, match="trailing dimension"):
        geometry.to_coordinates(np.zeros(geometry.n_free + 1))


def test_geometry_is_deterministic_for_a_fixed_seed() -> None:
    first = build_geometry(triangle(), model_id="triangle")
    second = build_geometry(triangle(), model_id="triangle")
    assert np.array_equal(first.basis, second.basis)
    assert np.array_equal(first.center, second.center)
    assert first.content_key() == second.content_key()


def test_a_different_model_id_names_a_different_rng_stream() -> None:
    """The seed is keyed on semantic coordinates, so two models do not share a probe sequence."""
    first = build_geometry(triangle(), model_id="triangle")
    second = build_geometry(triangle(), model_id="other")
    assert first.dimension == second.dimension  # the span is a property of the polytope
    assert not np.array_equal(first.basis, second.basis)  # the basis of it is not


def test_content_key_changes_with_the_basis() -> None:
    geometry = build_geometry(triangle(), model_id="triangle")
    perturbed = ReducedGeometry(**{**geometry.__dict__, "center": geometry.center + 1e-6})
    assert perturbed.content_key() != geometry.content_key()


# ---- what the certificate actually licenses (the M4 collab review)
# --------------------------------


def test_the_certificate_reports_a_sqrt_k_resolution_not_its_largest_width() -> None:
    """Width is subadditive, so ``k`` probes each below ε only bound an arbitrary complement
    direction by ``√k·ε``. A direction tilted equally across every probe hides that factor from all
    of them, so `resolution` — not `max_width` — is what the certificate is entitled to claim."""
    geometry = build_geometry(two_free_dimensions(), model_id="resolution")
    certificate = geometry.certificate

    expected = math.sqrt(certificate.n_complement) * certificate.max_width
    assert certificate.resolution == pytest.approx(expected)
    assert certificate.resolution >= certificate.max_width
    assert geometry.manifest()["span_resolution"] == pytest.approx(expected)


def test_a_wide_bound_range_does_not_hide_a_dimension_behind_the_dual_tolerance() -> None:
    """``c = p/s`` shrinks with the bound range, and HiGHS's dual tolerance is **absolute**.

    With ``s = 1e10`` the raw objective coefficients are ~1e-10, so an improving reduced cost
    can sit below the solver's 1e-9 dual tolerance: HiGHS reports kOptimal without leaving its
    starting vertex, both directions return the same point, and a wide-open dimension is
    measured as flat. Sup-normalizing the objective is what stops that, and this is the case
    that would catch its loss.
    """
    reduced = make_polytope([[1.0, -1.0]], [0.0, 0.0], [1e10, 1e10])
    geometry = build_geometry(reduced, model_id="wide")

    assert geometry.dimension == 1  # a 1e10-wide direction, not a flat one
    assert geometry.certificate.exhaustive
    assert geometry.certificate.worst_dual_error <= GeometryConfig().dual_tol


def test_a_dimension_below_the_blocked_resolution_is_dropped_and_that_is_documented() -> None:
    """The honest limitation, pinned as a test rather than left in a docstring.

    A reaction whose true range is 5e-16 is *blocked* — its dimension is real and it is discarded.
    `separation` does not object (it is 2e15× here), because it measures whether the split is
    **clustered**, not whether it is **right**. No absolute threshold can do better; what matters is
    that the behaviour is known, reported, and never described as a proof of exact constancy.
    """
    reduced = make_polytope(
        [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 5e-16, 1.0, 5e-16],
    )
    geometry = build_geometry(reduced, model_id="sub-resolution")

    assert geometry.dimension == 1  # the true affine dimension is 2
    assert geometry.diagnostics.n_blocked == 2
    assert geometry.diagnostics.blocked_separation > 1e14  # and the guard is perfectly happy


def test_the_config_forbids_resolutions_that_contradict_each_other() -> None:
    """``scale_floor ≥ blocked_tol / span_tol``, or the sweep reports an axis the projection
    removed.

    Below that relation a blocked reaction still shows a scaled width above `span_tol`: the sweep
    flags it as a missing dimension, the projection has already zeroed the residual it would need to
    append, and the build dies on a contradiction it cannot name. Refuse it up front instead.
    """
    with pytest.raises(ConfigError, match="blocked_tol/span_tol"):
        GeometryConfig(scale_floor=1e-6)  # the old default, with blocked_tol/span_tol = 1.0

    GeometryConfig(scale_floor=1e-6, blocked_tol=1e-15)  # consistent, and therefore allowed


def test_a_blocked_axis_the_sweep_still_flags_is_diagnosed_not_mystified() -> None:
    """If the resolutions *do* disagree, say which reaction and which tolerances — do not report
    that the basis 'already spans' a direction the projection just deleted.

    The config relation makes an *axis-aligned* contradiction unreachable, which is its job; a probe
    spread across many blocked axes could still get there via the same √k accumulation. So the guard
    is exercised directly, on exactly the state that used to produce the misleading message: a probe
    with real width whose residual the projection has zeroed.
    """
    reduced = triangle()
    config = GeometryConfig()
    basis = OrthonormalBasis(reduced.n_free, memory_limit_bytes=1 << 30)
    blocked = BlockedReactions(
        mask=np.array([False, True, False]),
        upper=np.zeros(3),
        lower=np.zeros(3),
        unresolved=np.zeros(3, dtype=bool),
        n_escalated=0,
        separation=np.inf,
    )
    killed = SupportProbe(
        direction=np.array([0.0, 1.0, 0.0]),  # points along the blocked reaction
        v_plus=np.zeros(3),
        v_minus=np.zeros(3),
        width=1.0,  # the LP says this direction is wide open …
        residual=np.zeros(3),  # … but the projection left nothing to append
        residual_norm=0.0,
        width_floor=1e-9,
        rank_floor=1e-9,
        is_conclusive=True,
        dual_error=0.0,
        width_upper=1.0,
        simplex_iterations=0,
    )

    with pytest.raises(GeometryError, match="admitted direction space") as caught:
        _append(basis, killed, reduced, blocked, config)
    assert "R1" in str(caught.value)  # names the reaction, not a mystery about spans


def test_an_ambiguous_equality_rank_is_refused() -> None:
    """A singular value straddling the numerical-rank cutoff means the arithmetic cannot say which
    equalities are real — and guessing samples a set the model does not describe.

    The matrix is one ULP from singular: exactly, its two rows are independent and the polytope is
    the single point 0; numerically, the second singular value falls under the rank cutoff and the
    equality is discarded. float64 cannot tell these apart — the LP cannot either — so what the
    geometry owes is a *report* of how confident the rank decision was, and a refusal when it is
    not.
    """
    reduced = make_polytope(
        [[1.0, -1.0], [1.0, -(1.0 + 2**-52)]], [-1.0, -1.0], [1.0, 1.0]
    )
    scales = reaction_scales(reduced, floor=1.0)
    blocked = np.zeros(reduced.n_free, dtype=bool)

    permissive = direction_space(reduced, scales, blocked, memory_limit_bytes=1 << 30)
    assert permissive.n_null == 1  # numerically rank-1, so it looks 1-dimensional

    with pytest.raises(GeometryError, match="no clear numerical rank"):
        direction_space(
            reduced, scales, blocked, memory_limit_bytes=1 << 30, min_singular_gap=1e30
        )


def test_the_support_points_span_every_discovered_direction() -> None:
    """M5 takes its rounding covariance from these. If they do not span all ``d`` directions, the
    ridge M5 adds would conceal a singular covariance instead of failing on it."""
    geometry = build_geometry(two_free_dimensions(), model_id="support")
    assert geometry.diagnostics.support_coordinate_rank == geometry.dimension == 2


# ---- round 2 of the collab review: the width can be *understated*, too --------------------------


def test_the_dual_gap_bounds_how_much_a_width_can_be_understated() -> None:
    """A width is a difference of two optima, so an unclaimed reduced cost makes it too *small*.

    ``max_dual_infeasibility`` cannot catch that: 1e-10 is dual-feasible anywhere, yet on a variable
    of range 1e10 it hides a whole unit of width. The bound that holds is
    ``Σⱼ dual_infeasibilityⱼ · (uⱼ − lⱼ)``, and the certificate's resolution rests on it.
    """
    reduced = make_polytope([[1.0, -1.0]], [0.0, 0.0], [1e10, 1e10])
    program = build_flux_lp(reduced)

    # Weak duality bounds max cᵀv from above for *any* row multipliers — no assumption that the
    # solver stopped anywhere sensible, and none that its point is exactly row-feasible.
    costs = np.array([1.0, 0.0])
    solution = program.maximize(costs)
    bound = dual_upper_bound(solution, costs, reduced)
    assert bound >= float(costs @ solution.primal) - 1e-6  # it really is an upper bound
    assert bound == pytest.approx(1e10, rel=1e-9)  # and here it is tight: the true optimum

    geometry = build_geometry(reduced, model_id="dual-gap")
    assert geometry.dimension == 1
    # The certificate's flatness claim rests on the *upper* bound, so the resolution it reports must
    # dominate what any single probe could have measured from the primal side alone.
    assert geometry.certificate.resolution >= geometry.certificate.max_width
    assert geometry.certificate.resolution < 1e-3


def test_the_svd_cutoff_is_reconciled_with_the_lp_feasibility_tolerance() -> None:
    """An equality the LP will not enforce must not be one the projector insists on.

    With ``σ_min = 1e-14`` the equality is *above* the machine-epsilon cutoff, so a spectral rank
    keeps it — while the LP, whose feasibility model is 1e-9, happily returns endpoints that move
    along it. The support LPs would then hand back a direction `DirectionSpace` projects to nothing:
    a contradiction, not a geometry. Taking the cutoff to be ``max(machine, feasibility_tol)`` makes
    the two components describe the same polytope.
    """
    reduced = make_polytope(
        [[1.0, -1.0], [1.0, -(1.0 + 1e-14)]], [-1.0, -1.0], [1.0, 1.0]
    )
    scales = reaction_scales(reduced, floor=1.0)
    blocked = np.zeros(reduced.n_free, dtype=bool)

    spectral = direction_space(
        reduced, scales, blocked, memory_limit_bytes=1 << 30, feasibility_tol=1e-300
    )
    assert spectral.n_null == 0  # a machine-epsilon rank keeps the near-singular equality

    reconciled = direction_space(
        reduced, scales, blocked, memory_limit_bytes=1 << 30, feasibility_tol=1e-9
    )
    assert reconciled.n_null == 1  # the LP's resolution discards it, and so must we

    assert build_geometry(reduced, model_id="reconciled").dimension == 1


def test_the_blocked_reactions_must_be_flat_together_not_only_one_at_a_time() -> None:
    """``‖r_blocked / s_blocked‖₂ ≤ span_tol`` — the same subadditivity that gives the certificate
    its ``√k``. Bounding each blocked range individually lets the *combination* reach
    ``√n_blocked · span_tol``, and a probe spread across them would then report a width the
    projection has already zeroed."""
    scales = np.array([1.0, 1.0, 1.0])
    config = GeometryConfig()
    upper = np.full(3, 0.9e-9)  # each below span_tol, but ‖·‖₂ = 1.56e-9 is not

    with pytest.raises(GeometryError, match="together"):
        _check_blocked_span(
            BlockedReactions(
                mask=np.ones(3, dtype=bool),
                upper=upper,
                lower=np.zeros(3),
                unresolved=np.zeros(3, dtype=bool),
                n_escalated=0,
                separation=np.inf,
            ),
            scales,
            config,
        )


def test_the_resolution_accounts_for_the_complements_floating_point_leakage() -> None:
    """``√k · max_width`` assumes ``Q`` exactly spans ``range(B)ᗮ``. Gram-Schmidt in float64 does
    not, quite — and a direction hiding in the gap would be probed by nothing at all. So the gap is
    measured (``‖I − BBᵀ − QQᵀ‖₂``) and charged at the polytope's own diameter, which is the most
    width any direction can have.

    Asserted on a **synthetic** certificate with *material* leakage. On a real one the leakage is
    ~1e-16, which is smaller than the outward-rounding inflation — so a real certificate cannot tell
    a correct ``√k·√(1+ε)`` from a wrong ``√k``, and a test built on one would pass either way.
    """
    material = SpanCertificate(
        exhaustive=True,
        n_probes=9,
        n_complement=9,
        max_width=1e-3,
        max_width_floor=1e-9,
        n_inconclusive=0,
        worst_dual_error=0.0,
        leakage=0.25,  # large enough that √(1 + leakage) = 1.118 is unmistakable
        diameter=4.0,
        complement_is_complete=True,
    )

    with_factor = 3.0 * math.sqrt(1.25) * 1e-3 + 0.25 * 4.0
    without_factor = 3.0 * 1e-3 + 0.25 * 4.0
    assert with_factor != pytest.approx(without_factor)  # the factor is detectable here

    assert material.resolution == pytest.approx(with_factor, rel=1e-12)
    assert material.resolution != pytest.approx(without_factor, rel=1e-12)
    assert material.resolution >= with_factor  # and it is rounded *outward*, never below

    # A real certificate carries the same terms, with the leakage that Gram-Schmidt actually leaves.
    real = build_geometry(two_free_dimensions(), model_id="leak").certificate
    assert real.leakage < 1e-12
    assert real.diameter > 0.0
    assert real.resolution >= math.sqrt(real.n_complement) * real.max_width


def test_the_fva_ranges_are_upper_bounds_not_lp_readings() -> None:
    """A reaction is called blocked because its range is small, so that range must be bounded from
    **above**. Reading it off the primal solutions gives a *lower* bound — the wrong end entirely:
    an LP that stopped short would report zero for a reaction that is wide open, and the projection
    would then delete a real dimension without anything noticing."""
    reduced = triangle()
    blocked = blocked_reactions(reduced, build_flux_lp(reduced), tol=1e-9)

    # Every triangle reaction moves, and the reported upper bounds are real widths, not zeros.
    assert blocked.n_blocked == 0
    assert np.all(blocked.upper >= 10.0 - 1e-6)
