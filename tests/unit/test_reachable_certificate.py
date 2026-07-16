"""The M9 reachable-state mass-balance certificate — the gate that replaced M5's ‖S·T‖ bar.

**Why this exists.** M5 gated on `RoundingDiagnostics.transform_mass_balance_error`, a per-direction
componentwise backward error. M9 measured it across RNG streams and found it rejects **8 of 24**
streams on a polytope that is perfectly samplable — the `model_id` *string* decided whether a
genome-scale model could be sampled, because the residual it divides is an inherited absolute floor
and some rows' cancellation scales are ~1e-5.

The replacement asks the operative question — *can the chain reach a state that violates mass
balance?* — and the load-bearing test is the first one: **the certificate must be able to refuse.**
A gate that certifies everything is not a gate, and this repository has twice shipped a test that
could not fail on the bug it existed to catch (M4's truncated basis, M6's `s_J` floor).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from tests.conftest import dense_polytope

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.rounding import (
    DEFAULT_MASS_BALANCE_CONTRACT,
    RoundingError,
    build_transform,
    certify_reachable_mass_balance,
)


def _dilution_polytope() -> Any:
    """Codex's counterexample polytope: two huge reactions and one tiny one.

    ``S = [[1,-1,0],[0,0,1]]``, ``|v₁|,|v₂| ≤ 1e12``, ``|v₃| ≤ 1``. Steady state forces ``v₁ = v₂``
    and ``v₃ = 0``, so the true direction space is one-dimensional: ``(1, 1, 0)``.

    The 1e12-to-1 bound ratio is the whole point. It makes the chord along a corrupted direction
    enormous, so an *infinitesimal* off-manifold component in ``T`` — 1e-10, far below any tolerance
    anyone would write down — reaches a mass-balance violation of order **1** at a perfectly
    bound-feasible interior point.
    """
    return dense_polytope(
        stoichiometry=[[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        lower=[-1e12, -1e12, -1.0],
        upper=[1e12, 1e12, 1.0],
    )


def test_the_certificate_refuses_a_transform_that_a_per_direction_bar_admits() -> None:
    """**The load-bearing test.** A T that the withdrawn per-column formula passes is refused.

    On this polytope a corrupted column ``T_k = (1, 1, 1e-10)`` gives ``S·T_k = (0, 1e-10)`` and
    ``|S|·|T_k| = (2, 1e-10)``. The per-*column* ratio of norms — the fix M9 first proposed —
    reports
    ``1e-10 / 2 = 5e-11``, comfortably inside any sane tolerance, because it divides by the
    *largest* row scale and lets the two huge reactions hide the violation on the tiny one.

    But ``v₃`` binds the chord at ``|y| ≲ 1e10``, so the reachable ``v₃`` runs to **1.0** — a
    mass-balance residual of order 1 with every reaction bound satisfied. The certificate sees it,
    because it asks about reachable *states* rather than unit *directions*.
    """
    reduced = _dilution_polytope()
    geometry = build_geometry(reduced, model_id="dilution")
    honest = build_transform(geometry, reduced)

    corrupt_column = np.array([[1.0], [1.0], [1e-10]], dtype=np.float64)
    corrupt = dataclasses.replace(honest, transform=corrupt_column)

    # The withdrawn formula's verdict, computed here so the test states the contrast rather than
    # asserting it on faith.
    residual = np.abs(reduced.stoichiometry.matvec(corrupt_column[:, 0]))
    scale = reduced.stoichiometry.cancellation_scale(corrupt_column[:, 0])
    per_column_ratio = residual.max() / scale.max()
    assert per_column_ratio < DEFAULT_MASS_BALANCE_CONTRACT, (
        "this counterexample only means something if the per-column formula ADMITS the transform"
    )

    certificate = certify_reachable_mass_balance(corrupt, reduced)
    assert not certificate.is_certified
    assert certificate.worst_absolute > 0.1, (
        f"the reachable violation is {certificate.worst_absolute:.3e}; the chord along this column "
        "runs to |y| ~ 1e10 against a 1e-10 leak, so it must reach order 1"
    )
    assert certificate.worst_row_id == "m1"


def test_the_certificate_admits_the_honest_transform_of_the_same_polytope() -> None:
    """The control for the test above: same polytope, uncorrupted T, must certify.

    Without this the refusal proves nothing — a certificate that refuses everything would pass it.

    It certifies with **zero LPs**, which is correct rather than a shortcut: the geometry projects
    the blocked ``v₃`` out exactly, so the honest ``T = (t, t, 0)`` gives ``S·T = 0`` *exactly* and
    every ``E_i`` is structurally zero. There is no reachable state to search over — the direction
    space provably never touches either metabolite — and the centre's own residual is the whole
    bound. A perfectly clean transform is the cheapest thing this certificate can see.
    """
    reduced = _dilution_polytope()
    transform = build_transform(build_geometry(reduced, model_id="dilution"), reduced)

    certificate = certify_reachable_mass_balance(transform, reduced)
    assert certificate.is_certified
    assert certificate.worst_absolute <= DEFAULT_MASS_BALANCE_CONTRACT
    assert certificate.n_lps == 0
    assert certificate.n_rows_certified == 0


def test_the_certified_bound_is_sound_rather_than_tight() -> None:
    """The bound must be **≥** the true reachable extreme. It is not required to equal it.

    The exact reachable extreme here is 1.0: ``v₃``'s bound gives ``|y| ≤ 1/1e-10 = 1e10``, and
    ``E₂·y = 1e-10 · 1e10 = 1``. A sound upper bound is anything ``≥ 1.0``, and this one comes back
    larger — which is the correct kind of wrong.

    **Two independent reasons it is loose, both worth knowing.** First, `_reachable_extreme` returns
    a weak-duality bound, not a primal reading, and weak duality is exact only at a perfectly solved
    optimum. Second — measured directly — **HiGHS drops the 1e-10 row from the scaled matrix
    entirely**: it reports that row's activity as 0.0 where the truth is 133.3, with
    ``max_primal_infeasibility = 0.0``. A coefficient that small beside 1.0 coefficients does not
    survive HiGHS's matrix scaling. That *relaxes* ``Y``, which can only enlarge the maximum.

    Both errors run outward, so the certificate stays sound. That is not luck but it is also not a
    guarantee anyone should lean on: relying on which way a solver's scaling errs is not a proof,
    which is exactly why the bound comes from weak duality — it holds whatever HiGHS returns.
    """
    reduced = _dilution_polytope()
    honest = build_transform(build_geometry(reduced, model_id="dilution"), reduced)
    corrupt = dataclasses.replace(
        honest, transform=np.array([[1.0], [1.0], [1e-10]], dtype=np.float64)
    )

    certificate = certify_reachable_mass_balance(corrupt, reduced)
    assert not certificate.is_certified
    assert certificate.worst_absolute >= 1.0, (
        f"the bound {certificate.worst_absolute:.3e} is BELOW the true reachable extreme of 1.0 — "
        "an upper bound that can be too small is not a bound"
    )


def test_the_certificate_deliberately_misses_a_structural_corruption_that_the_diagnostic_catches(
) -> None:
    """The two instruments answer different questions, and this documents the boundary.

    Codex's case: ``S = [1]``, one reaction bounded ``|v| ≤ δ`` with ``δ = 1e-12``. The true
    equality-constrained polytope has dimension **zero** — ``v`` must be 0 — but a corrupted ``T``
    invents motion along it. The reachable residual is only 1e-12, so the certificate passes: no
    state it can reach violates the contract, which is *exactly what the certificate claims* and it
    is true. The componentwise backward error reports ``|S·T| / (|S|·|T|) = 1`` and exposes the
    invented direction.

    That is why `transform_mass_balance_error` survives as a reported diagnostic rather than being
    deleted: it tests a structural invariant independent of reachable amplitude. It simply must not
    be a *gate*, because it also fires on rounding artifacts.
    """
    from gsmm_compiler.rounding import _transform_mass_balance

    delta = 1e-12
    reduced = dense_polytope(stoichiometry=[[1.0]], lower=[-delta], upper=[delta])
    invented = np.array([[delta]], dtype=np.float64)

    relative, absolute = _transform_mass_balance(invented, reduced)
    assert relative == pytest.approx(1.0), "the diagnostic must expose the invented direction"
    assert absolute == pytest.approx(delta)
    # And the reachable residual really is negligible — the certificate is not wrong, it is
    # answering the other question.
    assert delta < DEFAULT_MASS_BALANCE_CONTRACT


def test_the_certificate_refuses_a_foreign_polytope(
    toy_canonical: Any, simplex_polytope: Any
) -> None:
    """The M6 invariant at one more join: T and S must have been computed against each other.

    A bound proved against the wrong polytope is not a weaker bound — it is a statement about a
    different model, and it would come back looking exactly as reassuring.
    """
    reduced = toy_canonical.polytope.reduce()
    transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)

    with pytest.raises(RoundingError, match="not built from"):
        certify_reachable_mass_balance(transform, simplex_polytope)


def test_the_certificate_refuses_a_nonpositive_contract(toy_canonical: Any) -> None:
    reduced = toy_canonical.polytope.reduce()
    transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)

    with pytest.raises(RoundingError, match="contract must be positive"):
        certify_reachable_mass_balance(transform, reduced, contract=0.0)


def test_a_singleton_polytope_needs_no_lps(singleton_polytope: Any) -> None:
    """d = 0 reaches exactly one state, so the centre's own residual is the whole bound."""
    transform = build_transform(
        build_geometry(singleton_polytope, model_id="singleton"), singleton_polytope
    )
    certificate = certify_reachable_mass_balance(transform, singleton_polytope)

    assert certificate.n_lps == 0
    assert certificate.is_certified


def test_the_toy_model_certifies(toy_canonical: Any) -> None:
    reduced = toy_canonical.polytope.reduce()
    transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)

    certificate = certify_reachable_mass_balance(transform, reduced)
    assert certificate.is_certified
    assert certificate.margin > 1.0
    assert certificate.polytope_key == reduced.content_key()
    assert set(certificate.as_dict()) >= {
        "reachable_worst_absolute",
        "reachable_is_certified",
        "reachable_margin",
        "reachable_n_lps",
    }
