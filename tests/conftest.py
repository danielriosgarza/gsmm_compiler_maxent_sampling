"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope

if TYPE_CHECKING:
    from gsmm_compiler.model_input import CanonicalModel

REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_MODEL = (
    REPO_ROOT
    / "models"
    / "GCF_000010425_1_ASM1042v1_protein_non_gapfilled_latest_gapfilled_noO2.json"
)
"""Bifidobacterium adolescentis ATCC 15703, anaerobic medium (see CLAUDE.md)."""

TOY_MODEL = REPO_ROOT / "examples" / "toy_network.json"
TOY_CONFIG = REPO_ROOT / "examples" / "toy_config.toml"


@pytest.fixture(scope="session")
def example_model_path() -> Path:
    if not EXAMPLE_MODEL.is_file():
        pytest.skip(f"example model not present: {EXAMPLE_MODEL}")
    return EXAMPLE_MODEL


@pytest.fixture(scope="session")
def example_model(example_model_path: Path):  # type: ignore[no-untyped-def]
    from gsmm_compiler.model_input import load_model

    return load_model(example_model_path)


@pytest.fixture(scope="session")
def example_canonical(example_model_path: Path) -> CanonicalModel:
    from gsmm_compiler.model_input import load_canonical_model

    return load_canonical_model(example_model_path)


@pytest.fixture(scope="session")
def toy_path() -> Path:
    return TOY_MODEL


@pytest.fixture(scope="session")
def toy_config_path() -> Path:
    return TOY_CONFIG


@pytest.fixture(scope="session")
def toy_canonical(toy_path: Path) -> CanonicalModel:
    """The toy network: 7 reactions, 3 metabolites, one reaction fixed at a **nonzero** value."""
    from gsmm_compiler.model_input import load_canonical_model

    return load_canonical_model(toy_path)


# ---- synthetic polytopes whose uniform law is known on paper (M5) ------------------------------


def dense_polytope(
    stoichiometry: list[list[float]],
    lower: list[float],
    upper: list[float],
    rhs: list[float] | None = None,
    biomass_index: int = 0,
) -> ReducedPolytope:
    """A `ReducedPolytope` straight from a dense ``S`` — every reaction free, nothing fixed.

    The genome-scale model can tell us the sampler *runs*; only a polytope whose exact uniform law
    is derivable on paper can tell us it samples the **right** law. These are those polytopes.
    """
    from gsmm_compiler.native_csc import NativeCSC

    matrix = np.asarray(stoichiometry, dtype=np.float64)
    n_metabolites, n_reactions = matrix.shape

    return ReducedPolytope(
        reaction_ids=tuple(f"r{i}" for i in range(n_reactions)),
        metabolite_ids=tuple(f"m{i}" for i in range(n_metabolites)),
        free_indices=np.arange(n_reactions, dtype=np.intp),
        fixed_indices=np.empty(0, dtype=np.intp),
        fixed_values=np.empty(0, dtype=np.float64),
        stoichiometry=NativeCSC.from_dense(matrix),
        rhs=np.asarray(rhs if rhs is not None else [0.0] * n_metabolites, dtype=np.float64),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        biomass_index=biomass_index,
        biomass_full_index=biomass_index,  # nothing is fixed here, so the two coincide
        n_full=n_reactions,
    )


@pytest.fixture(scope="session")
def simplex_polytope() -> ReducedPolytope:
    """``{x + y + z = 1, 0 ≤ x,y,z ≤ 1}`` — the 2-simplex, ``d = 2``.

    The load-bearing analytic target, because its uniform marginal is **not uniform**: the
    density of ``x`` is proportional to the length of ``{y + z = 1 − x, y,z ≥ 0}``, so

        f(x) = 2(1 − x)        F(x) = 1 − (1 − x)²        on [0, 1].

    A sampler that quietly returned a uniform marginal — the failure a box cannot detect — is caught
    here by a KS test against that CDF.
    """
    return dense_polytope(
        stoichiometry=[[1.0, 1.0, 1.0]],
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        rhs=[1.0],
    )


@pytest.fixture(scope="session")
def coupled_box_polytope() -> ReducedPolytope:
    """``{v0 = v1 ∈ [0, 2], v2 ∈ [−1, 3]}`` — a 2-D box behind a real equality, ``d = 2``.

    Uniform on the polytope means uniform in each of ``v0`` and ``v2`` *independently*, so both the
    marginals and their independence are checkable. The equality is not decorative: it forces the
    sampler to move on the affine hull rather than in the raw coordinates.
    """
    return dense_polytope(
        stoichiometry=[[1.0, -1.0, 0.0]],
        lower=[0.0, 0.0, -1.0],
        upper=[2.0, 2.0, 3.0],
    )


@pytest.fixture(scope="session")
def anisotropic_polytope() -> ReducedPolytope:
    """The same shape, stretched 1000:1 — what rounding exists for, and a trap without it.

    Unrounded, the basis axes are near-useless here: a coordinate step along the long direction is
    the same size as one along the short direction, so every chord in one of them is clipped almost
    immediately. The marginals are still exactly uniform, so the target is unchanged and only the
    *mixing* is at stake — which is precisely the claim `rounding` makes about itself.
    """
    return dense_polytope(
        stoichiometry=[[1.0, -1.0, 0.0]],
        lower=[0.0, 0.0, 0.0],
        upper=[1000.0, 1000.0, 1.0],
    )


@pytest.fixture(scope="session")
def singleton_polytope() -> ReducedPolytope:
    """``{v0 = v1 = 0}`` with a mass balance that pins everything — ``d = 0`` (spec §16)."""
    return dense_polytope(
        stoichiometry=[[1.0, 1.0], [1.0, -1.0]],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
    )


# ---- synthetic polytopes whose TILTED law is known on paper (M6) --------------------------------
#
# The M5 fixtures above give a ReducedPolytope directly. M6's need a `FluxPolytope` as well, because
# the objective is built by `SparseFluxObjective.from_polytope` and lowered by `lower_objective` —
# the production path — so the statistical gate exercises the real lowering rather than an
# `L1Objective` the test wrote out by hand with the indices it expected.


def dense_flux_polytope(
    stoichiometry: list[list[float]],
    lower: list[float],
    upper: list[float],
    biomass_index: int = 0,
) -> FluxPolytope:
    """A `FluxPolytope` from a dense ``S``. ``.reduce()`` yields the matching `ReducedPolytope`."""
    from gsmm_compiler.native_csc import NativeCSC

    matrix = np.asarray(stoichiometry, dtype=np.float64)
    n_metabolites, n_reactions = matrix.shape

    return FluxPolytope(
        reaction_ids=tuple(f"r{i}" for i in range(n_reactions)),
        metabolite_ids=tuple(f"m{i}" for i in range(n_metabolites)),
        stoichiometry=NativeCSC.from_dense(matrix),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        biomass_index=biomass_index,
    )


@pytest.fixture(scope="session")
def line_flux_polytope() -> FluxPolytope:
    """``{v0 = v1 ∈ [0, 1]}`` — ``d = 1``, biomass ``v0``, and ``J = v0`` when ``λ = 0``.

    The simplest possible tilted target: ``π ∝ e^{κ v0}`` on ``[0, 1]``, a **truncated exponential**
    whose CDF is ``expm1(κx)/expm1(κ)``. No breakpoints at all, so it isolates the exponential draw
    from the piecewise machinery — and it stays exact at ``κ = 1000``, which is the large-β stress.

    It is also the one polytope where ``λ = 0`` diverges the two penalty sets: ``C(v) = |v1|`` is a
    real number the run must report, while *nothing* bends ``J``. A `ReducedObjective` that reused
    one index set for both would report ``C = 0`` here.
    """
    return dense_flux_polytope(
        stoichiometry=[[1.0, -1.0]],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        biomass_index=0,
    )


@pytest.fixture(scope="session")
def laplace_box_flux_polytope() -> FluxPolytope:
    """``{v2 = v0 + v1, v0,v1 ∈ [−1,1]}`` — ``d = 2``, biomass ``v2``, penalty on ``v0, v1``.

    ``J(v) = v2 − λ(|v0| + |v1|) = Σᵢ (vᵢ − λ|vᵢ|)`` over ``i ∈ {0, 1}``, so at ``λ = 2`` the target
    factorizes into two identical **asymmetric truncated Laplaces** on ``[−1, 1]``:

        g(x) = 3x   for x < 0        (slope +3κ)
        g(x) = −x   for x ≥ 0        (slope −1κ)

    This is the M6 gate's load-bearing target. Its bend at ``x = 0`` is *strictly interior* to
    almost
    every chord, so unlike the simplex it genuinely exercises `build_piecewise_j`, the segment
    log-masses and the categorical choice — and the two slopes differ by 3×, so a sampler that
    symmetrized the Laplace, or picked the segment by the wrong mass, lands somewhere the KS test
    can
    see. Being a product law, it also makes the two coordinates' independence checkable.
    """
    return dense_flux_polytope(
        stoichiometry=[[1.0, 1.0, -1.0]],
        lower=[-1.0, -1.0, -2.0],
        upper=[1.0, 1.0, 2.0],
        biomass_index=2,
    )


@pytest.fixture(scope="session")
def simplex_flux_polytope() -> FluxPolytope:
    """The 2-simplex again, now with biomass ``v0`` and a penalty on the rest — a *coupled* tilt.

    A `FluxPolytope`'s mass balance is homogeneous (``S·v = 0``), so the simplex's ``x + y + z = 1``
    can only arrive the way a real model would produce it: through a **fixed reaction**. ``SRC`` is
    pinned at 1.0 and the row reads ``x + y + z − SRC = 0``, which `reduce()` turns into the affine
    ``rhs = 1`` over the three free reactions. That is not a workaround — it is the case worth
    testing, because ``SRC`` is *penalized* too, so ``cost_offset = 1.0`` and `lower_objective` has
    a
    nonzero fixed-flux constant to get right. Every fixed reaction of the example model sits at
    zero,
    so it is the only polytope here that can catch dropping it.

    On the simplex every flux is nonnegative and biomass is out of the penalty set, so
    ``C = (y + z) + |SRC| = (1 − x) + 1 = 2 − x`` and

        J = x − λ(2 − x) = (1 + λ)x − 2λ,

    a linear function of ``x`` alone (the constant cancels out of the target, as constants must).
    The
    tilted law is therefore ``uniform-on-simplex × e^{γx}`` with ``γ = κ(1 + λ)``, whose marginal is

        f(x) ∝ (1 − x)·e^{γx}        on [0, 1],

    the ``(1 − x)`` coming from the *geometry* (the length of the slice ``{y + z = 1 − x}``) and the
    exponential from the *objective*. Neither factor alone is the answer, so this catches a sampler
    that gets the tilt right on a shape it gets wrong, or the reverse — M5's simplex test and M6's
    line test each check only one of those halves.
    """
    return dense_flux_polytope(
        stoichiometry=[[1.0, 1.0, 1.0, -1.0]],
        lower=[0.0, 0.0, 0.0, 1.0],
        upper=[1.0, 1.0, 1.0, 1.0],
        biomass_index=0,
    )


def synthetic_optimum(objective, j_star: float, polytope_key: str | None = None):  # type: ignore[no-untyped-def]
    """An `LPOptimum` carrying a chosen ``J*``, **keyed to `objective`** (M7).

    `choose_energy_scale` takes a keyed `LPOptimum` rather than a bare ``float`` precisely so that a
    ``J*`` cannot drift across an objective *or a polytope* boundary — ``s_J = J* − Q_q(J(W))`` is
    only a *range* if both ends are the same ``J`` on the same polytope, and M7 is the first
    milestone where two objectives share a polytope (and where two polytopes can share an objective
    key — Codex, round 2).

    Tests that exercise the *arithmetic* of ``s_J`` (the ULP floor, the quantile, the
    additive-constant invariance) still want to name a ``J*`` outright. They may — but they must now
    also say which objective *and polytope* it came from, which is the whole discipline in one line.
    Defaults the polytope key to the objective's own, so a plain call keys them consistently.
    """
    from gsmm_compiler.sparse_objective import LPOptimum, ObjectiveValue

    return LPOptimum(
        v_full=np.zeros(0, dtype=np.float64),
        z=np.zeros(0, dtype=np.float64),
        value=ObjectiveValue(mu=0.0, cost=0.0, total=float(j_star)),
        solver_objective=float(j_star),
        max_z_deviation=0.0,
        max_mass_balance_residual=0.0,
        max_bound_violation=0.0,
        simplex_iterations=0,
        elapsed_seconds=0.0,
        objective_key=objective.objective_key,
        polytope_key=objective.polytope_key if polytope_key is None else polytope_key,
    )


@pytest.fixture(scope="session")
def pinned_nonzero_polytope() -> ReducedPolytope:
    """``{v0 = 1 (0 ≤ v0 ≤ 2), v1 = v2 ∈ [0, 1]}`` — an **immovable reaction at a nonzero value**.

    The minimal counterexample to "immovable ⇒ near zero", and it took Codex two rounds to find it.

    ``v0``'s bounds are ``[0, 2]``, so `FluxPolytope.reduce` leaves it *free* — but the mass balance
    row ``v0 = 1`` pins it anyway. FVA therefore gives it zero range, M4 projects it out of the
    basis (BUILD_PLAN §1.4.1), its row of ``T`` is exactly zero, and `movable_reactions` correctly
    reports that the chain cannot move it.

    **But its flux is 1.0, not 0.** Every one of the example model's 61 immovable reactions happens
    to sit at ~1e-13, so *cannot move* and *is at zero* are indistinguishable there, and an M6
    assertion quietly conflated them. Here they come apart.
    """
    return dense_polytope(
        stoichiometry=[[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
        lower=[0.0, 0.0, 0.0],
        upper=[2.0, 1.0, 1.0],
        rhs=[1.0, 0.0],
        biomass_index=1,
    )
