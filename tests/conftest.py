"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gsmm_compiler.flux_polytope import ReducedPolytope

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
