"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
