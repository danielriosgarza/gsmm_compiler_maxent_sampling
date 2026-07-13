"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_MODEL = (
    REPO_ROOT
    / "models"
    / "GCF_000010425_1_ASM1042v1_protein_non_gapfilled_latest_gapfilled_noO2.json"
)
"""Bifidobacterium adolescentis ATCC 15703, anaerobic medium (see CLAUDE.md)."""


@pytest.fixture(scope="session")
def example_model_path() -> Path:
    if not EXAMPLE_MODEL.is_file():
        pytest.skip(f"example model not present: {EXAMPLE_MODEL}")
    return EXAMPLE_MODEL


@pytest.fixture(scope="session")
def example_model(example_model_path: Path):  # type: ignore[no-untyped-def]
    from gsmm_compiler.model_input import load_model

    return load_model(example_model_path)
