"""The no-SciPy gate (BUILD_PLAN §4 / CLAUDE.md conventions).

Two independent checks, because each catches what the other misses:

* a **runtime** check — import the numerical core in a fresh interpreter and assert nothing pulled
  ``scipy`` (or the cobra/optlang parser stack) into ``sys.modules``;
* a **static** check — no core module names scipy at all, so an import hidden inside a function body
  cannot slip past the runtime check.

These start nearly vacuous (M0 modules are stubs) and gain teeth as the core fills in. That is the
point: the gate must already be in place when the first array code lands.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "gsmm_compiler"

NUMERICAL_CORE = (
    "native_csc",
    "flux_polytope",
    "highs_backend",
    "sparse_objective",
    "affine_geometry",
    "rounding",
    "line_geometry",
    "line_distribution",
    "maxent_sampler",
    "diagnostics",
    "output",
    "features",
    "logging_utils",
)
"""Modules a sampling worker may import. cobra and scipy are banned from all of them."""

FORBIDDEN_ROOTS = ("scipy", "cobra", "optlang", "sympy", "pandas")
"""``scipy``/``cobra`` are the conventions; the rest come with cobra and would betray a leak."""


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_numerical_core_imports_no_scipy_or_cobra() -> None:
    """Importing every core module in a fresh interpreter pulls in no forbidden package."""
    code = (
        "import importlib, sys\n"
        f"for name in {NUMERICAL_CORE!r}:\n"
        "    importlib.import_module('gsmm_compiler.' + name)\n"
        "roots = {m.split('.')[0] for m in sys.modules}\n"
        f"print(','.join(sorted(roots & set({FORBIDDEN_ROOTS!r}))))\n"
    )
    leaked = _run(code)
    assert leaked == "", f"numerical core leaked forbidden imports: {leaked}"


def test_highspy_import_pulls_no_scipy() -> None:
    """The LP backend's dependency is itself scipy-free — the M0 platform claim."""
    leaked = _run(
        "import sys, highspy, numpy\n"
        "print('scipy' if 'scipy' in {m.split('.')[0] for m in sys.modules} else '')\n"
    )
    assert leaked == "", "importing highspy pulled in scipy"


def test_scipy_is_absent_from_the_environment() -> None:
    """Empirical M0 result: the whole dependency graph resolves without scipy at all.

    Not an assumption the design rests on — the core is tested scipy-free above regardless — but if
    this ever starts failing, a dependency began shipping scipy and the tree is worth re-reading.
    """
    with pytest.raises(ImportError):
        __import__("scipy")


@pytest.mark.parametrize("module_name", NUMERICAL_CORE)
def test_core_module_source_never_names_scipy(module_name: str) -> None:
    """Static scan: catches a lazily-imported ``import scipy`` inside a function body."""
    tree = ast.parse((CORE_DIR / f"{module_name}.py").read_text())

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)

    offenders = sorted({name for name in imported if name.split(".")[0] in FORBIDDEN_ROOTS})
    assert not offenders, f"{module_name}.py imports {offenders}"
