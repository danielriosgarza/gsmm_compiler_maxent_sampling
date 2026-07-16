"""The M9 performance assertions — the five invariants the design rests on.

These are not benchmarks. A benchmark reports a number and a slow machine makes it a bad number; an
**invariant** is a structural claim about the code that is either true or false, and a machine that
is 10× slower does not change the answer. Each one here guards a decision BUILD_PLAN made and that
nothing else in the suite can see:

1. **No HiGHS solve in the MCMC inner loop** (BUILD_PLAN §1.3). The process-global counter is the
   instrument; `freeze()` is the prohibition.
2. **No SciPy in the numerical path.** Covered structurally by `tests/unit/test_no_scipy.py`; here
   it is asserted against a *live sampling run*, which is the thing the convention protects.
3. **No Python loop in the chord.** Structural, because a timing test cannot separate "a Python
   loop" from "a slow machine" — but a per-element *cost* can, and the two orders of magnitude
   between interpreted and vectorized are far wider than any noise.
4. **No element-wise highspy extraction** (the M0 finding: attribute reads return Python `list`s,
   not NumPy views, so element-wise access is a per-element Python round trip). `highs_backend`'s
   own docstring calls this "the M9 assertion" — this is it.
5. **No full reconstruction every step.** ``centre + T·y`` is O(n_free·d); doing it per coordinate
   update would make the incremental cache pointless. Asserted **differentially**, so the claim is
   about the walk and not about however many times the dispersed start happened to retry.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.config import ObjectiveConfig, SamplerConfig
from gsmm_compiler.highs_backend import SolverFrozenError, total_solve_count
from gsmm_compiler.line_geometry import chord_on_support
from gsmm_compiler.maxent_sampler import run_chain
from gsmm_compiler.rounding import build_transform
from gsmm_compiler.sparse_objective import lower_objective, resolve_objective

CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "gsmm_compiler"


@pytest.fixture(scope="module")
def geometry_bundle(example_canonical: Any) -> dict[str, Any]:
    """Geometry + transform + lowered objective for the genome-scale model, built once."""
    reduced = example_canonical.polytope.reduce()
    geometry = build_geometry(reduced, model_id=example_canonical.model_id)
    transform = build_transform(geometry, reduced)
    resolved = resolve_objective(
        example_canonical.polytope, reduced, ObjectiveConfig(l1_penalty_scaled=0.5)
    )
    return {
        "reduced": reduced,
        "geometry": geometry,
        "transform": transform,
        "objective": lower_objective(reduced, resolved.objective).line,
        "model_id": example_canonical.model_id,
    }


# ── 1. No HiGHS solve in the MCMC inner loop ────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("beta", [0.0, 4.0])
def test_sampling_makes_zero_highs_solves(geometry_bundle: dict[str, Any], beta: float) -> None:
    """The process-global solve counter does not move while a chain walks (BUILD_PLAN §1.3).

    Process-global rather than per-program on purpose (M3): a per-instance count could be evaded by
    building a fresh `HighsLinearProgram` inside the loop, which is exactly the regression this
    exists to catch.
    """
    bundle = geometry_bundle
    config = SamplerConfig(n_chains=1, n_samples=60, burn_in=60, refresh_interval=30)

    before = total_solve_count()
    run_chain(
        bundle["transform"],
        bundle["reduced"],
        config=config,
        model_id=bundle["model_id"],
        chain_index=0,
        beta=beta,
        beta_index=0,
        objective=bundle["objective"] if beta > 0 else None,
        energy_scale=32.5,
    )
    assert total_solve_count() == before, (
        f"sampling at β={beta} made {total_solve_count() - before} HiGHS solves; the inner loop "
        "must be solver-free"
    )


def test_a_frozen_program_refuses_to_solve(toy_canonical: Any) -> None:
    """`freeze()` turns the convention into an error, so the loop cannot solve even by mistake."""
    from gsmm_compiler.sparse_objective import build_flux_lp

    program = build_flux_lp(toy_canonical.polytope.reduce())
    program.solve()
    program.freeze()

    with pytest.raises(SolverFrozenError):
        program.solve()


# ── 2. No SciPy, asserted against a live run ────────────────────────────────────────────────────


def test_a_sampling_run_never_imports_scipy_or_cobra(toy_path: Path) -> None:
    """Import the sampler, build a transform, walk a chain — in a fresh interpreter, then look.

    `tests/unit/test_no_scipy.py` proves the *modules* are clean at import. This proves the
    *execution* is: a lazily-imported scipy inside a rarely-taken branch would pass the import scan
    and fail here. The worker path is the one that matters — M8 pins that a worker imports neither
    cobra nor HiGHS, and this pins that walking never reaches for scipy either.
    """
    code = f"""
import sys
import numpy as np
from gsmm_compiler.rounding import build_transform
from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.maxent_sampler import run_chain
from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.model_input import load_canonical_model

canonical = load_canonical_model({str(toy_path)!r})
reduced = canonical.polytope.reduce()
transform = build_transform(build_geometry(reduced, model_id="toy"), reduced)
run_chain(transform, reduced, config=SamplerConfig(n_chains=1, n_samples=20, burn_in=20,
          refresh_interval=10), model_id="toy", chain_index=0)
print("scipy" if "scipy" in {{m.split(".")[0] for m in sys.modules}} else "")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
    check=True)
    assert result.stdout.strip() == "", "a sampling run imported scipy"


# ── 3. No Python loop in the chord ──────────────────────────────────────────────────────────────


def _function_ast(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((CORE_DIR / f"{module}.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module}.{name} not found")


@pytest.mark.parametrize("name", ["feasible_chord", "chord_on_support"])
def test_the_chord_contains_no_python_loop(name: str) -> None:
    """Structural: the chord intersects every bound with array ops, never an interpreted loop.

    The chord runs ``d`` times per sweep over a support of ~199 reactions — ~15M times in a
    production run. A Python loop there would cost ~100 ns per reaction against NumPy's ~1 ns, and
    the honest way to state that requirement is structurally: there is no loop, so there is nothing
    to be slow.
    """
    node = _function_ast("line_geometry", name)
    loops = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.For | ast.While | ast.ListComp | ast.GeneratorExp)
    ]
    assert not loops, f"line_geometry.{name} contains {len(loops)} Python loop(s)"


def test_the_chord_costs_far_less_than_an_interpreted_loop_per_reaction() -> None:
    """Behavioural companion to the structural test, measured as cost **per reaction**.

    Timing alone cannot say "no Python loop" — a slow machine looks like a loop. A *per-element*
    cost can, and the two are not close. Measured on this Jetson: the vectorized chord costs
    **7.6 ns/reaction**, an equivalent interpreted loop **710 ns/reaction** — a **93× gap**. The bar
    sits at 50 ns, which is 6.6× above the real cost (so a throttled machine still passes) and 14×
    below a loop's (so no amount of luck gets one through). Both endpoints were measured before the
    bar was chosen, rather than the bar being picked and the code assumed to clear it.

    The slope across two sizes cancels the fixed call overhead, which at n=200 dominates and would
    otherwise make a perfectly vectorized chord look expensive per element.
    """
    import time

    rng = np.random.default_rng(0)

    def cost(n: int) -> float:
        v = rng.uniform(-1.0, 1.0, n)
        direction = rng.standard_normal(n)
        lower, upper = v - 1.0, v + 1.0
        repeats = 2000
        start = time.perf_counter()
        for _ in range(repeats):
            chord_on_support(v, direction, lower, upper)
        return (time.perf_counter() - start) / repeats

    small, large = 200, 20_000
    per_reaction = (cost(large) - cost(small)) / (large - small)
    assert per_reaction < 50e-9, (
        f"the chord costs {per_reaction * 1e9:.1f} ns per reaction — an interpreted loop's order "
        "(~710 ns on this machine), not an array op's (~7.6 ns)"
    )


# ── 4. No element-wise highspy extraction ───────────────────────────────────────────────────────


def test_highs_solution_extraction_is_one_asarray_shot_per_vector() -> None:
    """Every `getSolution()` field is read exactly once, straight into `np.asarray` (M0 finding).

    highspy's attribute reads return Python **lists**, not NumPy views — the pybind layer copies. So
    ``solution.col_value[i]`` in a loop is a per-element Python round trip over 773 columns, and the
    fix (M3) was to convert each vector in one shot. This asserts the shape of that code rather than
    its speed: a subscript or a loop over a solution field is the regression, whatever it costs.
    """
    node = _function_ast("highs_backend", "solve")

    loops = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.For | ast.While | ast.ListComp | ast.GeneratorExp)
    ]
    assert not loops, f"highs_backend.solve contains {len(loops)} Python loop(s) over the solution"

    # Every `<name>.col_value`-style read of a getSolution() result must be the direct argument of
    # an np.asarray call. Collect the attribute reads that are, then assert none are left over.
    solution_fields = {"col_value", "row_value", "col_dual", "row_dual"}
    wrapped: set[int] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "asarray"
            and child.args
            and isinstance(child.args[0], ast.Attribute)
        ):
            wrapped.add(id(child.args[0]))

    seen: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in solution_fields:
            seen.add(child.attr)
            assert id(child) in wrapped, (
                f"highs_backend.solve reads .{child.attr} outside a single np.asarray call — "
                "highspy returns Python lists, so that is a per-element round trip"
            )

    # Without this the test is vacuous: rename a field upstream and the loop above matches nothing,
    # asserts nothing, and reports success. The M4/M6 lesson — a test that cannot fail on the bug it
    # exists to catch is worse than no test, because it also stops anyone from writing a real one.
    assert seen == solution_fields, (
        f"expected solve() to extract {sorted(solution_fields)} but only found {sorted(seen)} — "
        "this test no longer checks what it claims to"
    )


# ── 5. No full reconstruction every step ────────────────────────────────────────────────────────


def _count_to_flux_calls(
    bundle: dict[str, Any], config: SamplerConfig, monkeypatch: pytest.MonkeyPatch
) -> int:
    from gsmm_compiler.rounding import RoundedTransform

    calls = 0
    original = RoundedTransform.to_flux

    def counting(self: Any, y: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, y)

    monkeypatch.setattr(RoundedTransform, "to_flux", counting)
    run_chain(
        bundle["transform"],
        bundle["reduced"],
        config=config,
        model_id=bundle["model_id"],
        chain_index=0,
        beta=0.0,
        beta_index=0,
    )
    return calls


@pytest.mark.slow
def test_full_reconstruction_happens_per_sample_and_per_refresh_never_per_step(
    geometry_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``centre + T·y`` is rebuilt once per stored sample and once per refresh — not per update.

    Asserted **differentially**: two schedules differing only in ``n_samples`` must differ in
    reconstruction count by exactly the extra samples plus the extra refreshes they imply. A raw
    count would fold in `dispersed_start`'s own reconstructions, whose number depends on how many
    times the start had to shrink to land feasible — real, variable, and not what this is about.

    The absolute bound is asserted too, because the differential test alone would pass a walk that
    reconstructed on every step *and* honoured the schedule on top of it.
    """
    d = geometry_bundle["transform"].dimension

    def schedule(n_samples: int) -> SamplerConfig:
        return SamplerConfig(
            n_chains=1, n_samples=n_samples, burn_in=100, thin=1, refresh_interval=25
        )

    low, high = schedule(100), schedule(200)
    calls_low = _count_to_flux_calls(geometry_bundle, low, monkeypatch)
    calls_high = _count_to_flux_calls(geometry_bundle, high, monkeypatch)

    # 100 extra sweeps → 100 extra stored samples + 100/25 = 4 extra refreshes.
    expected_delta = 100 + 100 // 25
    assert calls_high - calls_low == expected_delta, (
        f"adding 100 sweeps added {calls_high - calls_low} reconstructions, expected "
        f"{expected_delta} (one per stored sample + one per refresh)"
    )

    updates = (low.burn_in + low.n_samples * low.thin) * d
    assert calls_low < updates / 20, (
        f"{calls_low} reconstructions over {updates} coordinate updates — the incremental cache is "
        "not doing its job"
    )
