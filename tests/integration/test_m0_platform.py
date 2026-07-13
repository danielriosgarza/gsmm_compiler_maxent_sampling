"""M0 platform spike: prove the aarch64/Python-3.11 stack does what the design assumes.

Namely: cobra parses the example GSMM, a flux-balance LP assembled from **native NumPy CSC arrays**
(no ``scipy.sparse``) is accepted by ``highspy.Highs.passModel``, and it solves to the same optimum
cobra's own solver reports. That last cross-check is what makes the CSC assembly trustworthy rather
than merely accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

BIOMASS_ID = "bio1"


def _build_native_csc(model):  # type: ignore[no-untyped-def]
    """Assemble the stoichiometric matrix S as column-wise (CSC) NumPy arrays.

    Columns are reactions in model order, rows are metabolites in model order. Returns
    ``(n_rows, n_cols, starts, indices, values)`` with the integer widths highspy expects.
    """
    metabolite_index = {m.id: i for i, m in enumerate(model.metabolites)}

    starts = np.zeros(len(model.reactions) + 1, dtype=np.int32)
    indices: list[int] = []
    values: list[float] = []
    for j, reaction in enumerate(model.reactions):
        # Sort by row index: HiGHS does not require it, but it makes the arrays canonical.
        column = sorted(
            (metabolite_index[m.id], coeff) for m, coeff in reaction.metabolites.items()
        )
        for row, coeff in column:
            indices.append(row)
            values.append(coeff)
        starts[j + 1] = len(indices)

    return (
        len(model.metabolites),
        len(model.reactions),
        starts,
        np.asarray(indices, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )


def test_example_model_loads(example_model) -> None:  # type: ignore[no-untyped-def]
    """The example model parses with the counts BUILD_PLAN §0 was written against."""
    from gsmm_compiler.model_input import summarize

    summary = summarize(example_model, "models/example.json")
    assert summary.n_reactions == 773
    assert summary.n_metabolites == 894
    assert summary.n_fixed == 513
    assert summary.n_free == 260
    assert summary.n_infinite_bounds == 0
    assert summary.objective_reaction_ids == (BIOMASS_ID,)


def test_native_csc_lp_solves_and_matches_cobra(example_model) -> None:  # type: ignore[no-untyped-def]
    """A biomass-maximizing LP built from native CSC arrays solves, and agrees with cobra's FBA."""
    import highspy

    n_rows, n_cols, starts, indices, values = _build_native_csc(example_model)
    assert starts[-1] == indices.size == values.size
    assert indices.max() < n_rows

    cost = np.array(
        [1.0 if r.id == BIOMASS_ID else 0.0 for r in example_model.reactions],
        dtype=np.float64,
    )
    col_lower = np.array([r.lower_bound for r in example_model.reactions], dtype=np.float64)
    col_upper = np.array([r.upper_bound for r in example_model.reactions], dtype=np.float64)

    lp = highspy.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows
    lp.col_cost_ = cost
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    # Steady state: S v = 0, so every row is an equality at zero.
    lp.row_lower_ = np.zeros(n_rows, dtype=np.float64)
    lp.row_upper_ = np.zeros(n_rows, dtype=np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = values
    lp.sense_ = highspy.ObjSense.kMaximize

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("threads", 1)
    assert highs.passModel(lp) == highspy.HighsStatus.kOk

    assert highs.run() == highspy.HighsStatus.kOk
    assert highs.getModelStatus() == highspy.HighsModelStatus.kOptimal

    info = highs.getInfo()
    native_objective = info.objective_function_value
    assert native_objective > 0.0, "anaerobic Bifido model should grow"

    solution = np.asarray(highs.getSolution().col_value, dtype=np.float64)
    assert solution.size == n_cols

    # The solution really is a steady-state flux vector, mass-balanced and within bounds.
    residual = np.zeros(n_rows, dtype=np.float64)
    np.add.at(
        residual,
        indices,
        values * np.repeat(solution, np.diff(starts)),
    )
    assert np.max(np.abs(residual)) < 1e-6
    assert np.all(solution >= col_lower - 1e-9)
    assert np.all(solution <= col_upper + 1e-9)

    cobra_objective = example_model.optimize().objective_value
    assert native_objective == pytest.approx(cobra_objective, rel=1e-6)


def test_highspy_returns_python_lists_not_arrays(example_model) -> None:  # type: ignore[no-untyped-def]
    """M0 finding, pinned so M3 cannot forget it.

    Reading ``HighsLp``/solution attributes back out of highspy yields Python **lists**, not NumPy
    views: the pybind layer copies. So the LP layer must (a) keep its own float64 arrays rather than
    round-tripping through highspy, and (b) extract a solution in one ``np.asarray`` shot instead of
    indexing element-wise (the M9 assertion).
    """
    import highspy

    lp = highspy.HighsLp()
    lp.col_lower_ = np.zeros(3, dtype=np.float64)

    assert isinstance(lp.col_lower_, list)
    assert not isinstance(lp.col_lower_, np.ndarray)


def test_multiprocessing_workers_inherit_thread_limits() -> None:
    """BUILD_PLAN §1.2: MCMC workers must run single-threaded BLAS and never import cobra/HiGHS."""
    import multiprocessing as mp

    context = mp.get_context("spawn")
    with context.Pool(processes=2, initializer=_worker_init) as pool:
        results = pool.map(_worker_probe, [3, 4])

    for imported_modules, thread_env in results:
        assert "cobra" not in imported_modules
        assert "highspy" not in imported_modules
        assert "scipy" not in imported_modules
        assert thread_env == dict.fromkeys(THREAD_ENV_VARS, "1")

    # A spawned worker really did do the numpy work, rather than returning an empty probe.
    assert "numpy" in results[0][0]


THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _worker_init() -> None:
    import os

    for var in THREAD_ENV_VARS:
        os.environ[var] = "1"


def _worker_probe(seed: int) -> tuple[frozenset[str], dict[str, str]]:
    import os
    import sys

    import numpy as np  # the only heavy import a worker is allowed

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    _ = rng.standard_normal(8) @ rng.standard_normal(8)

    thread_env = {var: os.environ.get(var, "unset") for var in THREAD_ENV_VARS}
    return frozenset(sys.modules), thread_env
