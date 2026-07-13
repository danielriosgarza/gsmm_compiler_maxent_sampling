"""The sparse biomass objective ``J(v) = v_b − λ Σ_r w_r |v_r|`` and the LPs that maximize it.

Three LPs live here, all over the **reduced** polytope (the ``l == u`` reactions are gone, §1.5):

* `build_sparse_objective_lp` — the ``(v, z)`` linearization of §3.3. Auxiliary ``z_r ≥ ±v_r``
  carrying a negative objective coefficient, so ``z_r = |v_r|`` at any optimum.
* `build_flux_lp` — flux variables only (§14). Geometry re-solves this one under hundreds of
  objectives; it never sees a ``z``.
* `biomass_maximum` — the ``max v_b`` diagnostic (§12), which says how much growth the sparse
  objective gave up.

**z is LP-only** (spec §3.4). The auxiliaries linearize an absolute value; they are not part of the
flux vector and must never enter the sampled state, or every flux would acquire artificial volume
from the ``z`` values it could have been paired with. Nothing here returns a ``z`` inside a flux
vector; `LPOptimum` keeps them in a separate field, for the ``z == |v|`` check and nothing else.

**The objective constant is J at the fixed fluxes.** The eliminated reactions still contribute to
``J`` — a forced ATP maintenance demand really does cost L1 — but they are no longer LP variables.
That contribution is exactly ``J(c)``, where ``c`` is the fixed-flux vector: the supports of the
free and fixed reactions are disjoint, so ``J(v_full) = J(c) + [the free part]`` with no
cross-terms. It rides into HiGHS as the objective offset, so the solver reports the true ``J*`` and
the gate check "solver objective == directly recomputed J" is one complete equation.

**Direct CSC assembly** (§12): the expanded matrix is written straight into ``starts``/``indices``/
``values``, never stacked from blocks. There is no SciPy here to stack with, but the better reason
is that a block form would materialize a copy of ``S`` and a ±1 identity pair for a matrix whose
columns can simply be written down.

Implemented in **M3** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope
from gsmm_compiler.highs_backend import HighsLinearProgram, LPNotOptimalError, LPSolution
from gsmm_compiler.native_csc import INDEX_DTYPE, VALUE_DTYPE, NativeCSC
from gsmm_compiler.provenance import content_key

OBJECTIVE_IMPL_VERSION: Final = 1
"""Feeds the L2 cache key. Bump when a change here can move ``J*`` or ``v*`` (§1.1)."""

DEFAULT_Z_TOL: Final = 1e-7
"""How far ``z`` may sit from ``|v|`` at the optimum before we refuse the solution (§12 check 6)."""

DEFAULT_OBJECTIVE_TOL: Final = 1e-6
"""Relative agreement required between HiGHS's objective and a recomputed ``J(v*)`` (§12 ch. 8)."""

DEFAULT_LP_FEASIBILITY_TOL: Final = 1e-7
"""Mass-balance and bound slack tolerated in a returned optimum (§12 checks 3–5)."""


class ObjectiveError(ValueError):
    """The objective is malformed, or an LP was asked of a polytope that cannot supply one."""


class LPCheckError(RuntimeError):
    """A solved LP failed one of the §12 result checks. The solution is not trusted."""


# ---- the objective ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveValue:
    """``J`` and its two components, always reported separately (spec §3.2)."""

    mu: float
    """``μ(v) = v_b`` — biomass flux."""
    cost: float
    """``C(v) = Σ_r w_r |v_r|`` — the weighted L1 cost, *before* λ."""
    total: float
    """``J(v) = μ − λ C``."""


@dataclass(frozen=True)
class SparseFluxObjective:
    """``J(v) = v_b − λ Σ_{r ∈ R_p} w_r |v_r|`` over the **full** reaction set.

    Defined on full-model coordinates, because that is what the manifest must record and what M7's
    reweighting updates. It is *lowered* onto a `ReducedPolytope` when an LP is built.
    """

    reaction_ids: tuple[str, ...]
    biomass_index: int
    l1_penalty: float
    """λ ≥ 0. Zero collapses ``J`` to pure biomass, and the LP to the flux-only model."""
    penalty_mask: NDArray[np.bool_]
    """``R_p``, full length. Recorded in the manifest — spec §3.2 requires the exact set."""
    weights: NDArray[np.float64]
    """``w``, full length, zero outside the penalty set so ``C(v) = w · |v|`` is a plain dot."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        n = len(self.reaction_ids)

        for name, array in (("penalty_mask", self.penalty_mask), ("weights", self.weights)):
            if array.shape != (n,):
                raise ObjectiveError(f"{name} has shape {array.shape}, expected ({n},)")
        if self.weights.dtype != VALUE_DTYPE:
            raise ObjectiveError(f"weights have dtype {self.weights.dtype}, expected float64")
        if self.penalty_mask.dtype != np.bool_:
            raise ObjectiveError(f"penalty_mask has dtype {self.penalty_mask.dtype}, expected bool")

        if not np.isfinite(self.l1_penalty) or self.l1_penalty < 0.0:
            raise ObjectiveError(f"l1_penalty (λ) must be finite and >= 0, got {self.l1_penalty}")
        if not np.all(np.isfinite(self.weights)):
            raise ObjectiveError("weights contain NaN or inf")
        if np.any(self.weights < 0.0):
            raise ObjectiveError("weights must be >= 0")
        if np.any(self.weights[~self.penalty_mask] != 0.0):
            offenders = np.flatnonzero(~self.penalty_mask & (self.weights != 0.0))[:5]
            raise ObjectiveError(
                "weights must be exactly zero outside the penalty set; nonzero at "
                f"{[self.reaction_ids[i] for i in offenders]}"
            )
        if not 0 <= self.biomass_index < n:
            raise ObjectiveError(f"biomass_index {self.biomass_index} outside [0, {n})")

    # ---- evaluation -----------------------------------------------------------------------------

    def evaluate(self, v_full: NDArray[np.float64]) -> ObjectiveValue:
        """``J`` evaluated **directly from the flux vector**, never from the LP's ``z`` (spec §3.4).

        This is the function the sampler tilts by, and the reference the LP's own objective is
        checked against. It has no idea an auxiliary variable ever existed.
        """
        flux = np.asarray(v_full, dtype=VALUE_DTYPE)
        if flux.shape != (len(self.reaction_ids),):
            raise ObjectiveError(f"v has shape {flux.shape}, expected ({len(self.reaction_ids)},)")
        mu = float(flux[self.biomass_index])
        cost = float(self.weights @ np.abs(flux))
        return ObjectiveValue(mu=mu, cost=cost, total=mu - self.l1_penalty * cost)

    # ---- structure ------------------------------------------------------------------------------

    @cached_property
    def penalty_indices(self) -> NDArray[np.intp]:
        """``R_p`` as indices, for the manifest."""
        return np.flatnonzero(self.penalty_mask).astype(np.intp)

    @cached_property
    def effective_costs(self) -> NDArray[np.float64]:
        """``λ w_r`` — what one unit of ``|v_r|`` actually costs ``J``.

        A reaction gets a ``z`` column *in the LP* only where this is strictly positive. A ``z``
        with zero objective cost has nothing pushing it down onto ``|v_r|``: it would settle
        anywhere in ``[|v_r|, z_max]``, and the ``z == |v|`` check would then fail on a solution
        that is perfectly correct in ``v``. So λ = 0, or a zero weight, means no ``z`` column at
        all — which is also the honest answer, since such a reaction contributes nothing to ``J``.
        """
        return self.l1_penalty * self.weights

    def with_weights(self, weights: NDArray[np.float64]) -> SparseFluxObjective:
        """A copy carrying new weights.

        M7's reweighting steps through this, producing a *new* frozen objective each iteration
        rather than mutating one a chain might already be sampling against.
        """
        weight_vector = np.asarray(weights, dtype=VALUE_DTYPE).copy()
        return SparseFluxObjective(
            reaction_ids=self.reaction_ids,
            biomass_index=self.biomass_index,
            l1_penalty=self.l1_penalty,
            penalty_mask=self.penalty_mask,
            weights=weight_vector,
        )

    def content_key(self) -> str:
        """The objective's share of the L2 cache key (§1.1)."""
        return content_key(
            impl_version=OBJECTIVE_IMPL_VERSION,
            biomass_id=self.reaction_ids[self.biomass_index],
            l1_penalty=self.l1_penalty,
            penalty_indices=self.penalty_indices.astype(np.int64),
            weights=self.weights,
        )

    def manifest(self) -> dict[str, object]:
        """The exact penalty set and weight vector, as spec §3.2 requires the run to record them."""
        penalized = self.penalty_indices
        return {
            "impl_version": OBJECTIVE_IMPL_VERSION,
            "biomass_id": self.reaction_ids[self.biomass_index],
            "l1_penalty": self.l1_penalty,
            "n_penalized": int(penalized.size),
            "penalty_reaction_ids": [self.reaction_ids[i] for i in penalized],
            "weights": self.weights[penalized].tolist(),
        }

    # ---- construction ---------------------------------------------------------------------------

    def with_l1_penalty(self, l1_penalty: float) -> SparseFluxObjective:
        """A copy carrying a different λ. The penalty set and weights are unchanged, so ``λ*`` —
        which depends only on those — is unchanged too. This is how `resolve_objective` turns a
        dimensionless ``λ̃`` into the raw λ that ``J`` actually uses."""
        return SparseFluxObjective(
            reaction_ids=self.reaction_ids,
            biomass_index=self.biomass_index,
            l1_penalty=l1_penalty,
            penalty_mask=self.penalty_mask,
            weights=self.weights,
        )

    @classmethod
    def from_polytope(
        cls,
        polytope: FluxPolytope,
        *,
        l1_penalty: float = 0.0,
        exclude_biomass_from_penalty: bool = True,
        penalty_ids: tuple[str, ...] | None = None,
        weights: NDArray[np.float64] | None = None,
    ) -> SparseFluxObjective:
        """Build the objective: by default every reaction penalized except biomass, unit weights.

        Takes a **raw** λ and touches no solver — `resolve_objective` is the config-driven entry
        point that computes λ from the model's own scale. Exchange, demand and sink reactions stay
        **in** the penalty set (spec §3.2). Excluding them would change what the cost means: it
        would make import free.
        """
        n = polytope.n_reactions

        if penalty_ids is None:
            mask = np.ones(n, dtype=np.bool_)
            if exclude_biomass_from_penalty:
                mask[polytope.biomass_index] = False
        else:
            index_of = {rid: i for i, rid in enumerate(polytope.reaction_ids)}
            unknown = sorted(set(penalty_ids) - set(index_of))
            if unknown:
                raise ObjectiveError(f"penalty_ids name reactions not in the model: {unknown}")
            mask = np.zeros(n, dtype=np.bool_)
            mask[[index_of[rid] for rid in penalty_ids]] = True

        if weights is None:
            weight_vector = mask.astype(VALUE_DTYPE)
        else:
            weight_vector = np.asarray(weights, dtype=VALUE_DTYPE).copy()
            if weight_vector.shape != (n,):
                raise ObjectiveError(
                    f"weights have shape {weight_vector.shape}, expected ({n},) — full length"
                )
            weight_vector[~mask] = 0.0

        return cls(
            reaction_ids=polytope.reaction_ids,
            biomass_index=polytope.biomass_index,
            l1_penalty=l1_penalty,
            penalty_mask=mask,
            weights=weight_vector,
        )


# ---- the (v, z) linear program ------------------------------------------------------------------


@dataclass(frozen=True)
class LPOptimum:
    """A verified optimum of the sparse-objective LP: every §12 check has passed."""

    v_full: NDArray[np.float64]
    """``v*``, full model length, with the fixed reactions restored to their fixed values."""
    z: NDArray[np.float64]
    """``z*``, one per LP-penalized free reaction. LP-only — never sampled (spec §3.4)."""
    value: ObjectiveValue
    """``J*``, ``μ(v*)``, ``C(v*)`` — recomputed from ``v*``, not read off the solver."""
    solver_objective: float
    """What HiGHS reported, offset included. Agrees with ``value.total`` to `DEFAULT_OBJECTIVE_TOL`.
    """
    max_z_deviation: float
    max_mass_balance_residual: float
    max_bound_violation: float
    simplex_iterations: int
    elapsed_seconds: float

    @property
    def j_star(self) -> float:
        """``J*``. Not a strict numeric upper bound on ``J(v)`` — see BUILD_PLAN §1.6 delta 4."""
        return self.value.total


@dataclass(frozen=True)
class SparseObjectiveLP:
    """The ``(v, z)`` LP of §12, bound to the objective and the polytope it was assembled from."""

    program: HighsLinearProgram
    reduced: ReducedPolytope
    objective: SparseFluxObjective
    z_columns: NDArray[np.intp]
    """For each ``z`` column, the *reduced* flux column it linearizes. Length ``p``."""

    @property
    def n_flux_columns(self) -> int:
        return self.reduced.n_free

    @property
    def n_z_columns(self) -> int:
        return int(self.z_columns.size)

    def solve(
        self,
        *,
        z_tol: float = DEFAULT_Z_TOL,
        objective_tol: float = DEFAULT_OBJECTIVE_TOL,
        feasibility_tol: float = DEFAULT_LP_FEASIBILITY_TOL,
    ) -> LPOptimum:
        """Solve, run every §12 result check, and return the optimum only if all of them pass."""
        solution = self.program.solve()
        return self._verify(
            solution, z_tol=z_tol, objective_tol=objective_tol, feasibility_tol=feasibility_tol
        )

    def _verify(
        self,
        solution: LPSolution,
        *,
        z_tol: float,
        objective_tol: float,
        feasibility_tol: float,
    ) -> LPOptimum:
        n_free = self.reduced.n_free
        v_reduced = solution.primal[:n_free]
        z = solution.primal[n_free:]
        v_full = self.reduced.to_full(v_reduced)

        # §12 check 3 — mass balance, recomputed against our own S rather than read off the solver's
        # row activities, which would only tell us HiGHS is self-consistent.
        residual = float(np.max(np.abs(self.reduced.mass_balance_residual(v_reduced)), initial=0.0))
        if residual > feasibility_tol:
            raise LPCheckError(
                f"LP optimum violates mass balance: max |S v − rhs| = {residual:.3e} "
                f"> {feasibility_tol:.3e}"
            )

        # §12 check 4 — flux bounds.
        violation = float(
            max(
                np.max(self.reduced.lower_bounds - v_reduced, initial=0.0),
                np.max(v_reduced - self.reduced.upper_bounds, initial=0.0),
                0.0,
            )
        )
        if violation > feasibility_tol:
            raise LPCheckError(
                f"LP optimum violates flux bounds by {violation:.3e} > {feasibility_tol:.3e}"
            )

        # §12 checks 5–6 — z ≥ |v| (the rows), and z == |v| (the objective pressure).
        abs_v = np.abs(v_reduced[self.z_columns])
        z_deviation = float(np.max(np.abs(z - abs_v), initial=0.0))
        below = float(np.max(abs_v - z, initial=0.0))
        if below > feasibility_tol:
            raise LPCheckError(
                f"auxiliary z fell {below:.3e} below |v| — the linearizing rows are not binding"
            )
        if z_deviation > z_tol:
            raise LPCheckError(
                f"z differs from |v| by {z_deviation:.3e} > {z_tol:.3e} at the optimum; "
                "the L1 linearization did not close"
            )

        # §12 checks 7–8 — recompute J from v* alone, and compare it with what the solver reported.
        value = self.objective.evaluate(v_full)
        gap = abs(value.total - solution.objective_value)
        scale = max(1.0, abs(value.total), abs(solution.objective_value))
        if gap > objective_tol * scale:
            raise LPCheckError(
                f"HiGHS reported an objective of {solution.objective_value!r}, but J(v*) "
                f"recomputed directly is {value.total!r} "
                f"(gap {gap:.3e}, relative tol {objective_tol:.1e})"
            )

        return LPOptimum(
            v_full=v_full,
            z=z,
            value=value,
            solver_objective=float(solution.objective_value),
            max_z_deviation=z_deviation,
            max_mass_balance_residual=residual,
            max_bound_violation=violation,
            simplex_iterations=solution.simplex_iterations,
            elapsed_seconds=solution.elapsed_seconds,
        )


def build_sparse_objective_lp(
    reduced: ReducedPolytope,
    objective: SparseFluxObjective,
    *,
    threads: int = 1,
) -> SparseObjectiveLP:
    """Assemble and pass the ``(v, z)`` LP of §12, directly in CSC.

    Columns are ``[v_0 … v_{n_free−1}, z_0 … z_{p−1}]``; rows are the mass balance, then the pair
    ``v_r − z_k ≤ 0``, ``−v_r − z_k ≤ 0`` for each penalized reaction.
    """
    _reject_singleton(reduced, "sparse-objective LP")

    n_free = reduced.n_free
    n_metabolites = len(reduced.metabolite_ids)

    # Which *free* reactions get a z column: those with a strictly positive λw (see
    # `effective_costs`). These are reduced column indices, in increasing order.
    effective = objective.effective_costs[reduced.free_indices]
    penalized = np.flatnonzero(effective > 0.0).astype(np.intp)
    p = int(penalized.size)

    matrix = _assemble_expanded_csc(reduced.stoichiometry, penalized)

    col_cost = np.zeros(n_free + p, dtype=VALUE_DTYPE)
    if reduced.biomass_index is not None:
        col_cost[reduced.biomass_index] = 1.0
    col_cost[n_free:] = -effective[penalized]

    # z_k ∈ [0, max(|l_r|, |u_r|)]. The upper bound is not needed for correctness — the objective
    # already pushes z down — but it keeps the LP bounded, so an unbounded status means something
    # about the fluxes rather than about the auxiliaries (spec §12).
    z_upper = np.maximum(
        np.abs(reduced.lower_bounds[penalized]), np.abs(reduced.upper_bounds[penalized])
    )
    col_lower = np.concatenate([reduced.lower_bounds, np.zeros(p, dtype=VALUE_DTYPE)])
    col_upper = np.concatenate([reduced.upper_bounds, z_upper])

    # Mass balance is an equality at the affine RHS; the absolute-value rows are one-sided at 0.
    row_lower = np.concatenate([reduced.rhs, np.full(2 * p, -np.inf, dtype=VALUE_DTYPE)])
    row_upper = np.concatenate([reduced.rhs, np.zeros(2 * p, dtype=VALUE_DTYPE)])

    program = HighsLinearProgram(
        matrix=matrix,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        col_cost=col_cost,
        maximize=True,
        offset=objective.evaluate(reduced.offset).total,
        threads=threads,
        name="sparse_objective",
    )
    assert program.n_rows == n_metabolites + 2 * p  # noqa: S101 - the §12 row layout, in one line
    return SparseObjectiveLP(
        program=program, reduced=reduced, objective=objective, z_columns=penalized
    )


def _assemble_expanded_csc(stoichiometry: NativeCSC, penalized: NDArray[np.intp]) -> NativeCSC:
    """The ``(m + 2p) × (n + p)`` matrix of §12, written straight into CSC arrays.

    Column ``v_r`` holds its stoichiometry and then — if it is the ``k``-th penalized reaction — a
    ``+1`` in row ``m + 2k`` and a ``−1`` in row ``m + 2k + 1``. Column ``z_k`` holds a ``−1`` in
    each of those two rows. Every stoichiometric row index is ``< m``, so row indices stay strictly
    increasing within each column and the result is canonical without a sort.
    """
    m, n = stoichiometry.n_rows, stoichiometry.n_cols
    p = int(penalized.size)

    is_penalized = np.zeros(n, dtype=np.bool_)
    is_penalized[penalized] = True

    stoich_counts = np.diff(stoichiometry.starts).astype(np.intp)
    counts = np.concatenate([stoich_counts + 2 * is_penalized, np.full(p, 2, dtype=np.intp)])

    starts = np.zeros(n + p + 1, dtype=INDEX_DTYPE)
    np.cumsum(counts, out=starts[1:], dtype=np.intp)

    indices = np.empty(int(starts[-1]), dtype=INDEX_DTYPE)
    values = np.empty(int(starts[-1]), dtype=VALUE_DTYPE)

    # The stoichiometric block: entry i of column j lands at starts[j] + i. Same "flat position
    # minus this column's own start" arithmetic as `NativeCSC.select_columns`.
    flux_starts = starts[:n].astype(np.intp)
    within_column = np.arange(stoichiometry.nnz, dtype=np.intp) - np.repeat(
        stoichiometry.starts[:-1].astype(np.intp), stoich_counts
    )
    destination = np.repeat(flux_starts, stoich_counts) + within_column
    indices[destination] = stoichiometry.indices
    values[destination] = stoichiometry.values

    if p:
        abs_rows = (m + 2 * np.arange(p, dtype=np.intp)).astype(INDEX_DTYPE)

        # The ±1 pair appended to each penalized flux column, just past its stoichiometry.
        after_stoich = flux_starts[penalized] + stoich_counts[penalized]
        indices[after_stoich] = abs_rows
        values[after_stoich] = 1.0
        indices[after_stoich + 1] = abs_rows + 1
        values[after_stoich + 1] = -1.0

        # The z columns: −1 in both of their rows.
        z_starts = starts[n : n + p].astype(np.intp)
        indices[z_starts] = abs_rows
        indices[z_starts + 1] = abs_rows + 1
        values[z_starts] = -1.0
        values[z_starts + 1] = -1.0

    return NativeCSC(
        n_rows=m + 2 * p, n_cols=n + p, starts=starts, indices=indices, values=values
    )


# ---- the flux-only model (§14) ------------------------------------------------------------------


def build_flux_lp(reduced: ReducedPolytope, *, threads: int = 1) -> HighsLinearProgram:
    """``{S_F v = rhs, l ≤ v ≤ u}`` with a zero objective — the model geometry re-solves (§14).

    No ``z`` columns: this LP exists to find feasible points and probe directions, and the
    linearization would only make each of its hundreds of solves bigger. Callers set the objective.
    """
    _reject_singleton(reduced, "flux-only LP")
    return HighsLinearProgram(
        matrix=reduced.stoichiometry,
        col_lower=reduced.lower_bounds,
        col_upper=reduced.upper_bounds,
        row_lower=reduced.rhs,
        row_upper=reduced.rhs,
        maximize=True,
        threads=threads,
        name="flux_only",
    )


def biomass_maximum(
    reduced: ReducedPolytope, objective: SparseFluxObjective, *, threads: int = 1
) -> float:
    """``μ_max = max_{v ∈ P} v_b`` — the §12 diagnostic.

    It changes nothing about the formulation. It says how much growth the sparse objective traded
    away for sparsity, which is the number a reader wants standing next to ``J*``.
    """
    if reduced.biomass_index is None:
        # Biomass itself is fixed (l == u), so it was eliminated: its maximum is its only value.
        return float(reduced.offset[objective.biomass_index])

    program = build_flux_lp(reduced, threads=threads)
    cost = np.zeros(reduced.n_free, dtype=VALUE_DTYPE)
    cost[reduced.biomass_index] = 1.0
    return float(program.maximize(cost).objective_value)


# ---- the L2 artifact: the optimum together with what it cost -------------------------------------


@dataclass(frozen=True)
class SparseObjectiveSolution:
    """The optimum of ``J``, next to the biomass it gave up to get there.

    This is the bundle spec §12 asks for: ``J*``, ``v*``, ``μ(v*)``, ``C(v*)``, **and** the
    biomass-only ``μ_max``, whose whole purpose is to reveal how much growth the sparse objective
    retains. It is the L2 cache artifact (§1.1), and the object M6 draws ``s_J`` from.
    """

    optimum: LPOptimum
    biomass_maximum: float
    """``μ_max`` — the most biomass the polytope allows, ignoring the L1 cost entirely."""

    @property
    def biomass_retention(self) -> float:
        """``μ(v*) / μ_max`` — the fraction of achievable growth the sparse optimum keeps.

        1.0 means λ bought no sparsity; 0.0 means it bought sparsity by abolishing growth.
        """
        if self.biomass_maximum == 0.0:
            return 1.0 if self.optimum.value.mu == 0.0 else float("inf")
        return self.optimum.value.mu / self.biomass_maximum

    @property
    def is_sparsity_dominated(self) -> bool:
        """True when λ is so large that the LP optimum abandons growth altogether.

        Above a model-specific critical ``λ* = max_v μ(v)/C(v)``, the cheapest way to maximize
        ``μ − λC`` is to set every flux to zero: the origin is feasible (``S·0 = 0``), it costs
        nothing, and it earns nothing — and that beats any growth whose L1 cost outruns its biomass.
        The LP is not wrong when this happens; ``J`` is. Every downstream stage (``s_J``, the
        β-ladder, the reweighting loop) would then be tilting toward a distribution concentrated on
        *no metabolism at all*.

        **This is not a hypothetical.** On the example model (μ_max ≈ 41.6, C ≈ 4.5e4 at the growth
        optimum) the cliff sits at λ* ≈ 1.5e-3, so both our default λ = 1.0 and the spec's suggested
        λ = 0.01 land far past it. λ must be chosen against the model's own μ/C scale — it is not a
        dimensionless knob. See DEVELOPMENT_STATUS.md (M3 findings).
        """
        return self.biomass_maximum > 0.0 and self.optimum.value.mu <= 0.0

    def diagnostics(self) -> dict[str, object]:
        """The objective block of the run's diagnostics JSON."""
        optimum = self.optimum
        return {
            "j_star": optimum.j_star,
            "mu_at_optimum": optimum.value.mu,
            "cost_at_optimum": optimum.value.cost,
            "biomass_maximum": self.biomass_maximum,
            "biomass_retention": self.biomass_retention,
            "sparsity_dominated": self.is_sparsity_dominated,
            "solver_objective": optimum.solver_objective,
            "max_z_deviation": optimum.max_z_deviation,
            "max_mass_balance_residual": optimum.max_mass_balance_residual,
            "max_bound_violation": optimum.max_bound_violation,
            "simplex_iterations": optimum.simplex_iterations,
        }


def solve_sparse_objective(
    reduced: ReducedPolytope,
    objective: SparseFluxObjective,
    *,
    threads: int = 1,
    z_tol: float = DEFAULT_Z_TOL,
    objective_tol: float = DEFAULT_OBJECTIVE_TOL,
    feasibility_tol: float = DEFAULT_LP_FEASIBILITY_TOL,
) -> SparseObjectiveSolution:
    """Solve both LPs of §12 — the sparse objective and the biomass-only diagnostic — as one unit.

    The two belong together: ``J*`` alone cannot tell you whether λ was sane, and ``μ_max`` is what
    makes ``μ(v*)`` legible. Callers that skip the diagnostic can miss a sparsity-dominated
    objective entirely, because a collapsed optimum looks perfectly healthy from inside the LP —
    optimal status, zero residual, ``z == |v|`` exactly.
    """
    lp = build_sparse_objective_lp(reduced, objective, threads=threads)
    optimum = lp.solve(z_tol=z_tol, objective_tol=objective_tol, feasibility_tol=feasibility_tol)
    return SparseObjectiveSolution(
        optimum=optimum,
        biomass_maximum=biomass_maximum(reduced, objective, threads=threads),
    )


def _reject_singleton(reduced: ReducedPolytope, what: str) -> None:
    if reduced.is_singleton:
        raise ObjectiveError(
            f"cannot build a {what}: every reaction is fixed, so the reduced polytope is a single "
            "point with no variables. Callers must handle the singleton case (M4's dim-0 path)."
        )


# ---- λ's scale: the sparsity cliff (BUILD_PLAN §1.7) ---------------------------------------------


def origin_is_feasible(polytope: FluxPolytope) -> bool:
    """Is ``v = 0`` inside the polytope?

    ``S·0 = 0`` always, so this asks only whether any reaction is *forced* to carry flux — an
    ``l > 0`` or ``u < 0``, typically an ATP-maintenance demand. It is the precondition for the
    collapse in §1.7: a cell that must spend ATP to stay alive cannot answer a large λ by shutting
    down, because shutting down is not available to it.
    """
    return bool(np.all(polytope.lower_bounds <= 0.0) and np.all(polytope.upper_bounds >= 0.0))


def critical_l1_penalty(
    reduced: ReducedPolytope, objective: SparseFluxObjective, *, threads: int = 1
) -> float:
    """``λ* = max_{v ∈ P, C(v) > 0} μ(v) / C(v)`` — the λ at which growth stops paying for itself.

    Depends only on the polytope, the penalty set and the **weights** — not on λ itself, so it is
    the natural unit in which to express λ. Above it, ``μ − λC`` is maximized by ``v = 0`` (when the
    origin is feasible), and every downstream stage would tilt toward a cell that does nothing.

    Computed **exactly, with one LP**, not by bisection. ``μ/C`` is a linear-fractional program, and
    the Charnes–Cooper substitution ``y = v·t, t = 1/C(v)`` turns it into a linear one: maximizing
    ``μ(y)`` subject to a unit cost budget ``C(y) ≤ 1``. The bounds ``l ≤ v ≤ u`` are not a cone, so
    they homogenize into rows ``l·t ≤ y ≤ u·t`` rather than staying column bounds, and the absolute
    value in ``C`` linearizes with the same ``z ≥ ±y`` trick as §12.

    Returns ``inf`` when the LP is unbounded — a growth path whose weighted L1 cost is zero, so no λ
    can ever suppress it and there is no cliff at all.
    """
    _reject_singleton(reduced, "critical-λ LP")

    n = reduced.n_free
    m = len(reduced.metabolite_ids)
    weights_free = objective.weights[reduced.free_indices]
    penalized = np.flatnonzero(weights_free > 0.0).astype(np.intp)
    p = int(penalized.size)

    # The fixed reactions' share of C(v) is a constant, so under y = v·t it scales with t.
    fixed_cost = objective.evaluate(reduced.offset).cost

    row_of_lower = m
    row_of_upper = m + n
    row_of_z = m + 2 * n
    row_of_budget = m + 2 * n + 2 * p
    n_rows = row_of_budget + 1
    column_of_t = n + p

    lp_column_of: dict[int, int] = {int(j): k for k, j in enumerate(penalized)}
    columns: list[dict[int, float]] = []

    for j in range(n):
        column: dict[int, float] = {
            int(row): float(value)
            for row, value in zip(
                reduced.stoichiometry.indices[
                    reduced.stoichiometry.starts[j] : reduced.stoichiometry.starts[j + 1]
                ],
                reduced.stoichiometry.values[
                    reduced.stoichiometry.starts[j] : reduced.stoichiometry.starts[j + 1]
                ],
                strict=True,
            )
        }
        column[row_of_lower + j] = 1.0  # y_j − l_j·t ≥ 0
        column[row_of_upper + j] = 1.0  # y_j − u_j·t ≤ 0
        if j in lp_column_of:
            k = lp_column_of[j]
            column[row_of_z + 2 * k] = 1.0  # +y_j − z_k ≤ 0
            column[row_of_z + 2 * k + 1] = -1.0  # −y_j − z_k ≤ 0
        columns.append(column)

    for k, penalized_column in enumerate(penalized):
        columns.append(
            {
                row_of_z + 2 * k: -1.0,
                row_of_z + 2 * k + 1: -1.0,
                row_of_budget: float(weights_free[penalized_column]),
            }
        )

    t_column: dict[int, float] = {
        int(row): -float(value) for row, value in enumerate(reduced.rhs) if value != 0.0
    }
    for j in range(n):
        if reduced.lower_bounds[j] != 0.0:
            t_column[row_of_lower + j] = -float(reduced.lower_bounds[j])
        if reduced.upper_bounds[j] != 0.0:
            t_column[row_of_upper + j] = -float(reduced.upper_bounds[j])
    if fixed_cost != 0.0:
        t_column[row_of_budget] = fixed_cost
    columns.append(t_column)

    row_lower = np.concatenate(
        [
            reduced.rhs * 0.0,  # S_F y − rhs·t = 0
            np.zeros(n, dtype=VALUE_DTYPE),  # y − l·t ≥ 0
            np.full(n + 2 * p + 1, -np.inf, dtype=VALUE_DTYPE),
        ]
    )
    row_upper = np.concatenate(
        [
            reduced.rhs * 0.0,
            np.full(n, np.inf, dtype=VALUE_DTYPE),
            np.zeros(n + 2 * p, dtype=VALUE_DTYPE),  # y − u·t ≤ 0 and the z rows
            np.ones(1, dtype=VALUE_DTYPE),  # the unit cost budget: C(y) ≤ 1
        ]
    )

    # y is free (it is v·t, not v); z ≥ 0 and t ≥ 0 by construction.
    col_lower = np.concatenate([np.full(n, -np.inf), np.zeros(p + 1)]).astype(VALUE_DTYPE)
    col_upper = np.full(n + p + 1, np.inf, dtype=VALUE_DTYPE)

    cost = np.zeros(n + p + 1, dtype=VALUE_DTYPE)
    if reduced.biomass_index is not None:
        cost[reduced.biomass_index] = 1.0
    else:
        # Biomass is fixed, so μ(v) is the constant c_b and μ(y) = c_b·t.
        cost[column_of_t] = float(reduced.offset[objective.biomass_index])

    program = HighsLinearProgram(
        matrix=NativeCSC.from_columns(n_rows, columns),
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        col_cost=cost,
        maximize=True,
        threads=threads,
        name="critical_l1_penalty",
    )
    try:
        return float(program.solve().objective_value)
    except LPNotOptimalError as unbounded:
        if "Unbounded" not in unbounded.model_status:
            raise
        # Growth at zero weighted L1 cost: no λ can ever suppress it, so there is no cliff.
        return float("inf")


@dataclass(frozen=True)
class ObjectiveScale:
    """How the raw λ in `SparseFluxObjective` was arrived at (BUILD_PLAN §1.7).

    Recorded in full — spec §3.6 forbids hidden scaling — so a reader can always recover the raw λ
    the mathematics actually used, and compare ``λ̃`` across strains that have different ``λ*``.
    """

    l1_penalty_scaled: float
    """``λ̃``, dimensionless. The knob the user turns; comparable across models."""
    critical_l1_penalty: float
    """``λ*``, this model's own cliff. ``inf`` if the model has no cliff."""
    l1_penalty: float
    """``λ = λ̃ · λ*`` — the raw penalty ``J`` is actually computed with."""
    origin_is_feasible: bool
    """Whether ``v = 0`` is available to the model. If not, no λ can collapse it."""

    def manifest(self) -> dict[str, object]:
        return {
            "l1_penalty_scaled": self.l1_penalty_scaled,
            "critical_l1_penalty": self.critical_l1_penalty,
            "l1_penalty": self.l1_penalty,
            "origin_is_feasible": self.origin_is_feasible,
        }


@dataclass(frozen=True)
class ResolvedObjective:
    """A `SparseFluxObjective` together with the scale reasoning that produced its λ."""

    objective: SparseFluxObjective
    scale: ObjectiveScale

    def manifest(self) -> dict[str, object]:
        return {**self.objective.manifest(), "scale": self.scale.manifest()}


def resolve_objective(
    polytope: FluxPolytope,
    reduced: ReducedPolytope,
    config: ObjectiveConfig | None = None,
    *,
    penalty_ids: tuple[str, ...] | None = None,
    weights: NDArray[np.float64] | None = None,
    threads: int = 1,
) -> ResolvedObjective:
    """Turn the config's dimensionless ``λ̃`` into the raw λ that ``J`` uses (BUILD_PLAN §1.7).

    ``λ = λ̃ · λ*``, where ``λ*`` is *this model's* sparsity cliff. So ``λ̃ = 0`` is plain FBA and
    ``λ̃ → 1`` is the most sparsity pressure the model can carry while still growing — and the same
    ``λ̃`` means the same *selection pressure* in every strain of a batch, which a shared raw λ
    emphatically does not (§1.1's cross-model comparison depends on this).

    Costs one extra LP per model. Both λ̃ and the raw λ land in the manifest.
    """
    settings = config if config is not None else ObjectiveConfig()

    base = SparseFluxObjective.from_polytope(
        polytope,
        l1_penalty=0.0,
        exclude_biomass_from_penalty=settings.exclude_biomass_from_penalty,
        penalty_ids=penalty_ids,
        weights=weights,
    )
    critical = critical_l1_penalty(reduced, base, threads=threads)
    origin_feasible = origin_is_feasible(polytope)
    scaled = settings.l1_penalty_scaled

    if origin_feasible and scaled >= 1.0:
        raise ObjectiveError(
            f"objective.l1_penalty_scaled = {scaled} would put λ at or past this model's sparsity "
            f"cliff (λ* = {critical:.6g}), where the LP optimum is v = 0 and the sampler would "
            "tilt toward a cell that does nothing. The origin is feasible for this model (no "
            "reaction is forced to carry flux), so λ̃ must be < 1. See BUILD_PLAN §1.7."
        )
    if not np.isfinite(critical):
        raise ObjectiveError(
            "this model has a growth path with zero weighted L1 cost, so λ* is unbounded and a "
            "scale-referenced λ has nothing to reference. Check the penalty set and weights — "
            "some reaction carrying biomass flux is evidently unpenalized."
        )

    raw = scaled * critical
    return ResolvedObjective(
        objective=base.with_l1_penalty(raw),
        scale=ObjectiveScale(
            l1_penalty_scaled=scaled,
            critical_l1_penalty=critical,
            l1_penalty=raw,
            origin_is_feasible=origin_feasible,
        ),
    )
