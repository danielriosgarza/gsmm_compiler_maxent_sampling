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
from gsmm_compiler.line_distribution import L1Objective
from gsmm_compiler.logging_utils import get_logger
from gsmm_compiler.native_csc import INDEX_DTYPE, VALUE_DTYPE, NativeCSC
from gsmm_compiler.provenance import content_key

_log = get_logger(__name__)

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


def _frozen(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make a weight buffer physically read-only. See `rounding._freeze` for the full reasoning.

    Kept local rather than imported: `sparse_objective` is importable by an MCMC worker without
    loading HiGHS (§1.2), and `rounding` reaches the solver through `affine_geometry`.

    Every caller here freezes a buffer it has just created (a fancy-index copy or a fresh product),
    so this never reaches back and freezes an array its owner still expects to write.
    """
    array.flags.writeable = False
    return array


class IncompatibleObjectiveError(ObjectiveError):
    """The objective and the polytope do not describe the same model."""


def check_compatible(reduced: ReducedPolytope, objective: SparseFluxObjective) -> None:
    """The objective and the polytope must describe **the same model**. One guard, every join.

    A `SparseFluxObjective` is a biomass index, a mask and a weight vector; a `ReducedPolytope` is a
    matrix and some bounds. Neither knows the other exists, and an index is just an integer. So a
    mismatched pair is not *detectable* by anything downstream — it is *computable*, and computes
    confidently. Feed the LP a polytope whose biomass is reaction ``b`` and an objective whose
    biomass is reaction ``a``, and you get an "optimum" whose ``μ`` is one reaction's flux and whose
    ``biomass_maximum`` is another's, in the same bundle, with every §12 check passing.

    **Both halves of this check are load-bearing, and the M6 review found each one separately.**

    * **The same coordinate system.** Comparing biomass *indices* is not enough: objective biomass
      ``"a"`` at index 0 and polytope biomass ``"b"`` at index 0 agree numerically while naming
      different reactions of different models. The reaction IDs are the coordinate system, and they
      have to *be* the same one. (Round 4.)
    * **The same biomass reaction.** Two models identical but for which reaction is biomass share
      everything else — and when that reaction is *fixed*, ``μ`` **is** its constant value, so the
      whole objective is wrong while the numbers stay plausible. (Round 3.)

    This is called from every public function that joins an objective to a polytope. It was
    previously called from none of them, and the resulting bugs were found one join at a time until
    Codex pointed out that patching joins is not the same as having an invariant.
    """
    if objective.reaction_ids != reduced.reaction_ids:
        raise IncompatibleObjectiveError(
            "the objective and the polytope are not built on the same reaction set, so their "
            "indices do not mean the same thing. Every index in the objective would silently "
            f"address a different reaction (objective: {len(objective.reaction_ids)} reactions "
            f"starting {objective.reaction_ids[:3]}…; polytope: {len(reduced.reaction_ids)} "
            f"starting {reduced.reaction_ids[:3]}…)"
        )
    if objective.biomass_index != reduced.biomass_full_index:
        raise IncompatibleObjectiveError(
            f"the objective's biomass is {objective.reaction_ids[objective.biomass_index]!r} "
            f"(index {objective.biomass_index}) but the polytope's is {reduced.biomass_id!r} "
            f"(index {reduced.biomass_full_index}). ``J`` would reward one reaction while every "
            "diagnostic described the other, and nothing downstream could tell."
        )


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
            weights=_frozen(weight_vector),
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
            weights=_frozen(weight_vector),
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

    objective_key: str
    """`SparseFluxObjective.content_key` of the objective this optimum was solved for.

    ``J*`` is a **float**, and a float remembers nothing. Until M7 that was harmless, because a run
    had exactly one objective and every ``J*`` in it necessarily came from that one. M7 creates a
    *second* objective on the same polytope — the reweighted one — and then ``s_J = J* − Q_q(J(W))``
    can be assembled from a ``J*`` belonging to one and a ``J(W)`` belonging to the other. The
    subtraction succeeds. It returns a plausible number. It is the difference of two functions.

    Carrying the key turns that from *computable* into *refused* (`choose_energy_scale`).
    """

    polytope_key: str
    """`ReducedPolytope.content_key` of the polytope this optimum was solved on.

    ``objective_key`` alone is **not enough**, and the gap is subtle (Codex, M7 review round 2):
    `SparseFluxObjective.content_key` hashes the biomass id, λ, penalty indices and weights — *not*
    the polytope's bounds or stoichiometry. Two different polytopes with the same reaction order,
    biomass, λ and weights therefore produce the **same** ``objective_key`` while their ``J*``
    differ.
    Without this field, ``s_J = J*(solved on polytope A) − Q_q(J(W over polytope B))`` passes every
    key check `choose_energy_scale` makes and silently rescales the ladder — the M6 disease, one
    level up. `choose_energy_scale` checks this against the objective's own `polytope_key` too."""

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
            objective_key=self.objective.content_key(),
            polytope_key=self.reduced.content_key(),
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
    check_compatible(reduced, objective)

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
    check_compatible(reduced, objective)

    if reduced.biomass_index is None:
        # Biomass itself is fixed (l == u), so it was eliminated: its maximum is its only value.
        # Read from the POLYTOPE's own `biomass_full_index`, not the objective's — the polytope
        # knows which reaction it is, and asking the objective was the polytope trusting the
        # objective to tell it what it is. (Only possible since M6 round 3 added the field.)
        return float(reduced.offset[reduced.biomass_full_index])

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
    check_compatible(reduced, objective)

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
    check_compatible(reduced, objective)

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
        # Biomass is fixed, so μ(v) is the constant c_b and μ(y) = c_b·t. Read from the POLYTOPE's
        # own `biomass_full_index` — asking the objective which reaction the polytope's biomass is
        # was exactly backwards, and is how the round-3 identity hole survived unnoticed.
        cost[column_of_t] = float(reduced.offset[reduced.biomass_full_index])

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
    reduced: ReducedPolytope | None = None,
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

    **``reduced`` must be the reduction of ``polytope``, and it is checked.** This function reads
    `origin_is_feasible` off the **canonical** bounds and computes ``λ*`` off the **reduced** LP, so
    a mismatched pair makes the §1.7 guard describe one polytope while the mathematics runs on
    another: pass a forced-flux canonical polytope with the reduction of an origin-feasible variant
    and
    ``λ̃ ≥ 1`` is *accepted and recorded as safe*, while the polytope that actually gets sampled
    collapses to the origin. Omit ``reduced`` and it is derived here, which is the safe default.
    (Codex, M6 review round 5.)
    """
    if reduced is None:
        reduced = polytope.reduce()
    elif not reduced.is_reduction_of(polytope):
        raise IncompatibleObjectiveError(
            "the reduced polytope is not the reduction of the canonical one. `origin_is_feasible` "
            "would be read from one and λ* computed on the other, so the sparsity-cliff guard "
            "(BUILD_PLAN §1.7) would describe a polytope that is not the one being sampled. Pass "
            "`polytope.reduce()`, or omit the argument and let this derive it."
        )

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


# ---- the sampler's view: J lowered onto the reduced polytope (M6) --------------------------------


@dataclass(frozen=True)
class ReducedObjective:
    """``J`` as the **sampler** needs it: on reduced coordinates, with the fixed constants kept.

    The sampler's state is a length-``n_free`` flux, so the objective it tilts by must be indexed
    the same way. Lowering it is not just a re-indexing, because the eliminated ``l == u`` reactions
    still contribute to ``J``: a forced ATP-maintenance demand really does cost L1, and a *fixed
    biomass* reaction really does supply growth. What they no longer do is **vary**.

    So this object is deliberately two things at once, and keeping them apart is the whole point:

    * `line` — the `L1Objective` the M2 kernel draws from, in reduced indices. It is ``J`` **up to
      an additive constant**, and it must be: the constant cannot reach a probability without
      inviting the exact catastrophic cancellation that `line_distribution` delta 4 exists to
      prevent. ``p(t)`` is invariant to it algebraically and — because it never appears —
      numerically.
    * `mu_offset` / `cost_offset` — that constant, split into its two reported halves. **Every
      number a human reads is built from these**: a trace of ``J`` is compared with ``J*``, which
      the LP computed from the *full* flux vector, so a trace that quietly omitted the fixed
      reactions' L1 cost would sit a constant away from ``J*`` and make ``(J − J*)/s_J`` — and
      therefore ``s_J`` itself — wrong by that constant.

    The two are held together by `evaluate`, which reproduces `SparseFluxObjective.evaluate` on the
    lifted flux exactly; a test asserts that on both models, and the toy is the one that can fail it
    (its ``FIX = 2.0`` makes ``cost_offset`` nonzero, where all 513 of the example model's fixed
    reactions sit at zero and hide the bug).
    """

    line: L1Objective
    """What the kernel tilts by. Penalized set is ``λ·w > 0`` — the reactions that *bend* ``J``."""

    weights: NDArray[np.float64]
    """``w`` over the free reactions, ``(n_free,)``, zero outside the penalty set.

    Separate from `line.weights`, and deliberately: ``C(v) = Σ w_r |v_r|`` is defined **without λ**,
    so its set is ``w > 0``, while only ``λw > 0`` puts a bend in ``J``. The two sets coincide for
    every λ > 0 and diverge at λ = 0, where ``J`` is linear (plain FBA) but ``C`` is still a number
    the run must report. Keeping one array for both would silently report ``C = 0`` at λ = 0.
    """

    mu_offset: float
    """``μ`` at the fixed fluxes — nonzero only when biomass itself is a fixed reaction."""

    cost_offset: float
    """``C`` at the fixed fluxes: ``Σ_{fixed r} w_r·|c_r|``."""

    l1_penalty: float

    n_free: int

    polytope_key: str
    """`ReducedPolytope.content_key` of the polytope this was lowered from — **checked, not
    trusted**.

    Without it a `ReducedObjective` is just indices and weights, and *nothing anywhere* could tell
    that it belongs to a different model. Hand `run_ladder` one whose `line.biomass_index` is 0 on a
    polytope whose biomass is actually fixed, and: the kernel tilts by reaction 0; `evaluate_many`
    reports reaction 0's flux as ``μ``; the trace of ``J`` rises monotonically with β **because the
    chain really is maximizing the thing the trace is measuring**. The diagnostics confirm the wrong
    target, and feasibility, mass balance, chords and R̂ are all perfect, because none of them knows
    which reaction ``J`` is supposed to be about.

    That pairing is not hypothetical once M8 exists: the L2 objective and the L3 geometry are
    *separate cache artifacts*, and a stale key or a hand-edited manifest is all it takes to load
    two
    that were never computed against each other. `run_ladder` refuses the pair.
    """

    objective_key: str
    """`SparseFluxObjective.content_key` of the objective this was lowered from — the **weights**,
    not just the polytope.

    `polytope_key` answers "which model?". This answers "which *objective on* that model?", and
    until M7 nothing needed to ask, because a run had exactly one. **M7's whole job is to produce a
    second one** — same reactions, same bounds, same biomass, different ``w`` and different λ — so
    the two share a `polytope_key` *exactly*. On the toy, ``s_J`` is 0.68 under the base weights and
    0.0068 under the reweighted ones: pair the wrong one and every rung of the ladder is 100× colder
    than it claims, while ``J`` still rises monotonically with β and every diagnostic agrees —
    because the chain really is maximizing the thing the trace measures.

    That is the M6 defect exactly (*two artifacts never computed against each other, silently
    joined*), and M7 is the milestone that arms it. So the objective is keyed too.
    """

    def __post_init__(self) -> None:
        if self.weights.shape != (self.n_free,):
            raise ObjectiveError(
                f"reduced weights have shape {self.weights.shape}, expected ({self.n_free},)"
            )
        if not np.isfinite(self.mu_offset) or not np.isfinite(self.cost_offset):
            raise ObjectiveError("the fixed-flux objective constants are not finite")
        if self.line.lam != self.l1_penalty:
            # The kernel bends J by `line.lam`; the traces report J = μ − `l1_penalty`·C. If the two
            # disagree, the chain samples one distribution and the run reports another — and both
            # look entirely healthy on their own.
            raise ObjectiveError(
                f"the kernel's λ ({self.line.lam}) and the reported λ ({self.l1_penalty}) differ; "
                "the chain would sample one objective and the traces would describe another"
            )

    def binds_to(self, reduced: ReducedPolytope) -> bool:
        """Was this objective lowered from *this* polytope? (see `polytope_key`)"""
        return self.polytope_key == reduced.content_key()

    @property
    def j_offset(self) -> float:
        """``J`` at the fixed fluxes — what `line` omits, and `evaluate` puts back."""
        return self.mu_offset - self.l1_penalty * self.cost_offset

    @property
    def biomass_index(self) -> int | None:
        return self.line.biomass_index

    def evaluate(self, v_reduced: NDArray[np.float64]) -> ObjectiveValue:
        """The **full** ``μ``, ``C``, ``J`` of one reduced flux — as if evaluated on the lift."""
        mu, cost, total = self.evaluate_many(np.atleast_2d(v_reduced))
        return ObjectiveValue(mu=float(mu[0]), cost=float(cost[0]), total=float(total[0]))

    def evaluate_many(
        self, v_reduced: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``μ``, ``C``, ``J`` for a ``(n, n_free)`` batch of reduced fluxes.

        One gemv against the full-length ``weights`` rather than a fancy-index of the penalized
        set: the traces are computed over every stored sample at once, and this is both faster and —
        since ``weights`` is zero off the penalty set by construction — exactly the same sum.
        """
        v = np.asarray(v_reduced, dtype=VALUE_DTYPE)
        if v.ndim != 2 or v.shape[1] != self.n_free:
            raise ObjectiveError(
                f"reduced fluxes have shape {v.shape}, expected (n, {self.n_free})"
            )

        cost = self.cost_offset + np.abs(v) @ self.weights
        mu = np.full(v.shape[0], self.mu_offset, dtype=VALUE_DTYPE)
        if self.line.biomass_index is not None:
            mu = mu + v[:, self.line.biomass_index]

        return mu, cost, mu - self.l1_penalty * cost

    def manifest(self) -> dict[str, object]:
        return {
            "polytope_key": self.polytope_key,
            "n_free": self.n_free,
            "n_bending": int(self.line.penalized_indices.size),
            "n_costed": int(np.count_nonzero(self.weights)),
            "l1_penalty": self.l1_penalty,
            "mu_offset": self.mu_offset,
            "cost_offset": self.cost_offset,
            "j_offset": self.j_offset,
            "biomass_is_fixed": self.line.biomass_index is None,
        }


def lower_objective(
    reduced: ReducedPolytope, objective: SparseFluxObjective
) -> ReducedObjective:
    """Lower a full-model `SparseFluxObjective` onto the reduced polytope the sampler lives on.

    ``reduced.offset`` is the full-length vector holding each fixed reaction's value and **zero at
    every free position**, so evaluating the full objective on it yields precisely the fixed
    reactions' contribution to ``μ`` and to ``C`` — the supports being disjoint, there are no
    cross-terms to worry about. It is the same constant `build_sparse_objective_lp` hands HiGHS as
    the objective offset, arrived at the same way, so the LP's ``J*`` and the sampler's ``J`` are
    guaranteed to be in the same units rather than merely believed to be.
    """
    check_compatible(reduced, objective)

    free = reduced.free_indices
    weights_free = np.ascontiguousarray(objective.weights[free], dtype=VALUE_DTYPE)

    # Only λw > 0 puts a bend in J — the same test `build_sparse_objective_lp` uses to decide which
    # reactions get a `z` column, so the LP that finds J* and the kernel that samples around it
    # agree on which reactions are penalized rather than each deciding for itself.
    bending = np.flatnonzero(objective.effective_costs[free] > 0.0).astype(np.intp)

    fixed = objective.evaluate(reduced.offset)

    # The chain samples against these for its whole life. `@dataclass(frozen=True)` freezes the
    # *binding*, not the buffer (the M5 lesson), and M7's entire safety argument is that ``w`` does
    # not move once sampling starts — a weight that shifts mid-chain retargets every conditional and
    # destroys stationarity, silently. So freeze the buffers, and let the interpreter enforce it.
    kernel_weights = _frozen(weights_free[bending])
    weights_free = _frozen(weights_free)

    return ReducedObjective(
        line=L1Objective(
            biomass_index=reduced.biomass_index,
            penalized_indices=bending,
            weights=kernel_weights,
            lam=objective.l1_penalty,
        ),
        weights=weights_free,
        mu_offset=fixed.mu,
        cost_offset=fixed.cost,
        l1_penalty=objective.l1_penalty,
        n_free=reduced.n_free,
        polytope_key=reduced.content_key(),
        objective_key=objective.content_key(),
    )


# ---- the energy scale s_J (spec §3.6, §22.2) ----------------------------------------------------

DEFAULT_ENERGY_QUANTILE: Final = 0.05
"""``s_J = J* − Q_{0.05}(J(W))`` — the robust lower quantile spec §22.2 prescribes."""

ENERGY_SCALE_ULP_MARGIN: Final = 64.0
"""How many ULPs of the *operands* a warm-up range must clear to be a range rather than its
rounding.

``s_J = J* − Q_q(J(W))`` is a **difference of two numbers of magnitude ~|J*|**, so evaluating it in
float64 costs about ``eps·max(|J*|, |Q_q|)`` before anything else goes wrong. Below a few ULPs of
that, the subtraction has no significant digits left and the "range" is the rounding of the
subtraction — the M4 lesson (*never divide by a small number that is noise*) arriving at the
calibration layer.

**This replaces a floor of ``1e-9·max(1, |J*|)``, which was the M2 bug wearing a different hat.**
The *range* ``J* − Q_q`` is invariant when a constant is added to ``J``; a floor keyed on ``|J*|``
is not. So shifting ``J`` by ``+1e16`` — a constant that provably cannot change any probability —
turned a healthy ``s_J = 12`` into a fallback, and made every rung of the ladder 12× hotter. Codex
found it; it is reproduced in `test_the_floor_is_a_cancellation_floor_not_a_magnitude_floor`.

The ULP floor is not invariant either, and *that is the point*: at a baseline of 1e16 a range of 12
really is only 6 ULPs wide, and each ``J(w)`` in the quantile carries ~1 ULP (= 2.0) of error, so
17% of the "range" is noise. The floor now tracks the arithmetic instead of guessing at it. At a
plausible ``|J*| = 1e5`` the old floor was 1e-4 and this one is 9e-10 — so a real range of 1e-6 is
now *kept*, where before it was thrown away.
"""


class DegenerateEnergyScaleError(ObjectiveError):
    """The warm-up objective range is not resolvable, and no fallback scale was declared."""


@dataclass(frozen=True)
class EnergyScale:
    """``s_J`` — the units ``β`` is measured in, and how it was arrived at (spec §3.6: no hidden
    scaling).

    ``β`` multiplies ``(J − J*)/s_J``, so ``s_J`` is what makes ``β`` mean the same thing across two
    strains whose ``J`` differ by orders of magnitude — the same job ``λ̃`` does for λ (BUILD_PLAN
    §1.7), and needed for the same reason: the batch comparison is the point.

    Dividing by ``s_J`` cannot change *a* distribution — it reparameterizes ``β`` — but it decides
    **which** distribution a given ``β`` names, and that is exactly what a cross-model ladder is
    comparing. So it is recorded in full.
    """

    value: float
    """``s_J > 0``. The one number the sampler consumes."""

    mode: str
    """``warmup_range`` (spec §22.2) or ``declared`` (``energy_scale = <float>``, spec §3.6)."""

    j_star: float

    n_warmup_points: int

    quantile: float | None
    """The quantile used, or ``None`` in ``declared`` mode."""

    warmup_quantile_j: float | None
    """``Q_q(J(W))`` — the low end of the observed objective range."""

    warmup_max_j: float | None
    """``max J(W)``. Next to ``j_star`` it says how close the warm-up points get to the optimum."""

    resolution: float | None
    """The float64 resolution of ``J* − Q_q(J(W))`` — ``64·ulp(max(|J*|, |Q_q|))``.

    The bar the range had to clear (`ENERGY_SCALE_ULP_MARGIN`), recorded so a reader can see *how
    much* room it had. A range one ULP above the floor is technically resolvable and scientifically
    worthless, and only this number says which one you got."""

    polytope_key: str
    """`ReducedPolytope.content_key` of the polytope this scale was calibrated on."""

    objective_key: str
    """`SparseFluxObjective.content_key` of the **objective** it was calibrated from.

    M6 keyed this artifact on the *polytope* while its own docstring said ``s_J`` "is a property of
    one objective on one polytope". Both statements were in the file at once, and the second was the
    true one; the guard implemented the first. It never fired, because until now a run had exactly
    one objective per polytope — so the two keys were distinctions without a difference.

    **M7 makes them differ.** The reweighted objective has the same reactions, the same bounds and
    the same biomass as the base one, so it shares the `polytope_key` *exactly* — and on the toy its
    ``s_J`` is 0.0068 where the base objective's is 0.68. A hundredfold. Borrowed across, every β on
    the ladder names a selection pressure two orders of magnitude from the one it reports, and
    nothing downstream can see it: ``J`` still rises monotonically with β, because the chain really
    is maximizing the thing the trace measures.

    (Codex, M6 review round 6 — got the invariant right and the key wrong. Found by M7's
    reproduction, `test_energy_scale_refuses_a_borrowed_objective`.)
    """

    fell_back: bool
    """True when the range was below `resolution` and an explicitly **declared** fallback was used.

    Never silent. A degenerate range with no declared fallback raises `DegenerateEnergyScaleError`
    rather than quietly substituting ``s_J = 1``, which would rescale every β on the ladder and make
    this strain's rungs incomparable with the rest of the batch — the one thing ``s_J`` exists to
    prevent."""

    def manifest(self) -> dict[str, object]:
        return {
            "energy_scale": self.value,
            "energy_scale_mode": self.mode,
            "j_star": self.j_star,
            "n_warmup_points": self.n_warmup_points,
            "energy_scale_quantile": self.quantile,
            "warmup_quantile_j": self.warmup_quantile_j,
            "warmup_max_j": self.warmup_max_j,
            "energy_scale_resolution": self.resolution,
            "energy_scale_polytope_key": self.polytope_key,
            "energy_scale_objective_key": self.objective_key,
            "energy_scale_fell_back": self.fell_back,
        }


def energy_scale_resolution(j_star: float, low: float) -> float:
    """The float64 resolution of ``J* − Q_q(J(W))``: ``64·ulp(max(|J*|, |Q_q|, 1))``.

    See `ENERGY_SCALE_ULP_MARGIN`. This is a *cancellation* floor — it asks "does this subtraction
    have any significant digits left?" — and not a magnitude floor, which asks the wrong question
    and gets a non-invariant answer.
    """
    return ENERGY_SCALE_ULP_MARGIN * float(
        np.spacing(max(abs(float(j_star)), abs(float(low)), 1.0))
    )


def choose_energy_scale(
    objective: ReducedObjective,
    warmup_fluxes: NDArray[np.float64],
    *,
    optimum: LPOptimum,
    warmup_polytope_key: str,
    mode: str | float = "warmup_range",
    quantile: float = DEFAULT_ENERGY_QUANTILE,
    fallback: float | None = None,
) -> EnergyScale:
    """``s_J`` from the objective values of the warm-up points (spec §22.2), or a declared constant.

    ``warmup_fluxes`` is ``(K, n_free)`` — M4's **support points**, the vertices the geometry has
    already solved for. They are the cheapest honest sample of how far ``J`` ranges over this
    polytope, and they cost no extra LP.

    **``warmup_polytope_key`` is required, not decorative.** The warm-up array is the *third* input
    to a subtraction that must all come from one polytope (``s_J = J* − Q_q(J(W))``), and unlike the
    optimum and the objective it is a bare ``(K, n_free)`` array with no identity of its own. Two
    polytopes with the same free dimension produce same-shaped support sets, so a warm-up array from
    the wrong one is silently evaluable and silently changes ``s_J`` (Codex, M7 review round 3). It
    comes from a keyed `affine_geometry.ReducedGeometry`; pass that geometry's ``polytope_key`` and
    it is checked against the objective before a single ``J(W)`` is formed.

    **``J*`` arrives as a keyed `LPOptimum`, not as a float, and that is the point.** ``s_J = J* −
    Q_q(J(W))`` is a *difference of two evaluations of J*, and it is only a range if both are the
    same ``J``. A float carries no evidence of which objective produced it, so before M7 the two
    could not be checked against each other — and before M7 they never had to be, because a run held
    exactly one objective. M7 holds two. The subtraction of one objective's ``J*`` from another's
    quantile is arithmetically fine, returns a plausible number, and is meaningless; on the toy it
    gives 1.22 where the honest answers are 0.68 and 0.0068. So the key is checked here.

    ``s_J = J* − Q_q(J(W))`` measures ``β`` in units of *the objective range this model actually
    spans*, rather than in units of its raw numerical magnitude — so ``β = 2`` names a comparable
    selection pressure in a strain whose ``J`` runs to 40 and in one whose ``J`` runs to 4000.

    **``j_star`` and ``J(W)`` must be the same ``J``**, which is why this takes a `ReducedObjective`
    and not an `L1Objective`: the latter is the objective *minus a constant* (the fixed reactions'
    contribution), and subtracting a quantile of it from the LP's full ``J*`` would put that
    constant straight into ``s_J`` — silently rescaling every β on the ladder. On the toy, whose
    ``FIX = 2.0`` makes the constant nonzero, that is a double-digit error in ``s_J``; on the
    example model the constant is zero and nothing would ever have shown it.

    **A degenerate range raises unless a fallback is *declared*.** Spec §22.2 says to "fall back to
    a
    **declared** positive scale" — and a library default is not a declaration. A silent substitution
    of ``s_J = 1`` would rescale every β on this strain's ladder, so its rungs would no longer name
    the same selection pressure as any other strain's in the batch. That is precisely the failure
    ``s_J`` exists to prevent, and it would arrive as a log line nobody read. So: no fallback, no
    run. The caller who wants one declares it (``sampler.energy_scale_fallback``), and the manifest
    records that it was used.
    """
    if optimum.objective_key != objective.objective_key:
        raise IncompatibleObjectiveError(
            "the LP optimum was solved for a different objective than the one whose J is being "
            f"measured (optimum.objective_key = {optimum.objective_key[:16]}…, "
            f"objective.objective_key = {objective.objective_key[:16]}…). ``s_J = J* − Q_q(J(W))`` "
            "would subtract a quantile of one objective from the optimum of another — the two "
            "share a polytope, so every shape check passes and the answer is a plausible number "
            "that is the difference of two different functions. Re-solve the LP against this "
            "objective (`solve_sparse_objective`), which is what M7's reweighting does after it "
            "freezes the weights."
        )
    if optimum.polytope_key != objective.polytope_key:
        raise IncompatibleObjectiveError(
            "the LP optimum was solved on a different polytope than this objective was lowered "
            "from "
            f"(optimum.polytope_key = {optimum.polytope_key[:16]}…, objective.polytope_key = "
            f"{objective.polytope_key[:16]}…). The two objectives can hash identically — the "
            "objective key does not cover the polytope's bounds or stoichiometry — so ``J*`` from "
            "one polytope would be subtracted from a warm-up quantile of another, rescaling every "
            "β "
            "on the ladder while every shape and objective-key check passes (Codex, M7 review "
            "round 2). Solve the LP on the same polytope the warm-up points came from."
        )
    if warmup_polytope_key != objective.polytope_key:
        raise IncompatibleObjectiveError(
            "the warm-up points came from a different polytope than this objective was lowered "
            f"from (warmup_polytope_key = {warmup_polytope_key[:16]}…, objective.polytope_key = "
            f"{objective.polytope_key[:16]}…). ``Q_q(J(W))`` would then be a quantile of ``J`` "
            "the wrong support set — same shape, different geometry — and ``s_J`` would silently "
            "rescale the ladder (Codex, M7 review round 3). Pass the `polytope_key` of the "
            "`ReducedGeometry` whose `support_points` you are handing in."
        )
    j_star = optimum.j_star

    if not isinstance(mode, str):
        declared = float(mode)
        if not np.isfinite(declared) or declared <= 0.0:
            raise ObjectiveError(f"a declared energy_scale must be finite and > 0, got {declared}")
        return EnergyScale(
            value=declared,
            mode="declared",
            j_star=float(j_star),
            n_warmup_points=0,
            quantile=None,
            warmup_quantile_j=None,
            warmup_max_j=None,
            resolution=None,
            polytope_key=objective.polytope_key,
            objective_key=objective.objective_key,
            fell_back=False,
        )

    if mode != "warmup_range":
        raise ObjectiveError(
            f"energy_scale must be 'warmup_range' or a positive number, got {mode!r}"
        )
    if not 0.0 < quantile < 1.0:
        raise ObjectiveError(f"energy_scale_quantile must lie in (0, 1), got {quantile}")
    if not np.isfinite(j_star):
        raise ObjectiveError(f"j_star must be finite, got {j_star}")

    warmup = np.atleast_2d(np.asarray(warmup_fluxes, dtype=VALUE_DTYPE))
    if warmup.shape[0] == 0:
        raise ObjectiveError("the warm-up range needs at least one warm-up point")

    _, _, j_warmup = objective.evaluate_many(warmup)
    low = float(np.quantile(j_warmup, quantile))
    scale = float(j_star) - low

    resolution = energy_scale_resolution(j_star, low)
    fell_back = not (scale > resolution)
    if fell_back:
        if fallback is None:
            raise DegenerateEnergyScaleError(
                f"the warm-up objective range is not resolvable: s_J = J* − Q_{quantile}(J(W)) = "
                f"{scale:.6g}, at or below the float64 resolution of that subtraction "
                f"({resolution:.3e}, i.e. {ENERGY_SCALE_ULP_MARGIN:.0f} ULPs of |J*| = "
                f"{abs(j_star):.6g}). Every warm-up point has essentially the LP-optimal "
                "objective, so there is no observed range for β to be measured against and any β "
                "would be meaningless. Either the objective does not discriminate on this polytope "
                "(check λ and the penalty set), or you must DECLARE a scale: set "
                "sampler.energy_scale to a positive number, or sampler.energy_scale_fallback to "
                "accept one here. Falling back silently would rescale every β on this strain's "
                "ladder and make its rungs incomparable with the batch (spec §22.2)."
            )
        declared_fallback = float(fallback)
        if not np.isfinite(declared_fallback) or declared_fallback <= 0.0:
            raise ObjectiveError(
                f"the declared energy-scale fallback must be finite and > 0, got {fallback}"
            )
        _log.warning(
            "the warm-up objective range is not resolvable (s_J = %.3e, at or below the %.3e "
            "resolution of J* − Q(J(W))). Using the DECLARED fallback s_J = %.3g (spec §22.2). "
            "β is now in reciprocal raw-objective units for this model and is NOT comparable with "
            "any other strain's ladder.",
            scale,
            resolution,
            fallback,
        )

    return EnergyScale(
        value=declared_fallback if fell_back else scale,
        mode="warmup_range",
        j_star=float(j_star),
        n_warmup_points=int(warmup.shape[0]),
        quantile=quantile,
        warmup_quantile_j=low,
        warmup_max_j=float(np.max(j_warmup)),
        resolution=resolution,
        polytope_key=objective.polytope_key,
        objective_key=objective.objective_key,
        fell_back=fell_back,
    )
