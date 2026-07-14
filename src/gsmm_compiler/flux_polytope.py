"""`FluxPolytope` (canonical IR) and `ReducedPolytope` (the reduced IR).

The canonical polytope is ``{v : S v = 0, l ≤ v ≤ u}`` over **all** reactions. Reactions with
``l == u`` cannot vary, so the reduced IR eliminates them from the sampled state (BUILD_PLAN §1.5):

    v_full = R · v_red + c            R = the 0/1 scatter onto free columns, c = the fixed values
    S_F · v_red = −S_fixed · v_fixed  the mass balance becomes **affine**, not homogeneous

That right-hand side is the part it is easy to get wrong. It vanishes only when every fixed reaction
sits at zero (as they all happen to in the example model, where the 513 fixed reactions are
FVA-blocked); a model with a nonzero fixed flux — a forced ATP maintenance demand, say — has a
genuinely nonzero RHS, and treating the reduced system as homogeneous would silently sample a
different polytope. So the RHS is computed and carried explicitly, and `to_full`'s fast index
path is tested against the explicitly materialized ``R``.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.native_csc import VALUE_DTYPE, NativeCSC
from gsmm_compiler.provenance import IR_SCHEMA_VERSION, content_key

DEFAULT_FEASIBILITY_TOL = 1e-9


class InvalidPolytopeError(ValueError):
    """The polytope description is inconsistent (shapes, bounds, or the biomass reaction)."""


@dataclass(frozen=True)
class FluxPolytope:
    """The canonical flux polytope over every reaction in the model, in frozen model order."""

    reaction_ids: tuple[str, ...]
    metabolite_ids: tuple[str, ...]
    stoichiometry: NativeCSC
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    biomass_index: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        n, m = self.n_reactions, self.n_metabolites

        if self.stoichiometry.shape != (m, n):
            raise InvalidPolytopeError(
                f"stoichiometry is {self.stoichiometry.shape}, expected ({m}, {n}) "
                "= (metabolites, reactions)"
            )
        for name, bounds in (
            ("lower_bounds", self.lower_bounds),
            ("upper_bounds", self.upper_bounds),
        ):
            if bounds.shape != (n,):
                raise InvalidPolytopeError(f"{name} has shape {bounds.shape}, expected ({n},)")
            if bounds.dtype != VALUE_DTYPE:
                raise InvalidPolytopeError(f"{name} has dtype {bounds.dtype}, expected float64")
            if not np.all(np.isfinite(bounds)):
                offenders = np.flatnonzero(~np.isfinite(bounds))[:5]
                raise InvalidPolytopeError(
                    f"{name} must be finite; offending reactions: "
                    f"{[self.reaction_ids[i] for i in offenders]}"
                )

        violated = np.flatnonzero(self.lower_bounds > self.upper_bounds)
        if violated.size:
            raise InvalidPolytopeError(
                "lower bound exceeds upper bound for "
                f"{[self.reaction_ids[i] for i in violated[:5]]}"
            )

        if not 0 <= self.biomass_index < n:
            raise InvalidPolytopeError(f"biomass_index {self.biomass_index} outside [0, {n})")
        if len(set(self.reaction_ids)) != n:
            raise InvalidPolytopeError("reaction_ids contain duplicates")
        if len(set(self.metabolite_ids)) != m:
            raise InvalidPolytopeError("metabolite_ids contain duplicates")

    # ---- structure ----------------------------------------------------------------------------

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_ids)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolite_ids)

    @property
    def biomass_id(self) -> str:
        return self.reaction_ids[self.biomass_index]

    @cached_property
    def fixed_mask(self) -> NDArray[np.bool_]:
        """``l == u`` — reactions that cannot carry a variable flux.

        Exact equality, deliberately: a *tolerance* here would fix a reaction whose bounds merely
        sit close together, deleting a real (if narrow) degree of freedom from the polytope.
        """
        return np.equal(self.lower_bounds, self.upper_bounds)

    @cached_property
    def fixed_indices(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.fixed_mask).astype(np.intp)

    @cached_property
    def free_indices(self) -> NDArray[np.intp]:
        return np.flatnonzero(~self.fixed_mask).astype(np.intp)

    @property
    def n_free(self) -> int:
        return int(self.free_indices.size)

    @property
    def n_fixed(self) -> int:
        return int(self.fixed_indices.size)

    # ---- membership ---------------------------------------------------------------------------

    def mass_balance_residual(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """``S v`` — zero (to tolerance) for any steady-state flux vector."""
        return self.stoichiometry.matvec(np.asarray(v, dtype=VALUE_DTYPE))

    def contains(self, v: NDArray[np.float64], tol: float = DEFAULT_FEASIBILITY_TOL) -> bool:
        """True when ``v`` satisfies the bounds and the steady-state constraint within ``tol``."""
        flux = np.asarray(v, dtype=VALUE_DTYPE)
        if flux.shape != (self.n_reactions,):
            raise ValueError(f"v has shape {flux.shape}, expected ({self.n_reactions},)")
        within_bounds = bool(
            np.all(flux >= self.lower_bounds - tol) and np.all(flux <= self.upper_bounds + tol)
        )
        balanced = bool(np.max(np.abs(self.mass_balance_residual(flux)), initial=0.0) <= tol)
        return within_bounds and balanced

    # ---- keys ---------------------------------------------------------------------------------

    def content_key(self) -> str:
        """The L1 cache key: everything that can change the bytes of the reduced IR (§1.1)."""
        return content_key(
            schema_version=IR_SCHEMA_VERSION,
            reaction_ids=list(self.reaction_ids),
            metabolite_ids=list(self.metabolite_ids),
            starts=self.stoichiometry.starts,
            indices=self.stoichiometry.indices,
            values=self.stoichiometry.values,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            biomass_index=self.biomass_index,
        )

    # ---- reduction ----------------------------------------------------------------------------

    def reduce(self) -> ReducedPolytope:
        """Eliminate the ``l == u`` reactions, yielding the polytope the sampler actually walks."""
        free = self.free_indices
        fixed = self.fixed_indices

        fixed_flux_full = np.zeros(self.n_reactions, dtype=VALUE_DTYPE)
        fixed_flux_full[fixed] = self.lower_bounds[fixed]

        # S v_full = 0 with v_full = R v_red + c  ⇒  S_F v_red = −S c = −S_fixed v_fixed.
        rhs = -self.stoichiometry.matvec(fixed_flux_full)

        biomass_reduced: int | None = None
        if not bool(self.fixed_mask[self.biomass_index]):
            biomass_reduced = int(np.searchsorted(free, self.biomass_index))

        return ReducedPolytope(
            reaction_ids=self.reaction_ids,
            metabolite_ids=self.metabolite_ids,
            free_indices=free,
            fixed_indices=fixed,
            fixed_values=self.lower_bounds[fixed].copy(),
            stoichiometry=self.stoichiometry.select_columns(free),
            rhs=rhs,
            lower_bounds=self.lower_bounds[free].copy(),
            upper_bounds=self.upper_bounds[free].copy(),
            biomass_index=biomass_reduced,
            biomass_full_index=self.biomass_index,
            n_full=self.n_reactions,
        )


@dataclass(frozen=True)
class ReducedPolytope:
    """``{v_red : S_F v_red = rhs, l_F ≤ v_red ≤ u_F}`` — the sampled state space.

    Identity is preserved: `to_full` lifts any reduced vector back to a full-length flux vector in
    the original reaction order, so every saved sample stays a full model flux (CLAUDE.md).
    """

    reaction_ids: tuple[str, ...]
    """Full-model reaction IDs, in the original frozen order."""
    metabolite_ids: tuple[str, ...]
    free_indices: NDArray[np.intp]
    fixed_indices: NDArray[np.intp]
    fixed_values: NDArray[np.float64]
    stoichiometry: NativeCSC
    """``S_F``: the stoichiometry restricted to the free columns."""
    rhs: NDArray[np.float64]
    """``−S_fixed v_fixed``. Nonzero whenever any fixed reaction carries flux."""
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    biomass_index: int | None
    """Biomass in *reduced* coordinates, or ``None`` if biomass itself is fixed.

    A **coordinate**, not an identity. When biomass is fixed it has no reduced coordinate, and this
    is ``None`` for *every* such model — so it cannot say **which** reaction the biomass is.
    `biomass_full_index` can, and must.
    """
    biomass_full_index: int
    """Which reaction *is* the biomass, in full-model coordinates. **Always** known.

    Added because `biomass_index` alone cannot answer the question, and the difference is not
    academic. Two models identical except that biomass is a *different fixed reaction* both reduce
    to ``biomass_index = None``; they used to hash to the same `content_key`, so an objective
    lowered from one would **bind** to the other — and their `mu_offset` (the fixed biomass's own
    flux, which is the entire ``μ`` on such a model) differs. The chain would report one strain's
    growth as another's, with every diagnostic green. (Codex, M6 review round 3.)

    The reduced IR simply *did not retain* this: `reduce()` computed the reduced coordinate and
    threw the identity away. `sparse_objective.biomass_maximum` had to borrow the index from the
    *objective* to work around it — the polytope trusting the objective to tell it what it is,
    exactly backwards.
    """
    n_full: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        n_free = int(self.free_indices.size)

        if self.stoichiometry.n_cols != n_free:
            raise InvalidPolytopeError(
                f"reduced stoichiometry has {self.stoichiometry.n_cols} columns, "
                f"expected {n_free} free reactions"
            )
        if self.rhs.shape != (len(self.metabolite_ids),):
            raise InvalidPolytopeError(
                f"rhs has shape {self.rhs.shape}, expected ({len(self.metabolite_ids)},)"
            )
        if self.fixed_values.shape != self.fixed_indices.shape:
            raise InvalidPolytopeError("fixed_values and fixed_indices differ in length")
        if int(self.free_indices.size + self.fixed_indices.size) != self.n_full:
            raise InvalidPolytopeError(
                "free and fixed indices do not partition the full reaction set"
            )
        if np.intersect1d(self.free_indices, self.fixed_indices).size:
            raise InvalidPolytopeError("free and fixed indices overlap")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise InvalidPolytopeError("reduced lower bound exceeds upper bound")
        if self.biomass_index is not None and not 0 <= self.biomass_index < n_free:
            raise InvalidPolytopeError(
                f"reduced biomass_index {self.biomass_index} outside [0, {n_free})"
            )
        if not 0 <= self.biomass_full_index < self.n_full:
            raise InvalidPolytopeError(
                f"biomass_full_index {self.biomass_full_index} outside [0, {self.n_full})"
            )

        # `free_indices` must be **strictly increasing**. `reduce()` builds it with
        # `np.flatnonzero`, which always is — but a `ReducedPolytope` can be constructed directly,
        # and half this class (plus `reduce`'s own `searchsorted`) silently assumes the order. Left
        # unchecked, an unsorted `free_indices` makes the reduced coordinates mean something other
        # than what every consumer believes they mean. (Codex, M6 review round 4.)
        if np.any(np.diff(self.free_indices) <= 0):
            raise InvalidPolytopeError(
                "free_indices must be strictly increasing: the reduced coordinates are positions "
                "in this array, and every consumer assumes they are in full-model order"
            )

        # The two biomass fields describe the same reaction in two coordinate systems, and they must
        # agree — a reduced index that does not point at `biomass_full_index` is a `J` that rewards
        # the wrong reaction, and nothing downstream can see it.
        #
        # Checked by **dereferencing**, not by `searchsorted`: the direct lookup is correct whatever
        # the order of `free_indices`, where a `searchsorted` silently agrees with itself on an
        # unsorted array. (Codex, M6 review round 4 — this was the actual bypass.)
        is_free = bool(np.isin(self.biomass_full_index, self.free_indices))
        if is_free:
            if self.biomass_index is None:
                raise InvalidPolytopeError(
                    f"biomass (full index {self.biomass_full_index}) is a free reaction, so it "
                    "has a reduced coordinate; biomass_index must not be None"
                )
            if int(self.free_indices[self.biomass_index]) != self.biomass_full_index:
                raise InvalidPolytopeError(
                    f"biomass_index {self.biomass_index} points at full reaction "
                    f"{int(self.free_indices[self.biomass_index])}, but biomass_full_index says "
                    f"the biomass is {self.biomass_full_index}"
                )
        elif self.biomass_index is not None:
            raise InvalidPolytopeError(
                f"biomass (full index {self.biomass_full_index}) is a FIXED reaction, so it has no "
                f"reduced coordinate; biomass_index must be None, not {self.biomass_index}"
            )

    def is_reduction_of(self, polytope: FluxPolytope) -> bool:
        """Was this produced by ``polytope.reduce()``? Compared by **content**, not identity.

        Any function taking *both* a canonical and a reduced polytope is implicitly claiming they
        are the same model, and until M6 nothing checked it. `sparse_objective.resolve_objective` is
        the case that bit: it reads `origin_is_feasible` off the **canonical** bounds and computes
        ``λ*`` off the **reduced** LP. Hand it a forced-flux canonical polytope together with the
        reduction of an origin-feasible variant sharing the same reaction IDs, and BUILD_PLAN §1.7's
        guard inverts — ``λ̃ ≥ 1`` is accepted and *recorded as safe*, because the origin is
        infeasible in the polytope that was asked, while the polytope actually sampled collapses to
        the origin. (Codex, M6 review round 5.)

        Costs one `reduce()` — an O(nnz) pass, once per model, at config-resolution time. Nowhere
        near a loop, and the alternative is a run that documents the opposite of what it did.
        """
        return self.content_key() == polytope.reduce().content_key()

    @property
    def biomass_id(self) -> str:
        """Which reaction the biomass *is* — answerable whether or not it was eliminated."""
        return self.reaction_ids[self.biomass_full_index]

    @property
    def biomass_is_fixed(self) -> bool:
        return self.biomass_index is None

    @property
    def n_free(self) -> int:
        return int(self.free_indices.size)

    @property
    def n_fixed(self) -> int:
        return int(self.fixed_indices.size)

    @property
    def is_singleton(self) -> bool:
        """Every reaction fixed: the polytope is a single point and there is nothing to sample."""
        return self.n_free == 0

    @cached_property
    def offset(self) -> NDArray[np.float64]:
        """``c`` in ``v_full = R v_red + c``: the fixed fluxes, scattered to full length."""
        c = np.zeros(self.n_full, dtype=VALUE_DTYPE)
        c[self.fixed_indices] = self.fixed_values
        return c

    def content_key(self) -> str:
        """The **L1 key** (BUILD_PLAN §1.1): *which polytope this is*, by content, not by identity.

        Everything downstream — the geometry, the rounded transform, the lowered objective, the
        energy scale — is computed *against a particular reduced polytope*, and is meaningless
        against any other. This key is what lets each of them say so, and what lets the consumer
        check rather than assume.

        The check is not academic. A `sparse_objective.ReducedObjective` is just indices and
        weights: hand the sampler one lowered from a *different* model of the same size and the
        chain tilts by the wrong reactions, the traces report those same wrong reactions as
        biomass — so the diagnostics **confirm the wrong target** — and feasibility, mass balance,
        chords and R̂ all stay green, because nothing else in this package knows which reaction
        ``J`` is supposed to be about. M8's cache makes exactly that pairing possible: two
        artifacts,
        loaded from disk, that were never computed against each other.
        """
        return content_key(
            impl_version=IR_SCHEMA_VERSION,
            reaction_ids=list(self.reaction_ids),
            free_indices=self.free_indices,
            fixed_values=self.fixed_values,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            starts=self.stoichiometry.starts,
            indices=self.stoichiometry.indices,
            values=self.stoichiometry.values,
            rhs=self.rhs,
            # **The biomass reaction's IDENTITY, not its reduced coordinate.** Both halves of that
            # sentence were learned the hard way, in successive rounds of the M6 review.
            #
            # It has to be *in* the key at all (round 2): the geometry does not depend on which
            # reaction is biomass — the affine hull is the same either way — so it looks like a
            # spurious dependency and was omitted for exactly that reason. But the key's whole *job*
            # is to stop a `ReducedObjective` binding to a polytope it was not lowered from, and
            # two polytopes differing **only** in their biomass reaction would otherwise share a
            # key: the objective binds, tilts the wrong reaction, and its traces confirm it.
            #
            # And it has to be the **full-model** index (round 3): every model whose biomass is a
            # *fixed* reaction reduces to `biomass_index = None`, so hashing the reduced coordinate
            # collapses all of them onto the same value. Two such models — biomass fixed at 1.0 in
            # one, at 2.0 in the other — then share a key, cross-bind, and report one strain's
            # growth as the other's. `μ` *is* that constant on such a model, so the whole objective
            # is wrong.
            #
            # The cost is a false cache miss in M8 whenever `biomass_id` changes and the geometry
            # could in principle have been reused. A false miss recomputes; a false *hit* corrupts.
            biomass_full_index=self.biomass_full_index,
        )

    def reconstruction_matrix(self) -> NativeCSC:
        """``R``, the ``n_full × n_free`` scatter matrix, materialized.

        `to_full` does this by indexing instead — that is what runs in production. ``R`` exists so
        the tests can confirm the fast path really computes ``R v_red + c`` and not something
        adjacent to it.
        """
        return NativeCSC.from_columns(self.n_full, [{int(row): 1.0} for row in self.free_indices])

    def to_full(self, v_reduced: NDArray[np.float64]) -> NDArray[np.float64]:
        """Lift reduced coordinates to a full-length flux vector: ``v_full = R v_red + c``.

        Accepts one vector ``(n_free,)`` or a batch ``(n_samples, n_free)``.
        """
        reduced = np.asarray(v_reduced, dtype=VALUE_DTYPE)
        if reduced.ndim == 0 or reduced.shape[-1] != self.n_free:
            raise ValueError(
                f"v_reduced has shape {reduced.shape}, expected trailing dimension {self.n_free}"
            )
        full = np.broadcast_to(self.offset, (*reduced.shape[:-1], self.n_full)).copy()
        full[..., self.free_indices] = reduced
        return full

    def to_reduced(self, v_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project a full flux vector onto the free coordinates — the inverse of `to_full`."""
        full = np.asarray(v_full, dtype=VALUE_DTYPE)
        if full.ndim == 0 or full.shape[-1] != self.n_full:
            raise ValueError(
                f"v_full has shape {full.shape}, expected trailing dimension {self.n_full}"
            )
        return full[..., self.free_indices].copy()

    def mass_balance_residual(self, v_reduced: NDArray[np.float64]) -> NDArray[np.float64]:
        """``S_F v_red − rhs`` — zero (to tolerance) for a feasible reduced flux."""
        return self.stoichiometry.matvec(np.asarray(v_reduced, dtype=VALUE_DTYPE)) - self.rhs

    def contains(
        self, v_reduced: NDArray[np.float64], tol: float = DEFAULT_FEASIBILITY_TOL
    ) -> bool:
        """True when the reduced vector satisfies the affine mass balance and its bounds."""
        reduced = np.asarray(v_reduced, dtype=VALUE_DTYPE)
        if reduced.shape != (self.n_free,):
            raise ValueError(f"v_reduced has shape {reduced.shape}, expected ({self.n_free},)")
        within_bounds = bool(
            np.all(reduced >= self.lower_bounds - tol)
            and np.all(reduced <= self.upper_bounds + tol)
        )
        residual = self.mass_balance_residual(reduced)
        return within_bounds and bool(np.max(np.abs(residual), initial=0.0) <= tol)
