"""Reweighted-L1: a stronger zero pressure, with the weights **frozen before sampling** (M7).

Spec §13. Starting from base weights ``w^(0)``, iterate

    1. solve the sparse-objective LP with weights ``w^(k)``
    2. ``w_r^(k+1) = w_r^base / (|v_r^(k)| + ε)``
    3. clip to the configured limits
    4. renormalize the positive weights to unit median
    5. stop when the active set **and** the solution have both settled

and then freeze. The point is to approximate a penalty on the *number* of active reactions while
keeping every subproblem an LP: a reaction carrying a large flux gets a small weight, so its cost
stops growing with its magnitude, and each active reaction ends up costing roughly a constant. It is
**experimental** and it does **not** solve exact cardinality minimization (spec §13) — it is a
heuristic surrogate, and the run says so.

Nothing here may ever see a chain. The weights are computed from **LP optima only**; a weight that
moves once sampling has started retargets every one-dimensional conditional the kernel builds and
destroys the stationarity argument outright (BUILD_PLAN §1.6.2). That is enforced structurally
rather than by convention: this module cannot import the sampler, the sampler cannot import this
module (`test_the_sampler_cannot_reach_the_reweighter`), and the weight buffers the chain finally
holds are physically read-only.

## λ is re-resolved from the current weights, every iteration  (the M3 decision, settled here)

``λ* = max_v μ(v)/C_w(v)`` is a function of **w** — halve every weight and λ* doubles — so the fork
BUILD_PLAN §1.7 left M7 is: does the raw λ stay frozen at its base-weight value through the loop, or
is it re-resolved from the weights actually in use? It stayed open because "either is defensible".

**Measurement closed it, and it is not a close call.** On the example model, the reweighting loop
moves ``λ*`` from 1.9e-3 to ~4e2 at the default [1e-3, 1e3] clip — and to ~2.3e5 at a wider clip —
because ``C_w`` changes *units*: under base weights it is a sum of absolute fluxes (~1.6e4 here),
and under reweighted ones it is very nearly a count of active reactions. A λ calibrated against the
first means nothing against the second. Hold λ frozen and the *effective* selection pressure
``λ̃_eff = λ/λ*(w)`` falls from the requested **0.5 to ~4e-6** after a single iteration (to 4e-9 at
the wider clip): the reweighting loop annihilates the very sparsity pressure it exists to
strengthen.

And it does not fail quietly-but-harmlessly. It fails *loudly, one layer down*: many reactions end
up with an effective cost ``λw`` below HiGHS's dual feasibility tolerance, so their ``z`` columns
have nothing pushing them onto ``|v_r|`` (the M3 finding), and by the **second** iteration M3's
``z == |v|`` gate fires — a deviation of 25 at the default clip, 1e3 at the wider one. The frozen-λ
policy does not merely mean something different from what it says — **it hands the solver an LP
whose L1 linearization cannot close.** (Both measured; the crash reproduces at the config default,
not only at a pathological clip.)

So: **λ^(k) = λ̃ · λ*(w^(k))**, one `resolve_objective` call per iteration, and the frozen final
objective carries ``λ = λ̃ · λ*(w_final)``. The user's dial stays λ̃ — the dimensionless selection
pressure of §1.7 — and it goes on meaning the same thing across the loop and across the batch, which
is the entire reason §1.7 introduced it.

This choice also buys an *invariance*, and the invariance is what makes step 4 legitimate: scaling
every weight by ``c`` sends ``C → C/c`` and hence ``λ* → c·λ*`` and ``λ → c·λ``, so the product
``λ·w`` — the only thing ``J`` ever uses — is **exactly unchanged**. The median renormalization
therefore *cannot move the target distribution*; it is a conditioning step, not a modelling one.
Under a frozen λ it would have been neither: it would have rescaled the selection pressure by an
arbitrary median every iteration. (Tested in `test_normalization_cannot_move_the_target`. The
invariance is exact in exact arithmetic and holds to LP tolerance over the range the LP can
actually solve — push the scale far enough and the Charnes–Cooper LP degrades, which is the honest
reason the normalization exists.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gsmm_compiler.config import ObjectiveConfig
from gsmm_compiler.native_csc import VALUE_DTYPE
from gsmm_compiler.sparse_objective import (
    ObjectiveError,
    ObjectiveScale,
    SparseFluxObjective,
    SparseObjectiveSolution,
    _frozen,
    resolve_objective,
    solve_sparse_objective,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gsmm_compiler.flux_polytope import FluxPolytope, ReducedPolytope

__all__ = [
    "ReweightingError",
    "ReweightingNotConvergedError",
    "ReweightingReport",
    "ReweightingStep",
    "reweight_objective",
    "update_weights",
]

_log = logging.getLogger(__name__)


class ReweightingError(ObjectiveError):
    """The reweighting loop could not produce a usable frozen weight vector."""


class ReweightingNotConvergedError(ReweightingError):
    """The active set and solution had not both settled within the iteration budget."""


# ---- step 2–4: the weight update ----------------------------------------------------------------


def update_weights(
    base_weights: NDArray[np.float64],
    v_full: NDArray[np.float64],
    penalty_mask: NDArray[np.bool_],
    *,
    epsilon: float,
    clip: tuple[float, float],
) -> tuple[NDArray[np.float64], int, int]:
    """``w_r ← w_r^base/(|v_r| + ε)``, clipped, then renormalized to unit median (spec §13, 2–4).

    Returns the new weights and how many landed on the low and high clip bounds.

    ``base_weights`` is ``w^base`` and never changes: the update is always taken from the *base*
    weights divided by the *current* flux, not from the current weights. Iterating on the current
    weights instead would compound the division and drive the vector to the clip bounds
    geometrically — a different algorithm, and not the one spec §13 specifies.

    The flux enters as the **exact** ``|v_r|``. Nothing is snapped to zero here (CLAUDE.md): ``ε``
    and the clip ceiling bound the update where ``v_r`` vanishes, and they do it without pretending
    a small flux is not there.
    """
    flux = np.asarray(v_full, dtype=VALUE_DTYPE)
    low, high = clip
    if not 0.0 < low < high:
        raise ReweightingError(f"weight clip must satisfy 0 < min < max, got {low} / {high}")
    if epsilon <= 0.0:
        raise ReweightingError(f"epsilon must be > 0, got {epsilon}")

    raw = np.zeros_like(base_weights)
    raw[penalty_mask] = base_weights[penalty_mask] / (np.abs(flux[penalty_mask]) + epsilon)

    clipped = np.where(penalty_mask, np.clip(raw, low, high), 0.0)
    at_low = int(np.count_nonzero(penalty_mask & (raw <= low)))
    at_high = int(np.count_nonzero(penalty_mask & (raw >= high)))

    positive = clipped[clipped > 0.0]
    if positive.size == 0:
        raise ReweightingError(
            "every weight is zero, so there is nothing to reweight and no median to normalize by"
        )

    # Step 4. Mathematically a no-op — λ = λ̃·λ*(w) absorbs any scaling of w exactly (see the module
    # docstring) — so this cannot move the target. It exists to keep λ·w in a range HiGHS can solve.
    return clipped / float(np.median(positive)), at_low, at_high


def _relative_weight_change(
    new: NDArray[np.float64], old: NDArray[np.float64], penalty_mask: NDArray[np.bool_]
) -> float:
    """``max_r |new_r − old_r| / max(new_r, old_r)`` over the penalty set — the fixed-point metric.

    Per-reaction and relative, so it sees a sparsity-critical weight still halving even while a
    large-flux reaction's weight is dead still. Both weights are ≥ the clip floor on the penalty
    set, so the denominator is strictly positive — no reaction is divided by noise (the M4 rule)."""
    both = penalty_mask & ((new > 0.0) | (old > 0.0))
    if not np.any(both):
        return 0.0
    denominator = np.maximum(new[both], old[both])
    # denominator > 0 on `both` by construction; the config caps the clip ratio so it cannot
    # underflow to a subnormal, but guard anyway rather than divide by a number that is not there.
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        return float("inf")
    return float(np.max(np.abs(new[both] - old[both]) / denominator))


# ---- the record of what the loop did -------------------------------------------------------------


@dataclass(frozen=True)
class ReweightingStep:
    """One iteration: the weights that went **in**, and the LP optimum that came **out**.

    Spec §13 requires every weight vector and every LP solution to be saved. This is that record,
    and it is what lets a reader see the loop converge rather than take its word for it.
    """

    iteration: int
    weights: NDArray[np.float64]
    """``w^(k)`` — full length, read-only. The weights this iteration's LP was *solved with*."""
    l1_penalty: float
    """``λ^(k) = λ̃ · λ*(w^(k))`` — re-resolved from *these* weights (see the module docstring)."""
    critical_l1_penalty: float
    """``λ*(w^(k))``. Watch it move by 1e8 across the first step; that is the whole argument."""
    solution: SparseObjectiveSolution
    n_active: int
    """``|{r : |v_r| > active_tol}|`` — the active-set size, for the convergence test only."""
    max_weight_change: float
    """The convergence metric that actually protects the freeze: the largest **per-reaction**
    change between ``w^(k)`` and the candidate next weights ``w^(k+1) = F(v^(k))``, i.e.
    ``max_r |w^(k+1)_r − w^(k)_r| / max(w^(k)_r, w^(k+1)_r)``. ``inf`` on iterations that have no
    candidate yet.

    **Why not the flux change.** The frozen artifact is the *weights*, and convergence must mean
    "the weights have stopped moving" — that ``w^(k)`` is a fixed point of ``w ↦ F(solve(w))``, so
    freezing it samples that fixed point rather than one stale step short of it. A flux-based test
    (Codex, M7 review) is blind exactly where it must not be: with one flux at 1e3 and a
    sparsity-critical flux at 1e-3, a *global* relative ``max|Δv|`` is dominated by the large
    coordinate and can read 1e-6 while the small coordinate's weight is still halving. A
    per-reaction relative *weight* change sees the coordinate that matters, because it is
    normalized by that coordinate's own magnitude, not the vector's."""
    max_flux_change: float
    """Relative ``max|Δv|`` against the previous iterate — a **diagnostic**, no longer the stop test
    (see `max_weight_change`). ``inf`` on the first."""
    active_set_changes: int
    """Size of the symmetric difference with the previous active set. ``-1`` on the first."""
    n_at_clip_low: int
    n_at_clip_high: int
    """How many of the weights *produced* by this step hit a clip bound. If every one does, the clip
    has flattened the weight vector to two values and the surrogate has degenerated."""

    def manifest(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "l1_penalty": self.l1_penalty,
            "critical_l1_penalty": self.critical_l1_penalty,
            "n_active": self.n_active,
            "max_weight_change": self.max_weight_change,
            "max_flux_change": self.max_flux_change,
            "active_set_changes": self.active_set_changes,
            "n_at_clip_low": self.n_at_clip_low,
            "n_at_clip_high": self.n_at_clip_high,
            **self.solution.diagnostics(),
        }


@dataclass(frozen=True)
class ReweightingReport:
    """The **frozen** objective, the optimum rebuilt from it, and the whole history (spec §13).

    `objective` is what the sampler must be given. Its weights are physically read-only and its λ is
    ``λ̃ · λ*(w_final)``. `solution` is the L2 optimum **re-solved against those exact weights**, so
    ``J*`` and ``s_J`` are computed from the objective that will actually be sampled — which is the
    join `sparse_objective.choose_energy_scale` now refuses to make on trust.
    """

    objective: SparseFluxObjective
    scale: ObjectiveScale
    solution: SparseObjectiveSolution
    history: tuple[ReweightingStep, ...]
    converged: bool

    active_base: frozenset[int]
    """The active set at the **base** weights — what plain L1 alone lights up."""
    active_final: frozenset[int]
    """The active set at the frozen weights."""

    @property
    def n_active_base(self) -> int:
        return len(self.active_base)

    @property
    def n_active_final(self) -> int:
        return len(self.active_final)

    @property
    def n_turned_off(self) -> int:
        """Reactions plain L1 had **on** that reweighting switched **off** — sparsity it bought."""
        return len(self.active_base - self.active_final)

    @property
    def n_turned_on(self) -> int:
        """Reactions plain L1 had **off** that reweighting switched **on**. Nonzero here means the
        support was *rearranged*, not merely thinned — legitimate, but it makes the net count a
        misleading headline (Codex, M7 review)."""
        return len(self.active_final - self.active_base)

    @property
    def n_shed(self) -> int:
        """**Net** reactions shed: ``n_turned_off − n_turned_on``. The scientific headline — did the
        stronger zero pressure reduce the active count — but read it next to `n_turned_on`, because
        a reweighting that swaps one reaction for another reports a net of zero while having changed
        the support entirely."""
        return self.n_active_base - self.n_active_final

    @property
    def support_unchanged(self) -> bool:
        """The active set is **identical** before and after — the symmetric difference is empty.

        This, not ``n_shed == 0``, is the true "reweighting did nothing to the support" signal: net
        shed is zero under a straight swap too, and would warn spuriously. A clip ceiling below the
        "nearly off" band (``|v| ≈ 1e-3``) merges it with "off" and the surrogate has no lever —
        the loop converges, the LP is healthy, and the support is exactly plain L1's. At clip
        [1e-2, 1e2] on the example model that is precisely what happens; [1e-3, 1e3] sheds 3."""
        return self.active_base == self.active_final

    @property
    def n_iterations(self) -> int:
        return len(self.history)

    def manifest(self) -> dict[str, object]:
        return {
            "reweighting": {
                "converged": self.converged,
                "n_iterations": self.n_iterations,
                "n_active_base": self.n_active_base,
                "n_active_final": self.n_active_final,
                "n_shed": self.n_shed,
                "n_turned_off": self.n_turned_off,
                "n_turned_on": self.n_turned_on,
                "support_unchanged": self.support_unchanged,
                "experimental": True,
                "claims_exact_cardinality": False,
                "steps": [step.manifest() for step in self.history],
            },
            "objective": {**self.objective.manifest(), "scale": self.scale.manifest()},
        }


# ---- the loop ------------------------------------------------------------------------------------


def reweight_objective(
    polytope: FluxPolytope,
    reduced: ReducedPolytope | None = None,
    config: ObjectiveConfig | None = None,
    *,
    penalty_ids: tuple[str, ...] | None = None,
    base_weights: NDArray[np.float64] | None = None,
    threads: int = 1,
    allow_unconverged: bool = False,
) -> ReweightingReport:
    """Run the reweighting loop to a fixed point and **freeze** the result (spec §13).

    Takes a polytope and a config. It takes **no sampler, no chain and no flux** — there is no
    parameter through which MCMC state could reach the weights, which is the one invariant this
    whole milestone exists to protect. Every ``v`` it reweights from is an LP optimum it solved
    itself.

    Costs two LPs per iteration (Charnes–Cooper for ``λ*``, then the ``(v, z)`` sparse LP) plus the
    biomass-only diagnostic. That is cheap — M4's geometry spends 1089 — and it buys the guarantee
    that ``λ̃`` still names the selection pressure §1.7 defined it to name.

    Raises `ReweightingNotConvergedError` if the loop hits its iteration budget with the active set
    or the solution still moving. It does **not** quietly ship the last iterate: an unconverged
    weight vector is not a fixed point of anything, and freezing it would mean sampling a target
    chosen by the iteration budget. Pass ``allow_unconverged=True`` to accept one anyway; the report
    records ``converged = False`` and the manifest carries it.
    """
    settings = config if config is not None else ObjectiveConfig()
    reduction = reduced if reduced is not None else polytope.reduce()
    clip = (settings.weight_clip_min, settings.weight_clip_max)

    # w^base: fixed for the whole loop. Every update divides THIS, never the current weights.
    seed = SparseFluxObjective.from_polytope(
        polytope,
        l1_penalty=0.0,
        exclude_biomass_from_penalty=settings.exclude_biomass_from_penalty,
        penalty_ids=penalty_ids,
        weights=base_weights,
    )
    base = np.array(seed.weights, dtype=VALUE_DTYPE)
    mask = seed.penalty_mask

    weights = base.copy()
    history: list[ReweightingStep] = []
    previous_v: NDArray[np.float64] | None = None
    previous_active: set[int] = set()
    active_base: set[int] = set()
    active: set[int] = set()
    converged = False
    resolved = None

    for iteration in range(settings.reweighting_max_iterations):
        # λ^(k) = λ̃·λ*(w^(k)) — the whole λ policy, in one call. `resolve_objective` also re-checks
        # the §1.7 sparsity cliff at every iteration, which matters here precisely because
        # reweighting MOVES the cliff (by 1e8 on the example model).
        resolved = resolve_objective(
            polytope, reduction, settings, penalty_ids=penalty_ids, weights=weights, threads=threads
        )
        solution = solve_sparse_objective(reduction, resolved.objective, threads=threads)

        v = solution.optimum.v_full
        active = set(np.flatnonzero(np.abs(v) > settings.reweighting_active_tol).tolist())
        if iteration == 0:
            active_base = active

        # The candidate next weights ``w^(k+1) = F(v^(k))``. Computing them *before* the convergence
        # test is what lets the test ask the question that protects the freeze: has the weight the
        # map produces stopped changing? ``weights`` is ``w^(k) = F(v^(k-1))``, so this compares
        # ``F(v^(k))`` with ``F(v^(k-1))`` — the fixed-point condition itself.
        updated, at_low, at_high = update_weights(
            base, v, mask, epsilon=settings.reweighting_epsilon, clip=clip
        )
        weight_change = _relative_weight_change(updated, weights, mask)

        if previous_v is None:
            flux_change, set_changes = float("inf"), -1
        else:
            denominator = max(1.0, float(np.max(np.abs(previous_v))))
            flux_change = float(np.max(np.abs(v - previous_v))) / denominator
            set_changes = len(active ^ previous_active)
            # Converge on the WEIGHTS (the frozen artifact), not the fluxes — a flux-based test is
            # blind exactly on the small, sparsity-critical coordinates (Codex, M7 review). The
            # active set must also be stable: a support flip is a discontinuity the continuous
            # metric can under-weight when the flipped flux is tiny.
            converged = (
                set_changes == 0 and weight_change <= settings.reweighting_solution_tol
            )

        history.append(
            ReweightingStep(
                iteration=iteration,
                weights=_frozen(weights.copy()),
                l1_penalty=resolved.scale.l1_penalty,
                critical_l1_penalty=resolved.scale.critical_l1_penalty,
                solution=solution,
                n_active=len(active),
                max_weight_change=weight_change,
                max_flux_change=flux_change,
                active_set_changes=set_changes,
                n_at_clip_low=at_low,
                n_at_clip_high=at_high,
            )
        )

        if converged:
            break

        previous_v, previous_active = v.copy(), active
        weights = updated

    final = history[-1]
    if not converged:
        message = (
            f"the reweighting loop did not converge in {settings.reweighting_max_iterations} "
            f"iterations (last step: {final.active_set_changes} active-set changes, per-reaction "
            f"relative weight change {final.max_weight_change:.3e} against a tolerance of "
            f"{settings.reweighting_solution_tol:.1e}). An unconverged weight vector is not a "
            "fixed point of anything, so freezing it would mean sampling a target chosen by the "
            "iteration budget. Raise objective.reweighting_max_iterations, loosen the tolerances, "
            "or pass "
            "allow_unconverged=True to accept this iterate deliberately."
        )
        if not allow_unconverged:
            raise ReweightingNotConvergedError(message)
        _log.warning("%s Accepting it because allow_unconverged=True.", message)

    active_final = active
    if active_base == active_final:
        _log.warning(
            "reweighting left the active set unchanged (%d reactions, symmetric difference empty). "
            "The loop ran; it simply had no lever on the support. The usual cause is a weight-clip "
            "ceiling (objective.weight_clip_max = %.3g) at or below 1/|v| for the smallest fluxes "
            "you want treated as off — it merges 'nearly off' with 'off', which is the one "
            "distinction the surrogate is made of, and reweighted-L1's SUPPORT degenerates into "
            "plain L1's with no error anywhere. (The frozen weights may still differ from uniform, "
            "so the sampled law is not necessarily plain L1's — but the sparsity claim is empty.)",
            len(active_base),
            settings.weight_clip_max,
        )

    # The frozen objective is *the very object* the last solve used — not a rebuild of it. Calling
    # `resolve_objective` again on `final.weights` would spend another Charnes–Cooper LP and, worse,
    # would only be the same objective if HiGHS is bit-reproducible: a λ differing in its last ulp
    # gives a different `content_key`, and `choose_energy_scale` would then reject the pair — or,
    # far worse, accept it while `solution.optimum` belonged to a λ that is not quite this one.
    # M7 exists to stop two artifacts being joined that were never computed against each other, so
    # it will not create the pair in the first place.
    assert resolved is not None  # the loop body runs at least once (max_iterations >= 1)
    return ReweightingReport(
        objective=resolved.objective,
        scale=resolved.scale,
        solution=final.solution,
        history=tuple(history),
        converged=converged,
        active_base=frozenset(active_base),
        active_final=frozenset(active_final),
    )
