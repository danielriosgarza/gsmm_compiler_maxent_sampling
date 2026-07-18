"""The affine geometry of the flux polytope: scaled coordinates, an orthonormal basis of its
direction space, a feasible centre — and a **deterministic certificate** that the basis is
complete.

The polytope ``{S v = rhs, l ≤ v ≤ u}`` lives on a proper affine subspace of ℝⁿ, because every
feasible direction ``d`` must satisfy ``S d = 0``, and a random vector in reaction space
essentially never does. So before any MCMC can run we need a basis ``B`` of that direction space,
and we get it the way spec §15 prescribes: not from a sparse QR or an SVD of ``S``, but from
**differences of feasible points handed back by the LP**. Every column of ``B`` is therefore a
direction the solver has *proved* is feasible, rather than one a rank decision claimed was.

Three things in here are load-bearing, and each is a place where a subtle error would silently
sample the wrong set.

**1. The certificate is deterministic and complete — but *resolution-bounded*, not exact.**
Discovery stops when random probes stall, and random probes stalling is not a proof. So afterwards
we sweep an orthonormal basis ``{p₁…p_k}`` of ``range(B)ᗮ`` (``k = n_free − d``, BUILD_PLAN §1.4)
and demand that every one of them have zero LP width. *In exact arithmetic that is a proof.* Write
``V`` for the true direction space and ``S = range(B)``; every basis column came from a difference
of feasible points, so ``S ⊆ V``. A probe's width is ``max_{x∈X} pᵀx − min_{x∈X} pᵀx``, which is
zero exactly when ``p ⊥ V``. If every ``pᵢ`` has zero width then ``Sᗮ = span{pᵢ} ⊆ Vᗮ``, hence ``V
⊆ S``, hence ``V = S``. Contrapositively, a missed direction ``w ∈ V ∩ Sᗮ`` is nonzero, so some
``pⱼᵀw ≠ 0``, so that ``pⱼ`` is not orthogonal to ``V`` and its width is positive. A missing
dimension has nowhere to hide.

In float64 a solver can only ever report "flatter than this", so what the sweep licenses is a
**bound**, and the bound is *not* the largest width it saw. Width is a support-function difference
— subadditive and positively homogeneous — so a unit direction ``p = Σ aⱼ pⱼ`` spread across the
complement (``‖a‖₂ = 1``, hence ``‖a‖₁ ≤ √k``) obeys ``width(p) ≤ √k · maxⱼ width(pⱼ)``. A
direction tilted equally across all ``k`` probes therefore hides a factor of ``√k`` from every one
of them individually. `SpanCertificate.resolution` reports ``√k · max_width`` for exactly this
reason, and it is the number to quote: on the example model, 214 probes with a worst width of
7e-15 certify that **no missed direction is wider than 1e-13** — while the *tolerance* alone would
only have licensed 1.5e-8.

**2. The scaled coordinates amplify the solver's error by 1/sᵢ — so that error is measured, never
assumed.** Scaled coordinates are ``xᵢ = vᵢ/sᵢ`` with ``sᵢ = max(uᵢ − lᵢ, scale_floor)``. Dividing
by ``sᵢ`` divides the LP's feasibility error by ``sᵢ`` too, so a narrow-ranged reaction turns
solver noise into a large scaled coordinate, and a phantom "direction" of pure noise would inflate
the sampled dimension. Rather than hope the tolerances line up, every probe carries its own error
bar, built from what HiGHS reports **and** from what we can check ourselves
(`SupportProbe.width_floor` / `rank_floor`): a direction counts as real only if it clears that bar
by `NOISE_SAFETY`, and a probe whose noise swamps the configured tolerance is **inconclusive** —
recorded, and costing the certificate its exhaustiveness. `probe_noise_ceiling` reports the
headroom.

These are *resolution* bars, not proofs. Turning a constraint residual into a distance from the
exact feasible set needs a Hoffman constant we do not have, so they state what the solver could
have manufactured at its own admitted accuracy — which is the honest thing available.

Two error sources are kept apart on purpose. A **bound** violation is exact in float64 (it is a
comparison), so it enters the noise bar. A **mass-balance** residual does not: ‖S‖·‖v‖ ≈ 1e5 here,
so merely *evaluating* ``S·v`` costs ~1e-10 of rounding, and charging that to the solver would
fail a perfectly good certificate. It gets a relative check of its own instead
(`_check_mass_balance`).

**3. A width can be too *small*, and only the dual side can tell.** ``kOptimal`` means "optimal to
the configured tolerances". A width is a *difference of two optima*, so a solve that stopped short
of optimality reports it too small — and a real dimension is certified flat, silently. Primal
feasibility cannot see this. Worse, ``c = p/s`` makes the objective coefficients *tiny* when the
bound ranges are wide, so an improving reduced cost can fall below HiGHS's **absolute** dual
tolerance and the solver never leaves its starting vertex. Both are handled: each support
objective is **sup-normalized** before it goes to the solver (a positive rescaling cannot move the
argmax, and the width is recomputed from the returned primal points in the original units), and
every probe gates on ``max_dual_infeasibility``.

**4. A proven feasible direction is never discarded.** A probe yields ``Δx = (v⁺ − v⁻)/s``, the
difference of two feasible points, so ``diag(s)·Δx`` is a feasible direction *whatever the width
says*. Spec §15.3 appends it only when the width **and** the residual norm clear tolerance; but
when the width is zero and the residual is large — the argmax and the argmin landed on different
vertices of a wholly-optimal face — the spec's ``and`` throws away a dimension the solver has just
proved exists. We use ``or``. It cannot admit a direction that is not real (both endpoints are
feasible points), and it cannot loop forever (each acceptance raises ``d``, and ``d ≤ n_free``).

**5. A free reaction that cannot actually move — which the spec did not anticipate.**
`blocked_reactions` runs FVA over the reduced polytope first, and on the example model **61 of the
260 free reactions turn out to be unable to carry any flux at all**: the model file left ``l <
u``, but mass balance pins them. This is a correctness requirement, not a curiosity. If ``max vᵢ
== min vᵢ`` over ``P`` then every feasible direction has ``dᵢ = 0`` identically, so a nonzero
``B[i,:]`` is numerical error and nothing else — and that error is not harmless. A basis row of
~1e-15 in a coordinate whose centre sits ~1e-13 *outside* its own bound (both solver noise)
divides into a **chord limit of order 0.03–0.5**, squarely inside the legitimate chord. The
measured result was a chord of ``[−0.54, −0.39]`` that **excluded ``t = 0``**, which
`line_geometry` rightly refuses to sample: the sampler could not have started. So blocked
components are projected out of every candidate direction, exactly. This is *not* the forbidden
snapping of small fluxes (CLAUDE.md) — no flux is rounded, and a pinned reaction keeps whatever
value it holds; what is zeroed is a component of the **direction space** that an LP measured as
zero.

Say it precisely, though: this is *numerically fixed at resolution* `blocked_tol`, not *provably
constant*. A reaction whose true range is 5e-16 would be blocked and its (real, if absurdly thin)
dimension dropped — and `BlockedReactions.separation`, which is 5e11× on the example model, would
not notice, because it measures whether the split is *clustered*, not whether it is *right*. What
the separation does buy is the assurance that no tolerance in eleven orders of magnitude would
change the answer here. `GeometryConfig` also forbids the resolutions from contradicting each
other (``scale_floor ≥ blocked_tol / span_tol``); without that relation the sweep will report the
very axis the projection removed, and then be unable to append it.

**What is being certified, and about which set.** The two instruments here point in opposite
directions on purpose, and they do not have the same standing. The **upper** bound — the one the
certificate rests on — comes from weak duality, which requires nothing of the returned point at
all: not optimality, not even primal feasibility. It is therefore a statement about the **exact**
polytope ``{S v = rhs, l ≤ v ≤ u}``, evaluated with an outward rounding allowance so that float64
cannot round it below the truth. The **lower** bound — a width read off two returned endpoints,
which is what admits a direction into the basis — is only tolerance-qualified: those endpoints are
feasible to the solver's tolerance, not exactly.

That asymmetry licenses exactly one claim, and it is narrower than it is tempting to say. It is
**not** "cannot under-count a dimension": a direction thinner than the certified resolution can be
missed, and `blocked_tol` will drop one narrower than itself — a true 5e-16-wide dimension goes,
and nothing objects. Nor does the sweep prove ``range(B) ⊆ V``; the basis may be tilted off the
exact hull by ε. What it does prove is this: **every feasible direction of the exact polytope has
its component orthogonal to ``range(B)`` bounded in width by `SpanCertificate.resolution`.** (If a
wide exact direction ``w`` were missing, ``w_perp``, its part outside ``range(B)``, is nonzero,
and ``w·w_perp > 0`` puts ``w_perp`` off ``Vᗮ`` — so the sweep would have measured positive width
there.) Every exact direction therefore lies within the certified resolution of the span we
sample.

For a sampler, over-counting is the benign failure: the chain explores a slightly larger set,
every sample is still checked against the bounds and the mass balance, and ``‖S·diag(s)·B‖``
bounds how far any basis direction can stray. Omitting a *wide* direction is the malignant one —
it would silently delete part of the support, and no test downstream would ever see the samples
that were never drawn. So the resolution is the number to read: it says how thin a direction had
to be for us to have missed it.

Where the exact and the resolved sets genuinely diverge — a stoichiometry one ULP from singular
describes a point exactly, while every float64 method, the LP included, sees a line — the module
does not pretend to arbitrate. What it owes instead is that its components never disagree with
*each other*: the SVD's rank cutoff is taken to be at least the LP's feasibility tolerance, so an
equality the solver will not enforce is not one the projector insists on; a residual disagreement
is raised (`_append` names the reaction and the tolerances) rather than sampled; and the
resolution of every claim is measured and reported rather than implied.

Implemented in **M4** — see BUILD_PLAN.md §1.4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.config import GeometryConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.highs_backend import (
    BACKEND_IMPL_VERSION,
    DEFAULT_PRIMAL_FEASIBILITY_TOL,
    HighsLinearProgram,
    LPNotOptimalError,
    LPSolution,
    total_solve_count,
)
from gsmm_compiler.native_csc import VALUE_DTYPE
from gsmm_compiler.numerics import under_deterministic_blas
from gsmm_compiler.provenance import content_key, stream_seed
from gsmm_compiler.sparse_objective import build_flux_lp


class Maximizer(Protocol):
    """Anything the probing code can hand an objective to and get a checked `LPSolution` back.

    `HighsLinearProgram` satisfies it directly (the warm fast path); `_SolveSession` satisfies it
    while adding cold escalation on ``kUnknown`` (M11.2). Widening `probe_direction` and
    `sweep_complement` to this Protocol is what lets the escalation reach the support and span
    stages without either of them knowing whether the solve behind it was warm or cold.
    """

    def maximize(self, costs: NDArray[np.float64]) -> LPSolution: ...


def solve_fresh_once(reduced: ReducedPolytope, costs: NDArray[np.float64]) -> LPSolution:
    """One fresh LP instance, one solve — the escalation primitive (M11.2, Codex's factoring).

    A brand-new `HighsLinearProgram` has no inherited basis, so this is the "remove all warm-start
    history" solve that turned `blocked_reactions`' `kUnknown` crashes into clean builds. It is
    deliberately *single*: `blocked_reactions` needs a cold **pair** and calls this twice on two
    fresh instances, which keeps its measured one-pair retry budget from silently growing.
    """
    return build_flux_lp(reduced, threads=1).maximize(
        np.asarray(costs, dtype=VALUE_DTYPE)
    )


class _SolveSession:
    """Owns the LP-instance lifecycle for one `build_geometry` call (M11.2, §1.6.11).

    Warm fast path on a persistent instance until the **first** ``kUnknown``; then cold-only — a
    fresh instance per solve — for the rest of the build. That transition is the whole point: M11.1
    abandoned a degraded instance only *inside* `blocked_reactions` (a local ``warm = None``), while
    `build_geometry` kept using the original program for the initial solve, support discovery and
    the span sweep — so a strain recovered in one stage and still died ``kUnknown`` in the next. The
    session makes the abandonment build-wide.

    Escalates **only** ``kUnknown``. Infeasible, unbounded and limit statuses keep their
    hard-failure meaning — a ``kUnknown`` is "the solver could not decide", which a fresh basis can
    fix; the others are verdicts about the model, which it cannot (Codex, M11.2 review).

    A ``kUnknown`` does not *prove* later warm solves are wrong, but continuing to warm-start off an
    instance the solver could not certify makes the geometry a story nobody can retell. Cold-only
    for the tail is the conservative, M11.1-consistent choice; `degraded` and `n_cold_solves` record
    that it happened and what it cost so a run that leaned on it is visible. `n_cold_solves` counts
    the session's **own** cold retries (the initial/support/span solves); `blocked_reactions`' cold
    pairs are reported separately as `n_blocked_escalated`, and the total appears in `n_lp_solves`.
    """

    def __init__(self, reduced: ReducedPolytope, *, cold_only: bool = False) -> None:
        self._reduced = reduced
        self._program = build_flux_lp(reduced, threads=1)
        self._cold_only = cold_only
        self.n_cold_solves = 0
        self.degraded = cold_only

    @property
    def program(self) -> HighsLinearProgram:
        """The persistent warm instance, for `blocked_reactions`' warm sweep and cold-pair logic."""
        return self._program

    def mark_degraded(self) -> None:
        """`blocked_reactions` calls this when its use of the persistent instance hit ``kUnknown``.

        The instance is the session's own, so once `blocked_reactions` could not certify a solve on
        it, every later stage must stop warm-starting off it too. Recorded as a **boolean** rather
        than a solve index: `blocked_reactions`' own FVA solves do not pass through this session, so
        an index here would count from zero and mislead (Codex, M11.2 review).
        """
        self._cold_only = True
        self.degraded = True

    def maximize(self, costs: NDArray[np.float64]) -> LPSolution:
        """Warm attempt, then one cold retry on ``kUnknown`` — a drop-in for `program.maximize`."""
        if not self._cold_only:
            try:
                return self._program.maximize(costs)
            except LPNotOptimalError as error:
                _reraise_unless_kunknown(error)  # infeasible/unbounded/limit stay hard failures
                self.mark_degraded()
        self.n_cold_solves += 1
        return solve_fresh_once(self._reduced, costs)

GEOMETRY_IMPL_VERSION: Final = 4
"""Bumped when a change here can alter the bytes of a geometry. Feeds the L3 cache key (§1.1).

2 (M10.2): L3 now stores the **geometry** (``s``, ``B``, centre, support points, certificate) and
not only the transform built from it, and `ReducedGeometry.content_key` hashes ``support_points``.
The bump is load-bearing rather than cosmetic: `batch.geometry_cache_key` does *not* fold in
`ReducedGeometry.content_key`, so without it a v1 bundle — transform arrays, no ``basis`` — would
still be a cache **hit** and `from_bundle` would die on a missing key. §1.1's rule is that a schema
change must **miss**, never error on stale bytes.

3 (M11.1): `blocked_reactions` now brackets each width and cold-escalates an unresolved reaction, so
its **mask can differ** from a v2 build on the same polytope — where the warm duals were loose, a
reaction v2 called moving is now certified blocked, which moves the basis, the support points and
possibly ``d``. That is a genuine change of the geometry's bytes, so it must **miss** a v2 cache
rather than be served one. M11.0's `BACKEND_IMPL_VERSION` bump does not cover this: that identifies
the *solver*, this is an *algorithm* change above it.

4 (M11.2): the span certificate's ``exhaustive`` now means ``resolution ≤ span_tol`` rather than
``n_inconclusive == 0``, so a geometry that v3 refused (a noise-swamped probe under a certified
resolution) v4 accepts and stores. The *arrays* of an accepted geometry are unchanged, but its
cached **certificate/diagnostic semantics** are not, and a v3 bundle would carry the old verdict —
so v4 must miss it (Codex, M11.2 review).
"""

NOISE_SAFETY: Final = 10.0
"""How far a width must clear the solver's own admitted error before we call the direction real."""

BASIS_BLOCK: Final = 32
"""Basis columns are allocated in blocks, never reallocated per append (spec §15.5)."""

ORTHONORMALITY_TOL: Final = 1e-10
"""``‖BᵀB − I‖_max``. Two-pass projection delivers ~1e-15; this is a loose alarm, not a target."""

MASS_BALANCE_SAFETY: Final = 100.0
"""Slack over the float64 evaluation floor of ``S·v`` before an endpoint's residual is a defect."""

MAX_DEGENERATE_DRAWS: Final = 100
"""Consecutive random probes that may collapse onto the basis before we call it a defect."""

_BYTES_PER_FLOAT64: Final = 8


class GeometryError(RuntimeError):
    """The geometry could not be built, or could not be certified — never silently downgraded."""


# ---- scaled coordinates (spec §15.2) ------------------------------------------------------------


def reaction_scales(reduced: ReducedPolytope, *, floor: float) -> NDArray[np.float64]:
    """``sᵢ = max(uᵢ − lᵢ, floor)`` — the units the affine basis is orthonormal in.

    Every reaction in a *reduced* polytope is free (``l == u`` was eliminated in M1), so its range
    is strictly positive and the ``uᵢ == lᵢ → 1`` branch of spec §15.2 cannot arise here. The floor
    still earns its place: it bounds the ``1/sᵢ`` amplification of solver error described in the
    module docstring.
    """
    span = reduced.upper_bounds - reduced.lower_bounds
    if np.any(span <= 0.0):
        raise GeometryError(
            "a reduced polytope must have l < u for every reaction; the fixed ones are eliminated "
            "in M1, so a non-positive range here means the reduced IR is corrupt"
        )
    return np.maximum(span, floor).astype(VALUE_DTYPE, copy=False)


# ---- structural zeros of the direction space ----------------------------------------------------


@dataclass(frozen=True)
class BlockedReactions:
    """Free reactions that cannot carry flux — the zeros of the direction space, at a resolution.

    **Three states, because two instruments answer two different questions** (M11.1, BUILD_PLAN
    §1.6.11). Each reaction's width is *bracketed*, never collapsed to one number:

    * ``upper`` — ``U_i``, from weak duality. Sound for **any** multipliers, optimal or nonsense,
      needing no primal point at all. ``U_i ≤ blocked_tol`` is what certifies BLOCKED.
    * ``lower`` — ``L_i``, from the two returned endpoints, **minus** their own admitted error and
      rounded outward. ``L_i > blocked_tol`` (strictly) is what qualifies MOVING.
    * anything else is **UNRESOLVED**: the bracket straddles the bar and this instrument cannot say.

    **The two sides are not symmetric, and pretending otherwise is what broke 24 of 40 strains.**
    Only BLOCKED is *certified*: turning a constraint residual into a distance from the exact
    feasible set needs a Hoffman constant this package does not have (`probe_direction` records the
    same limit at its own noise bar), so the endpoints are only tolerance-feasible and MOVING is
    **resolution-qualified**, never proved. That is §1.4.1's own honesty — "numerically fixed at
    resolution ``blocked_tol``, not provably constant" — applied at last to *both* sides.

    The predecessor collapsed this to ``ranges[i] = max(U_i, W_i)`` and read ``> blocked_tol`` as
    "moving", when it licenses only *"not certified blocked by this dual"*. Since warm-started duals
    are sound but **loose**, a flat reaction's ``U_i`` drifted over the bar and was called moving.
    """

    mask: NDArray[np.bool_]
    """``(n_free,)``. True where ``U_i ≤ blocked_tol`` — certified flat by weak duality."""
    upper: NDArray[np.float64]
    """``(n_free,)`` rigorous upper bound on each width. The evidence for BLOCKED."""
    lower: NDArray[np.float64]
    """``(n_free,)`` resolution-qualified lower witness. The evidence for MOVING.

    Kept **separate** from `upper` rather than folded into one ``max``: the two bracket the true
    width, so ``lower > upper`` is a broken arithmetic contract and must be loud. A ``max`` would
    silently repair it (Codex, M11.1 review).
    """
    unresolved: NDArray[np.bool_]
    """``(n_free,)``. True where the bracket straddles the bar even after cold escalation."""
    n_escalated: int
    """How many reactions needed a cold solve pair. Reported: it is the warm path's error rate."""
    separation: float
    """Narrowest resolution-qualified-moving `lower` ÷ widest certified-blocked `upper`.

    **A diagnostic since M11.1, no longer a gate.** It measures whether ``d`` is *sensitive to
    moving* ``blocked_tol``; UNRESOLVED measures whether a bracket *straddles* it. Those are
    different properties, so retiring the guard does forfeit the tolerance-stability claim — which
    is acceptable only because this module is explicitly resolution-bounded, and the gate that
    replaces it ("no reaction is unresolved") tests each classification directly instead of
    inferring soundness from a gap between two clusters. Computed from the certified endpoints, not
    from the conflated ``ranges`` its predecessor used.
    """

    @property
    def n_blocked(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def n_unresolved(self) -> int:
        return int(np.count_nonzero(self.unresolved))


def blocked_reactions(
    reduced: ReducedPolytope,
    program: HighsLinearProgram,
    *,
    tol: float,
    min_separation: float = 100.0,
    on_degraded: Callable[[], None] | None = None,
) -> BlockedReactions:
    """FVA over the free reactions: which of them cannot move at all? (2·n_free warm-started LPs.)

    A reaction whose feasible range is a single point contributes an *identically zero* component to
    every feasible direction, because feasible directions are differences of feasible points. That
    makes this an exact statement about the geometry, not a threshold applied to a flux — and the
    module docstring explains why leaving those components at their noise value corrupts the chord.

    `min_separation` guards the one thing that could make this a guess: if the widest blocked range
    is not far below the narrowest moving one, the split depends on `tol` and the affine dimension
    is ambiguous. We refuse rather than quietly pick a dimension.

    **M11.1 — the warm pass is a fast path, and its verdicts are checked, not trusted.** The
    predecessor ran one persistent warm-started program over all ``2·n_free`` solves and believed
    whatever came back. Measured across the 40 real strains that is wrong two ways: 9 die outright
    when HiGHS returns ``kUnknown``, and 24 refuse because the warm duals — sound, but *loose for
    the objective they were not solved for* — push a flat reaction's ``U_i`` over the bar. Both are
    the **same case**: a reaction whose bracket straddles ``tol``, including one whose solve gave no
    usable evidence at all. Each such reaction gets **one** cold solve pair on a fresh instance;
    everything else keeps the warm start. Measured, that is ~5 reactions in 446.

    This is not "retry until it passes" (§1.6.7 r4 rejected that): the acceptance criterion never
    moves, the retry is *declared* — the inherited basis is removed, which is a different
    computation, not the same one re-rolled — and it is bounded at one. A reaction still unresolved
    after its cold pair **refuses**, reporting the resolution rather than picking a class.
    """
    n = reduced.n_free
    upper = np.empty(n, dtype=VALUE_DTYPE)
    lower = np.empty(n, dtype=VALUE_DTYPE)
    unresolved = np.zeros(n, dtype=np.bool_)
    warm: HighsLinearProgram | None = program
    n_escalated = 0

    for i in range(n):
        bracket = _warm_width_bracket(warm, reduced, i, n) if warm is not None else None
        if bracket is None:
            # A ``kUnknown`` warm solve yields no witness in *either* direction — the same state as
            # a bracket that straddles the bar, and it escalates the same way. Only ``kUnknown``
            # reaches here: `_warm_width_bracket` re-raises infeasible/unbounded/limit, model
            # verdicts a fresh basis cannot change. The catch is caller-side, not in
            # `HighsLinearProgram.solve`, which must go on refusing for everyone —
            # `sparse_objective.critical_l1_penalty` needs `LPNotOptimalError` to detect an
            # unbounded `J*`.
            #
            # And the failed instance is abandoned: later solves warm-started off a basis HiGHS
            # could not certify would still be *usable*, but their provenance would be a story
            # nobody can retell (Codex, M11.1 review). `on_degraded` propagates that abandonment
            # to the *rest of the build* (M11.2): M11.1's `warm = None` was local to this function,
            # so `build_geometry` kept warm-starting the next stages off the same degraded instance.
            warm = None
            if on_degraded is not None:
                on_degraded()
        elif not _straddles(bracket, tol):
            upper[i], lower[i] = bracket
            continue

        # Escalate on a **fully** cold pair — a fresh instance per *solve*, not per reaction. One
        # instance shared by the two solves would warm-start `min` off `max`'s basis, and that one
        # step of inherited history is enough to leave a floor reaction straddling: measured, it
        # turned genuinely-resolvable reactions on several strains into false "unresolved" refusals
        # while the truly-at-floor cases (Hafnia, `U ≈ 2e-9`) stayed unresolved either way. Removing
        # *all* warm history is what the escalation is for; leaving one step was leaving the bug.
        n_escalated += 1
        cold = _cold_width_bracket(reduced, i, n)
        if cold is None:
            # No complete `U_i` exists, so this reaction's resolution is *unavailable* — not a
            # number to be maxed into an aggregate as if it had been measured.
            upper[i], lower[i], unresolved[i] = np.inf, -np.inf, True
            continue
        upper[i], lower[i] = cold
        unresolved[i] = _straddles(cold, tol)

    _reject_contradictory_bracket(upper, lower)
    mask = upper <= tol
    moving = lower > tol  # strict: `>=` would overlap `mask` when both bounds sit on the bar
    unresolved |= ~(mask | moving)

    blocked_max = float(np.max(upper[mask], initial=0.0))
    moving_min = float(np.min(lower[moving], initial=np.inf))
    separation = np.inf if blocked_max <= 0.0 else moving_min / blocked_max

    if np.any(unresolved):
        worst = float(np.max(upper[unresolved]))
        raise GeometryError(
            f"{int(np.count_nonzero(unresolved))} of {n} reactions cannot be resolved as blocked "
            f"or moving at geometry.blocked_tol ({tol:.1e}), even after a cold re-solve: their "
            f"widths are bracketed to at most {worst:.3e} from above and no further than {tol:.1e} "
            "below, so this instrument cannot tell a flat reaction from a narrow one here. The "
            "affine dimension would depend on the tolerance rather than on the polytope. Set "
            "geometry.blocked_tol deliberately above the reported bracket, and know that doing so "
            "declares any dimension narrower than it to be absent."
        )

    return BlockedReactions(
        mask=mask,
        upper=upper,
        lower=lower,
        unresolved=unresolved,
        n_escalated=n_escalated,
        separation=separation,
    )


def _reraise_unless_kunknown(error: LPNotOptimalError) -> None:
    """Escalation is for ``kUnknown`` — "the solver could not decide" — and nothing else (M11.2).

    Infeasible, unbounded and limit statuses are verdicts about the *model* that a fresh basis
    cannot change, so re-raising them keeps their hard-failure meaning across the whole build, not
    only inside `_SolveSession`. On a reduced polytope (feasible by construction, finite bounds) a
    single-coordinate FVA solve should only ever be optimal or ``kUnknown``, so an infeasible or
    unbounded status here signals a corrupt IR — exactly the thing that must surface, not be quietly
    re-solved. (Codex, M11.2 review.)
    """
    if "kUnknown" not in error.model_status:
        raise error


def _bracket_from_solutions(
    reduced: ReducedPolytope, high: LPSolution, low: LPSolution, index: int, n: int
) -> tuple[float, float]:
    """``(U_i, L_i)`` from a max/min solution pair — the one bracket computation, warm or cold.

    ``U_i`` is bounded **from above** by weak duality on each solve, and that direction is the whole
    point: a primal reading is a *lower* bound, so an LP that stopped short would report a range of
    zero for a reaction that is wide open and the projection would delete a real dimension. Same
    instrument as the span certificate, same reason.

    ``L_i`` is that primal reading, kept for the *opposite* question — is this reaction certainly
    moving? — and charged for its own error first. It is **not** a proof: the endpoints are only
    tolerance-feasible, and no Hoffman constant is available to turn that into a distance from the
    exact feasible set. Hence the outward subtraction, and hence "resolution-qualified".
    """
    unit = np.zeros(n, dtype=VALUE_DTYPE)
    unit[index] = 1.0
    upper = dual_upper_bound(high, unit, reduced) + dual_upper_bound(low, -unit, reduced)

    admitted = max(
        _bound_violation(high.primal, reduced),
        _bound_violation(low.primal, reduced),
        high.max_primal_infeasibility,
        low.max_primal_infeasibility,
    )
    width = float(high.primal[index] - low.primal[index])
    eps = float(np.finfo(VALUE_DTYPE).eps)
    reach = float(max(abs(reduced.lower_bounds[index]), abs(reduced.upper_bounds[index])))
    lower = width - NOISE_SAFETY * 2.0 * admitted - eps * reach
    return upper, float(np.nextafter(lower, -np.inf))


def _warm_width_bracket(
    program: HighsLinearProgram, reduced: ReducedPolytope, index: int, n: int
) -> tuple[float, float] | None:
    """Warm bracket: both solves on the persistent instance (they warm-start off each other)."""
    cost = np.zeros(n, dtype=VALUE_DTYPE)
    cost[index] = 1.0
    try:
        high = program.maximize(cost)
        low = program.maximize(-cost)
    except LPNotOptimalError as error:
        _reraise_unless_kunknown(error)
        return None
    return _bracket_from_solutions(reduced, high, low, index, n)


def _cold_width_bracket(
    reduced: ReducedPolytope, index: int, n: int
) -> tuple[float, float] | None:
    """Cold bracket: each leg is a fresh instance via `solve_fresh_once` — no inherited history.

    Uses the single approved escalation primitive twice rather than driving two fresh programs by
    hand, so there is one place a fresh solve is defined (Codex, M11.2 review).
    """
    cost = np.zeros(n, dtype=VALUE_DTYPE)
    cost[index] = 1.0
    try:
        high = solve_fresh_once(reduced, cost)
        low = solve_fresh_once(reduced, -cost)
    except LPNotOptimalError as error:
        _reraise_unless_kunknown(error)
        return None
    return _bracket_from_solutions(reduced, high, low, index, n)


def _straddles(bracket: tuple[float, float], tol: float) -> bool:
    """Neither certified blocked (``U ≤ tol``) nor resolution-qualified moving (``L > tol``)."""
    upper, lower = bracket
    return not (upper <= tol or lower > tol)


def _reject_contradictory_bracket(
    upper: NDArray[np.float64], lower: NDArray[np.float64]
) -> None:
    """``L_i ≤ U_i`` is an arithmetic contract, not a preference — so a violation is loud.

    The two bound the same width from opposite sides. The predecessor wrote ``max(U, W)``, which
    cannot *observe* a contradiction because it resolves it: a primal witness wider than a rigorous
    upper bound would silently become the reported range, and the fact that weak duality had just
    been contradicted would go unrecorded (Codex, M11.1 review).
    """
    finite = np.isfinite(upper) & np.isfinite(lower)
    broken = np.flatnonzero(finite & (lower > upper))
    if broken.size:
        i = int(broken[0])
        raise GeometryError(
            f"reaction {i} has a resolution-qualified lower width ({lower[i]:.6e}) above its "
            f"rigorous upper bound ({upper[i]:.6e}), and {broken.size} reaction(s) do. These "
            "bracket the same quantity from opposite sides, so one of the two is wrong: either "
            "weak duality was contradicted or an endpoint is further outside the feasible set than "
            "its admitted error. Neither is a tolerance to widen."
        )


# ---- the space every feasible direction must lie in ---------------------------------------------


@dataclass(frozen=True)
class DirectionSpace:
    """Projector onto ``{x : x_blocked = 0, S·diag(s)·x = 0}`` — the two constraints that *every*
    feasible direction satisfies exactly, whatever the LP's floating-point opinion of it.

    **Why this is here at all.** Without it, the mass-balance error of the basis *accumulates*. Each
    column is a Gram-Schmidt residual ``r = Δx − B(BᵀΔx)``, so

        ``S·diag(s)·r = S·Δv − S·diag(s)·B·(BᵀΔx)``

    — the LP's row residual (~1e-12), **plus the error already in ``B``, amplified by ‖BᵀΔx‖**. That
    second term is a feedback loop: column 46 inherits the errors of columns 1–45. Measured on the
    example model, it grew the worst column to 8e-10 against a 6e-12 median, within 1.25× of the
    tolerance the basis must satisfy — and an earlier attempt to fix it by re-probing along each new
    direction made it *worse* (1.4e-9, a hard failure), because a sharper probe drives ``Δx`` to
    extreme vertices and so *raises* ‖BᵀΔx‖. The chain has to be cut, not managed: project each
    candidate into this space *before* it is appended, and every column is mass-balanced to machine
    precision, so there is nothing left to accumulate.

    **This does not decide the dimension — the LP still does.** Spec §15.1 rejects computing a
    null-space basis of ``S`` and calling it the answer, and it is right to: the null space is only
    an *upper bound* on the affine dimension, because the bounds can flatten the polytope inside it.
    The example model proves the point — ``dim null = 46`` here, but before the blocked reactions
    were removed the naive count was 55, and the true dimension is what the support LPs find. This
    class supplies a projector, not a dimension. `n_null` is retained purely as an independent
    ceiling to check ``d`` against.
    """

    blocked: NDArray[np.bool_]
    row_space: NDArray[np.float64]
    """``Q``: orthonormal basis of the row space of ``S·diag(s)`` over the *moving* columns. The
    null space is its orthogonal complement, so projecting is ``x -= Q(Qᵀx)`` — no null-space basis
    is ever formed, and the economy SVD suffices whatever the shape of ``S``."""
    moving: NDArray[np.intp]
    n_null: int
    """``n_moving − rank``: an upper bound on the affine dimension, never a claim about it."""
    singular_gap: float
    """Smallest **retained** singular value ÷ largest **discarded** one — a *two-sided* gap.

    A one-sided margin (retained vs the cutoff) says nothing about the decision: it can be huge
    while
    a discarded value sits just under the bar. Only the ratio across the split says the rank was
    read
    off the spectrum rather than chosen from it. Ambiguity here is a hard failure — an equality
    wrongly discarded lets the sampler walk off the exact affine hull along a near-null direction
    whose mass-balance residual is far too small for the ``‖S·diag(s)·B‖`` check to notice.
    """

    def project(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """The component of ``vector`` that is a legal feasible direction. Twice, for accuracy."""
        out = np.zeros_like(vector)
        moving = vector[self.moving]
        if self.row_space.size:
            for _ in range(2):
                moving = moving - self.row_space @ (self.row_space.T @ moving)
        out[self.moving] = moving
        return out


def direction_space(
    reduced: ReducedPolytope,
    scales: NDArray[np.float64],
    blocked: NDArray[np.bool_],
    *,
    memory_limit_bytes: int,
    feasibility_tol: float = 1e-9,
    min_singular_gap: float = 1e3,
) -> DirectionSpace:
    """Factor ``S·diag(s)`` once, so every candidate direction can be cleaned in O(n·rank).

    Dense NumPy linear algebra, which spec §7 permits for reduced-coordinate work *with an explicit
    memory check* — so there is one. The numerical rank uses the standard machine-precision cutoff
    (``σ_max · max(shape) · eps``), and errs safe: reading the rank too *low* merely cleans less,
    while reading it too high would need a genuinely zero singular value to exceed a cutoff three
    orders of magnitude above the noise floor.
    """
    moving = np.flatnonzero(~blocked).astype(np.intp)
    if moving.size == 0:
        return DirectionSpace(
            blocked=blocked,
            row_space=np.zeros((0, 0), dtype=VALUE_DTYPE),
            moving=moving,
            n_null=0,
            singular_gap=np.inf,
        )

    n_rows = reduced.stoichiometry.n_rows
    needed = _BYTES_PER_FLOAT64 * n_rows * moving.size * 3  # the matrix, plus SVD working copies
    if needed > memory_limit_bytes:
        raise GeometryError(
            f"the direction-space factorization needs ~{needed / 2**30:.3f} GiB "
            f"({n_rows}×{moving.size} float64), but geometry.max_geometry_memory_gb allows "
            f"{memory_limit_bytes / 2**30:.3f} GiB"
        )

    matrix = reduced.stoichiometry.to_dense()[:, moving] * scales[moving]
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)

    # The cutoff is the larger of two resolutions, and it must be, because they must agree.
    # Machine epsilon says which singular values are indistinguishable from zero *in the
    # arithmetic*. The LP's feasibility tolerance says which equalities it will actually
    # *enforce*: a singular value below ε means a unit step along that right-singular vector moves
    # the mass balance by less than the solver tolerates, so the LP treats the direction as free.
    # Keeping such an equality while the LP ignores it is the worst of both — the support LPs hand
    # back a moving direction and `DirectionSpace` projects it to nothing, which is a
    # contradiction, not a geometry. Take the max and the two components describe the same
    # polytope. (On the example model the eps-based cutoff is 1.9e-8 and already dominates, so
    # this is a no-op there; it is the σ ≈ 1e-14 case it saves.)
    machine_cutoff = (
        float(singular.max(initial=0.0)) * max(matrix.shape) * float(np.finfo(VALUE_DTYPE).eps)
    )
    cutoff = max(machine_cutoff, feasibility_tol)
    rank = int(np.count_nonzero(singular > cutoff))

    retained = float(singular[rank - 1]) if rank > 0 else np.inf
    discarded = float(singular[rank]) if rank < singular.size else 0.0
    gap = np.inf if discarded <= 0.0 else retained / discarded
    if gap < min_singular_gap:
        raise GeometryError(
            "the equality constraints have no clear numerical rank: the smallest "
            f"retained singular value is {retained:.3e} and the largest discarded one is "
            f"{discarded:.3e}, a gap of "
            f"only {gap:.1f}× (need {min_singular_gap:.0f}×). S is near-rank-deficient, so which "
            "equalities are real is a decision the arithmetic cannot make — and the wrong answer "
            "samples a set the model does not describe."
        )

    return DirectionSpace(
        blocked=blocked,
        row_space=np.ascontiguousarray(right[:rank].T),
        moving=moving,
        n_null=int(moving.size - rank),
        singular_gap=gap,
    )


# ---- the orthonormal basis ----------------------------------------------------------------------


def _allocate(n: int, columns: int, memory_limit_bytes: int, what: str) -> NDArray[np.float64]:
    """Allocate an ``(n, columns)`` float64 block — but only after checking we are allowed to.

    Spec §15.5: *"Do not allow the operating system to discover the limit by crashing."* The guard
    runs before **every** allocation rather than once up front, because ``d`` is not known in
    advance and the complement (``n − d`` columns) is usually the larger of the two.
    """
    needed = _BYTES_PER_FLOAT64 * n * max(columns, 1)
    if needed > memory_limit_bytes:
        raise GeometryError(
            f"{what} needs {needed / 2**30:.4f} GiB ({n}×{columns} float64), but "
            f"geometry.max_geometry_memory_gb allows {memory_limit_bytes / 2**30:.4f} GiB"
        )
    return np.zeros((n, max(columns, 1)), dtype=VALUE_DTYPE, order="F")


class OrthonormalBasis:
    """The discovered feasible directions, orthonormal in scaled coordinates.

    Projection is **two-pass** (``v -= B(Bᵀv)``, applied twice) rather than one. A single pass loses
    orthogonality when the vector being projected lies nearly inside the current span — which is
    exactly the situation late in discovery, when the residuals under test are close to `rank_tol`.
    The second pass restores orthogonality to machine precision (spec §15.5).
    """

    def __init__(self, n: int, *, memory_limit_bytes: int, block: int = BASIS_BLOCK) -> None:
        self._n = n
        self._block = block
        self._columns = 0
        self._memory_limit_bytes = memory_limit_bytes
        self._storage = _allocate(n, min(block, max(n, 1)), memory_limit_bytes, "affine basis")

    @property
    def n_columns(self) -> int:
        return self._columns

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The ``(n, d)`` basis — a Fortran-contiguous view of the block storage (spec §15.5)."""
        return self._storage[:, : self._columns]

    def remove_components(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """``v − B Bᵀ v``, twice."""
        residual = np.array(vector, dtype=VALUE_DTYPE, copy=True)
        if self._columns == 0:
            return residual
        basis = self.matrix
        for _ in range(2):
            residual -= basis @ (basis.T @ residual)
        return residual

    def append_normalized(self, vector: NDArray[np.float64], *, rank_tol: float) -> None:
        """Re-orthogonalize, normalize, and append. Refuses a vector the basis already spans."""
        if self._columns >= self._n:
            raise GeometryError(
                f"cannot append a {self._n + 1}-th direction to a basis of ℝ^{self._n}"
            )
        residual = self.remove_components(vector)
        norm = float(np.linalg.norm(residual))
        if norm <= rank_tol:
            raise GeometryError(
                f"refusing to append a direction whose residual norm is {norm:.3e} (≤ rank_tol "
                f"{rank_tol:.3e}) — it already lies in the discovered span"
            )
        if self._columns == self._storage.shape[1]:
            self._grow()
        self._storage[:, self._columns] = residual / norm
        self._columns += 1

    def _grow(self) -> None:
        capacity = min(self._storage.shape[1] + self._block, self._n)
        grown = _allocate(self._n, capacity, self._memory_limit_bytes, "affine basis")
        grown[:, : self._columns] = self._storage[:, : self._columns]
        self._storage = grown


# ---- one support-LP probe (spec §15.3) ----------------------------------------------------------


@dataclass(frozen=True)
class SupportProbe:
    """One LP pair: maximize and minimize ``cᵀv`` along a direction orthogonal to the basis.

    `width_floor` and `rank_floor` are what make this honest. They are not configuration — they are
    computed from the infeasibility HiGHS reported *for these two solves*, so a probe knows how much
    of its own width could be an artefact of the solver rather than a property of the polytope.
    """

    direction: NDArray[np.float64]
    """``p``: unit norm, scaled coordinates, orthogonal to the current basis."""
    v_plus: NDArray[np.float64]
    v_minus: NDArray[np.float64]
    width: float
    """``cᵀ(v⁺ − v⁻)`` with ``c = p/s`` — the support width of the scaled polytope along ``p``."""
    residual: NDArray[np.float64]
    """``Δx − B Bᵀ Δx``: the part of this proven feasible direction the basis does not span."""
    residual_norm: float
    width_floor: float
    rank_floor: float
    is_conclusive: bool
    """False when the solver's own admitted error swamps the tolerance we were asked to resolve.

    Such a probe proves nothing in *either* direction: it can neither certify the direction flat nor
    claim it is real. It is recorded, and it costs the certificate its exhaustiveness.
    """
    dual_error: float
    """``max_dual_infeasibility`` over the two solves, on the sup-normalized objective."""
    width_upper: float
    """A **rigorous** upper bound on the true width, from weak duality (`dual_upper_bound`).

    The certificate's flatness claim rests on this, never on `width`. `width` is a *lower* bound —
    the objective difference between two feasible points, which proves a direction is at least that
    open — and certifying flatness from it would be reading the wrong end of the interval. On the
    example model the two agree to ~1e-13."""
    simplex_iterations: int

    @property
    def is_new_direction(self) -> bool:
        """Did this probe prove a feasible direction outside the current span?

        ``or``, not the spec's ``and`` — see the module docstring, point 3. The two tests cannot
        contradict each other: ``p`` is a unit vector orthogonal to ``B``, so
        ``width = pᵀΔx = pᵀ·residual ≤ ‖residual‖``, and `GeometryConfig` enforces
        ``rank_tol ≤ span_tol``. A positive width therefore always brings a residual with it.
        """
        return self.width > self.width_floor or self.residual_norm > self.rank_floor


def probe_direction(
    program: Maximizer,
    reduced: ReducedPolytope,
    basis: OrthonormalBasis,
    direction: NDArray[np.float64],
    scales: NDArray[np.float64],
    inverse_scale_norm: float,
    config: GeometryConfig,
    space: DirectionSpace | None = None,
) -> SupportProbe:
    """Solve ``max cᵀv`` and ``min cᵀv`` along ``p``, and report what the answer proves.

    ``c = p/s`` turns a scaled-coordinate direction into a flux-space objective (§15.3 step 5), so
    that ``cᵀv = pᵀ(v/s) = pᵀx``: the LP objective *is* the scaled coordinate along ``p``, and its
    optimized range *is* the width of the scaled polytope there.

    ``Δx`` is projected into `DirectionSpace` **before** the residual is formed, so what reaches the
    basis is exactly zero on blocked reactions and exactly mass-balanced. The **width** is left as
    the LP reported it: that is the honest support width, and the projection removes only components
    that no feasible direction can have anyway.
    """
    p = np.asarray(direction, dtype=VALUE_DTYPE)
    costs = p / scales

    # Hand HiGHS a **sup-normalized** objective. Its dual feasibility tolerance is *absolute*, so a
    # cost vector whose entries are all tiny — which ``c = p/s`` is whenever the bound ranges are
    # wide — can leave an improving reduced cost below that tolerance. HiGHS then reports kOptimal
    # at a vertex it never left, and the width comes back **zero for a direction that is genuinely
    # open**: a real dimension, certified flat. A positive rescaling cannot move the argmax, so this
    # costs nothing; and the width below is recomputed from the returned *primal* points with the
    # original `costs`, so it stays in the units we want.
    scale = float(np.max(np.abs(costs), initial=0.0))
    lp_costs = costs / scale if scale > 0.0 else costs

    plus = program.maximize(lp_costs)
    minus = program.maximize(-lp_costs)

    delta_v = plus.primal - minus.primal
    width = float(costs @ delta_v)

    delta_x = delta_v / scales
    if space is not None:
        delta_x = space.project(delta_x)
    residual = basis.remove_components(delta_x)
    residual_norm = float(np.linalg.norm(residual))

    # How much of that could the solver have invented? Take the worse of what HiGHS admits to and
    # what we can check ourselves — a solver's residual is only as believable as the model it was
    # measured in, and HiGHS's is internally scaled.
    #
    # Only the **bound** violation joins the noise bar, and deliberately. Comparing `v` against `l`
    # and `u` is exact in float64, so it measures the point. The mass-balance residual is *not* the
    # same kind of number: ‖S‖·‖v‖ ≈ 1e5 here, so evaluating `S·v` at all costs ~1e-10 of rounding,
    # and folding that in would charge the solver for our own arithmetic — enough, when measured, to
    # push `rank_floor` past `rank_tol` and fail the certificate on a geometry that is perfectly
    # good. It gets a *relative* check of its own instead, below.
    admitted = max(
        _bound_violation(plus.primal, reduced),
        _bound_violation(minus.primal, reduced),
        plus.max_primal_infeasibility,
        minus.max_primal_infeasibility,
    )
    _check_mass_balance(plus.primal, reduced, config)
    _check_mass_balance(minus.primal, reduced, config)
    # Push that through the two quantities we test:
    #     |cᵀe| ≤ ‖c‖₁·‖e‖_∞          (the width)
    #     ‖e/s‖₂ ≤ ‖1/s‖₂·‖e‖_∞       (the residual norm — a projection cannot lengthen it)
    #
    # These are *resolution* bars, not proofs: converting a constraint residual into a distance from
    # the exact feasible set needs a Hoffman constant we do not have. They say what the solver could
    # have manufactured at its own admitted accuracy, which is the honest thing we can say.
    width_noise = NOISE_SAFETY * float(np.abs(costs).sum()) * 2.0 * admitted
    rank_noise = NOISE_SAFETY * inverse_scale_norm * 2.0 * admitted

    # `width` above is a **lower** bound on the true width: it is the objective difference between
    # two (nearly) feasible points, so it proves a direction is *at least* that open. It does not
    # bound the width from above — and the certificate's whole claim is an upper bound. A solve that
    # stops short of optimality reports the width too **small**, and a real dimension is certified
    # flat. `max_dual_infeasibility` cannot catch that either: 1e-10 is dual-feasible anywhere, yet
    # on a variable of range 1e10 it hides a whole unit of width.
    #
    # So the two claims get two different instruments. Weak duality bounds the width from above with
    # no assumption about the solver at all — not optimality, and not even primal feasibility:
    #
    #     true width  ≤  α · [ U(y⁺; a) + U(y⁻; −a) ]
    #
    # On the example model that comes to ~1e-13, so it costs nothing where nothing is wrong.
    #
    # `dual_upper_bound` charges the roundings *inside* itself. The ones out here still need paying:
    # the division ``costs/scale`` perturbs each coefficient by a relative eps, and multiplying back
    # by `scale` plus the final addition each cost one more. The coefficient perturbation is the one
    # that is not purely relative — its effect on a width is bounded by what those perturbed
    # coefficients can do across the box — so it is charged in absolute terms, and the rest by a
    # relative inflation. Small (~1e-13 here), and the difference between a bound and an estimate.
    eps = float(np.finfo(VALUE_DTYPE).eps)
    width_upper = scale * (
        dual_upper_bound(plus, lp_costs, reduced)
        + dual_upper_bound(minus, -lp_costs, reduced)
    )
    coefficient_error = eps * float(
        np.abs(costs) @ (reduced.upper_bounds - reduced.lower_bounds)
    )
    width_upper = float(
        np.nextafter(width_upper * (1.0 + 8.0 * eps) + coefficient_error, np.inf)
    )
    if not np.isfinite(width_upper):
        raise GeometryError(
            "the width's upper bound overflowed to infinity; nothing can be certified from it"
        )
    dual_error = max(plus.max_dual_infeasibility, minus.max_dual_infeasibility)

    return SupportProbe(
        direction=p,
        v_plus=plus.primal,
        v_minus=minus.primal,
        width=width,
        residual=residual,
        residual_norm=residual_norm,
        width_floor=max(config.span_tol, width_noise),
        rank_floor=max(config.rank_tol, rank_noise),
        is_conclusive=(
            width_noise <= config.span_tol
            and rank_noise <= config.rank_tol
            and dual_error <= config.dual_tol
        ),
        dual_error=dual_error,
        width_upper=max(width_upper, width),
        simplex_iterations=plus.simplex_iterations + minus.simplex_iterations,
    )


def probe_noise_ceiling(inverse_scale_norm: float) -> float:
    """The scaling/LP-tolerance coupling, as a single number — reported, never trusted as a verdict.

    HiGHS is *permitted* to return a point that misses feasibility by its own primal tolerance
    ``ε``.
    Scaled coordinates divide by ``sᵢ``, so that error reaches a probe multiplied by at most
    ``‖1/s‖₂`` (Cauchy–Schwarz bounds ``‖c‖₁ ≤ ‖1/s‖₂`` for a unit direction, so one factor covers
    the width and the residual alike). Doubling for the difference ``v⁺ − v⁻`` and applying
    `NOISE_SAFETY`::

        ceiling = NOISE_SAFETY · 2 · ‖1/s‖₂ · ε

    That is the worst noise floor a probe could have **if the solver spent its whole error budget**.
    It never does — measured on the example model, HiGHS misses by ~1e-12 against a permitted 1e-9 —
    so gating on this bound would refuse geometries that are in fact perfectly resolved, which is
    why
    it is a diagnostic and not a veto. Each probe measures its *own* error instead
    (`SupportProbe.is_conclusive`), and an inconclusive probe is what actually costs the certificate
    its exhaustiveness. This number says how much room that mechanism has: on the example model the
    ceiling is 2.4e-10 against a 1e-9 tolerance, so even a worst-case-legal solve stays conclusive.
    """
    return NOISE_SAFETY * 2.0 * inverse_scale_norm * DEFAULT_PRIMAL_FEASIBILITY_TOL


def _check_blocked_span(
    blocked: BlockedReactions, scales: NDArray[np.float64], config: GeometryConfig
) -> None:
    """The blocked reactions must be flat **together**, not merely one at a time.

    A probe ``p`` supported on the blocked coordinates has width at most
    ``Σᵢ |pᵢ|·rᵢ/sᵢ ≤ ‖r_blocked/s_blocked‖₂`` when ``‖p‖₂ = 1`` — the same subadditivity
    that gives the certificate its ``√k``. `GeometryConfig` bounds each ``rᵢ/sᵢ`` by
    `span_tol` individually, which permits the *combination* to reach
    ``√n_blocked · span_tol``, and a probe spread across them would then report a width the
    projection has already zeroed. This is the check that actually holds: it uses the
    measured FVA widths rather than the config's worst case, and on the example model it
    comes to 5.7e-16 against a 1e-9 bar. The **upper** bracket is the width to use here — the
    same rigorous bound that certified each of these reactions blocked one at a time.
    """
    if not np.any(blocked.mask):
        return
    extent = float(np.linalg.norm(blocked.upper[blocked.mask] / scales[blocked.mask]))
    if extent > config.span_tol:
        raise GeometryError(
            f"the blocked reactions span {extent:.3e} in scaled coordinates *together* "
            f"(‖r/s‖₂ over {blocked.n_blocked} of them), above span_tol "
            f"{config.span_tol:.1e}. Individually each is flat, but a probe spread across "
            "them would show a width the blocked projection has already removed. "
            "already removed. Raise geometry.scale_floor, or lower geometry.blocked_tol."
        )


def dual_upper_bound(
    solution: LPSolution, costs: NDArray[np.float64], reduced: ReducedPolytope
) -> float:
    """A **rigorous** upper bound on ``max cᵀv`` over the polytope, from weak duality.

    For ``max cᵀv s.t. S v = rhs, l ≤ v ≤ u`` and *any* row multipliers ``y``::

        max cᵀv  ≤  rhsᵀy + Σⱼ max(dⱼ·lⱼ, dⱼ·uⱼ),      d = c − Sᵀy

    because ``cᵀv = (rhsᵀy) + dᵀv`` for every row-feasible ``v``, and each ``dⱼvⱼ`` is maximized at
    one end of its box. The bound holds *whatever* ``y`` is — optimal, sloppy, or
    nonsense — which is
    exactly why it is the right instrument here.

    The alternative, a gap built from complementary slackness, quietly assumes the returned point is
    exactly row-feasible: the true identity is
    ``cᵀv − cᵀv̂ = d̂ᵀ(v − v̂) + yᵀ(rhs − S v̂)``, and dropping that second term is unsound when
    ``S v̂ ≠ rhs``, which it never exactly is. ``d`` is recomputed here from ``c − Sᵀy`` rather than
    read from HiGHS's ``col_dual``, so a stationarity residual in the solver's own arrays cannot
    leak into the bound either.
    """
    duals = solution.row_duals
    if not np.all(np.isfinite(duals)):
        raise GeometryError("HiGHS returned non-finite row duals; no bound can be built from them")

    row_cost = reduced.stoichiometry.rmatvec(duals)
    d = costs - row_cost
    box = np.maximum(d * reduced.lower_bounds, d * reduced.upper_bounds)
    value = float(reduced.rhs @ duals + box.sum())

    # Every operation above rounds to **nearest**, so the computed number can land *below* the exact
    # one — and a bound that can be too small is not a bound. Add an outward allowance covering the
    # forward error of each step: forming Sᵀy, subtracting it from c, multiplying by the bounds, and
    # summing. Cheap insurance, and it is the difference between "an estimate of an upper bound" and
    # an upper bound. (Negligible on the example model — it changes nothing but what may be
    # claimed.)
    # Every operation above rounds to **nearest**, so the computed number can land *below* the exact
    # one — and a bound that can be too small is not a bound. Add an outward allowance covering the
    # forward error of each step, and the allowance is no formality: ``U`` is a cancellation of
    # terms of size ~5e3, so evaluating it in float64 costs ~1e-9 absolutely. **That is the floor on
    # what this certificate can honestly resolve** — orders coarser than the ~1e-13 the arithmetic
    # appears to produce. Reporting the appearance would mean reporting a number we cannot stand
    # behind.
    #
    # The terms, tightly: forming ``(Sᵀy)ⱼ`` sums ``nnzⱼ`` products (a handful, not ``n``); the
    # subtraction and the box product each cost one rounding; and `np.sum` is **pairwise**, so its
    # error grows as ``log₂ n`` rather than ``n``.
    eps = float(np.finfo(VALUE_DTYPE).eps)
    magnitude = reduced.stoichiometry.cancellation_scale_transpose(duals)
    nnz_per_column = np.diff(reduced.stoichiometry.starts).astype(VALUE_DTYPE)
    reach = np.maximum(np.abs(reduced.lower_bounds), np.abs(reduced.upper_bounds))
    depth = float(np.log2(max(reduced.n_free, 2))) + 1.0

    column_error = eps * (nnz_per_column + 1.0) * (magnitude + np.abs(d))
    # Two summations, two error models. `np.sum(box)` is **pairwise**, so ``log₂ n`` suffices. But
    # ``rhs @ duals`` is a BLAS dot: no pairwise guarantee, and its term count is ``n_rows``, not
    # ``n_free``. Charging it with the pairwise depth would be assuming an accumulation order BLAS
    # never promised.
    box_error = eps * depth * float(np.abs(box).sum())
    rhs_error = (
        eps
        * float(reduced.stoichiometry.n_rows)
        * float(np.abs(reduced.rhs) @ np.abs(duals))
    )
    allowance = float(np.sum(column_error * reach)) + box_error + rhs_error
    bound = value + allowance

    if not np.isfinite(bound):
        raise GeometryError("the dual upper bound is not finite; the LP's duals are unusable")
    return bound


def _bound_violation(flux: NDArray[np.float64], reduced: ReducedPolytope) -> float:
    """How far a returned LP point sits outside its box. Exact in float64 — it is three
    comparisons."""
    return float(
        np.max(
            np.maximum(reduced.lower_bounds - flux, flux - reduced.upper_bounds),
            initial=0.0,
        )
    )


def _check_mass_balance(
    flux: NDArray[np.float64], reduced: ReducedPolytope, config: GeometryConfig
) -> None:
    """Verify a returned endpoint against *our* equalities, on a **relative** bar.

    An absolute bar is the wrong instrument. ``S·v`` is a sum of terms as large as ‖S‖·‖v‖ ≈ 1e5 on
    this model, so merely *evaluating* it in float64 costs ~1e-10 — a residual of that size says
    nothing whatever about the LP. What would be damning is a residual large *relative to the
    magnitudes being cancelled*, and that is what this measures.
    """
    residual = float(np.max(np.abs(reduced.mass_balance_residual(flux)), initial=0.0))
    magnitude = float(np.max(reduced.stoichiometry.cancellation_scale(flux), initial=0.0))
    floor = magnitude * float(np.finfo(VALUE_DTYPE).eps) * reduced.n_free
    if residual > max(config.feasibility_tol, MASS_BALANCE_SAFETY * floor):
        raise GeometryError(
            f"a support LP returned a point whose mass-balance residual is {residual:.3e}, beyond "
            f"both the feasibility tolerance ({config.feasibility_tol:.1e}) and the {floor:.3e} "
            "that evaluating S·v in float64 could account for — the solver's answer is not a "
            "steady state"
        )


# ---- the deterministic span certificate (BUILD_PLAN §1.4) ---------------------------------------


def _span_resolution(
    n_complement: int, leakage: float, max_width: float, diameter: float
) -> float:
    """The width bound a completed sweep licenses — the **one** definition, used by two callers.

    `SpanCertificate.resolution` reports it; `sweep_complement` gates on it. Extracting it (M11.2)
    keeps the gate and the report from drifting apart — a recurring shape in this repo, where a
    duplicated formula is how a claim and its check come to disagree. See the `resolution` property
    for the ``√k`` subadditivity and the leakage term.
    """
    value = (
        float(np.sqrt(max(n_complement, 1))) * float(np.sqrt(1.0 + leakage)) * max_width
        + leakage * diameter
    )
    eps = float(np.finfo(np.float64).eps)
    rounded = float(np.nextafter(value * (1.0 + 8.0 * eps), np.inf))
    if not np.isfinite(rounded):
        raise GeometryError("the span resolution is not finite; this certificate certifies nothing")
    return rounded


@dataclass(frozen=True)
class SpanCertificate:
    """The evidence that ``B`` spans every feasible direction — or the admission that it may not."""

    exhaustive: bool
    """The span is certified to the requested resolution: ``resolution ≤ span_tol`` on a **complete,
    uncapped** sweep with no probe having proved a new direction (M11.2, §1.6.11).

    It is **not** "every probe conclusive" — that was the v1 rule, and it conflated *"a probe's
    primal/residual discovery signal was noise-swamped"* with *"the span may be incomplete"*, which
    are different propositions (§1.6.10). `is_conclusive` governs whether a probe's endpoints can
    *drive discovery*; the flatness claim rests only on each probe's **rigorous** `width_upper`
    (weak duality, assuming nothing of the returned point), aggregated by `resolution`. So an
    inconclusive probe may sit under a certified geometry, and a conclusive probe with a loose dual
    bound may fail: the new predicate is stronger about what it *claims*, not a subset of the old.
    A cap or a truncated complement still forbid it (those leave directions genuinely unprobed).
    `n_inconclusive` survives as a reported diagnostic."""
    n_probes: int
    n_complement: int
    """``n_free − d``: how many probes a complete certificate needs."""
    max_width: float
    """The largest width any *single* probe found. See `resolution` for what that licenses."""
    max_width_floor: float
    n_inconclusive: int
    worst_dual_error: float
    """Largest ``max_dual_infeasibility`` over the sweep."""
    leakage: float
    """``‖I − BBᵀ − QQᵀ‖₂`` — how much of ℝⁿ neither the basis nor the *computed* complement covers.

    The ``√k`` argument assumes ``Q`` exactly spans ``range(B)ᗮ``. Gram-Schmidt in float64 does not,
    quite, and a direction hiding in the gap would be probed by nothing. So the gap is measured, and
    it enters `resolution` multiplied by the polytope's own diameter — which is the most width any
    direction can have."""
    diameter: float
    """``‖(u − l)/s‖₂`` — an upper bound on the scaled polytope's width along *any* unit
    direction."""
    complement_is_complete: bool
    """False when Gram-Schmidt could not produce all ``n_free − d`` complement directions."""

    @property
    def resolution(self) -> float:
        """What the certificate actually licenses: **no missed direction is wider than this.**

        Not `max_width`. Width is a support-function difference, hence subadditive and positively
        homogeneous, so for a unit ``p = Σ aⱼ pⱼ`` in the complement (``‖a‖₂ = 1``, thus
        ``‖a‖₁ ≤ √k``)::

            width(p) ≤ Σ |aⱼ|·width(pⱼ) ≤ √k · maxⱼ width(pⱼ)

        A direction tilted equally across all ``k`` probes therefore hides a factor of ``√k`` from
        every individual one. Probing ``k = 214`` axes at a 1e-9 tolerance would only bound an
        arbitrary complement direction by 1.5e-8 — so the number to *report* is this one, and on the
        example model the measured widths make it 8.9e-14 rather than the tolerance's 1.5e-8.

        This is why the certificate is **resolution-bounded, not exact**: in exact arithmetic zero
        width is zero and the span proof is airtight, but a float64 LP can only ever say "flatter
        than this", and this is the honest value of *this*.
        """
        return _span_resolution(self.n_complement, self.leakage, self.max_width, self.diameter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_certificate_exhaustive": self.exhaustive,
            "n_probes": self.n_probes,
            "n_complement": self.n_complement,
            "max_width": self.max_width,
            "max_width_floor": self.max_width_floor,
            "span_resolution": self.resolution,
            "n_inconclusive": self.n_inconclusive,
            "worst_dual_error": self.worst_dual_error,
            "complement_leakage": self.leakage,
            "scaled_diameter": self.diameter,
            "complement_is_complete": self.complement_is_complete,
        }

    def to_cache(self) -> dict[str, Any]:
        """A round-trippable dict keyed by the **real field names** (not `as_dict`'s report keys).

        `as_dict` is the human-facing manifest: it renames fields (``exhaustive`` →
        ``span_certificate_exhaustive``) and injects the *derived* `resolution`, which is not a
        field and must not be fed back to the constructor. A cache reconstructs the object, so it
        keeps the field names — the same split `rounding.RoundingDiagnostics` already draws.
        """
        return asdict(self)

    @classmethod
    def from_cache(cls, data: dict[str, Any]) -> SpanCertificate:
        names = {f.name for f in fields(cls)}
        missing = names - data.keys()
        if missing:
            raise GeometryError(f"cached span certificate lacks fields: {sorted(missing)}")
        return cls(**{name: data[name] for name in names})


def complement_basis(
    basis_matrix: NDArray[np.float64],
    n: int,
    *,
    rank_tol: float,
    max_columns: int | None,
    memory_limit_bytes: int,
) -> NDArray[np.float64]:
    """An orthonormal basis of ``range(B)ᗮ``, ordered by residual norm (BUILD_PLAN §1.4).

    This is a column-pivoted QR of the projector ``I − BBᵀ``, run as Gram-Schmidt over the
    coordinate axes, pivoting on the largest remaining residual (Businger–Golub). Ordering by
    residual norm puts the axes *least* represented in ``B`` first, so a sweep that is going to
    fail tends to fail early. The completeness argument needs all of them, and does not care in
    what order they come.

    The residual norm of axis ``i`` against an orthonormal set ``Q`` is ``1 − Σⱼ Q[i,j]²``, so the
    pivot scores downdate in O(n) per column instead of being recomputed from scratch.
    """
    d = int(basis_matrix.shape[1])
    target = n - d
    if max_columns is not None:
        target = min(target, max_columns)
    if target <= 0:
        return np.zeros((n, 0), dtype=VALUE_DTYPE, order="F")

    complement = _allocate(n, target, memory_limit_bytes, "span-certificate complement")

    scores = np.ones(n, dtype=VALUE_DTYPE)
    if d:
        scores -= np.einsum("ij,ij->i", basis_matrix, basis_matrix)

    added = 0
    while added < target:
        axis = int(np.argmax(scores))
        if scores[axis] <= rank_tol**2:
            break  # every axis is (numerically) spanned — the complement is exhausted early

        vector = np.zeros(n, dtype=VALUE_DTYPE)
        vector[axis] = 1.0
        for _ in range(2):  # two-pass, against B and against what we have built of the complement
            if d:
                vector -= basis_matrix @ (basis_matrix.T @ vector)
            if added:
                built = complement[:, :added]
                vector -= built @ (built.T @ vector)

        norm = float(np.linalg.norm(vector))
        if norm <= rank_tol:
            scores[axis] = -np.inf  # spanned after all; never pick this axis again
            continue

        column = vector / norm
        complement[:, added] = column
        scores -= column**2
        np.maximum(scores, 0.0, out=scores)
        added += 1

    return complement[:, :added]


def sweep_complement(
    program: Maximizer,
    reduced: ReducedPolytope,
    basis: OrthonormalBasis,
    scales: NDArray[np.float64],
    inverse_scale_norm: float,
    config: GeometryConfig,
    memory_limit_bytes: int,
    space: DirectionSpace | None = None,
) -> tuple[SpanCertificate, SupportProbe | None]:
    """Probe every direction orthogonal to the basis; stop at the first that proves a dimension.

    Returns the certificate *and* the failing probe, when there was one. A failure is not merely a
    verdict: it hands back the very ``Δx`` that disproves the basis, so the caller can append that
    direction instead of sending discovery out to look for it again.

    The complement deliberately spans the **whole** of ``range(B)ᗮ`` in ℝ^n_free, blocked coordinate
    axes included. Probing those costs one LP pair each and buys something worth having: the
    certificate re-derives, from its own LPs, that a blocked reaction has zero width — so the
    completeness proof never has to *assume* what `blocked_reactions` told it.
    """
    n = int(scales.size)
    complement = complement_basis(
        basis.matrix,
        n,
        rank_tol=config.rank_tol,
        max_columns=config.max_span_probes,
        memory_limit_bytes=memory_limit_bytes,
    )
    n_complement = n - basis.n_columns
    capped = config.max_span_probes is not None and config.max_span_probes < n_complement
    complete = int(complement.shape[1]) == n_complement

    # What neither B nor the computed complement covers. In exact arithmetic this is zero; in
    # float64 it is where a direction could hide from every probe, so it is measured rather than
    # assumed. The spectral norm, because it bounds ‖(I − BBᵀ − QQᵀ)p‖ for *every* unit p.
    span = basis.matrix @ basis.matrix.T + complement @ complement.T
    eps = float(np.finfo(VALUE_DTYPE).eps)
    raw_leakage = float(np.linalg.norm(np.eye(n) - span, 2)) if complete else 1.0
    raw_diameter = float(np.linalg.norm((reduced.upper_bounds - reduced.lower_bounds) / scales))
    # `np.linalg.norm` is an ordinary float64 computation, so neither is a certified bound as it
    # stands. Both feed an upper bound, so both round *outward* — generously, since an SVD of an
    # n×n matrix accumulates more than a single ulp.
    leakage = float(np.nextafter(raw_leakage * (1.0 + 16.0 * n * eps), np.inf))
    diameter = float(np.nextafter(raw_diameter * (1.0 + 16.0 * n * eps), np.inf))

    max_width = 0.0
    max_width_floor = 0.0
    worst_dual_error = 0.0
    n_inconclusive = 0
    n_probes = 0
    failing: SupportProbe | None = None

    for column in range(int(complement.shape[1])):
        probe = probe_direction(
            program,
            reduced,
            basis,
            complement[:, column],
            scales,
            inverse_scale_norm,
            config,
            space,
        )
        n_probes += 1
        max_width = max(max_width, probe.width_upper)
        max_width_floor = max(max_width_floor, probe.width_floor)
        worst_dual_error = max(worst_dual_error, probe.dual_error)
        if not probe.is_conclusive:
            n_inconclusive += 1
        if probe.is_new_direction:
            failing = probe
            break

    # Gate on the quantity the certificate actually claims (the resolution), not on `n_inconclusive`
    # (M11.2, §1.6.11). The resolution rests only on each probe's rigorous `width_upper`; a probe
    # being inconclusive says its primal discovery signal was noise-swamped, a different proposition
    # (§1.6.10). `failing is None`, `complete` and `not capped` stay: those are the cases where a
    # direction was genuinely left unprobed, and no resolution bound covers them.
    resolution = _span_resolution(n_complement, leakage, max_width, diameter)
    certificate = SpanCertificate(
        exhaustive=(
            failing is None and complete and not capped and resolution <= config.span_tol
        ),
        n_probes=n_probes,
        n_complement=n_complement,
        max_width=max_width,
        max_width_floor=max_width_floor,
        n_inconclusive=n_inconclusive,
        worst_dual_error=worst_dual_error,
        leakage=leakage,
        diameter=diameter,
        complement_is_complete=complete,
    )
    return certificate, failing


# ---- the artifact -------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryDiagnostics:
    """What the geometry phase saw. Written to the run manifest (spec §15.5)."""

    n_free: int
    dimension: int
    n_blocked: int
    """Free reactions FVA proved cannot carry flux — exact zeros of the direction space."""
    blocked_separation: float
    """How unambiguous that split was: narrowest moving range ÷ widest blocked range."""
    n_blocked_escalated: int
    """How many reactions the warm FVA pass could not classify and a cold re-solve had to (M11.1).

    Zero on a model whose warm duals stay tight; nonzero is the warm path's own error rate, and
    worth watching — a model that needs many cold re-solves is one whose geometry sits near the
    instrument's floor.
    """
    n_support_points: int
    n_lp_solves: int
    """Every HiGHS solve this build ran — warm, failed-warm, `blocked_reactions`' cold pairs and the
    session's cold retries — from the process-global counter. Exact for a **serialized** build; two
    `build_geometry` calls running in one process concurrently would overlap their deltas, so the
    batch runs geometry in the parent, one strain at a time (§1.2)."""
    n_cold_solves: int
    """The **session's** cold retries — the initial/support/span solves it ran on a fresh instance
    after the warm one returned ``kUnknown`` (M11.2). It does **not** include `blocked_reactions`'
    cold pairs (those are `n_blocked_escalated`) nor a refusal's fully-cold re-sweep (that path
    raises). Zero when the support/span stages never degraded."""
    degraded: bool
    """True if the build abandoned its warm instance at any point (in `blocked_reactions` or a later
    stage). A boolean, not an index: `blocked_reactions`' FVA solves bypass the session,
    so an index would count from zero and mislead (Codex, M11.2 review)."""
    n_random_probes: int
    n_sweeps: int
    simplex_iterations: int
    orthonormality_error: float
    """``‖BᵀB − I‖_max``."""
    mass_balance_error: float
    """``‖S·diag(s)·B‖_max`` — every basis direction must be a *steady-state* direction."""
    center_mass_balance_residual: float
    center_bound_slack: float
    """The centre's smallest distance to any bound, after the clamp — so never negative."""
    center_clamp: float
    """How far the raw mean of the support points had to move to satisfy its bounds exactly.

    A convex combination of feasible points is feasible in exact arithmetic; in float64 it lands a
    solver-noise distance outside a bound it touches. That distance is this number, and it is
    bounded by the LP feasibility tolerance or the geometry refuses to build."""
    min_chord_at_center: float
    """The shortest chord through the centre along any basis direction (M5's start condition).

    Every chord must strictly contain ``t = 0`` — this is what the blocked-reaction projection buys,
    and it is checked here rather than left for the sampler to discover."""
    support_coordinate_rank: int
    """Rank of the support points in reduced coordinates. **Must equal ``d``.**

    M5 rounds the polytope with the Cholesky factor of the support-point covariance. If the support
    points do not actually span all ``d`` directions, that covariance is singular in some direction
    and the ridge M5 adds would quietly paper over it — producing a transform that is invertible,
    plausible, and wrong. The geometry is where this is knowable, so it is checked here."""
    n_scales_floored: int
    inverse_scale_norm: float
    """``‖1/s‖₂`` — how much the scaled coordinates magnify the solver's feasibility error."""
    probe_noise_ceiling: float
    """Worst noise floor a *legal* solve can give a probe. Must stay under span_tol and rank_tol."""
    basis_memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_free": self.n_free,
            "dimension": self.dimension,
            "n_blocked": self.n_blocked,
            "blocked_separation": self.blocked_separation,
            "n_blocked_escalated": self.n_blocked_escalated,
            "n_support_points": self.n_support_points,
            "n_lp_solves": self.n_lp_solves,
            "n_cold_solves": self.n_cold_solves,
            "degraded": self.degraded,
            "n_random_probes": self.n_random_probes,
            "n_sweeps": self.n_sweeps,
            "simplex_iterations": self.simplex_iterations,
            "orthonormality_error": self.orthonormality_error,
            "mass_balance_error": self.mass_balance_error,
            "center_mass_balance_residual": self.center_mass_balance_residual,
            "center_bound_slack": self.center_bound_slack,
            "center_clamp": self.center_clamp,
            "min_chord_at_center": self.min_chord_at_center,
            "support_coordinate_rank": self.support_coordinate_rank,
            "n_scales_floored": self.n_scales_floored,
            "inverse_scale_norm": self.inverse_scale_norm,
            "probe_noise_ceiling": self.probe_noise_ceiling,
            "basis_memory_bytes": self.basis_memory_bytes,
        }

    def to_cache(self) -> dict[str, Any]:
        """Round-trippable, keyed by the real field names — see `SpanCertificate.to_cache`."""
        return asdict(self)

    @classmethod
    def from_cache(cls, data: dict[str, Any]) -> GeometryDiagnostics:
        names = {f.name for f in fields(cls)}
        missing = names - data.keys()
        if missing:
            raise GeometryError(f"cached geometry diagnostics lack fields: {sorted(missing)}")
        return cls(**{name: data[name] for name in names})


@dataclass(frozen=True)
class ReducedGeometry:
    """The L3 artifact: ``v = centre + diag(s)·B·q`` maps reduced coordinates ``q ∈ ℝᵈ`` to flux.

    The centre and the support points live in **reduced** flux coordinates (length ``n_free``);
    `ReducedPolytope.to_full` lifts them to full-length flux vectors when a sample is saved.

    M5 preconditions this further — ``T = diag(s)·B·L``, with ``L`` the Cholesky factor of the
    support covariance — so what is stored here is deliberately the *unrounded* transform.
    """

    scaling: NDArray[np.float64]
    basis: NDArray[np.float64]
    """``(n_free, d)``, orthonormal, Fortran-contiguous."""
    center: NDArray[np.float64]
    support_points: NDArray[np.float64]
    """``(K, n_free)`` feasible LP vertices. M5's rounding takes its covariance from these."""
    certificate: SpanCertificate
    diagnostics: GeometryDiagnostics
    polytope_key: str
    """The L1 key of the polytope this belongs to — a geometry is meaningless without it."""

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_free(self) -> int:
        return int(self.scaling.size)

    @property
    def is_singleton(self) -> bool:
        """Dimension zero: the feasible set is a point, and every sample is the centre (§16)."""
        return self.dimension == 0

    def to_flux(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """``v = centre + diag(s)·B·q``. Takes one ``q`` or a batch of them."""
        q = np.asarray(coordinates, dtype=VALUE_DTYPE)
        if q.shape[-1:] != (self.dimension,):
            raise ValueError(
                f"coordinates have shape {q.shape}, expected trailing dimension {self.dimension}"
            )
        return self.center + (q @ self.basis.T) * self.scaling

    def to_coordinates(self, flux: NDArray[np.float64]) -> NDArray[np.float64]:
        """``q = Bᵀ·diag(s)⁻¹·(v − centre)`` — the inverse of `to_flux` on the affine hull."""
        v = np.asarray(flux, dtype=VALUE_DTYPE)
        if v.shape[-1:] != (self.n_free,):
            raise ValueError(f"flux has shape {v.shape}, expected trailing dimension {self.n_free}")
        return ((v - self.center) / self.scaling) @ self.basis

    def content_key(self) -> str:
        """The L3 cache key (§1.1): the polytope, the scaling, the basis, the code behind them.

        ``support_points`` is hashed too (M10.2). It is a *field of this artifact*, and it is
        load-bearing downstream in two places that outlive the geometry: `rounding.build_transform`
        takes ``C_q`` from it, and `sparse_objective.choose_energy_scale` reads ``warmup_range``'s
        ``s_J`` off it. Omitting it let two geometries agreeing on ``B`` but disagreeing on their
        support hull hash **identically** — the M6 join again, one artifact further on.
        """
        return content_key(
            polytope_key=self.polytope_key,
            scaling=self.scaling,
            basis=self.basis,
            center=self.center,
            support_points=self.support_points,
            dimension=self.dimension,
            geometry_impl_version=GEOMETRY_IMPL_VERSION,
            backend_impl_version=BACKEND_IMPL_VERSION,
            numpy_version=np.__version__,
        )

    def to_bundle(self) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Split into ``(arrays, meta)`` for the content-addressed cache (`cache.ArtifactCache`).

        M10.2 exists because this method did not. §1.1 has always said L3 "holds B, support_points,
        centre, L (Cholesky), T, dimension, span certificate", but the only serializer in the
        package was `rounding.RoundedTransform.to_bundle` — which holds ``T`` and ``L`` and neither
        ``B`` nor ``s``. So a cache **hit** returned a transform and no geometry, and
        `rounding.reround_transform` (which needs ``B``, ``s``, and this key) could not run: M10's
        pilot DAG was reachable from the library and unreachable from the CLI. That was recorded as
        a design fork; it was plan/code drift.
        """
        return (
            {
                "scaling": np.ascontiguousarray(self.scaling),
                "basis": np.ascontiguousarray(self.basis),
                "center": np.ascontiguousarray(self.center),
                "support_points": np.ascontiguousarray(self.support_points),
            },
            {
                "content_key": self.content_key(),
                "polytope_key": self.polytope_key,
                "certificate": self.certificate.to_cache(),
                "diagnostics": self.diagnostics.to_cache(),
            },
        )

    @classmethod
    def from_bundle(
        cls,
        arrays: dict[str, NDArray[Any]],
        meta: dict[str, Any],
        reduced: ReducedPolytope,
    ) -> ReducedGeometry:
        """Rebuild a geometry cached by `to_bundle`, against the polytope it was built for.

        Two guards, and they catch different things. The ``polytope_key`` check refuses the M6 join
        — a geometry rebuilt against a *different* model's bounds. The ``content_key`` round-trip
        then proves the reconstructed object **hashes to the key it was stored under**: the cache is
        content-addressed, so an artifact that does not reproduce its own key is not the artifact
        the key names, whatever its arrays look like.

        The arrays are left writable, matching `build_geometry` exactly — a cached geometry must be
        indistinguishable from a fresh one, and a reconstruction that hardened them differently
        would be its own drift (M5 left ``center`` writable for its owner on purpose).
        """
        if meta["polytope_key"] != reduced.content_key():
            raise GeometryError(
                "cached geometry was built for a different polytope "
                f"({str(meta['polytope_key'])[:16]}…) than the one it is being rebuilt against "
                f"({reduced.content_key()[:16]}…)"
            )
        geometry = cls(
            scaling=np.asarray(arrays["scaling"], dtype=VALUE_DTYPE),
            # `(n_free, d)`, Fortran-contiguous — the layout the dataclass documents and the column
            # reads want. `hash_array` C-orders before hashing, so this is a performance contract,
            # never an identity one: the key below is order-blind by construction.
            basis=np.asfortranarray(np.asarray(arrays["basis"], dtype=VALUE_DTYPE)),
            center=np.asarray(arrays["center"], dtype=VALUE_DTYPE),
            support_points=np.asarray(arrays["support_points"], dtype=VALUE_DTYPE),
            certificate=SpanCertificate.from_cache(meta["certificate"]),
            diagnostics=GeometryDiagnostics.from_cache(meta["diagnostics"]),
            polytope_key=str(meta["polytope_key"]),
        )
        if geometry.content_key() != meta["content_key"]:
            raise GeometryError(
                "the rebuilt geometry does not reproduce its own cached content key "
                f"({geometry.content_key()[:16]}… vs {str(meta['content_key'])[:16]}…); the stored "
                "bytes are not the artifact this key names, so nothing downstream may trust it"
            )
        return geometry

    def manifest(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n_free": self.n_free,
            "geometry_impl_version": GEOMETRY_IMPL_VERSION,
            "content_key": self.content_key(),
            **self.certificate.as_dict(),
            **self.diagnostics.as_dict(),
        }


# ---- discovery (spec §15.3, §15.6) --------------------------------------------------------------


@under_deterministic_blas
def build_geometry(
    reduced: ReducedPolytope,
    *,
    config: GeometryConfig | None = None,
    model_id: str = "model",
) -> ReducedGeometry:
    """Discover the affine basis, certify that it spans the polytope, and build a feasible centre.

    The outer loop is what makes the certificate binding: discovery hands the sweep a basis, and a
    sweep that *fails* hands discovery back the direction it was missing. Every failure raises the
    dimension by one, so the loop runs at most ``n_free`` times — and it can only exit through a
    clean sweep.

    The decorator is load-bearing (M10.2e): the re-orthogonalization below is a BLAS matmul, and
    multi-threaded OpenBLAS reduces in a different order — so the *ambient thread count* used to
    pick between two valid bases under one L3 key, and the pilot amplified the 2.7e-15 between them
    to O(1). See `numerics` for the measurements.
    """
    cfg = config if config is not None else GeometryConfig()
    memory_limit_bytes = int(cfg.max_geometry_memory_gb * 2**30)

    if reduced.is_singleton:
        return _singleton_geometry(reduced, cfg)

    n = reduced.n_free
    scales = reaction_scales(reduced, floor=cfg.scale_floor)
    n_floored = int(np.sum(reduced.upper_bounds - reduced.lower_bounds < cfg.scale_floor))
    inverse_scale_norm = float(np.linalg.norm(1.0 / scales))
    noise_ceiling = probe_noise_ceiling(inverse_scale_norm)

    # One solve session owns the LP instance for the whole build (M11.2): warm until the first
    # `kUnknown`, cold-only after — so a degradation in *any* stage propagates to the ones that
    # follow, which M11.1's function-local abandonment could not do. `solves_at_start` reads the
    # **process-global** counter, which records every solve on every instance (warm, failed, and the
    # fresh cold retries), so the reported LP count no longer undercounts the escalations.
    session = _SolveSession(reduced)
    solves_at_start = total_solve_count()
    basis = OrthonormalBasis(n, memory_limit_bytes=memory_limit_bytes)
    rng = np.random.default_rng(stream_seed(model_id=model_id, stage="geometry", seed=cfg.seed))

    # FVA first: the reactions that cannot move are exact zeros of every feasible direction, and a
    # basis that carries their noise instead corrupts the chord (module docstring, point 4). Then
    # factor the space those directions must live in, so no candidate can pollute the basis.
    # `blocked_reactions` runs its own warm sweep + cold-pair escalation on the session's instance,
    # and tells the session (`mark_degraded`) if that instance ever returned `kUnknown`.
    blocked = blocked_reactions(
        reduced, session.program, tol=cfg.blocked_tol, on_degraded=session.mark_degraded
    )
    space = direction_space(
        reduced,
        scales,
        blocked.mask,
        memory_limit_bytes=memory_limit_bytes,
        feasibility_tol=cfg.feasibility_tol,
    )
    _check_blocked_span(blocked, scales, cfg)

    # A zero objective asks HiGHS for *any* feasible point — the seed of the support set (§15.6).
    initial = session.maximize(np.zeros(n, dtype=VALUE_DTYPE))
    support: list[NDArray[np.float64]] = [initial.primal]
    iterations = initial.simplex_iterations

    n_random_probes = 0
    stall = 0
    degenerate_draws = 0
    while stall < cfg.stall_probes and basis.n_columns < n:
        candidate = basis.remove_components(rng.normal(size=n))
        norm = float(np.linalg.norm(candidate))
        if norm <= cfg.rank_tol:
            # The draw collapsed onto the basis. With d < n that has probability zero, so a run of
            # them means the basis is not what it claims to be — fail loudly rather than spin.
            degenerate_draws += 1
            if degenerate_draws > MAX_DEGENERATE_DRAWS:
                raise GeometryError(
                    f"{MAX_DEGENERATE_DRAWS} consecutive random probes collapsed onto a "
                    f"{basis.n_columns}-column basis of ℝ^{n} — the basis is not orthonormal"
                )
            continue
        degenerate_draws = 0
        n_random_probes += 1

        probe = probe_direction(
            session, reduced, basis, candidate / norm, scales, inverse_scale_norm, cfg, space
        )
        iterations += probe.simplex_iterations
        if probe.is_new_direction:
            _append(basis, probe, reduced, blocked, cfg)
            support.extend([probe.v_plus, probe.v_minus])
            stall = 0
        else:
            stall += 1

    n_sweeps = 0
    while True:
        n_sweeps += 1
        certificate, failing = sweep_complement(
            session, reduced, basis, scales, inverse_scale_norm, cfg, memory_limit_bytes, space
        )
        if failing is None:
            break
        iterations += failing.simplex_iterations
        # The sweep did not merely report "incomplete" — it produced the missing direction. Take it.
        _append(basis, failing, reduced, blocked, cfg)
        support.extend([failing.v_plus, failing.v_minus])

    if cfg.exhaustive_span_certificate and not certificate.exhaustive:
        # Before a *final* refusal, re-sweep fully cold and re-check (M11.2, Codex). The sweep's
        # `width_upper` is a weak-duality bound, sound but able to be *loose* under a warm-started
        # basis — the same mechanism that made `blocked_reactions` refuse valid strains — so a
        # `resolution > span_tol` from a warm sweep is not yet a statement about the polytope. Only
        # when a cold sweep, on fresh instances with no inherited history, still fails to resolve it
        # do we call the geometry uncertifiable. This costs one extra sweep, and only on the refusal
        # path. (`failing`/`complete`/`capped` refusals are structural, not warm-dual artifacts, so
        # they are not re-confirmed — a genuinely unprobed direction stays unprobed cold.)
        confirmed = certificate
        if certificate.complement_is_complete and not (
            cfg.max_span_probes is not None and cfg.max_span_probes < certificate.n_complement
        ):
            cold_session = _SolveSession(reduced, cold_only=True)
            confirmed, _ = sweep_complement(
                cold_session, reduced, basis, scales, inverse_scale_norm, cfg,
                memory_limit_bytes, space,
            )
        if not confirmed.exhaustive:
            raise GeometryError(
                "the span certificate does not resolve the polytope to span_tol, even after a "
                f"fully-cold re-sweep (resolution {confirmed.resolution:.3e} vs span_tol "
                f"{cfg.span_tol:.1e}; {confirmed.n_probes}/{confirmed.n_complement} probes, "
                f"{confirmed.n_inconclusive} inconclusive [diagnostic], complement complete="
                f"{confirmed.complement_is_complete}) — spec §15.4 forbids sampling a "
                "lower-dimensional subset on an uncertified basis. Set "
                "geometry.exhaustive_span_certificate = false to accept a randomized partial check."
            )
        certificate = confirmed

    support_points = np.asarray(support, dtype=VALUE_DTYPE)
    center, clamp = _feasible_center(support_points, reduced, cfg)

    diagnostics = _validate(
        reduced=reduced,
        basis=basis.matrix,
        scales=scales,
        center=center,
        clamp=clamp,
        support_points=support_points,
        blocked=blocked,
        space=space,
        config=cfg,
        n_lp_solves=total_solve_count() - solves_at_start,
        n_random_probes=n_random_probes,
        n_sweeps=n_sweeps,
        simplex_iterations=iterations,
        n_floored=n_floored,
        inverse_scale_norm=inverse_scale_norm,
        noise_ceiling=noise_ceiling,
        n_cold_solves=session.n_cold_solves,
        degraded=session.degraded,
    )

    return ReducedGeometry(
        scaling=scales,
        basis=np.asfortranarray(basis.matrix.copy()),
        center=center,
        support_points=support_points,
        certificate=certificate,
        diagnostics=diagnostics,
        polytope_key=_polytope_key(reduced),
    )


def _append(
    basis: OrthonormalBasis,
    probe: SupportProbe,
    reduced: ReducedPolytope,
    blocked: BlockedReactions,
    config: GeometryConfig,
) -> None:
    """Append a proven direction — or explain precisely why the tolerances make that impossible.

    A probe can report a real width while `DirectionSpace` zeroes the residual that width came from.
    That is not "the basis already spans it": it means the LP found motion along a direction the
    admitted space **excludes** — a blocked reaction that can actually move, or an equality the SVD
    rank dropped. The two resolutions have contradicted each other, and the geometry must say so
    rather than raise a message about a span, which is what it used to do.

    `GeometryConfig` forbids the common cause up front (``scale_floor ≥ blocked_tol /
    span_tol``), so
    reaching here means something the config relation cannot rule out — a probe spread across many
    blocked axes, or a near-singular equality. Either way the user needs the numbers, not a guess.
    """
    if probe.residual_norm > config.rank_tol:
        basis.append_normalized(probe.residual, rank_tol=config.rank_tol)
        return

    carried = np.abs(probe.direction) * blocked.mask
    worst = int(np.argmax(carried)) if carried.size else 0
    raise GeometryError(
        f"a support probe found width {probe.width:.3e} (floor {probe.width_floor:.3e}), but the "
        f"direction it came from has no component left in the admitted direction space "
        f"(residual {probe.residual_norm:.3e}). The LP can move along a direction that "
        "`blocked_reactions` or the mass-balance projection has excluded, so the resolutions "
        f"disagree. Largest blocked component of the probe: reaction "
        f"{reduced.reaction_ids[reduced.free_indices[worst]]!r} at {carried[worst]:.3e}. "
        f"Tolerances: span_tol={config.span_tol:.1e}, blocked_tol={config.blocked_tol:.1e}, "
        f"scale_floor={config.scale_floor:g}."
    )


def _feasible_center(
    support_points: NDArray[np.float64], reduced: ReducedPolytope, config: GeometryConfig
) -> tuple[NDArray[np.float64], float]:
    """The mean of the support points, made **exactly** bound-feasible (spec §16).

    A convex average of feasible points is feasible in exact arithmetic. In float64 it is not quite:
    the LP's own solutions sit a solver-noise distance outside the bounds they touch, and the mean
    inherits that. The residue is ~1e-13, and left alone it is anything but cosmetic — it is one
    half of the divide-by-noise that yields a chord excluding ``t = 0``.

    So we clamp, but under a hard rule: the move must be smaller than the LP feasibility tolerance,
    or we raise. That keeps this an act of rounding a coordinate already ambiguous at that scale,
    and not the silent repair of a broken centre that spec §16 forbids — and mass balance is
    re-checked afterwards by `_validate`, so a clamp large enough to matter cannot pass unnoticed.
    """
    raw = support_points.mean(axis=0)
    if not np.all(np.isfinite(raw)):
        raise GeometryError("the centre contains non-finite fluxes")

    violation = float(
        np.max(
            np.maximum(reduced.lower_bounds - raw, raw - reduced.upper_bounds),
            initial=0.0,
        )
    )
    if violation > config.feasibility_tol:
        raise GeometryError(
            f"the mean of the support points violates a bound by {violation:.3e}, above the "
            f"feasibility tolerance {config.feasibility_tol:.1e}. Re-solve the support LPs at a "
            "tighter tolerance — do not clip a centre that is genuinely infeasible (spec §16)"
        )

    center = np.clip(raw, reduced.lower_bounds, reduced.upper_bounds)
    return center, max(violation, 0.0)


def _singleton_geometry(reduced: ReducedPolytope, config: GeometryConfig) -> ReducedGeometry:
    """Every reaction is fixed: the polytope is one point and MCMC has nothing to do (spec §16).

    Returned, not raised. A strain whose medium blocks it completely is a *result*, and a batch has
    to be able to report it alongside the strains that grew.
    """
    empty = np.zeros(0, dtype=VALUE_DTYPE)
    return ReducedGeometry(
        scaling=empty,
        basis=np.zeros((0, 0), dtype=VALUE_DTYPE, order="F"),
        center=empty,
        support_points=np.zeros((1, 0), dtype=VALUE_DTYPE),
        certificate=SpanCertificate(
            exhaustive=True,
            n_probes=0,
            n_complement=0,
            max_width=0.0,
            max_width_floor=config.span_tol,
            n_inconclusive=0,
            worst_dual_error=0.0,
            leakage=0.0,
            diameter=0.0,
            complement_is_complete=True,
        ),
        diagnostics=GeometryDiagnostics(
            n_free=0,
            dimension=0,
            n_blocked=0,
            blocked_separation=np.inf,
            n_blocked_escalated=0,
            n_support_points=1,
            n_lp_solves=0,
            n_cold_solves=0,
            degraded=False,
            n_random_probes=0,
            n_sweeps=0,
            simplex_iterations=0,
            orthonormality_error=0.0,
            mass_balance_error=0.0,
            center_mass_balance_residual=0.0,
            center_bound_slack=0.0,
            center_clamp=0.0,
            min_chord_at_center=0.0,
            support_coordinate_rank=0,
            n_scales_floored=0,
            inverse_scale_norm=0.0,
            probe_noise_ceiling=0.0,
            basis_memory_bytes=0,
        ),
        polytope_key=_polytope_key(reduced),
    )


def _validate(
    *,
    reduced: ReducedPolytope,
    basis: NDArray[np.float64],
    scales: NDArray[np.float64],
    center: NDArray[np.float64],
    clamp: float,
    support_points: NDArray[np.float64],
    blocked: BlockedReactions,
    space: DirectionSpace,
    config: GeometryConfig,
    n_lp_solves: int,
    n_random_probes: int,
    n_sweeps: int,
    simplex_iterations: int,
    n_floored: int,
    inverse_scale_norm: float,
    noise_ceiling: float,
    n_cold_solves: int,
    degraded: bool,
) -> GeometryDiagnostics:
    """The checks spec §15.4 and §16 demand — each a hard failure, none of them a warning."""
    d = int(basis.shape[1])

    orthonormality_error = 0.0
    if d:
        gram = basis.T @ basis
        orthonormality_error = float(np.max(np.abs(gram - np.eye(d))))
        if orthonormality_error > ORTHONORMALITY_TOL:
            raise GeometryError(
                f"‖BᵀB − I‖_max = {orthonormality_error:.3e} exceeds {ORTHONORMALITY_TOL:.1e}"
            )

    # A blocked reaction contributes an identically zero component to every feasible direction. The
    # projection in `probe_direction` makes that true by construction, so this is an invariant
    # check, not a tolerance: any nonzero here means a direction escaped the projection.
    if d and np.any(basis[blocked.mask, :] != 0.0):
        offender = float(np.max(np.abs(basis[blocked.mask, :])))
        raise GeometryError(
            f"the basis is nonzero ({offender:.3e}) on a reaction FVA proved cannot move — a "
            "candidate direction reached the basis without being projected"
        )

    # ‖S·diag(s)·B‖_max, column by column through the reference CSC matvec (spec §15.4). A basis
    # direction that fails this is not a direction of the polytope at all: moving along it breaks
    # the steady state — and the chord, which only ever looks at bounds, would never notice.
    #
    # The bar is *relative*, for the same reason `_check_mass_balance` uses one. Forming
    # ``S·diag(s)·b``
    # sums terms of size ‖S·diag(s)‖, so simply evaluating it costs that much times eps: ~1e-11 on
    # the
    # example model, but ~1e-6 on a polytope with bounds of 1e10. An absolute 1e-9 would call the
    # second one broken while it is merely *large*, and no basis could ever pass.
    mass_balance_error = 0.0
    for column in range(d):
        direction = scales * basis[:, column]
        residual = float(
            np.max(np.abs(reduced.stoichiometry.matvec(direction)), initial=0.0)
        )
        magnitude = float(
            np.max(reduced.stoichiometry.cancellation_scale(direction), initial=0.0)
        )
        floor = magnitude * float(np.finfo(VALUE_DTYPE).eps) * reduced.n_free
        if residual > max(config.feasibility_tol, MASS_BALANCE_SAFETY * floor):
            raise GeometryError(
                f"basis direction {column} has a mass-balance residual of {residual:.3e}, beyond "
                f"both the feasibility tolerance ({config.feasibility_tol:.1e}) and the "
                f"{floor:.3e} that evaluating S·diag(s)·b in float64 could account for — it is not "
                "a direction of the polytope, and the chord would never notice"
            )
        mass_balance_error = max(mass_balance_error, residual)

    center_residual = float(np.max(np.abs(reduced.mass_balance_residual(center)), initial=0.0))
    if center_residual > config.feasibility_tol:
        raise GeometryError(
            f"the centre's mass-balance residual is {center_residual:.3e}, above the feasibility "
            f"tolerance {config.feasibility_tol:.1e}. Re-solve the support LPs at a tighter "
            "tolerance — do not clip the centre, which would break S·v = rhs (spec §16)"
        )

    lower_slack = float(np.min(center - reduced.lower_bounds, initial=np.inf))
    upper_slack = float(np.min(reduced.upper_bounds - center, initial=np.inf))
    center_bound_slack = min(lower_slack, upper_slack)
    if center_bound_slack < 0.0:
        raise GeometryError(
            f"the centre violates a bound by {-center_bound_slack:.3e} after clamping — "
            "np.clip should have made this impossible"
        )

    min_chord = _min_chord_at_center(reduced, basis, scales, center, config)

    # The support points must span every discovered direction, or M5's covariance is rank-deficient
    # and its ridge would hide the fact. Coordinates, not fluxes: the flux vectors also vary in
    # directions the basis does not span.
    support_rank = 0
    if d:
        coordinates = ((support_points - center) / scales) @ basis
        support_rank = int(np.linalg.matrix_rank(coordinates))
        if support_rank != d:
            raise GeometryError(
                f"the {support_points.shape[0]} support points span only {support_rank} of the {d} "
                "discovered directions, so their covariance is singular — M5's rounding ridge "
                "would conceal that rather than fail on it"
            )

    return GeometryDiagnostics(
        n_free=reduced.n_free,
        dimension=d,
        n_blocked=blocked.n_blocked,
        blocked_separation=blocked.separation,
        n_blocked_escalated=blocked.n_escalated,
        n_support_points=int(support_points.shape[0]),
        n_lp_solves=n_lp_solves,
        n_cold_solves=n_cold_solves,
        degraded=degraded,
        n_random_probes=n_random_probes,
        n_sweeps=n_sweeps,
        simplex_iterations=simplex_iterations,
        orthonormality_error=orthonormality_error,
        mass_balance_error=mass_balance_error,
        center_mass_balance_residual=center_residual,
        center_bound_slack=float(center_bound_slack),
        center_clamp=clamp,
        min_chord_at_center=min_chord,
        support_coordinate_rank=support_rank,
        n_scales_floored=n_floored,
        inverse_scale_norm=inverse_scale_norm,
        probe_noise_ceiling=noise_ceiling,
        basis_memory_bytes=_BYTES_PER_FLOAT64 * reduced.n_free * d,
    )


def _min_chord_at_center(
    reduced: ReducedPolytope,
    basis: NDArray[np.float64],
    scales: NDArray[np.float64],
    center: NDArray[np.float64],
    config: GeometryConfig,
) -> float:
    """Every basis direction must give a chord through the centre that contains ``t = 0``.

    M5's sampler starts here, and `line_geometry.feasible_chord` — which by design keeps *every*
    nonzero direction component, however small — raises outright on a point outside its bounds. So
    the geometry proves its own centre is samplable rather than shipping the failure downstream. It
    is a cheap check (``d`` chords) and it is the one that catches a corrupted basis row: on the
    example model, before blocked reactions were projected out, this produced a chord of
    ``[−0.54, −0.39]`` that excluded the origin.
    """
    from gsmm_compiler.line_geometry import feasible_chord

    if basis.shape[1] == 0:
        return 0.0

    transform = scales[:, None] * basis
    shortest = np.inf
    for column in range(int(basis.shape[1])):
        chord = feasible_chord(
            center,
            transform[:, column],
            reduced.lower_bounds,
            reduced.upper_bounds,
            feasibility_tol=config.feasibility_tol,
        )
        if not chord.contains(0.0):
            raise GeometryError(
                f"the chord through the centre along basis direction {column} is "
                f"[{chord.t_lo:.3e}, {chord.t_hi:.3e}] and does not contain t = 0 — the sampler "
                "could not start from this centre"
            )
        # A zero-length chord is legal *mid-chain* (the kernel self-loops on it), but at the start
        # it means this coordinate can never move: the chain would be reducible in that coordinate,
        # and `d` would overstate the dimension actually explored. The centre must be in the
        # relative interior, and this is what says so.
        if not chord.is_samplable:
            raise GeometryError(
                f"basis direction {column} has a zero-length chord at the centre "
                f"[{chord.t_lo:.3e}, {chord.t_hi:.3e}] — the centre is on the boundary of that "
                "coordinate and a chain started here could never move along it"
            )
        shortest = min(shortest, chord.length)

    if not np.isfinite(shortest):
        raise GeometryError("a chord through the centre is unbounded — the polytope is not compact")
    return float(shortest)


def _polytope_key(reduced: ReducedPolytope) -> str:
    """The L1 key: which polytope this geometry is the geometry *of*.

    Now `ReducedPolytope.content_key`, so the geometry, the rounded transform and M6's lowered
    objective all name the polytope the **same way** — which is the only thing that lets a consumer
    check that two artifacts were computed against each other rather than merely assume it.
    """
    return reduced.content_key()
