"""Coordinate hit-and-run for ``π_β``, in rounded reduced coordinates (spec §18).

The chain lives on ``y ∈ ℝᵈ`` and carries a synchronized flux vector ``v = centre + T·y``. One step
picks a coordinate ``k`` uniformly, forms the flux direction ``T[:, k]``, intersects it with the
bounds to get the chord, and replaces ``y_k`` by a draw from its **exact** conditional along that
chord. There is no Metropolis correction because there is nothing to correct: it is a Gibbs update.

**Why this leaves ``π_β`` invariant, and where each piece of the argument is enforced.** For a fixed
``k``, resampling ``y_k`` from ``π_β(y_k | y_{-k})`` is a Gibbs transition and preserves ``π_β``. A
mixture over ``k`` with weights *independent of the state* preserves it too, because the invariance
can then be pulled out through the mixture. Four things must hold, and none is merely assumed:

1. **The coordinate law is independent of the state.** ``k`` comes from `Generator.integers`, drawn
   before the chord is even looked at. This is why a degenerate chord must be a **self-loop** and
   never a redraw of a different coordinate (BUILD_PLAN §1.6.6): a redraw makes the mixture weights
   a function of the state and the invariance argument collapses. `sample_line` already returns the
   correct self-loop; here it is simply applied.
2. **The line conditional is exact.** M2's oracle, tested against a 60-digit `decimal` reference and
   against analytic CDFs. At ``β = 0`` it is a uniform draw on the chord, and that *is* the whole
   inner loop for M5.
3. **The transform is frozen.** ``T`` comes from `rounding.RoundedTransform`, which is immutable,
   and no pilot re-estimation runs mid-chain (spec §17.4 defers that to a separate DAG stage, M10).
   An adapted ``T`` makes the kernel depend on the chain's own history, and its draws are then not
   from ``π_β`` at all.
4. **``T``'s columns span the affine direction space.** M4 certifies the basis, and `rounding`
   preserves its span exactly, ``L`` being invertible.

**The one place float64 makes this approximate, and exactly how far the claim goes.** ``v`` is
mathematically a function of ``y``, so it is a cache and not part of the state. Maintaining it
*incrementally* — ``v += t·d`` rather than ``v = centre + T·y`` — lets the cache accumulate rounding
error, so the chord is formed at a point a hair from where ``y`` says the chain is.

The consequence has to be stated precisely, because it is easy to overclaim. **In float64 the chain
is not Markov in ``y`` alone**: its true state is ``(y, cache error, refresh phase)``. The four
points above establish exact Gibbs invariance *in exact arithmetic*; they do not establish exact
invariance of the float64 process, and nothing here should be read as claiming they do. What is done
instead is to make the perturbation small, corrigible, and **observed**:

* ``v`` is rebuilt exactly from ``y`` every ``refresh_interval`` sweeps — an identity operation in
  exact arithmetic, hence a *phase*-dependent but never a *state*-dependent modification, and one
  that strictly discards accumulated error rather than adding any;
* the discrepancy is measured at every refresh **and at every stored sample**, and reported as
  `ChainDiagnostics.max_refresh_drift` (max ~6e-12 on the example model, against fluxes of ~1e3);
* the rebuilt point is re-checked against its bounds, and the chain **raises** rather than quietly
  sampling a state outside the support;
* the stored flux is the exact ``centre + T·y`` of the stored state, so what is written out is a
  function of the state and not of the cache.

A measured per-step drift is *not* a bound on the error induced in the stationary law — that needs a
spectral-gap or contraction argument this package does not have, and does not pretend to.

**Time is measured in sweeps.** One sweep is ``d`` random-scan coordinate updates. It is still a
random scan — each update draws its own ``k`` — but counting single updates would make
``burn_in = 1000`` mean 21 passes over a 46-dimensional model and 1000 over a 1-dimensional one,
which is not a schedule anyone means to write.

**No solver, ever.** Nothing in this module's import graph loads `highspy` or `cobra`, and the
process-global solve counter is asserted unchanged across a run (BUILD_PLAN §1.3, the M5 gate).

**M6 — what the tilt adds, and what it deliberately does not.** At ``β > 0`` the *only* change to
the loop above is that `sample_line` builds M2's piecewise-exponential conditional instead of
drawing uniformly. Every word of the invariance argument survives verbatim, because it never
mentioned the conditional's shape — only that it is *exact*, which M2 established against a 60-digit
reference. So M6 supplies inputs, not machinery: the objective in reduced coordinates
(`run_ladder`), the energy scale ``s_J`` (`sparse_objective.choose_energy_scale`), the ladder, and
the traces.

Three things M6 does **not** do, each a place the spec invites a subtle error:

* **It does not maintain ``J`` incrementally**, though spec §1.3 step 4 suggests it. Nothing needs
  it: `build_piecewise_j` derives every slope from ``v`` and the direction on the spot, and the
  conditional depends on ``J`` only through peak-relative *heights*. A running ``J`` would be a
  second cache to drift, reconcile and mistrust — M5 paid that price for ``v``, which is genuinely
  needed for the chord, and there is no reason to pay it twice for a quantity that is only
  *reported*. Traces are therefore computed **exactly**, after the fact, from the stored fluxes.
* **It does not feed ``J*`` into the draw.** ``J*`` cancels out of ``p(t)``, and carrying it invites
  the catastrophic cancellation M2 measured reversing which segment the sampler favoured. It enters
  only the reported log-energy ``(J − J*)/s_J``, which is a diagnostic.
* **It does not change ``T``, ``w`` or ``s_J`` between ``β`` rungs.** All three are frozen before
  the first chain starts. A ladder that re-rounded per β would be sampling a different chain each
  time and calling the difference physics.

The **mean-J monotonicity** the gate checks is a theorem, not a hope. Writing ``κ = β/s_J``, the
target is ``π_κ ∝ e^{κJ}`` and

    d E_κ[J] / dκ = Var_κ(J) ≥ 0,

so ``E_β[J]`` is nondecreasing in ``β`` for any ``s_J > 0``, strictly unless ``J`` is a.s. constant.
`LadderResult.monotonicity` checks it in units of **Monte-Carlo standard error** computed from the
ESS of the ``J`` trace — not from the raw sample count, which on a chain with an ESS of 1% would
understate the error by tenfold and fail a perfectly good ladder.

Implemented in **M5** (β = 0) and **M6** (β > 0) — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.config import DEFAULT_NEAR_ZERO_THRESHOLDS, SamplerConfig
from gsmm_compiler.diagnostics import effective_sample_size, mcse, split_r_hat
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.line_distribution import L1Objective, sample_line
from gsmm_compiler.line_geometry import chord_on_support
from gsmm_compiler.provenance import stream_seed
from gsmm_compiler.rounding import RoundedTransform
from gsmm_compiler.sparse_objective import EnergyScale, ReducedObjective

SAMPLER_IMPL_VERSION = 1
"""Bump when the transition kernel changes — invalidates every cached sample artifact."""

VALUE_DTYPE = np.float64

SUGGESTED_BETA_LADDER: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
"""Spec §22.1's *starting* ladder, in units of ``s_J``. Not a default, and not a recommendation.

§22.1 is explicit that "no universal ladder should be hard-coded as scientifically correct", and it
is right: the β at which a given strain's flux distribution actually concentrates depends on the
shape of its own polytope. This constant exists so a caller can write the spec's suggestion without
retyping it, and so that a run which *used* it says so in its manifest. `SamplerConfig.betas`
defaults to ``(0.0,)`` — the one rung that needs no objective at all.
"""

DEFAULT_START_CONCENTRATION = 0.5
"""Dirichlet concentration for a chain's start over the support points (BUILD_PLAN §1.2).

Below 1, so the weights concentrate on a few vertices and the starts are genuinely **dispersed**.
With ``α = 1`` (the flat Dirichlet) a convex combination of ``K = 93`` support points is a near-mean
point with variance ``~1/K`` of the vertex spread: every chain would start in a huddle around the
centre, and the between-chain variance R̂ divides by would be measuring nothing at all. Overdispersed
starts are what make R̂ *conservative*, which is the only direction in which it is useful.
"""

MAX_START_SHRINKS = 60
"""Halvings of a start toward the centre before it is called impossible rather than unlucky."""


class SamplerError(RuntimeError):
    """The chain could not run, or left the polytope."""


@dataclass(frozen=True)
class ChainDiagnostics:
    """What one chain did. Written to the run manifest (spec §15.5, §22)."""

    chain_index: int

    beta: float

    dimension: int

    n_sweeps: int
    """Sweeps run, burn-in included. Coordinate updates are ``n_sweeps · d``."""

    n_samples: int

    n_degenerate_steps: int
    """Steps whose chord had no positive width, so the exact conditional was a point mass.

    A **self-loop** — not a rejection, and not a redraw (BUILD_PLAN §1.6.6). Expect ~0 in the
    interior of a well-rounded polytope; a large count means the chain is pressed against a face.
    """

    n_refreshes: int

    max_refresh_drift: float
    """``‖v_incremental − (centre + T·y)‖_∞``, worst over **every refresh and every stored sample**.

    The cost of the incremental update, measured rather than assumed (module docstring). Measured at
    the samples too, not only at the refreshes: drift can peak and partly cancel between two
    refreshes, and a `refresh_interval` longer than the run would otherwise report 0.0 having
    measured nothing. It is still an observed maximum over the visited states, **not** a proof — and
    emphatically not a bound on the error it induces in the stationary law, which would need a
    spectral-gap argument this package does not have.
    """

    max_bound_violation: float
    """Worst ``max(l − v, v − u, 0)`` over the stored samples. Zero is expected: the chord's inward
    ``nextafter`` is what buys it."""

    max_mass_balance_residual: float
    """Worst ``‖S·v − rhs‖_∞`` over the stored samples, **relative** to ``|S|·|v|``. (The M4 lesson:
    an absolute bar on a residual summing terms of size 1e5 charges float64 rounding to the
    sampler.)"""

    mean_chord_length: float

    start_shrink: float
    """``ρ`` — how far the dispersed start had to be pulled back toward the centre to be feasible.
    ``1.0`` means not at all."""

    spawn_key: tuple[int, ...]
    """The RNG stream's semantic coordinates. Reproducing this chain needs nothing else."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_index": self.chain_index,
            "beta": self.beta,
            "dimension": self.dimension,
            "n_sweeps": self.n_sweeps,
            "n_samples": self.n_samples,
            "n_degenerate_steps": self.n_degenerate_steps,
            "n_refreshes": self.n_refreshes,
            "max_refresh_drift": self.max_refresh_drift,
            "max_bound_violation": self.max_bound_violation,
            "max_mass_balance_residual": self.max_mass_balance_residual,
            "mean_chord_length": self.mean_chord_length,
            "start_shrink": self.start_shrink,
            "spawn_key": list(self.spawn_key),
        }


@dataclass(frozen=True)
class ChainResult:
    """One chain's draws. Fluxes are **reduced**; `ReducedPolytope.to_full` lifts them to save."""

    coordinates: NDArray[np.float64]
    """``(n_samples, d)`` — the rounded state ``y``. The chain's actual state space."""

    fluxes: NDArray[np.float64]
    """``(n_samples, n_free)`` — the synchronized flux. A function of `coordinates`, stored because
    every downstream question (feasibility, activity, exchange) is asked of the flux."""

    diagnostics: ChainDiagnostics


@dataclass(frozen=True)
class SamplerResult:
    """Every chain at one ``β``."""

    beta: float
    chains: tuple[ChainResult, ...]
    model_id: str

    @property
    def coordinates(self) -> NDArray[np.float64]:
        """``(n_chains, n_samples, d)`` — the shape `diagnostics.convergence_report` wants."""
        return np.stack([chain.coordinates for chain in self.chains])

    @property
    def fluxes(self) -> NDArray[np.float64]:
        """``(n_chains, n_samples, n_free)``."""
        return np.stack([chain.fluxes for chain in self.chains])

    def manifest(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "model_id": self.model_id,
            "n_chains": len(self.chains),
            "sampler_impl_version": SAMPLER_IMPL_VERSION,
            "chains": [chain.diagnostics.as_dict() for chain in self.chains],
        }


def dispersed_start(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    rng: np.random.Generator,
    *,
    concentration: float = DEFAULT_START_CONCENTRATION,
    optimum_coordinates: NDArray[np.float64] | None = None,
    feasibility_tol: float = 1e-9,
) -> tuple[NDArray[np.float64], float]:
    """A feasible, dispersed start: a random convex combination of the support points (§1.2).

    Built in ``y`` rather than in flux, so the combination is of points *already on the affine hull*
    and cannot drift off it. Convexity does the rest: a convex combination of feasible points is
    feasible, so the only thing float64 can cost is a bound touched from the outside by a rounding
    error — and shrinking toward ``y = 0``, the centre, which `rounding.build_transform` has already
    proved bound-feasible with a positive chord on every axis, repairs that in a bounded number of
    halvings.

    Dispersion is the whole reason for this rather than starting every chain at the centre. R̂
    divides between-chain variance by within-chain variance, and chains that all start at the same
    interior point make the numerator small for reasons that have nothing to do with convergence.

    **``optimum_coordinates`` adds ``v*`` to the hull (M6, BUILD_PLAN §1.2).** At large ``β`` the
    target's mass sits near the LP optimum, and a chain started at a random vertex has to *walk*
    there — with a step scale set by the polytope's width, not by the width of the region that
    carries the mass. §1.2's warning is exactly this: "embarrassingly parallel" describes the
    compute, not the mixing, and cold high-β chains can stay trapped near their init. Mixing ``v*``
    into the hull means some chains begin where the mass is and some begin far from it, which is
    what makes R̂ able to *see* a chain that never arrived.

    Passing ``None`` reproduces M5's draws bit for bit — the Dirichlet is over ``K`` weights rather
    than ``K + 1``, so it consumes the same numbers from the same stream.

    One honest caveat: ``v*`` reaches this function through `RoundedTransform.to_coordinates`, which
    *projects* onto the affine hull. If the LP returned a point a hair off the hull, the
    projection — not ``v*`` — is what joins the hull, and it may sit a rounding error outside a
    bound. That is precisely the case the shrink loop below already exists to absorb, and the start
    it finally returns is verified feasible regardless of where it came from.
    """
    if transform.is_singleton:
        return np.zeros(0, dtype=VALUE_DTYPE), 1.0

    hull = transform.support_coordinates
    if optimum_coordinates is not None:
        optimum = np.asarray(optimum_coordinates, dtype=VALUE_DTYPE)
        if optimum.shape != (transform.dimension,):
            raise SamplerError(
                f"optimum_coordinates has shape {optimum.shape}, expected "
                f"({transform.dimension},) — the reduced coordinates of v*"
            )
        hull = np.vstack([hull, optimum[np.newaxis, :]])

    weights = rng.dirichlet(np.full(int(hull.shape[0]), concentration))
    proposal = np.asarray(weights @ hull, dtype=VALUE_DTYPE)

    shrink = 1.0
    for _ in range(MAX_START_SHRINKS):
        start = shrink * proposal
        if _start_is_feasible(transform, reduced, start, feasibility_tol=feasibility_tol):
            return np.ascontiguousarray(start, dtype=VALUE_DTYPE), shrink
        shrink *= 0.5

    raise SamplerError(
        f"no feasible start after {MAX_START_SHRINKS} halvings toward the centre, which "
        "`build_transform` certified feasible — the transform and the polytope disagree"
    )


def _start_is_feasible(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    start: NDArray[np.float64],
    *,
    feasibility_tol: float,
) -> bool:
    """Bound-feasible **and** able to move along every axis — or the chain is stuck at step 1."""
    v = transform.to_flux(start)

    violation = max(
        float(np.max(reduced.lower_bounds - v, initial=0.0)),
        float(np.max(v - reduced.upper_bounds, initial=0.0)),
    )
    if violation > feasibility_tol:
        return False

    precompute = transform.precompute
    for k in range(transform.dimension):
        support = precompute.support[k]
        try:
            chord = chord_on_support(
                v[support],
                precompute.direction[k],
                precompute.lower[k],
                precompute.upper[k],
                feasibility_tol=feasibility_tol,
            )
        except ValueError:
            return False
        if not chord.is_samplable:
            return False
    return True


def run_chain(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    chain_index: int,
    beta: float = 0.0,
    beta_index: int = 0,
    objective: L1Objective | None = None,
    energy_scale: float = 1.0,
    feasibility_tol: float = 1e-9,
    start: NDArray[np.float64] | None = None,
    optimum_coordinates: NDArray[np.float64] | None = None,
) -> ChainResult:
    """Run one chain of coordinate hit-and-run and return its thinned draws (spec §18.2).

    ``objective`` and ``energy_scale`` are needed only at ``β > 0`` (M6). At ``β = 0`` the target is
    flat, ``J`` provably never enters the draw, and the whole step is a uniform draw on the chord.

    ``objective`` is the `L1Objective` in **reduced** coordinates, which
    `sparse_objective.lower_objective` builds. A full-model objective would index ``v`` (length
    ``n_free``) with full-model reaction indices: on the example model that reads reaction 772's
    weight at position 772 of a 260-long vector and raises, but on a model where the index happens
    to be in range it would silently tilt by the wrong reactions.

    The RNG is keyed on ``(model_id, "sample", β_index, chain_index)`` — semantic coordinates, not a
    position in a `spawn()` sequence (`provenance.stream_seed`). Add a chain, and the others still
    draw the same numbers; reorder a batch, and every stream keeps its name.
    """
    if not np.isfinite(beta) or beta < 0.0:
        raise SamplerError(f"beta must be finite and >= 0, got {beta}")
    if beta > 0.0:
        if objective is None:
            raise SamplerError("an objective is required to sample at β > 0")
        if not np.isfinite(energy_scale) or energy_scale <= 0.0:
            raise SamplerError(f"energy_scale must be finite and > 0 at β > 0, got {energy_scale}")
    if transform.polytope_key != reduced.content_key():
        raise SamplerError(
            "the transform was not built from this polytope: the chain would step along one "
            "model's directions and be bounds-checked against another's. (`run_ladder` guards this "
            "too; `run_chain` is the low-level entry point and must guard it itself.)"
        )
    if objective is not None and not transform.is_singleton:
        _check_objective_is_reduced(objective, transform.n_free)

    seed_sequence = stream_seed(
        model_id=model_id,
        stage="sample",
        seed=config.seed,
        beta_index=beta_index,
        chain_index=chain_index,
    )
    rng = np.random.default_rng(seed_sequence)
    spawn_key = tuple(int(key) for key in seed_sequence.spawn_key)

    if transform.is_singleton:
        return _singleton_chain(transform, config, beta, chain_index, spawn_key)

    if start is None:
        start_state, shrink = dispersed_start(
            transform,
            reduced,
            rng,
            optimum_coordinates=optimum_coordinates,
            feasibility_tol=feasibility_tol,
        )
    else:
        start_state = np.ascontiguousarray(start, dtype=VALUE_DTYPE)
        shrink = 1.0
        if not _start_is_feasible(transform, reduced, start_state, feasibility_tol=feasibility_tol):
            raise SamplerError("the supplied start is not feasible, or is pinned on some axis")

    return _walk(
        transform=transform,
        reduced=reduced,
        config=config,
        rng=rng,
        start=start_state,
        start_shrink=shrink,
        beta=beta,
        objective=objective,
        energy_scale=energy_scale,
        feasibility_tol=feasibility_tol,
        chain_index=chain_index,
        spawn_key=spawn_key,
    )


def _check_objective_is_reduced(objective: L1Objective, n_free: int) -> None:
    """The objective must be indexed in **reduced** coordinates, or it tilts by the wrong reactions.

    The sampler's ``v`` is length ``n_free``; a full-model `sparse_objective.SparseFluxObjective`
    lowered by hand, or an `L1Objective` built against full-model indices, would read
    ``v[biomass_index]`` at a position that means a different reaction — or none at all.

    An out-of-range index raises inside NumPy anyway, so this check buys nothing on the example
    model (773 reactions, 260 free). It buys everything on a model whose biomass happens to sit at a
    low index: the read succeeds, the chain tilts by *some other reaction's* flux, and every
    downstream check — feasibility, mass balance, chords, R̂ — passes, because nothing else in this
    package knows which reaction ``J`` is supposed to reward.
    """
    if objective.biomass_index is not None and not 0 <= objective.biomass_index < n_free:
        raise SamplerError(
            f"objective.biomass_index is {objective.biomass_index}, outside the reduced polytope's "
            f"[0, {n_free}). The sampler takes the objective in REDUCED coordinates — see "
            "`sparse_objective.lower_objective`"
        )
    if objective.penalized_indices.size:
        highest = int(objective.penalized_indices.max())
        if int(objective.penalized_indices.min()) < 0 or highest >= n_free:
            raise SamplerError(
                f"objective.penalized_indices reach {highest}, outside the reduced polytope's "
                f"[0, {n_free}). The sampler takes the objective in REDUCED coordinates — see "
                "`sparse_objective.lower_objective`"
            )


def _walk(
    *,
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    config: SamplerConfig,
    rng: np.random.Generator,
    start: NDArray[np.float64],
    start_shrink: float,
    beta: float,
    objective: L1Objective | None,
    energy_scale: float,
    feasibility_tol: float,
    chain_index: int,
    spawn_key: tuple[int, ...],
) -> ChainResult:
    """The inner loop. Pure NumPy on frozen arrays; no Python loop over reactions, no solver."""
    d = transform.dimension
    precompute = transform.precompute
    support = precompute.support
    direction = precompute.direction
    lower = precompute.lower
    upper = precompute.upper
    # Contiguous views into the Fortran-contiguous T. Needed full-length only on the β > 0 path,
    # where `sample_line` reads the objective's own reactions out of the direction.
    columns = tuple(transform.transform[:, k] for k in range(d))

    y = np.array(start, dtype=VALUE_DTYPE, copy=True)
    v = np.ascontiguousarray(transform.to_flux(y), dtype=VALUE_DTYPE)

    total_sweeps = config.burn_in + config.n_samples * config.thin
    coordinates = np.empty((config.n_samples, d), dtype=VALUE_DTYPE)
    fluxes = np.empty((config.n_samples, transform.n_free), dtype=VALUE_DTYPE)

    n_degenerate = 0
    n_refreshes = 0
    max_drift = 0.0
    chord_total = 0.0
    n_sampled_steps = 0
    stored = 0

    for sweep in range(total_sweeps):
        # Drawn for the whole sweep at once, and *before* any chord is looked at: the coordinate
        # law must not depend on the state (module docstring, point 1).
        for raw in rng.integers(0, d, size=d):
            k = int(raw)
            chord = chord_on_support(
                v[support[k]],
                direction[k],
                lower[k],
                upper[k],
                feasibility_tol=feasibility_tol,
            )
            if chord.is_samplable:
                chord_total += chord.length
                n_sampled_steps += 1
            else:
                n_degenerate += 1

            t = sample_line(v, columns[k], chord, objective, beta, energy_scale, rng)

            y[k] += t
            # Only the support moves: T is exactly zero off it, so `v[off] += t·0` is a no-op that
            # would cost a full-length pass. Identical result, O(nnz) instead of O(n_free).
            v[support[k]] += t * direction[k]

        if (sweep + 1) % config.refresh_interval == 0:
            v, drift = _refresh(transform, reduced, y, v, feasibility_tol=feasibility_tol)
            max_drift = max(max_drift, drift)
            n_refreshes += 1

        recording = sweep >= config.burn_in and (sweep - config.burn_in) % config.thin == 0
        if recording and stored < config.n_samples:
            # Store the **exact** flux of the stored state, ``centre + T·y``, not the incremental
            # cache. Two reasons, and the second is the one that matters:
            #
            #  * the pair ``(y, v)`` we hand out is then consistent *by construction* — a reader who
            #    recomputes ``to_flux(coordinates)`` gets back exactly ``fluxes``, so there is no
            #    third quantity to reconcile;
            #  * it makes the drift **measurable at every sample** rather than only at the refresh
            #    instants. `max_refresh_drift` alone is no bound on the cache error: drift can peak
            #    and partly cancel between two refreshes, and with `refresh_interval` larger than
            #    the run it would report a serene 0.0 having measured nothing at all.
            exact = np.ascontiguousarray(transform.to_flux(y), dtype=VALUE_DTYPE)
            max_drift = max(max_drift, float(np.max(np.abs(v - exact), initial=0.0)))

            coordinates[stored] = y
            fluxes[stored] = exact
            stored += 1

    if stored != config.n_samples:
        raise SamplerError(f"stored {stored} samples, expected {config.n_samples}")

    violation, residual = _sample_feasibility(fluxes, reduced)

    return ChainResult(
        coordinates=coordinates,
        fluxes=fluxes,
        diagnostics=ChainDiagnostics(
            chain_index=chain_index,
            beta=beta,
            dimension=d,
            n_sweeps=total_sweeps,
            n_samples=config.n_samples,
            n_degenerate_steps=n_degenerate,
            n_refreshes=n_refreshes,
            max_refresh_drift=max_drift,
            max_bound_violation=violation,
            max_mass_balance_residual=residual,
            mean_chord_length=chord_total / max(n_sampled_steps, 1),
            start_shrink=start_shrink,
            spawn_key=spawn_key,
        ),
    )


def _refresh(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    y: NDArray[np.float64],
    v: NDArray[np.float64],
    *,
    feasibility_tol: float,
) -> tuple[NDArray[np.float64], float]:
    """Rebuild ``v`` exactly from ``y``, measure the drift, re-check the bounds (§1.3).

    ``v`` is a cache of ``centre + T·y``, so this is a no-op in exact arithmetic — which is exactly
    why it is safe to run on a fixed schedule and why it cannot disturb stationarity. What it does
    in float64 is discard the rounding the incremental updates accumulated, and *report* how much
    there was, so that a claim about the chain's feasibility rests on a measurement rather than on
    an argument about error growth.

    The rebuilt point is what the chain continues from, not the incremental one: ``y`` is the state,
    and the target is defined at the flux that ``y`` actually maps to.
    """
    exact = np.ascontiguousarray(transform.to_flux(y), dtype=VALUE_DTYPE)
    drift = float(np.max(np.abs(v - exact), initial=0.0))

    violation = max(
        float(np.max(reduced.lower_bounds - exact, initial=0.0)),
        float(np.max(exact - reduced.upper_bounds, initial=0.0)),
    )
    if violation > feasibility_tol:
        raise SamplerError(
            f"rebuilding v from y put the chain {violation:.3e} outside its bounds, above the "
            f"feasibility tolerance {feasibility_tol:.1e}: the incremental state and the transform "
            "have diverged, and the chain is no longer sampling the polytope"
        )

    return exact, drift


def _sample_feasibility(
    fluxes: NDArray[np.float64], reduced: ReducedPolytope
) -> tuple[float, float]:
    """Worst bound violation and worst **relative** mass-balance residual over the stored samples.

    Relative, per the M4 lesson, and floored, per what that lesson does to its own instrument — see
    `NativeCSC.relative_residual`, which owns both halves of the argument.
    """
    violation = max(
        float(np.max(reduced.lower_bounds - fluxes, initial=0.0)),
        float(np.max(fluxes - reduced.upper_bounds, initial=0.0)),
    )

    worst = 0.0
    for sample in fluxes:
        contiguous = np.ascontiguousarray(sample)
        relative = reduced.stoichiometry.relative_residual(contiguous, reduced.rhs)
        worst = max(worst, float(relative.max(initial=0.0)))

    return violation, worst


def _singleton_chain(
    transform: RoundedTransform,
    config: SamplerConfig,
    beta: float,
    chain_index: int,
    spawn_key: tuple[int, ...],
) -> ChainResult:
    """``d = 0``: the polytope is a point, so every sample *is* that point (spec §16)."""
    coordinates = np.zeros((config.n_samples, 0), dtype=VALUE_DTYPE)
    fluxes = np.repeat(transform.center[np.newaxis, :], config.n_samples, axis=0)

    return ChainResult(
        coordinates=coordinates,
        fluxes=np.ascontiguousarray(fluxes, dtype=VALUE_DTYPE),
        diagnostics=ChainDiagnostics(
            chain_index=chain_index,
            beta=beta,
            dimension=0,
            n_sweeps=0,
            n_samples=config.n_samples,
            n_degenerate_steps=0,
            n_refreshes=0,
            max_refresh_drift=0.0,
            max_bound_violation=0.0,
            max_mass_balance_residual=0.0,
            mean_chord_length=0.0,
            start_shrink=1.0,
            spawn_key=spawn_key,
        ),
    )


def run_chains(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    beta: float = 0.0,
    beta_index: int = 0,
    objective: L1Objective | None = None,
    energy_scale: float = 1.0,
    feasibility_tol: float = 1e-9,
    optimum_coordinates: NDArray[np.float64] | None = None,
) -> SamplerResult:
    """Every chain at one ``β``, each from its own dispersed start and its own named RNG stream.

    Serial here. The chains are independent given the frozen transform, so M8 hands exactly these
    units to a process pool without changing a line of the mathematics — and because each stream is
    named by ``(model_id, stage, β, chain)`` rather than by its position in a spawn sequence, the
    parallel run draws the *same numbers* as this serial one.

    "Embarrassingly parallel" describes the compute, not the mixing (BUILD_PLAN §1.2). Independent
    chains are what make R̂ meaningful; they are not a substitute for checking it.
    """
    chains = tuple(
        run_chain(
            transform,
            reduced,
            config=config,
            model_id=model_id,
            chain_index=chain_index,
            beta=beta,
            beta_index=beta_index,
            objective=objective,
            energy_scale=energy_scale,
            feasibility_tol=feasibility_tol,
            optimum_coordinates=optimum_coordinates,
        )
        for chain_index in range(config.n_chains)
    )

    return SamplerResult(beta=beta, chains=chains, model_id=model_id)


# ---- objective traces (spec §24.2) --------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveTrace:
    """``μ``, ``C``, ``J`` and the log-energy at every stored sample of one chain (spec §24.2).

    Computed **exactly, after the chain**, from the stored fluxes — which are themselves the exact
    ``centre + T·y`` of the stored state (M5). Spec §1.3 suggests maintaining these incrementally
    alongside ``v``; we deliberately do not. The kernel never needs a running ``J`` (it rebuilds
    every slope from ``v`` on the spot), so an incremental trace would be a second cache with its
    own drift, its own reconciliation and its own way of being quietly wrong — bought with nothing,
    since the exact recomputation is one gemv over the stored samples.
    """

    mu: NDArray[np.float64]
    """``μ(v) = v_b`` per stored sample. Includes a fixed biomass reaction's constant, if any."""

    cost: NDArray[np.float64]
    """``C(v) = Σ_r w_r |v_r|`` per sample, **before** λ — including the fixed reactions' share."""

    j: NDArray[np.float64]
    """``J(v) = μ − λC``, the *full* objective: directly comparable with the LP's ``J*``."""

    normalized_log_energy: NDArray[np.float64]
    """``(J(v) − J*)/s_J`` (spec §24.2). The log-density is ``β`` times this.

    Reported without ``β`` on purpose: it is the quantity the ladder *shares*, so two rungs' traces
    can be laid side by side. Usually ≤ 0, but **not guaranteed** to be — ``J*`` is an LP optimum
    computed to a solver tolerance, not a strict numeric upper bound on ``J`` (BUILD_PLAN §1.6
    delta 4), and a sample can legitimately land a few 1e-9 above it.
    """

    near_zero_thresholds: tuple[float, ...]

    near_zero_counts: NDArray[np.int64]
    """``(n_samples, n_thresholds)`` — how many **movable** reactions carry ``|v_r| < threshold``.

    *Movable*, not merely free, and the distinction is not a nicety — it is the whole content of the
    number. On the example model 61 of the 260 free reactions are **FVA-blocked**: mass balance pins
    them at zero however wide the file's bounds are (BUILD_PLAN §1.4.1), so their row of ``T`` is
    exactly zero and their flux is the centre's residual noise, ~1e-13, forever. Counted among the
    "near-zero" reactions they contribute **61 at every threshold, every β, every sample** — and
    measured over the free set the count is *exactly* 61.0 at β = 0 and 61.0 at β = 16, because the
    199 reactions that can actually move contribute nothing to it at all.

    That is not an uninformative number; it is a **misleading** one. It looks like a sparsity
    signal,
    it is constant in β, and it would go straight into a cross-model activity table as though
    selection had switched 61 reactions off. What it counts is the geometry, not the objective. So
    the structural zeros are excluded, `n_movable` says how many reactions were eligible, and a
    count
    that does not move with β now means the objective did not move it.

    This is *analysis*, and the only place a threshold may come near a flux. The chain never rounds,
    snaps or truncates — that would move the stationary distribution and can break mass balance
    (spec §3.7).
    """

    near_zero_counts_all_free: NDArray[np.int64]
    """The same counts over **every** free reaction — the movable ones *and* the immovable ones.

    Reported alongside rather than instead, because structural blockage is a real biological fact
    and not an artefact to filter away: 61 of this model's reactions genuinely cannot carry flux
    under this medium, and a run should be able to say so. What it must not do is let that constant
    masquerade as a *selection* signal.

    **The gap between the two counts is a per-threshold constant, and it is _not_ ``n_blocked``.**
    An immovable reaction is pinned at *its own* value, and that value need not be zero: mass
    balance can force a *free* reaction to a nonzero constant (`{v₀ = 1, 0 ≤ v₀ ≤ 2}` is the minimal
    case), leaving it immovable and nowhere near zero. It happens that all 61 of *this* model's
    immovable reactions sit at ~1e-13, so here the gap **is** 61 at every threshold — but that is a
    measured property of this model, not an identity, and an earlier version of this docstring
    asserted it as one. (Codex, M6 review round 2: *immovable does not imply near-zero*.)
    """

    n_free: int

    n_movable: int
    """Reactions with a structurally nonzero row in ``T``: the denominator `near_zero_counts` is out
    of. 199 of 260 on the example model."""

    n_blocked: int
    """``n_free − n_movable`` — the reactions the chain **cannot move** (BUILD_PLAN §1.4.1).

    *Immovable*, not *at zero*: their flux is whatever the centre pins them at. On this model that
    is solver noise (~1e-13), which is why they dominate every near-zero count; in general it need
    not be. See `near_zero_counts_all_free`.
    """

    @property
    def mean_j(self) -> float:
        return float(self.j.mean())

    def as_dict(self) -> dict[str, Any]:
        """Summaries for the manifest. The raw per-sample arrays go to the sample artifact (M8)."""
        return {
            "n_samples": int(self.j.size),
            "n_free": self.n_free,
            "n_movable": self.n_movable,
            "n_blocked": self.n_blocked,
            "mean_mu": float(self.mu.mean()),
            "mean_cost": float(self.cost.mean()),
            "mean_j": float(self.j.mean()),
            "std_j": float(self.j.std(ddof=1)) if self.j.size > 1 else 0.0,
            "min_j": float(self.j.min()),
            "max_j": float(self.j.max()),
            "mean_normalized_log_energy": float(self.normalized_log_energy.mean()),
            "max_normalized_log_energy": float(self.normalized_log_energy.max()),
            "near_zero_thresholds": list(self.near_zero_thresholds),
            "mean_near_zero_counts": self.near_zero_counts.mean(axis=0).tolist(),
            "mean_near_zero_counts_all_free": self.near_zero_counts_all_free.mean(axis=0).tolist(),
        }


def movable_reactions(transform: RoundedTransform) -> NDArray[np.intp]:
    """The reactions the chain can actually move: those with a structurally nonzero row in ``T``.

    An exact test, and it is allowed to be: M4 projects the FVA-blocked reactions out of the basis
    and `rounding._check_structural_zeros` holds the multiply to producing an **exactly** zero row
    for each of them. So this is the same set `CoordinatePrecompute` derives its supports from, by
    the same `flatnonzero`, and no tolerance decides membership.
    """
    return np.flatnonzero(np.any(transform.transform != 0.0, axis=1)).astype(np.intp)


def trace_objective(
    fluxes: NDArray[np.float64],
    objective: ReducedObjective,
    *,
    j_star: float,
    energy_scale: float,
    thresholds: tuple[float, ...] = DEFAULT_NEAR_ZERO_THRESHOLDS,
    movable: NDArray[np.intp] | None = None,
) -> ObjectiveTrace:
    """The §24.2 trace of one chain's stored (reduced) fluxes.

    ``j_star`` and ``energy_scale`` come from the same `sparse_objective.EnergyScale` the chain was
    tilted with, so the reported log-energy is the *actual* exponent of the law that was sampled —
    divided by ``β`` — rather than a plausible-looking recomputation from different inputs.

    ``movable`` is `movable_reactions` of the transform the chain walked in; `run_ladder` supplies
    it. Passing ``None`` counts every free reaction, which is right only when none of them is
    structurally pinned — true of the synthetic polytopes, false of every real model (see
    `ObjectiveTrace.near_zero_counts`).
    """
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise SamplerError(f"energy_scale must be finite and > 0, got {energy_scale}")
    if not thresholds:
        raise SamplerError("at least one near-zero threshold must be declared (spec §3.7)")

    v = np.atleast_2d(np.asarray(fluxes, dtype=VALUE_DTYPE))
    mu, cost, j = objective.evaluate_many(v)

    eligible = np.arange(objective.n_free, dtype=np.intp) if movable is None else movable
    if eligible.size and (int(eligible.min()) < 0 or int(eligible.max()) >= objective.n_free):
        raise SamplerError(
            f"`movable` indexes reactions outside [0, {objective.n_free}); it must come from "
            "`movable_reactions(transform)` of the transform the chain actually walked"
        )

    def count(magnitude: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.stack(
            [np.count_nonzero(magnitude < threshold, axis=1) for threshold in thresholds], axis=1
        ).astype(np.int64)

    return ObjectiveTrace(
        mu=mu,
        cost=cost,
        j=j,
        normalized_log_energy=(j - j_star) / energy_scale,
        near_zero_thresholds=tuple(thresholds),
        near_zero_counts=count(np.abs(v[:, eligible])),
        near_zero_counts_all_free=count(np.abs(v)),
        n_free=objective.n_free,
        n_movable=int(eligible.size),
        n_blocked=objective.n_free - int(eligible.size),
    )


# ---- the β-ladder -------------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaRung:
    """One rung: every chain at one ``β``, with its objective traces."""

    beta: float
    beta_index: int
    result: SamplerResult
    traces: tuple[ObjectiveTrace, ...]

    @property
    def j(self) -> NDArray[np.float64]:
        """``(n_chains, n_samples)`` — the ``J`` trace, shaped for `diagnostics`."""
        return np.stack([trace.j for trace in self.traces])

    @property
    def mean_j(self) -> float:
        return float(self.j.mean())

    @property
    def ess_j(self) -> float:
        """ESS **of ``J`` itself**, not of a coordinate.

        ``J`` is the statistic the monotonicity claim is about, and its autocorrelation is its own:
        a chain can mix well in most coordinates while crawling along the one direction ``J``
        actually varies in. Taking the error bar from the coordinates' ESS would then be an error
        bar on the wrong quantity.
        """
        return float(effective_sample_size(self.j)[0])

    @property
    def r_hat_j(self) -> float:
        """Split-R̂ **of ``J``**. The one number that can refute the monotonicity check's premise.

        The check compares two rungs' means with error bars from their ESS, and an ESS-based error
        bar assumes the draws are from the stationary law. They may not be: at ``β = 16`` the chains
        start dispersed over the support hull (``v*`` included), and an under-burned high-β chain
        that has simply *retained* its high-``J`` start produces a rising curve for a reason that
        has
        nothing to do with the tilt.

        R̂ is what distinguishes those. If the chains — started far apart — agree with each other
        about ``E[J]``, they are not each sitting in their own initial neighbourhood. Read this
        before believing `MonotonicityReport.is_monotone`; the integration test asserts it.
        """
        return float(split_r_hat(self.j)[0])

    @property
    def standard_error_j(self) -> float:
        """MC standard error of `mean_j`: ``√(var⁺/ESS)``, both from `diagnostics`.

        Two things it is deliberately not:

        * **not ``sd/√N``.** At an ESS of ~1% of the draws, ``√N`` understates the error tenfold and
          a monotonicity check built on it would reject ladders that are perfectly consistent (the
          M5 lesson, applied to a new statistic).
        * **not ``sd_pooled/√ESS``.** The ESS is estimated against ``var⁺``, the overdispersed
          variance that counts between-chain disagreement; pairing it with the *pooled sample*
          variance throws that conservatism away. Two chains trapped at ``±a`` have ``var⁺ = 2a²``
          and a pooled variance of ``a²``, so the naive form under-reports the error by ``√2``
          **exactly when the chains disagree** — which is when the error bar matters. Codex caught
          this in the M6 review; `diagnostics.mcse` owns the argument.
        """
        return float(mcse(self.j)[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "beta_index": self.beta_index,
            "mean_j": self.mean_j,
            "ess_j": self.ess_j,
            "r_hat_j": self.r_hat_j,
            "standard_error_j": self.standard_error_j,
            "chains": [trace.as_dict() for trace in self.traces],
            **self.result.manifest(),
        }


@dataclass(frozen=True)
class MonotonicityReport:
    """Is ``E_β[J]`` nondecreasing along the ladder, within Monte-Carlo error? (spec §24.2)

    It is a theorem that it must be: with ``κ = β/s_J`` and ``π_κ ∝ e^{κJ}``,

        d E_κ[J] / dκ = Var_κ(J) ≥ 0.

    So a *violation* is never physics — it is either Monte-Carlo noise or a bug, and the whole value
    of this report is telling those two apart. It measures each drop in units of the pooled standard
    error of the two rungs it spans, and calls the ladder monotone when no drop exceeds `n_sigma`.

    **A ladder that passes proves less than it appears to, in two distinct ways, and both are
    reported rather than buried.**

    * *The error bars may be wide.* Two rungs with an ESS of 30 admit a real defect between them.
      `ess_j` is there to be read first.
    * *The error bars assume stationarity, and the chains may not have reached it.* An ESS says
      nothing about shared burn-in bias. A high-β chain that merely **retained** a high-``J`` start
      produces a rising curve for a reason that has nothing to do with the tilt. `r_hat_j` is the
      check that bites there: chains started far apart which nonetheless agree about ``E[J]`` are
      not each sitting in their own initial neighbourhood.

    Neither is a hypothetical — this package's own example model has an ESS of ~60 out of 4800 and
    is
    explicitly run below convergence. Codex raised both in the M6 review; the answer is to *report*
    them, because the alternative is a green check mark that means less than it looks like.
    """

    betas: tuple[float, ...]
    mean_j: tuple[float, ...]
    standard_error_j: tuple[float, ...]
    ess_j: tuple[float, ...]
    r_hat_j: tuple[float, ...]
    n_sigma: float

    worst_drop_sigma: float
    """The largest ``(E_i[J] − E_{i+1}[J]) / √(se_i² + se_{i+1}²)`` over consecutive rungs.

    Negative when every rung rises — the ladder is then monotone with room to spare.
    """

    is_monotone: bool

    @property
    def max_r_hat_j(self) -> float:
        return max(self.r_hat_j) if self.r_hat_j else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "betas": list(self.betas),
            "mean_j": list(self.mean_j),
            "standard_error_j": list(self.standard_error_j),
            "ess_j": list(self.ess_j),
            "r_hat_j": list(self.r_hat_j),
            "max_r_hat_j": self.max_r_hat_j,
            "n_sigma": self.n_sigma,
            "worst_drop_sigma": self.worst_drop_sigma,
            "mean_j_is_monotone": self.is_monotone,
        }


@dataclass(frozen=True)
class LadderResult:
    """The whole β-ladder for one model, with the frozen scale every rung was tilted by."""

    model_id: str
    rungs: tuple[BetaRung, ...]
    energy_scale: EnergyScale
    objective: ReducedObjective

    @property
    def betas(self) -> tuple[float, ...]:
        return tuple(rung.beta for rung in self.rungs)

    def rung_at(self, beta: float) -> BetaRung:
        for rung in self.rungs:
            if rung.beta == beta:
                return rung
        raise SamplerError(f"no rung at β = {beta}; the ladder is {list(self.betas)}")

    def monotonicity(self, *, n_sigma: float = 4.0) -> MonotonicityReport:
        """Check ``E_β[J]`` is nondecreasing, comparing rungs in **ascending β** order.

        Sorted here rather than assumed: a config is free to list its ladder in any order, and
        comparing rungs in *config* order would report a spurious "drop" the moment someone wrote
        ``betas = [1.0, 0.0]``.
        """
        order = sorted(range(len(self.rungs)), key=lambda i: self.rungs[i].beta)
        rungs = [self.rungs[i] for i in order]

        # A non-finite mean or error is not weak evidence of monotonicity — it is **no evidence at
        # all**, and it must not be able to masquerade as a pass. It very nearly could: `max(-inf,
        # nan)` is `-inf` in Python (the comparison `nan > -inf` is False), so a single NaN σ would
        # sail through the fold below and be reported as the most monotone ladder imaginable. The
        # same shape of trap as M5's `np.min(x, initial=0.0)`, and it is refused rather than ranked.
        #
        # Checked *before* the standard errors are computed, so the failure names the quantity that
        # is actually broken. `diagnostics` would otherwise reject the NaN draws first — correctly,
        # but opaquely: "draws contain NaN or inf" does not say which rung, nor that E[J] was the
        # thing being asked about.
        means = [rung.mean_j for rung in rungs]
        _reject_non_finite(rungs, means, "mean J")

        errors = [rung.standard_error_j for rung in rungs]
        _reject_non_finite(rungs, errors, "standard error of J")

        worst = -np.inf
        for i in range(len(rungs) - 1):
            drop = means[i] - means[i + 1]
            pooled = float(np.hypot(errors[i], errors[i + 1]))
            # A pooled error of exactly zero means J is constant on both rungs — a singleton
            # polytope, or a tilt so strong the chain never moves. The means are then exact, so any
            # drop at all is real and none is noise.
            zero_error = 0.0 if drop <= 0.0 else np.inf
            sigma = zero_error if pooled == 0.0 else drop / pooled
            worst = max(worst, float(sigma))

        worst_drop = -np.inf if len(rungs) < 2 else float(worst)

        return MonotonicityReport(
            betas=tuple(rung.beta for rung in rungs),
            mean_j=tuple(means),
            standard_error_j=tuple(errors),
            ess_j=tuple(rung.ess_j for rung in rungs),
            r_hat_j=tuple(rung.r_hat_j for rung in rungs),
            n_sigma=n_sigma,
            worst_drop_sigma=worst_drop,
            is_monotone=bool(worst_drop <= n_sigma),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "sampler_impl_version": SAMPLER_IMPL_VERSION,
            "betas": list(self.betas),
            "objective": self.objective.manifest(),
            **self.energy_scale.manifest(),
            "monotonicity": self.monotonicity().as_dict(),
            "rungs": [rung.as_dict() for rung in self.rungs],
        }


def _reject_non_finite(rungs: list[BetaRung], values: list[float], what: str) -> None:
    """A NaN is not a small number, and must not win a comparison by losing every one of them."""
    bad = [i for i, x in enumerate(values) if not np.isfinite(x)]
    if bad:
        raise SamplerError(
            f"the {what} is not finite at β = {[rungs[i].beta for i in bad]}, so the monotonicity "
            "of E_β[J] cannot be assessed. Refused rather than reported: a NaN compares false "
            "against everything, so it would otherwise be read as a pass."
        )


def _check_bindings(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    objective: ReducedObjective,
    energy_scale: EnergyScale,
) -> None:
    """The transform and the objective must both have been built from **this** polytope.

    Three artifacts meet here — the L1 polytope, the L3 transform and the L2 objective — and until
    now nothing checked that they had ever met before. They are all just arrays; a mismatched trio
    produces no bad numbers.

    The dangerous pairing is the objective. Hand `run_ladder` one lowered from a *different* model
    of
    the same size and the chain tilts by whatever reactions that objective names, while
    `ReducedObjective.evaluate_many` reports **those same reactions** as ``μ`` and ``C``. So the
    trace of ``J`` rises with β exactly as the theorem says it must — because the chain really is
    maximizing the thing the trace is measuring. Every diagnostic in this package agrees, and every
    one of them is describing the wrong model. Feasibility, mass balance, the chords and R̂ cannot
    help: none of them knows which reaction ``J`` is supposed to be about.

    One string comparison per run buys that away. It costs nothing and it is what makes M8's cache
    safe to build: the L2 and L3 artifacts are stored separately, and a stale key is all it takes to
    load two that were never computed against each other.
    """
    key = reduced.content_key()

    if not objective.binds_to(reduced):
        raise SamplerError(
            "the objective was not lowered from this polytope (objective.polytope_key = "
            f"{objective.polytope_key[:16]}…, reduced.content_key() = {key[:16]}…). It would tilt "
            "the chain by whatever reactions IT names, and its own traces would confirm them — see "
            "`_check_bindings`. Rebuild it with `sparse_objective.lower_objective(reduced, …)`."
        )
    if transform.polytope_key != key:
        raise SamplerError(
            "the rounded transform was not built from this polytope (transform.polytope_key = "
            f"{transform.polytope_key[:16]}…, reduced.content_key() = {key[:16]}…). The chain "
            "would step along directions from one polytope and be bounds-checked against another."
        )
    if objective.n_free != reduced.n_free or transform.n_free != reduced.n_free:
        raise SamplerError(
            f"n_free disagrees: polytope {reduced.n_free}, objective {objective.n_free}, transform "
            f"{transform.n_free}"
        )
    if energy_scale.polytope_key != objective.polytope_key:
        raise SamplerError(
            "the energy scale was calibrated from a different objective. ``s_J`` is the range "
            "``J`` spans over *this* objective on *this* polytope; borrowed from another, every β "
            "on the ladder silently names a different selection pressure — the exact failure "
            "``s_J`` exists to prevent. Rebuild it with `sparse_objective.choose_energy_scale`."
        )


def run_ladder(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    objective: ReducedObjective,
    energy_scale: EnergyScale,
    optimum_coordinates: NDArray[np.float64] | None = None,
    near_zero_thresholds: tuple[float, ...] = DEFAULT_NEAR_ZERO_THRESHOLDS,
    feasibility_tol: float = 1e-9,
) -> LadderResult:
    """Run every ``β`` in ``config.betas``, with objective and scale **frozen across the ladder**.

    Frozen is the load-bearing word, and it is why this is one function rather than a loop the
    caller writes. ``T``, ``w``, ``λ`` and ``s_J`` are inputs here, computed once, before the first
    chain starts. Re-deriving ``s_J`` per rung from that rung's own samples — an easy and
    plausible-looking thing to do — would make ``β`` mean something different on every rung, so the
    "mean ``J`` rises with ``β``" curve would no longer be a statement about ``β`` at all. M7 will
    need the same discipline for ``w``, for the sharper reason that a weight moving mid-chain
    destroys stationarity outright.

    ``β_index`` is the rung's position in ``config.betas``, and it names the RNG stream. So two runs
    with different ladders that share a β do **not** share its draws — which is honest: they are
    different experiments, and pretending otherwise would let a ladder's composition leak into a
    single rung's numbers.
    """
    _check_bindings(transform, reduced, objective, energy_scale)

    rungs: list[BetaRung] = []
    movable = movable_reactions(transform)

    for beta_index, beta in enumerate(config.betas):
        result = run_chains(
            transform,
            reduced,
            config=config,
            model_id=model_id,
            beta=beta,
            beta_index=beta_index,
            objective=objective.line,
            energy_scale=energy_scale.value,
            feasibility_tol=feasibility_tol,
            optimum_coordinates=optimum_coordinates,
        )
        traces = tuple(
            trace_objective(
                chain.fluxes,
                objective,
                j_star=energy_scale.j_star,
                energy_scale=energy_scale.value,
                thresholds=near_zero_thresholds,
                movable=movable,
            )
            for chain in result.chains
        )
        rungs.append(
            BetaRung(beta=beta, beta_index=beta_index, result=result, traces=traces)
        )

    return LadderResult(
        model_id=model_id,
        rungs=tuple(rungs),
        energy_scale=energy_scale,
        objective=objective,
    )
