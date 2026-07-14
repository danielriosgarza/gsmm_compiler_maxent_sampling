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

Implemented in **M5** (β = 0). The β-ladder, the energy scale ``s_J`` and the objective traces are
**M6**; the tilted branch below is M2's `sample_line` verbatim — wired up, not yet gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.config import SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.line_distribution import L1Objective, sample_line
from gsmm_compiler.line_geometry import chord_on_support
from gsmm_compiler.provenance import stream_seed
from gsmm_compiler.rounding import RoundedTransform

SAMPLER_IMPL_VERSION = 1
"""Bump when the transition kernel changes — invalidates every cached sample artifact."""

VALUE_DTYPE = np.float64

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
    """
    if transform.is_singleton:
        return np.zeros(0, dtype=VALUE_DTYPE), 1.0

    n_support = int(transform.support_coordinates.shape[0])
    weights = rng.dirichlet(np.full(n_support, concentration))
    proposal = np.asarray(weights @ transform.support_coordinates, dtype=VALUE_DTYPE)

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
) -> ChainResult:
    """Run one chain of coordinate hit-and-run and return its thinned draws (spec §18.2).

    ``objective`` and ``energy_scale`` are needed only at ``β > 0`` (M6). At ``β = 0`` the target is
    flat, ``J`` provably never enters the draw, and the whole step is a uniform draw on the chord.

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
            transform, reduced, rng, feasibility_tol=feasibility_tol
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
        )
        for chain_index in range(config.n_chains)
    )

    return SamplerResult(beta=beta, chains=chains, model_id=model_id)
