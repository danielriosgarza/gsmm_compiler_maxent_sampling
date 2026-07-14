"""Preconditioning the reduced polytope: ``T = diag(s)·B·L`` (spec §17).

The geometry (M4) hands over an *orthonormal* basis ``B`` of the direction space. Orthonormal in
the scaled flux coordinates is not the same as well-proportioned inside the polytope: flux
polytopes are extremely anisotropic, and an axis that is long in one direction is often needle-thin
in another. Coordinate hit-and-run along such axes mixes appallingly — every chord is short, so
every step is. Rounding fixes the *axes*, not the target. ``L`` is the Cholesky factor of the
support-point covariance in reduced coordinates, so the rounded axes are stretched to match how far
the polytope actually reaches in each direction.

Measured on the example model, that is not a small effect: the shortest chord through the centre
goes from 0.018 to 0.744 — 41× — and the spread between the shortest and longest axis collapses
from 77× to 3.8×.

**Nothing here can change the distribution, and that is the point.** ``L`` is ``d × d`` and
invertible, so

    range(T) = range(diag(s)·B·L) = range(diag(s)·B)

*exactly* — the same affine direction space, whatever ``L`` is. The map ``y ↦ v = centre + T·y`` is
affine and injective with a constant Jacobian, so uniform-in-``y`` is uniform-on-the-polytope, and
``π_β`` in ``y`` is ``π_β`` in flux, for **every** invertible ``L``. A badly chosen ``L`` costs
mixing speed and nothing else. This is worth stating precisely: it is what lets the ridge be an
engineering parameter rather than a knob that quietly bends the target, and it makes
transform-invariance a *testable* claim rather than a hope.

Three things the spec does not say, which the implementation has to get right anyway:

1. **The ridge is escalated but never *needed* for invertibility.** ``C_q + εI`` is symmetric
   positive definite for any ``ε > 0`` and any PSD ``C_q``, so in exact arithmetic Cholesky cannot
   fail. It fails in float64 only when ``ε`` is small enough to be lost in the rounding of ``C_q``
   — when the ridge was too small to exist. The geometric escalation walks out of exactly that, and
   the ridge that finally worked is recorded: the transform is not reproducible without it.

2. **A rank-deficient covariance is a *mixing* defect, not a correctness one — and the geometry's
   own rank check does not rule it out.** `GeometryDiagnostics.support_coordinate_rank` is the rank
   of the support points about the **centre**; the covariance is taken about the support points'
   own **mean**. Those differ, and a centred matrix can lose a rank the uncentred one has. So the
   rank of ``C_q`` is measured *here*. If it is below ``d``, the ridge is the only thing holding
   that direction open, its rounded step scale is ``√ε`` against a mean of ``√(trace/d)``, and the
   chain will crawl along it while looking perfectly healthy. The honest response is to report it —
   `RoundingDiagnostics.step_scale_ratio` is that number — not to hide it behind a Cholesky that
   succeeds regardless.

3. **``‖S·T‖`` must be judged on a relative bar** (the M4 lesson). ``T``'s columns are as long as
   the polytope is wide — hundreds of flux units — so ``S·T`` sums terms of that size, and
   evaluating it in float64 costs ``~eps·‖|S|·|T|‖`` before the solver contributes anything at all.
   An absolute 1e-9 bar charges that arithmetic to the geometry and fails a good transform.

Implemented in **M5** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.affine_geometry import (
    GEOMETRY_IMPL_VERSION,
    GeometryError,
    ReducedGeometry,
)
from gsmm_compiler.config import GeometryConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.line_geometry import feasible_chord
from gsmm_compiler.provenance import content_key

_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])

ROUNDING_IMPL_VERSION = 1
"""Bump when the transform's arithmetic changes — invalidates every cached ``T`` and its samples."""

VALUE_DTYPE = np.float64

_BYTES_PER_FLOAT64 = 8


class RoundingError(GeometryError):
    """The rounding transform could not be built, or failed one of its own checks."""


def _freeze(array: _ArrayT) -> _ArrayT:
    """Make an array physically read-only, so "frozen" is an invariant and not a convention.

    ``@dataclass(frozen=True)`` freezes the *bindings*, not the buffers: it stops
    ``transform.transform = X`` and does nothing whatever about ``transform.transform[0, 0] = X``.
    That gap is not academic here. `CoordinatePrecompute` holds **copies** of ``T``'s columns and
    its bound slices, taken at construction and validated against ``T`` once. Mutate ``T`` in place
    afterwards and the two silently disagree: the chord would be computed from the stale precompute
    while ``to_flux`` and the refresh used the mutated matrix — a chain sampling one polytope and
    reporting fluxes from another, with no error anywhere.

    Spec §17.4 and §18.3 both rest on the transform being frozen during production. This is what
    makes that enforceable rather than aspirational; an in-place write now raises `ValueError`.

    **What this does and does not guarantee.** It is accident-proof, not adversary-proof: a caller
    can set ``writeable = True`` back on an array that owns its buffer, and a writable *alias* taken
    before freezing would still be writable. So the arrays here are frozen at construction from
    buffers nothing else holds — every one is a fresh product, solve, or fancy-index copy — and
    `center` is explicitly `.copy()`-ed, so that freezing the transform does not reach back and
    freeze the geometry's own centre array underneath its owner.
    """
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class RoundingDiagnostics:
    """What the rounding step saw. Written to the run manifest alongside the geometry's."""

    dimension: int

    n_support_points: int

    covariance_rank: int
    """Rank of ``C_q`` — the support coordinates about **their own mean**, not the centre.

    Below ``d`` means the support vertices lie in a proper affine subspace of the direction space,
    and the ridge alone is holding the missing direction open. Not a correctness failure (see the
    module docstring), but `step_scale_ratio` then says how slowly the chain will move there.
    """

    covariance_trace: float

    ridge: float
    """The **absolute** ridge finally used, ``ε``."""

    ridge_relative: float
    """``ε·d / trace(C_q)`` — the ridge as the spec asks it be chosen: a fraction of the mean
    coordinate variance rather than an unexplained absolute constant."""

    n_escalations: int

    min_eigenvalue: float

    max_eigenvalue: float

    condition_number: float
    """``κ(C_ε)``. The rounded axes' *step* scales go as its square root."""

    step_scale_ratio: float
    """``√(λ_min/λ_max)`` — the shortest rounded axis as a fraction of the longest.

    This is the number that predicts mixing. At 1 the polytope is a ball in rounded coordinates and
    coordinate hit-and-run is as good as it gets; at 1e-3 one direction takes a million times as
    many steps to traverse, and an ESS from a chain too short to notice will still look fine.
    """

    coordinate_mean_norm: float
    """``‖q̄‖₂`` — how far the support points' mean sits from the geometry's centre, in reduced
    coordinates. The spec expects ≈ 0, the centre being their (clamped) mean."""

    transform_mass_balance_error: float
    """``max_k ‖S·T_k‖_∞ / ‖|S|·|T_k|‖_∞`` — the **relative** ``‖S·T‖`` (module docstring)."""

    transform_mass_balance_absolute: float
    """``‖S·T‖_max``, for reference. Do not put a fixed tolerance on this one."""

    min_chord_at_center: float
    """Shortest chord through the centre along any **rounded** axis — the start condition.

    M4 checked this along the *basis* columns. The sampler steps along ``T``'s columns, which are
    different directions, so the guarantee is re-established here rather than inherited.
    """

    transform_memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounding_dimension": self.dimension,
            "rounding_n_support_points": self.n_support_points,
            "covariance_rank": self.covariance_rank,
            "covariance_trace": self.covariance_trace,
            "ridge": self.ridge,
            "ridge_relative": self.ridge_relative,
            "n_escalations": self.n_escalations,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "step_scale_ratio": self.step_scale_ratio,
            "coordinate_mean_norm": self.coordinate_mean_norm,
            "transform_mass_balance_error": self.transform_mass_balance_error,
            "transform_mass_balance_absolute": self.transform_mass_balance_absolute,
            "min_chord_at_center": self.min_chord_at_center,
            "transform_memory_bytes": self.transform_memory_bytes,
        }


@dataclass(frozen=True)
class CoordinatePrecompute:
    """The frozen per-axis arrays the inner loop needs, and nothing else (BUILD_PLAN §1.3).

    For each rounded coordinate ``k`` the chord needs the *structurally nonzero* rows of
    ``T[:, k]`` and, on exactly those rows, the direction values and the bounds. Re-deriving them
    every step costs four fancy-index allocations over ``n_free``, for arrays that cannot have
    changed — ``T`` and the bounds are frozen.

    **The support is derived from ``T``, never supplied.** M2 removed a caller-supplied ``support``
    argument from `feasible_chord` precisely because an unvalidated one silently reintroduces the
    §1.6.1 tolerance bug: drop a *small* ``d_i`` whose ``v_i`` sits near its bound, and the chain
    steps straight through that bound. Here the support is `np.flatnonzero` of the column — an
    exact test, no tolerance — and `validate` re-derives every array from ``T`` and refuses any
    disagreement. The invariant holds by construction rather than by discipline.
    """

    support: tuple[NDArray[np.intp], ...]
    direction: tuple[NDArray[np.float64], ...]
    lower: tuple[NDArray[np.float64], ...]
    upper: tuple[NDArray[np.float64], ...]

    @property
    def dimension(self) -> int:
        return len(self.support)

    def validate(
        self,
        transform: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> None:
        """Re-derive every array from ``T`` and the bounds; refuse any disagreement."""
        d = int(transform.shape[1])
        lengths = {len(self.support), len(self.direction), len(self.lower), len(self.upper), d}
        if len(lengths) != 1:
            raise RoundingError(
                f"precompute has {len(self.support)} coordinates, transform has {d} columns"
            )
        for k in range(d):
            expected = np.flatnonzero(transform[:, k])
            if not np.array_equal(self.support[k], expected):
                raise RoundingError(
                    f"precomputed support of coordinate {k} is not the structural support of "
                    f"T[:, {k}] ({self.support[k].size} entries vs {expected.size}) — a truncated "
                    "support drops a binding bound and lets the chain step outside the polytope"
                )
            if not np.array_equal(self.direction[k], transform[expected, k]):
                raise RoundingError(f"precomputed direction of coordinate {k} does not match T")
            if not np.array_equal(self.lower[k], lower_bounds[expected]):
                raise RoundingError(f"precomputed lower bounds of coordinate {k} do not match")
            if not np.array_equal(self.upper[k], upper_bounds[expected]):
                raise RoundingError(f"precomputed upper bounds of coordinate {k} do not match")

    @classmethod
    def build(
        cls,
        transform: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> CoordinatePrecompute:
        support: list[NDArray[np.intp]] = []
        direction: list[NDArray[np.float64]] = []
        lower: list[NDArray[np.float64]] = []
        upper: list[NDArray[np.float64]] = []

        for k in range(int(transform.shape[1])):
            column = transform[:, k]
            nonzero = np.flatnonzero(column)
            if nonzero.size == 0:
                raise RoundingError(
                    f"rounded coordinate {k} has an all-zero column: the transform is rank "
                    "deficient, and the chain could never move along that axis"
                )
            support.append(_freeze(np.ascontiguousarray(nonzero, dtype=np.intp)))
            direction.append(_freeze(np.ascontiguousarray(column[nonzero], dtype=VALUE_DTYPE)))
            lower.append(_freeze(np.ascontiguousarray(lower_bounds[nonzero], dtype=VALUE_DTYPE)))
            upper.append(_freeze(np.ascontiguousarray(upper_bounds[nonzero], dtype=VALUE_DTYPE)))

        return cls(
            support=tuple(support),
            direction=tuple(direction),
            lower=tuple(lower),
            upper=tuple(upper),
        )


@dataclass(frozen=True)
class RoundedTransform:
    """The frozen preconditioned transform: ``v = centre + T·y`` (spec §17.3).

    Everything the sampler needs and nothing it does not: an MCMC worker gets one of these plus the
    bounds, and never imports cobra or HiGHS (BUILD_PLAN §1.2). Fluxes here are **reduced** (length
    ``n_free``); `ReducedPolytope.to_full` lifts a sample to a full-length flux vector when saved.

    Frozen is load-bearing, not decorative. Spec §17.4 and §18.3 both turn on it: an adaptive ``T``
    makes the transition kernel depend on the chain's own history, and the samples it produces are
    then not from ``π_β`` at all. No code path here mutates one after construction.
    """

    transform: NDArray[np.float64]
    """``T``, ``(n_free, d)``, Fortran-contiguous — so ``T[:, k]`` is a view, not a copy."""

    inverse_transform: NDArray[np.float64]
    """``T⁺ = L⁻¹·Bᵀ·diag(s)⁻¹``, ``(d, n_free)`` — flux → rounded coordinates, for the starts.

    Built with `numpy.linalg.solve` against ``L`` (spec §17), not by forming ``L⁻¹`` and
    multiplying. It is a left inverse on the affine hull and a *projector* off it: a flux that has
    drifted off the hull maps to the coordinates of its projection, which is the only sensible
    answer.
    """

    cholesky: NDArray[np.float64]
    """``L``, ``(d, d)`` lower triangular, with ``L·Lᵀ = C_q + εI``."""

    center: NDArray[np.float64]

    support_coordinates: NDArray[np.float64]
    """``(K, d)`` — the support points in **rounded** coordinates. Chain starts are drawn from
    their convex hull, which lies inside the polytope by convexity."""

    precompute: CoordinatePrecompute

    diagnostics: RoundingDiagnostics

    geometry_key: str

    polytope_key: str

    @property
    def dimension(self) -> int:
        return int(self.transform.shape[1])

    @property
    def n_free(self) -> int:
        return int(self.transform.shape[0])

    @property
    def is_singleton(self) -> bool:
        return self.dimension == 0

    def to_flux(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """``v = centre + T·y``. Takes one ``y`` or a batch of them."""
        y = np.asarray(coordinates, dtype=VALUE_DTYPE)
        if y.shape[-1:] != (self.dimension,):
            raise ValueError(
                f"coordinates have shape {y.shape}, expected trailing dim {self.dimension}"
            )
        return np.asarray(self.center + y @ self.transform.T, dtype=VALUE_DTYPE)

    def to_coordinates(self, flux: NDArray[np.float64]) -> NDArray[np.float64]:
        """``y = T⁺·(v − centre)`` — the inverse of `to_flux` on the affine hull."""
        v = np.asarray(flux, dtype=VALUE_DTYPE)
        if v.shape[-1:] != (self.n_free,):
            raise ValueError(f"flux has shape {v.shape}, expected trailing dim {self.n_free}")
        return np.asarray((v - self.center) @ self.inverse_transform.T, dtype=VALUE_DTYPE)

    def content_key(self) -> str:
        """The rounded-transform cache key: the geometry it came from, plus this code."""
        return content_key(
            geometry_key=self.geometry_key,
            transform=self.transform,
            center=self.center,
            ridge=self.diagnostics.ridge,
            rounding_impl_version=ROUNDING_IMPL_VERSION,
            geometry_impl_version=GEOMETRY_IMPL_VERSION,
            numpy_version=np.__version__,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "content_key": self.content_key(),
            "geometry_key": self.geometry_key,
            "polytope_key": self.polytope_key,
            "rounding_impl_version": ROUNDING_IMPL_VERSION,
            **self.diagnostics.as_dict(),
        }


def build_transform(
    geometry: ReducedGeometry,
    reduced: ReducedPolytope,
    *,
    config: GeometryConfig | None = None,
) -> RoundedTransform:
    """Precondition the polytope: covariance → ridge → Cholesky → ``T`` (spec §17.2–17.3).

    Every check here is one the sampler would otherwise discover the hard way, thousands of steps
    in: a zero column it can never move along, a chord at the centre excluding ``t = 0``, a ``T``
    whose columns have drifted off the mass-balance manifold.
    """
    config = config or GeometryConfig()

    # The geometry and the polytope must be the same polytope. `T` is built from the geometry's
    # basis and scaling, but the *bounds* the chain is held to — and the `CoordinatePrecompute` it
    # steps with — come from `reduced`. Mixing them produces a **hybrid**: it records the geometry's
    # `polytope_key`, so it sails through `run_ladder`'s binding check, and then walks bounds and a
    # mass balance belonging to another model entirely. (Codex, M6 review round 6.)
    if geometry.polytope_key != reduced.content_key():
        raise RoundingError(
            "the geometry was not built from this polytope. The transform would carry the "
            "geometry's directions and the polytope's bounds — a hybrid of two models that "
            "passes every downstream key check, because the key it reports is the geometry's."
        )

    if geometry.is_singleton:
        return _singleton_transform(geometry, reduced)

    d = geometry.dimension
    coordinates = geometry.to_coordinates(geometry.support_points)  # (K, d), spec §17.2
    n_support = int(coordinates.shape[0])
    if n_support < 2:
        raise RoundingError(
            f"a covariance needs at least 2 support points, got {n_support}; the geometry should "
            "have produced 2 per discovered direction"
        )

    covariance, trace, mean_norm = _support_covariance(coordinates)
    eigenvalues = np.linalg.eigvalsh(covariance)

    mean_variance = trace / d
    if not mean_variance > 0.0:
        raise RoundingError(
            f"the support points have zero spread in reduced coordinates (trace {trace:.3e}); "
            f"they cannot span the {d} directions the geometry says exist"
        )

    # The rank of C_q, on a bar *relative* to the mean coordinate variance. An absolute cutoff
    # would mean something different on every model; a relative one asks the question that has an
    # answer: is this direction's spread a real fraction of the others', or is it rounding noise?
    rank_cutoff = mean_variance * config.covariance_rank_tol
    covariance_rank = int(np.count_nonzero(eigenvalues > rank_cutoff))

    cholesky, ridge, n_escalations = _cholesky_with_ridge(
        covariance,
        initial_ridge=config.ridge_relative * mean_variance,
        growth=config.ridge_growth,
        max_escalations=config.max_ridge_escalations,
    )

    ridged = eigenvalues + ridge  # the eigenvalues of C_q + εI, exactly
    min_eigenvalue = float(ridged.min())
    max_eigenvalue = float(ridged.max())
    if not (np.isfinite(min_eigenvalue) and np.isfinite(max_eigenvalue) and min_eigenvalue > 0.0):
        # `eigvalsh` can return a slightly negative eigenvalue for a PSD matrix, and Cholesky can
        # still succeed on the ridged one. Then `step_scale_ratio = √(λ_min/λ_max)` is `√(negative)`
        # — a **NaN silently entering the manifest**, in exactly the regime (a near-singular
        # covariance) where the diagnostic is the only thing that would have warned anyone. The
        # finiteness half matters too: an infinite λ_max leaves λ_min positive, so a bare `> 0`
        # test passes while the condition number goes to infinity and the step scale to zero.
        raise RoundingError(
            f"the ridged covariance has a nonpositive or non-finite eigenvalue (λ_min = "
            f"{min_eigenvalue:.3e}, λ_max = {max_eigenvalue:.3e}) although Cholesky succeeded; "
            f"ridge = {ridge:.3e} is too small to make C_q numerically definite"
        )

    _guard_memory(reduced.n_free, d, config.max_geometry_memory_gb)
    # T = diag(s)·(B·L). Rows of B that are structurally zero — the FVA-blocked reactions M4
    # projects out (§1.4.1) — stay *exactly* zero through the multiply, and `_check_structural_
    # zeros` holds the multiply to that. Those exact zeros keep a blocked reaction out of the chord
    # entirely, rather than letting it contribute a noise-valued bound ratio.
    transform = np.asfortranarray((geometry.basis @ cholesky) * geometry.scaling[:, np.newaxis])
    _check_structural_zeros(geometry.basis, transform)
    _check_full_column_rank(transform, d, rank_tol=config.rank_tol)

    # y = L⁻¹·Bᵀ·diag(s)⁻¹·(v − centre), by a solve against L rather than an explicit inverse.
    scaled_basis = geometry.basis / geometry.scaling[:, np.newaxis]
    inverse_transform = np.linalg.solve(cholesky, scaled_basis.T)
    support_coordinates = np.linalg.solve(cholesky, coordinates.T).T

    relative_error, absolute_error = _transform_mass_balance(transform, reduced)
    if relative_error > config.span_tol:
        raise RoundingError(
            f"‖S·T‖ relative to its own cancellation scale is {relative_error:.3e}, above "
            f"span_tol {config.span_tol:.1e}: the rounded axes have left the mass-balance "
            "manifold, so stepping along them would break the steady-state constraint"
        )

    precompute = CoordinatePrecompute.build(transform, reduced.lower_bounds, reduced.upper_bounds)
    precompute.validate(transform, reduced.lower_bounds, reduced.upper_bounds)

    min_chord = _min_chord_at_center(
        transform, geometry.center, reduced, feasibility_tol=config.feasibility_tol
    )
    if not min_chord > 0.0:
        raise RoundingError(
            f"the shortest chord through the centre along a rounded axis is {min_chord:.3e}, so "
            "the sampler has an axis it cannot move along from its own starting point: the centre "
            "is pinned in a direction the geometry says is free"
        )

    diagnostics = RoundingDiagnostics(
        dimension=d,
        n_support_points=n_support,
        covariance_rank=covariance_rank,
        covariance_trace=float(trace),
        ridge=float(ridge),
        ridge_relative=float(ridge / mean_variance),
        n_escalations=n_escalations,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        condition_number=float(max_eigenvalue / min_eigenvalue),
        step_scale_ratio=float(np.sqrt(min_eigenvalue / max_eigenvalue)),
        coordinate_mean_norm=float(mean_norm),
        transform_mass_balance_error=float(relative_error),
        transform_mass_balance_absolute=float(absolute_error),
        min_chord_at_center=float(min_chord),
        transform_memory_bytes=_BYTES_PER_FLOAT64 * reduced.n_free * d,
    )

    return RoundedTransform(
        transform=_freeze(transform),
        inverse_transform=_freeze(np.ascontiguousarray(inverse_transform, dtype=VALUE_DTYPE)),
        cholesky=_freeze(np.ascontiguousarray(cholesky, dtype=VALUE_DTYPE)),
        center=_freeze(np.ascontiguousarray(geometry.center, dtype=VALUE_DTYPE).copy()),
        support_coordinates=_freeze(
            np.ascontiguousarray(support_coordinates, dtype=VALUE_DTYPE)
        ),
        precompute=precompute,
        diagnostics=diagnostics,
        geometry_key=geometry.content_key(),
        polytope_key=geometry.polytope_key,
    )


def _support_covariance(
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """``C_q`` about the support points' own mean, plus its trace and ``‖q̄‖`` (spec §17.2).

    Centred on ``q̄``, **not** on the origin. The geometry's centre is the (clamped) mean of the
    support points, so ``q̄ ≈ 0`` — but "approximately zero" is not zero, and a covariance taken
    about the wrong point is a second-moment matrix: it picks up ``q̄·q̄ᵀ``, inflating the variance
    along whichever direction the clamp happened to move the centre.
    """
    mean = coordinates.mean(axis=0)
    centered = coordinates - mean
    n_support = int(coordinates.shape[0])

    covariance = (centered.T @ centered) / (n_support - 1)
    # Exactly symmetric, or `eigvalsh`/`cholesky` read a matrix that is not the one we computed:
    # the Gram product is symmetric in exact arithmetic, but its float64 rounding is not.
    covariance = 0.5 * (covariance + covariance.T)

    return covariance, float(np.trace(covariance)), float(np.linalg.norm(mean))


def _cholesky_with_ridge(
    covariance: NDArray[np.float64],
    *,
    initial_ridge: float,
    growth: float,
    max_escalations: int,
) -> tuple[NDArray[np.float64], float, int]:
    """``chol(C_q + εI)``, escalating ``ε`` geometrically until it factors (spec §17.2).

    In exact arithmetic this loop never runs twice: ``C_q`` is PSD and ``εI`` is positive definite,
    so the sum is positive definite for every ``ε > 0``. It runs twice when ``ε`` is small enough
    that adding it to ``C_q``'s diagonal is lost to rounding — that is, when the ridge was too
    small to exist. Escalating is the only sensible repair, and the ridge that finally worked is
    what the manifest must record: the transform is not reproducible without it.
    """
    d = int(covariance.shape[0])
    identity = np.eye(d, dtype=VALUE_DTYPE)
    ridge = float(initial_ridge)

    for escalation in range(max_escalations + 1):
        try:
            factor = np.linalg.cholesky(covariance + ridge * identity)
        except np.linalg.LinAlgError:
            ridge *= growth
            continue
        return np.asarray(factor, dtype=VALUE_DTYPE), ridge, escalation

    raise RoundingError(
        f"Cholesky still failed after {max_escalations} geometric escalations of the ridge (last "
        f"ε = {ridge:.3e}); the support covariance is not merely ill-conditioned but corrupt"
    )


def _guard_memory(n_free: int, dimension: int, limit_gb: float) -> None:
    needed = _BYTES_PER_FLOAT64 * n_free * dimension
    limit = int(limit_gb * (1 << 30))
    if needed > limit:
        raise RoundingError(
            f"the transform needs {needed / (1 << 20):.1f} MiB ({n_free}×{dimension} float64), "
            f"above the {limit_gb:.2f} GiB limit"
        )


def _check_structural_zeros(basis: NDArray[np.float64], transform: NDArray[np.float64]) -> None:
    """A reaction with an all-zero basis row must have an all-zero ``T`` row — exactly.

    Those rows are the FVA-blocked reactions the geometry projected out (BUILD_PLAN §1.4.1), and
    the exactness is what keeps them out of the chord: `np.flatnonzero` is an exact test, so an
    exact zero contributes no bound ratio at all. A row of 1e-300 instead of ``0.0`` would
    contribute one, and M4 measured what that does — a chord limit of 0.03–0.5 from pure noise.

    Since ``T = diag(s)·B·L``, a zero row of ``B`` gives a zero row of ``T`` in float64 as surely
    as in exact arithmetic (a sum of ``0·x`` is ``0.0``) — *unless* ``L`` holds a NaN or an
    infinity, which is precisely the corruption this catches.
    """
    blocked = ~np.any(basis != 0.0, axis=1)
    if not np.any(blocked):
        return
    leaked = np.flatnonzero(np.any(transform[blocked] != 0.0, axis=1))
    if leaked.size:
        raise RoundingError(
            f"{leaked.size} reactions with an all-zero basis row picked up a nonzero row in T; "
            "the Cholesky factor must hold a NaN or an infinity"
        )


def _check_full_column_rank(
    transform: NDArray[np.float64], dimension: int, *, rank_tol: float
) -> None:
    """``T`` must have full column rank ``d`` — the claim the whole module rests on.

    In exact arithmetic this is free: ``s_i ≥ scale_floor > 0`` makes ``diag(s)`` invertible, ``B``
    has full column rank by construction, and ``L`` is invertible, so ``rank(T) = d`` and
    ``range(T) = range(diag(s)·B)`` — the identity that lets *any* ridge be chosen without moving
    the target.

    In float64 it is an *assumption*, and it was an unchecked one. A ``T`` that has quietly lost a
    column would span a strict subspace of the direction space: the chain would then explore a
    lower-dimensional slice of the polytope, every sample would be perfectly feasible, every chord
    would be positive, the mass balance would hold exactly — and part of the support would simply
    never be visited. No other check in this package looks for that, because a *missing* dimension
    produces no bad numbers, only absent ones.

    It costs one SVD of a 260×46 matrix. Measured on the example model: rank 46/46 with a condition
    number of 165, so it is nowhere near failing — which is the point of knowing rather than hoping.
    """
    if not (np.isfinite(rank_tol) and rank_tol > 0.0):
        raise RoundingError(f"rank_tol must be finite and > 0, got {rank_tol}")

    # Compare the rank against ``T``'s **own** column count, not only against the ``d`` handed in.
    # Checking `rank == dimension` alone would pass an ``n × (d+1)`` matrix of rank ``d`` — the very
    # rank deficiency this exists to catch — because the extra dependent column is never counted.
    columns = int(transform.shape[1])
    if columns != dimension:
        raise RoundingError(f"T has {columns} columns, expected {dimension}")
    if columns == 0:
        return  # a singleton polytope has no axes; `.max()` below would raise on the empty SVD

    singular_values = np.linalg.svd(transform, compute_uv=False)
    if not np.all(np.isfinite(singular_values)):
        raise RoundingError("T has non-finite singular values; the Cholesky factor is corrupt")

    cutoff = float(singular_values.max()) * rank_tol
    numerical_rank = int(np.count_nonzero(singular_values > cutoff))
    if numerical_rank != columns:
        raise RoundingError(
            f"T has numerical rank {numerical_rank}, not {columns}: the rounded axes span only "
            f"part of the direction space (σ_min = {singular_values.min():.3e}, σ_max = "
            f"{singular_values.max():.3e}). The chain would sample a lower-dimensional slice "
            "of the polytope and never once look infeasible"
        )


def _transform_mass_balance(
    transform: NDArray[np.float64], reduced: ReducedPolytope
) -> tuple[float, float]:
    """``‖S·T‖``, relative to its own cancellation scale, and absolute.

    The relative bar is the M4 lesson: ``S·T_k`` sums terms of size ``|S|·|T_k|``, so float64
    evaluation alone costs ``~eps`` of that before any solver error enters. Dividing by it asks the
    question that has an answer — *did this direction leave the manifold by more than the arithmetic
    that measured it could resolve?*

    **Unfloored, deliberately — unlike `NativeCSC.relative_residual`, and the asymmetry is the
    point.** That floor exists because a sampled *flux* carries solver noise at the FVA-blocked
    reactions (~1e-14 rather than 0), so a row touched only by blocked reactions divides a noise
    value by itself and reports a relative residual of exactly 1.0. **A direction has no such
    noise**: ``T``'s rows at blocked reactions are *exactly* ``0.0`` (`_check_structural_zeros`), so
    such a row's cancellation scale is exactly zero, its residual is a sum of no terms, and it is
    excluded rather than divided. Every remaining row has a scale earned by a real, nonzero entry of
    ``T``, and dividing by it is legitimate however small it is.

    So the floor would only make this gate *weaker* with nothing to buy: 2049 of the 41124
    (column, row) pairs on the example model have a cancellation scale below one flux unit, and
    flooring them at 1.0 loosens the bar on every one. Measured, the unfloored figure is 3.5e-10
    against a `span_tol` of 1e-9 — it passes on its own merits, and it is the honest instrument.

    **What a per-column check does and does not certify.** It bounds every *combination* in
    absolute terms — ``‖S·T·y‖_∞ ≤ ‖y‖₁ · max_k ‖S·T_k‖_∞`` by the triangle inequality — but it is
    **not** a support-wide bound on the *relative* residual, because the cancellation scale at a
    point ``T·y`` can shrink through inter-column cancellation while the column residuals reinforce.
    This function does not claim otherwise. The operative guarantee is elsewhere and is stronger for
    being empirical: the mass balance of ``v = centre + T·y`` is recomputed and checked on **every
    stored sample** (`diagnostics.feasibility_report`, and `is_feasible` now fails on it), so the
    points the chain actually emits are verified rather than bounded a priori. Measured over 2000
    genome-scale samples: 2.8e-12 relative, 2.6e-11 absolute.
    """
    worst_relative = 0.0
    worst_absolute = 0.0

    for k in range(int(transform.shape[1])):
        column = np.ascontiguousarray(transform[:, k])
        residual = np.abs(reduced.stoichiometry.matvec(column))
        scale = reduced.stoichiometry.cancellation_scale(column)

        worst_absolute = max(worst_absolute, float(residual.max(initial=0.0)))

        active = scale > 0.0
        if np.any(active):
            worst_relative = max(worst_relative, float((residual[active] / scale[active]).max()))
        if np.any(residual[~active] != 0.0):
            raise RoundingError(
                "a mass-balance row with no active reaction has a nonzero residual, which is "
                "arithmetically impossible; S·T is not the matrix we think it is"
            )

    return worst_relative, worst_absolute


def _min_chord_at_center(
    transform: NDArray[np.float64],
    center: NDArray[np.float64],
    reduced: ReducedPolytope,
    *,
    feasibility_tol: float,
) -> float:
    """Shortest chord through the centre along any rounded axis — the start condition.

    Uses the `feasible_chord` **oracle** rather than the precompute, deliberately: it is a second
    opinion on the same geometry, from the code that derives its own support.
    """
    shortest = np.inf
    for k in range(int(transform.shape[1])):
        chord = feasible_chord(
            center,
            np.ascontiguousarray(transform[:, k]),
            reduced.lower_bounds,
            reduced.upper_bounds,
            feasibility_tol=feasibility_tol,
        )
        shortest = min(shortest, chord.length)
    return float(shortest)


def _singleton_transform(geometry: ReducedGeometry, reduced: ReducedPolytope) -> RoundedTransform:
    """``d = 0``: the feasible set is the centre, and every sample is that point (spec §16)."""
    n_free = reduced.n_free

    return RoundedTransform(
        transform=_freeze(np.asfortranarray(np.zeros((n_free, 0), dtype=VALUE_DTYPE))),
        inverse_transform=_freeze(np.zeros((0, n_free), dtype=VALUE_DTYPE)),
        cholesky=_freeze(np.zeros((0, 0), dtype=VALUE_DTYPE)),
        center=_freeze(np.ascontiguousarray(geometry.center, dtype=VALUE_DTYPE).copy()),
        support_coordinates=_freeze(np.zeros((0, 0), dtype=VALUE_DTYPE)),
        precompute=CoordinatePrecompute(support=(), direction=(), lower=(), upper=()),
        diagnostics=RoundingDiagnostics(
            dimension=0,
            n_support_points=0,
            covariance_rank=0,
            covariance_trace=0.0,
            ridge=0.0,
            ridge_relative=0.0,
            n_escalations=0,
            min_eigenvalue=0.0,
            max_eigenvalue=0.0,
            condition_number=1.0,
            step_scale_ratio=1.0,
            coordinate_mean_norm=0.0,
            transform_mass_balance_error=0.0,
            transform_mass_balance_absolute=0.0,
            min_chord_at_center=0.0,
            transform_memory_bytes=0,
        ),
        geometry_key=geometry.content_key(),
        polytope_key=geometry.polytope_key,
    )
