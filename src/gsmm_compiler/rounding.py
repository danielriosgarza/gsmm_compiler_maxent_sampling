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

3. **``‖S·T‖`` cannot be gated on *any* per-direction bar, relative or absolute** — corrected in
   **M9**, and the correction is the module's sharpest lesson. M5 reasoned, rightly, that an
   absolute 1e-9 bar charges float64's own arithmetic to the geometry, and so divided by the
   cancellation scale. But the residual ``S·T`` is **not** produced by that multiply: it is an
   absolute floor (~4e-12, constant to 1.8× across RNG streams) inherited from the basis
   construction. Dividing a fixed floor by a per-row scale that happens to be ~1e-5 manufactures a
   "relative error" of 3e-8 — and it rejected **8 of 24 RNG streams on the same polytope**, making
   the `model_id` *string* decide whether a genome-scale model could be sampled.

   The measurement that settles it: a log-log fit of residual against row scale over 61009
   (column, row) pairs has slope **+0.165**, not the **+1** a locally-generated error requires.

   The deeper point is that a per-direction bar answers the wrong question. What matters is not how
   far a *unit* step along ``T_k`` leaves the manifold, but whether any state the chain can
   **reach**
   violates mass balance — and ``Y`` is a coupled polytope, not a box, so no per-column quantity
   bounds it. `certify_reachable_mass_balance` asks the reachable question directly, with two LPs
   per metabolite, against the *same* contract `diagnostics` applies to emitted samples.
   `RoundingDiagnostics.transform_mass_balance_error` survives as a **reported structural
   diagnostic** — it is a genuine Oettli–Prager componentwise backward error, and it catches a
   corruption the reachable certificate deliberately misses (a transform inventing motion along a
   direction of true width zero) — but it never raises. See BUILD_PLAN §1.4.2.

Implemented in **M5**; the mass-balance gate replaced in **M9** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
from gsmm_compiler.native_csc import NativeCSC
from gsmm_compiler.provenance import content_key

_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])

ROUNDING_IMPL_VERSION = 2
"""Bump when the transform's arithmetic changes — invalidates every cached ``T`` and its samples.

2 (M10): `RoundingDiagnostics` gained `covariance_source`, so a v1 bundle cannot say which
estimator produced its ``C_q``. `from_cache` would reject it on the missing field anyway; bumping
the version makes such a bundle **miss** the cache instead of erroring on a hit — BUILD_PLAN §1.1's
rule that a refactor changing artifact semantics must miss, never load stale.
"""

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
    """How many points the covariance was estimated from — support vertices, or pilot draws when
    `covariance_source` says ``pilot``."""

    covariance_source: str
    """``support_points`` (M4's LP vertices), ``pilot`` (M10's frozen β=0 chain), or ``singleton``.

    Not cosmetic: the two estimators estimate *different things*. The support vertices are extreme
    points and describe the polytope's **outline**; a β=0 pilot describes the **uniform measure's
    own** covariance, which is what spec §17.4's rounding actually wants. Both give a valid ``T`` —
    the transform is a preconditioner and provably cannot move the target (§1.6.1) — but they give
    different *conditioning*, so a reader comparing two runs' `condition_number` has to know which
    estimator each used.
    """

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
    """``max_k max_i |S_i·T_k| / (|S|·|T_k|)_i`` — the **componentwise backward error** of
    ``S·T=0``.

    **This docstring used to claim ``max_k ‖S·T_k‖_∞ / ‖|S|·|T_k|‖_∞`` — a per-*column* ratio of
    norms — which is not what the code computes and never was** (M9). The two differ:
    ``max_i(r_i/s_i) ≠ max_i(r_i)/max_i(s_i)``, and on the example model by five orders of
    magnitude.

    What it is, exactly (Oettli–Prager): for a row with ``q = (|S|·|T_k|)_i > 0``,
    ``|S_i·T_k| / q`` is the **smallest componentwise relative perturbation of ``S_i``** that would
    make ``(S_i + ΔS_i)·T_k = 0`` hold exactly. So it is a real, well-defined structural quantity —
    and it is **reported, never gated on**. See `certify_reachable_mass_balance` for the gate and
    BUILD_PLAN §1.4.2 for why the two are different questions.

    **Why it must not be a gate.** The residual it divides is not generated by row ``i``'s own
    multiply — it is an absolute floor inherited from the basis construction. Measured over 61009
    (column, row) pairs across 8 RNG streams: a log-log fit of residual against row scale has slope
    **+0.165**, not the **+1** a locally-generated error would give; across ≥4 decades of row scale
    the median residual rises only **6.6×** where ``~1e4×`` would be expected. So at a row whose
    scale is ~1e-5 this ratio is a fixed ~1e-13 divided by a small number, and it crossed a
    ``span_tol`` of 1e-9 on **8 of 24 RNG streams for the same polytope** — the `model_id` *string*
    decided whether a genome-scale model could be sampled at all.
    """

    transform_mass_balance_absolute: float
    """``‖S·T‖_max``, for reference. Do not put a fixed tolerance on this one.

    Nearly constant across RNG streams (3.1e-12 … 5.5e-12, a 1.8× spread, against the relative
    figure's 373×) — which is the measurement that exposed the error as an inherited absolute floor.
    """

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
            "covariance_source": self.covariance_source,
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

    def to_cache(self) -> dict[str, Any]:
        """A round-trippable dict keyed by the **real field names** (not `as_dict`'s report keys).

        `as_dict` renames fields for the human-facing manifest (``dimension`` →
        ``rounding_dimension``); a cache must instead reconstruct the object exactly, so it keeps
        the field names and `from_cache` feeds them straight back to the constructor.
        """
        return asdict(self)

    @classmethod
    def from_cache(cls, data: dict[str, Any]) -> RoundingDiagnostics:
        names = {f.name for f in fields(cls)}
        missing = names - data.keys()
        if missing:
            raise RoundingError(f"cached rounding diagnostics lack fields: {sorted(missing)}")
        return cls(**{name: data[name] for name in names})


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

    # ---- caching (L3 artifact, M8) ------------------------------------------------------------

    def to_bundle(self) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Split into ``(arrays, meta)`` for the content-addressed cache (`cache.ArtifactCache`).

        The `precompute` is *not* stored: it is a pure function of ``T`` and the bounds, and storing
        it would only be a second copy to fall out of step. `from_bundle` rebuilds it and validates
        it against ``T`` — the same by-construction discipline `build_transform` uses (BUILD_PLAN
        §1.3), so a cached transform is exactly as trustworthy as a freshly built one.
        """
        arrays = {
            "transform": np.ascontiguousarray(self.transform),
            "inverse_transform": np.ascontiguousarray(self.inverse_transform),
            "cholesky": np.ascontiguousarray(self.cholesky),
            "center": np.ascontiguousarray(self.center),
            "support_coordinates": np.ascontiguousarray(self.support_coordinates),
        }
        meta = {
            "content_key": self.content_key(),
            "geometry_key": self.geometry_key,
            "polytope_key": self.polytope_key,
            "diagnostics": self.diagnostics.to_cache(),
        }
        return arrays, meta

    @classmethod
    def from_bundle(
        cls,
        arrays: dict[str, NDArray[Any]],
        meta: dict[str, Any],
        reduced: ReducedPolytope,
    ) -> RoundedTransform:
        """Rebuild a transform cached by `to_bundle`, against the polytope it was rounded for.

        ``reduced`` supplies the bounds `CoordinatePrecompute` is derived and validated from, and
        its ``content_key`` is checked against the cached ``polytope_key`` — reconstructing a
        transform against a *different* polytope is exactly the "two artifacts that never met" join
        M6 spent six rounds hunting, so it is refused rather than silently producing a transform for
        the wrong model.
        """
        if meta["polytope_key"] != reduced.content_key():
            raise RoundingError(
                "cached transform was rounded for a different polytope "
                f"({str(meta['polytope_key'])[:16]}…) than the one it is being rebuilt against "
                f"({reduced.content_key()[:16]}…)"
            )
        transform = np.asfortranarray(np.asarray(arrays["transform"], dtype=VALUE_DTYPE))
        precompute = CoordinatePrecompute.build(
            transform, reduced.lower_bounds, reduced.upper_bounds
        )
        precompute.validate(transform, reduced.lower_bounds, reduced.upper_bounds)
        rebuilt = cls(
            transform=transform,
            inverse_transform=np.ascontiguousarray(arrays["inverse_transform"], dtype=VALUE_DTYPE),
            cholesky=np.ascontiguousarray(arrays["cholesky"], dtype=VALUE_DTYPE),
            center=np.ascontiguousarray(arrays["center"], dtype=VALUE_DTYPE),
            support_coordinates=np.ascontiguousarray(
                arrays["support_coordinates"], dtype=VALUE_DTYPE
            ),
            precompute=precompute,
            diagnostics=RoundingDiagnostics.from_cache(meta["diagnostics"]),
            geometry_key=meta["geometry_key"],
            polytope_key=meta["polytope_key"],
        )
        # A cheap tripwire: if any array was silently altered, the content key drifts and we would
        # rather fail loudly than sample against a transform that is not what was cached.
        if rebuilt.content_key() != meta["content_key"]:
            raise RoundingError(
                "rebuilt transform does not match its cached content key — the bundle is corrupt "
                "or was written by an incompatible version"
            )
        return rebuilt


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

    coordinates = geometry.to_coordinates(geometry.support_points)  # (K, d), spec §17.2
    n_support = int(coordinates.shape[0])
    if n_support < 2:
        raise RoundingError(
            f"a covariance needs at least 2 support points, got {n_support}; the geometry should "
            "have produced 2 per discovered direction"
        )

    return _transform_from_coordinates(
        geometry, reduced, coordinates, config=config, source="support_points"
    )


def reround_transform(
    geometry: ReducedGeometry,
    reduced: ReducedPolytope,
    bootstrap: RoundedTransform,
    *,
    pilot_coordinates: NDArray[np.float64],
    config: GeometryConfig | None = None,
) -> RoundedTransform:
    """Re-round from a frozen β=0 pilot's own covariance (spec §17.4) — M10's ``T₁``.

    `build_transform` estimates ``C_q`` from M4's **support-LP vertices**, which are extreme points:
    they describe the polytope's *outline*, not the uniform measure living inside it. Spec §17.4
    says to re-round from a pilot chain instead, and it is worth having — measured on the example
    model, ``cond(C_q)`` falls 1.54e4 → 5.11e3 and ESS rises ~2.5×.

    **This cannot move the target, and that is a theorem.** ``L`` is ``d×d`` and invertible, so
    ``range(T₁) = range(diag(s)·B·L₁) = range(diag(s)·B) = range(T₀)`` *exactly*, and
    ``y ↦ centre + T₁·y`` is affine and injective with a constant Jacobian (§1.6.1). What must be
    guarded is not the algebra — it was never in doubt — but the implementation: rank loss,
    feasibility tolerance, and above all that the pilot is **frozen before production** and never
    re-read mid-chain.

    ``pilot_coordinates`` is ``(N, d)`` in ``bootstrap``'s **rounded** frame — the chain's own
    ``y``. They are mapped back to geometry coordinates *exactly*, by ``q = L₀·y``: from
    ``diag(s)·B·q = T₀·y = diag(s)·B·L₀·y`` with ``diag(s)·B`` of full column rank. That is an
    identity, not a projection — which is why this takes the bootstrap transform rather than
    re-deriving ``q`` from the pilot's fluxes through ``T₀⁺``, which would silently absorb drift
    instead of carrying it.

    **Arrays, not a pilot object, deliberately.** `rounding` is imported by MCMC workers (§1.2) and
    must not import the sampler; `calibration` owns the `NeutralPilot` and calls this with its
    arrays. The binding is still checked — the caller must have run its pilot under this
    ``bootstrap``.
    """
    config = config or GeometryConfig()

    if geometry.polytope_key != reduced.content_key():
        raise RoundingError(
            "the geometry was not built from this polytope; re-rounding would carry one model's "
            "directions and another's bounds"
        )
    if bootstrap.polytope_key != reduced.content_key():
        raise RoundingError(
            "the bootstrap transform was not built from this polytope, so its `cholesky` does not "
            "map this pilot's coordinates back into this geometry's frame"
        )
    if bootstrap.geometry_key != geometry.content_key():
        raise RoundingError(
            "the bootstrap transform was not built from this geometry "
            f"({bootstrap.geometry_key[:16]}… vs {geometry.content_key()[:16]}…). ``q = L₀·y`` is "
            "the right change of coordinates only when ``L₀`` is *this* geometry's Cholesky: with "
            "another's, the re-rounded covariance describes a frame the new ``T₁`` is not built "
            "in — and every downstream check still passes, because the shapes agree."
        )

    if geometry.is_singleton:
        return _singleton_transform(geometry, reduced)

    y = np.asarray(pilot_coordinates, dtype=VALUE_DTYPE)
    if y.ndim != 2 or y.shape[1] != geometry.dimension:
        raise RoundingError(
            f"pilot coordinates have shape {y.shape}, expected (N, {geometry.dimension})"
        )
    if int(y.shape[0]) < 2:
        raise RoundingError(f"a covariance needs at least 2 pilot draws, got {int(y.shape[0])}")
    if not np.all(np.isfinite(y)):
        raise RoundingError("the pilot coordinates are not all finite")

    coordinates = y @ bootstrap.cholesky.T  # q = L₀·y, exactly

    return _transform_from_coordinates(
        geometry, reduced, coordinates, config=config, source="pilot"
    )


def _transform_from_coordinates(
    geometry: ReducedGeometry,
    reduced: ReducedPolytope,
    coordinates: NDArray[np.float64],
    *,
    config: GeometryConfig,
    source: str,
) -> RoundedTransform:
    """covariance → ridge → Cholesky → ``T``, from *any* point set in geometry coordinates.

    One implementation for both estimators. They differ only in which points they hand in;
    everything after the covariance — ridge escalation, the rank check, the structural zeros, the
    chord at the centre — is the same mathematics and must not drift into two versions of itself.
    """
    d = geometry.dimension
    n_points = int(coordinates.shape[0])

    covariance, trace, mean_norm = _support_covariance(coordinates)
    eigenvalues = np.linalg.eigvalsh(covariance)

    mean_variance = trace / d
    if not mean_variance > 0.0:
        raise RoundingError(
            f"the {source} points have zero spread in reduced coordinates (trace {trace:.3e}); "
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
    # Always the **support vertices** in the new rounded frame — never `coordinates`, which on the
    # pilot path is the pilot's own N draws. `support_coordinates` seeds dispersed chain starts from
    # a hull that lies inside the polytope by convexity of vertices; a hull of *interior* pilot
    # points is also inside it, but far less dispersed — exactly when M10's purpose is to detect
    # retained initialization. On the support path this is `coordinates` and the line is a no-op.
    support_coordinates = np.linalg.solve(
        cholesky, geometry.to_coordinates(geometry.support_points).T
    ).T

    # Reported, **not** gated (M9, BUILD_PLAN §1.4.2). This is a componentwise backward error, and
    # its denominator is not the scale of the arithmetic that produced the residual — so a small-
    # scale row divides an inherited ~1e-13 floor and manufactures a "relative error" that rejected
    # 8 of 24 RNG streams on a polytope that is perfectly samplable. The gate that replaced it is
    # `certify_reachable_mass_balance`, which bounds what the chain can actually *emit*.
    relative_error, absolute_error = _transform_mass_balance(transform, reduced)

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
        n_support_points=n_points,
        covariance_source=source,
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
            covariance_source="singleton",
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


# ──────────────────────────────────────────────────────────────────────────────────────────────
# The reachable-state mass-balance certificate (M9)
# ──────────────────────────────────────────────────────────────────────────────────────────────

DEFAULT_MASS_BALANCE_CONTRACT = 1e-9
"""``η`` — the declared relative mass-balance contract, shared with `diagnostics`.

**One number, two checks.** `diagnostics.feasibility_report` already holds every *emitted sample* to
``|S_i·v − b_i| / max((|S|·|v|)_i, 1) ≤ η``. This certificate proves the same predicate holds for
**every state the chain can reach**, a priori. Deriving a second, separate tolerance here — from a
measured residual, a typical flux, or the LP's own feasibility tolerance — would mean the package
declared two different definitions of "mass balanced" and enforced each in a different place.
The solver's tolerance is an *implementation capability* that must be tight enough to support the
contract; it is not the contract.
"""


@dataclass(frozen=True)
class ReachabilityCertificate:
    """A bound on the mass-balance residual of **every state the chain can reach** (M9).

    This is the gate `_transform_mass_balance` was wrongly asked to be. The question it answers is
    the operative one — *can this transform put the chain somewhere that violates steady state?* —
    and unlike a per-direction backward error it is a statement about the emitted support.

    **The construction.** With ``E = S·T``, ``r_c = S·c − b`` and the reachable set
    ``Y = {y : l ≤ c + T·y ≤ u}``, the worst residual on metabolite ``i`` is

        ``R_i = max(|r_c,i + min_{y∈Y} E_i·y|, |r_c,i + max_{y∈Y} E_i·y|)``

    which is two LPs per metabolite over a **fixed** ``Y`` — only the objective changes, so every
    solve after the first is a warm start off the retained basis (M3's validated pattern).

    **Why not a cheap box bound.** Bounding each ``|y_k| ≤ ρ_k`` and summing ``Σ_k |E_ik|·ρ_k`` is
    sound but can be loose without limit, because ``Y`` is *not* a box — its coordinates are
    coupled.
    Codex's counterexample: ``Y = {|y₁| ≤ 1, |y₂| ≤ 1, |y₁ − y₂| ≤ δ}`` with ``E_i = (1, −1)`` has
    ``ρ₁ = ρ₂ = 1``, so the box bound is ``2`` while the true maximum is ``δ``. The ratio ``2/δ``
    has no bound in the dimension, and `step_scale_ratio` does not control it — it says nothing
    about
    how
    an ``E_i`` row aligns with a thin coupled direction of ``Y``. So the maximum is taken exactly.

    **The one conservatism, and why it is sound.** The contract's denominator is
    ``max((|S|·|v|)_i, 1)``, which is **always ≥ 1**. Hence ``R_i / max((|S|·|v|)_i, 1) ≤ R_i``, and
    proving ``R_i ≤ η`` proves the contract for *every* reachable ``v`` without ever computing the
    denominator. That collapses a per-metabolite constraint matrix (the exact relative form needs a
    row ``d ≥ Σ_j |S_ij|·p_j``, whose coefficients change with ``i``, defeating the warm start) into
    one fixed ``Y``. The price is that a model whose fluxes are large enough to earn a denominator
    ``≫ 1`` is held to a stricter bar than the contract demands. That errs toward **refusing** a
    good
    transform, never toward admitting a bad one — and on the example model the certificate passes
    with ~100× of margin, so the conservatism costs nothing that has been observed. If a strain ever
    fails *only* on this margin, the exact relative LP is the escalation, and `worst_absolute` says
    by how much.
    """

    worst_absolute: float
    """``max_i R_i`` — the largest mass-balance residual any reachable state can exhibit."""

    worst_row: int
    """Which metabolite attains it."""

    worst_row_id: str

    contract: float
    """``η``. The same number `diagnostics.feasibility_report` applies to emitted samples."""

    n_rows: int
    n_rows_certified: int
    """Rows needing an LP. A row with ``E_i`` structurally zero is bounded by ``|r_c,i|``
    directly."""

    n_lps: int
    elapsed_seconds: float
    polytope_key: str

    @property
    def is_certified(self) -> bool:
        """True when **every** reachable state meets the declared contract."""
        return bool(self.worst_absolute <= self.contract)

    @property
    def margin(self) -> float:
        """``η / worst_absolute`` — how much room the certificate had. Below 1 it failed."""
        return self.contract / self.worst_absolute if self.worst_absolute > 0.0 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "reachable_worst_absolute": self.worst_absolute,
            "reachable_worst_row": self.worst_row,
            "reachable_worst_row_id": self.worst_row_id,
            "reachable_contract": self.contract,
            "reachable_n_rows": self.n_rows,
            "reachable_n_rows_certified": self.n_rows_certified,
            "reachable_n_lps": self.n_lps,
            "reachable_elapsed_seconds": self.elapsed_seconds,
            "reachable_is_certified": self.is_certified,
            "reachable_margin": self.margin,
        }


def certify_reachable_mass_balance(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    contract: float = DEFAULT_MASS_BALANCE_CONTRACT,
    threads: int = 1,
) -> ReachabilityCertificate:
    """Prove that every state ``c + T·y`` the chain can reach satisfies the mass-balance contract.

    Two LPs per metabolite over the fixed reachable set ``Y``; see `ReachabilityCertificate` for the
    construction and the single conservatism. Raises `RoundingError` on a mismatched polytope — the
    M6 invariant: ``T`` and ``S`` must have been computed against each other or the bound is about
    two different models.

    This *replaces* the M5 gate on `RoundingDiagnostics.transform_mass_balance_error`, which was a
    componentwise backward error and rejected 8 of 24 RNG streams on a samplable polytope.
    """
    import time

    from gsmm_compiler.highs_backend import HighsLinearProgram  # local: keeps rounding solver-free

    if transform.polytope_key != reduced.content_key():
        raise RoundingError(
            "certifying a transform against a polytope it was not built from: "
            f"{transform.polytope_key[:16]}… vs {reduced.content_key()[:16]}…"
        )
    if contract <= 0.0:
        raise RoundingError(f"the mass-balance contract must be positive, got {contract}")

    start = time.perf_counter()
    matrix = np.asarray(transform.transform, dtype=VALUE_DTYPE)
    centre = np.asarray(transform.center, dtype=VALUE_DTYPE)
    n_rows = int(reduced.stoichiometry.n_rows)

    # E = S·T, column by column: S is CSC and `matvec` is the one primitive that exists.
    energy = np.empty((n_rows, transform.dimension), dtype=VALUE_DTYPE)
    for k in range(transform.dimension):
        energy[:, k] = reduced.stoichiometry.matvec(np.ascontiguousarray(matrix[:, k]))
    residual_at_centre = reduced.stoichiometry.matvec(centre) - reduced.rhs

    if transform.dimension == 0:
        # A singleton polytope reaches exactly one state, so the centre's own residual is the bound.
        worst = np.abs(residual_at_centre)
        row = int(np.argmax(worst)) if worst.size else 0
        return ReachabilityCertificate(
            worst_absolute=float(worst.max(initial=0.0)),
            worst_row=row,
            worst_row_id=reduced.metabolite_ids[row] if worst.size else "",
            contract=contract,
            n_rows=n_rows,
            n_rows_certified=0,
            n_lps=0,
            elapsed_seconds=time.perf_counter() - start,
            polytope_key=transform.polytope_key,
        )

    # Y = {y : l − c ≤ T·y ≤ u − c}. Fixed for every solve; only the objective moves.
    row_lower = reduced.lower_bounds - centre
    row_upper = reduced.upper_bounds - centre
    omega = _reachable_coordinate_box(transform, row_lower, row_upper)

    program = HighsLinearProgram(
        matrix=NativeCSC.from_dense(matrix),
        col_lower=-omega,
        col_upper=omega,
        row_lower=row_lower,
        row_upper=row_upper,
        threads=threads,
        name="reachable_mass_balance",
    )

    worst_absolute, worst_row, n_lps, n_certified = 0.0, 0, 0, 0
    for row in range(n_rows):
        objective = energy[row]
        if not objective.any():
            # E_i is structurally zero: the direction space never touches this metabolite, so the
            # centre's own residual is the exact bound and no LP can say more. On the example model
            # 727 of 894 rows are like this, which is why the certificate costs 334 LPs, not 1788.
            bound = abs(float(residual_at_centre[row]))
        else:
            offset = float(residual_at_centre[row])
            high = offset + _reachable_extreme(program, matrix, objective, row_lower, row_upper,
            omega)
            low = offset - _reachable_extreme(
                program, matrix, -objective, row_lower, row_upper, omega
            )
            n_lps += 2
            n_certified += 1
            bound = max(abs(low), abs(high))

        if bound > worst_absolute:
            worst_absolute, worst_row = bound, row

    return ReachabilityCertificate(
        worst_absolute=worst_absolute,
        worst_row=worst_row,
        worst_row_id=reduced.metabolite_ids[worst_row],
        contract=contract,
        n_rows=n_rows,
        n_rows_certified=n_certified,
        n_lps=n_lps,
        elapsed_seconds=time.perf_counter() - start,
        polytope_key=transform.polytope_key,
    )


def _reachable_coordinate_box(
    transform: RoundedTransform,
    row_lower: NDArray[np.float64],
    row_upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A **provable outer box** ``|y_k| ≤ Ω_k`` containing the reachable set ``Y``.

    Needed because ``y`` is free in ``Y = {y : lo ≤ T·y ≤ hi}``, and the weak-duality bound on
    ``max E_i·y`` carries a term ``Σ_k |d_k|·Ω_k`` for the stationarity residual ``d = E_i − Tᵀπ``.
    With ``y`` unbounded, *any* ``d ≠ 0`` sends the dual bound to ``+∞`` — and ``d`` is never
    exactly
    zero in float64. So the box is not a convenience; without it there is no finite rigorous bound.

    ``T`` has full column rank (`_check_full_column_rank`), so ``T⁺·T = I`` and for ``y ∈ Y`` with
    ``a = T·y ∈ [lo, hi]``::

        y = T⁺·a + (I − T⁺T)·y
        ‖y‖_∞ ≤ ‖T⁺‖_∞·‖a‖_∞ + ‖I − T⁺T‖_∞·‖y‖_∞
        ‖y‖_∞ ≤ ‖T⁺‖_∞·A / (1 − ‖I − T⁺T‖_∞),      A = max_j max(|lo_j|, |hi_j|)

    **The direction of error matters and runs the safe way.** A box that is too *large* only loosens
    the ``Σ|d_k|Ω_k`` term — and since ``|d| ~ 1e-16``, an Ω generous by orders costs nothing
    measurable. A box too *small* would cut off part of ``Y``, understate the maximum, and certify a
    transform that reaches further: unsound. So Ω is doubled outright, and the residual
    ``‖I − T⁺T‖_∞`` is checked rather than assumed.

    **``T⁺`` is recomputed here rather than read from `RoundedTransform.inverse_transform`.** The
    certificate's subject is ``T``; a bound derived from a *stored* inverse would be a claim about
    whatever pair of arrays happens to be in the dataclass, and a ``T`` that had drifted from its
    recorded inverse would get a box built for a different matrix. Same principle as
    `RoundedTransform.from_bundle` rebuilding the precompute from ``T`` instead of trusting a cached
    copy: an artifact does not get to vouch for itself. One SVD on a ``(n_free, d)`` matrix is
    nothing beside the LPs that follow.
    """
    matrix = np.asarray(transform.transform, dtype=VALUE_DTYPE)
    pseudo_inverse = np.linalg.pinv(matrix)

    identity_error = float(
        np.abs(pseudo_inverse @ matrix - np.eye(transform.dimension)).sum(axis=1).max()
    )
    if identity_error >= 0.5:
        raise RoundingError(
            f"‖I − T⁺T‖_∞ = {identity_error:.3e} — the stored pseudo-inverse does not invert the "
            "transform, so no outer box on the reachable coordinates can be derived from it"
        )

    reach = np.maximum(np.abs(row_lower), np.abs(row_upper))
    box = 2.0 * (np.abs(pseudo_inverse) @ reach) / (1.0 - identity_error)

    # A degenerate polytope (every bound at the centre) gives reach = 0 and a box of zeros, which is
    # a legitimate answer — Y is the single point y = 0 — but HiGHS wants a nondegenerate column.
    return np.asarray(np.maximum(box, 1.0), dtype=VALUE_DTYPE)


def _reachable_extreme(
    program: Any,
    matrix: NDArray[np.float64],
    objective: NDArray[np.float64],
    row_lower: NDArray[np.float64],
    row_upper: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> float:
    """A **rigorous upper bound** on ``max_{y ∈ Y} objective·y``, from weak duality.

    **Never from the primal.** M4 established this and named the next two milestones as the places
    the temptation would return: a returned ``objective_value`` is a *lower* bound on the maximum —
    it is the value at one point — so a solve that stops short of optimality reports the reachable
    residual too **small** and certifies a transform that actually reaches further. That is the
    unsound direction. Weak duality assumes nothing about the returned point: not optimality, not
    even feasibility.

    For ``max e·y s.t. lo ≤ T·y ≤ hi, |y| ≤ Ω`` and **any** row multipliers ``π``::

        e·y = πᵀ(T·y) + (e − Tᵀπ)·y  ≤  Σ_j max(π_j·lo_j, π_j·hi_j) + Σ_k |d_k|·Ω_k

    since each row activity lies in ``[lo_j, hi_j]`` and each ``|y_k| ≤ Ω_k``. ``d`` is recomputed
    from ``e − Tᵀπ`` rather than read from HiGHS's reduced costs, so a stationarity residual in the
    solver's own arrays cannot leak into the bound.

    **The objective is normalized first, and that is load-bearing.** ``E_i`` is ~1e-13 on a real
    model; handed to HiGHS raw it sits below the dual feasibility tolerance, every reduced cost
    reads as zero, and the solver returns whatever vertex it started from — duals included. Measured
    directly: with a 1e-10 coefficient beside 1.0 coefficients HiGHS **drops the row from the scaled
    matrix entirely**, reporting a row activity of 0.0 where the truth is 133.3, and calls it
    feasible with zero primal infeasibility. (That relaxation happens to enlarge ``Y`` and so stays
    conservative — but relying on which way a solver's scaling errs is not a certificate.)
    """
    norm = float(np.abs(objective).max())
    unit = objective / norm
    solution = program.maximize(unit)

    duals = np.asarray(solution.row_duals, dtype=VALUE_DTYPE)
    if not np.all(np.isfinite(duals)):
        raise RoundingError("HiGHS returned non-finite row duals; no bound can be built from them")

    stationarity = unit - matrix.T @ duals
    rows = np.maximum(duals * row_lower, duals * row_upper)
    box = np.abs(stationarity) * omega
    value = float(rows.sum() + box.sum())

    # Every operation above rounds to nearest, so the computed number can land *below* the exact one
    # — and a bound that can be too small is not a bound (M4). An outward allowance covers forming
    # Tᵀπ (n_free products per column), the subtraction, the two products, and the two pairwise
    # sums.
    eps = float(np.finfo(VALUE_DTYPE).eps)
    depth = float(np.log2(max(matrix.shape[0], 2))) + 1.0
    transpose_error = eps * float(matrix.shape[0]) * (np.abs(matrix).T @ np.abs(duals))
    allowance = float(
        eps * depth * (float(np.abs(rows).sum()) + float(np.abs(box).sum()))
        + float(((transpose_error + eps * np.abs(stationarity)) * omega).sum())
    )

    bound = (value + allowance) * norm
    if not np.isfinite(bound):
        raise RoundingError("the reachable dual bound is not finite; the LP's duals are unusable")
    return bound
