"""The batch runner: a models manifest → per-model geometry (cached) → a global pool over units.

This is the orchestration layer BUILD_PLAN §1.1–§1.2 asks for. It reads a *models manifest* (one row
per strain), and for each model runs the pipeline the M6/M7 integration tests wire by hand — parse
(L0), reduce (L1), resolve the objective + LP optimum + ``s_J`` (L2), build the geometry and rounded
transform (L3), then sample the β-ladder — writing a per-strain result directory plus, at the end,
the cross-model tables that answer §2's comparative question.

Two properties matter and shape everything here:

* **The expensive, β-independent stage is the geometry**, so it goes through the content-addressed
  cache (`cache.ArtifactCache`): a re-run with more β rungs, or a second strain sharing a polytope,
  reuses it, and two concurrent jobs never both compute it.

* **The sampling of one ``(model, β, chain)`` unit needs no solver**, so it runs in a worker process
  that receives only frozen NumPy arrays (a pickled `RoundedTransform` + `ReducedPolytope` + the
  objective) and *never imports cobra or HiGHS* (§1.2). The parent does all the parsing and LP work;
  workers only walk the chain and write their own files. Restart is per-unit: a unit whose directory
  already carries a ``COMPLETE`` marker is skipped, so a killed batch resumes only what it lost.

Nothing at module scope imports cobra or HiGHS — both are pulled lazily inside `prepare_model`,
which runs only in the parent — so this module is safe for a ``spawn``-ed worker to import.

Implemented in **M8** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.affine_geometry import (
    GEOMETRY_IMPL_VERSION,
    ReducedGeometry,
    build_geometry,
)
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.calibration import calibrate
from gsmm_compiler.config import Config
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.maxent_sampler import (
    SAMPLER_IMPL_VERSION,
    movable_reactions,
    run_chain,
    trace_objective,
)
from gsmm_compiler.output import (
    OUTPUT_IMPL_VERSION,
    RunLayout,
    SampleStorage,
    is_complete,
    read_json,
    write_json,
)
from gsmm_compiler.provenance import Provenance, content_key
from gsmm_compiler.rounding import (
    ROUNDING_IMPL_VERSION,
    ReachabilityCertificate,
    RoundedTransform,
    build_transform,
    certify_reachable_mass_balance,
    require_certified_transform,
)
from gsmm_compiler.sparse_objective import (
    ReducedObjective,
    lower_objective,
    resolve_objective,
    solve_sparse_objective,
)

_GEOMETRY_ARRAY_NAMES = (
    "scaling",
    "basis",
    "center",
    "support_points",
)

_TRANSFORM_ARRAY_NAMES = (
    "transform",
    "inverse_transform",
    "cholesky",
    "center",
    "support_coordinates",
)

_WORKER_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class BatchError(RuntimeError):
    """The batch manifest is malformed, or a model could not be prepared for sampling."""


# ---- the models manifest -----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One row of the models manifest: a strain to sample (mirrors ``strains.tsv``, §1.1)."""

    model_path: str
    biomass_id: str | None = None
    model_id: str | None = None
    """An explicit id for the result directory; defaults to the parsed model's own id."""


def load_models_manifest(path: str | Path) -> list[ModelSpec]:
    """Read a models manifest — ``.json`` (a list or ``{"models": [...]}``) or a ``.tsv`` + header.

    The TSV mirrors ``metabolicSubcommunities/metadata/strains.tsv``: a header row naming columns,
    of which ``model_path`` is required and ``biomass_id`` / ``model_id`` are optional.
    """
    source = Path(path)
    if not source.is_file():
        raise BatchError(f"models manifest not found: {source}")

    if source.suffix.lower() == ".json":
        rows = _rows_from_json(read_json(source))
    elif source.suffix.lower() in {".tsv", ".txt"}:
        rows = _rows_from_tsv(source.read_text())
    else:
        raise BatchError(f"unsupported manifest format {source.suffix!r} (expected .json or .tsv)")

    specs = [_spec_from_row(row, base=source.parent) for row in rows]
    if not specs:
        raise BatchError(f"models manifest {source} lists no models")
    _reject_duplicate_ids(specs)
    return specs


def _rows_from_json(payload: Any) -> list[dict[str, Any]]:
    rows = payload["models"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise BatchError("JSON manifest must be a list of rows or an object with a 'models' list")
    return rows


def _rows_from_tsv(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    header = lines[0].split("\t")
    if "model_path" not in header:
        raise BatchError("TSV manifest header must include a 'model_path' column")
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append({key: value for key, value in zip(header, values, strict=False) if value != ""})
    return rows


def _spec_from_row(row: dict[str, Any], *, base: Path) -> ModelSpec:
    if "model_path" not in row or not row["model_path"]:
        raise BatchError(f"manifest row is missing 'model_path': {row}")
    model_path = Path(row["model_path"])
    if not model_path.is_absolute():
        model_path = base / model_path
    return ModelSpec(
        model_path=str(model_path),
        biomass_id=row.get("biomass_id") or None,
        model_id=row.get("model_id") or None,
    )


def _reject_duplicate_ids(specs: Sequence[ModelSpec]) -> None:
    """Two rows landing in the same result directory would overwrite each other's samples."""
    seen: dict[str, str] = {}
    for spec in specs:
        key = spec.model_id or Path(spec.model_path).stem
        if key in seen:
            raise BatchError(
                f"two manifest rows resolve to model_id {key!r} ({seen[key]} and "
                f"{spec.model_path}); give one an explicit distinct model_id"
            )
        seen[key] = spec.model_path


# ---- per-model preparation (parent-only: this is the sole cobra/HiGHS path) ---------------------


@dataclass(frozen=True)
class ModelPlan:
    """Everything a model's ``(β, chain)`` units need — frozen arrays, all picklable to workers."""

    model_id: str
    reduced: ReducedPolytope
    transform: RoundedTransform
    objective: ReducedObjective
    energy_scale_value: float
    j_star: float
    optimum_coordinates: NDArray[np.float64] | None
    movable: NDArray[np.intp]
    storage: SampleStorage
    sampler: Any  # SamplerConfig
    near_zero_thresholds: tuple[float, ...]
    feasibility_tol: float
    reports: dict[str, Any] = field(default_factory=dict)
    """The heavy human-facing manifests (model report, objective, geometry, energy scale) — kept out
    of the pickled worker payload by living only on the parent's plan, never on a `SampleJob`."""


def prepare_model(
    spec: ModelSpec, config: Config, *, cache: ArtifactCache | None = None
) -> ModelPlan:
    """Run L0→L3, calibrate, and return the frozen artifacts its chains sample from.

    Imports cobra lazily (parsing is the one place it is needed) so the module stays worker-safe.
    The geometry — the expensive, β-independent stage — goes through the cache; the rest is cheap
    enough to recompute each run.

    **Every run goes through `calibration.calibrate`, with no branch** (M10.2). With the pilot
    switches off it returns ``T₀`` and the support-vertex ``s_J`` unchanged, so v1's numbers keep
    v1's labels; with them on it runs the pilot DAG. Routing the default path through the same call
    is deliberate — a second, uncalibrated code path is how the two would drift apart, and this
    package has spent two milestones on artifacts that never met.
    """
    from gsmm_compiler.model_input import load_canonical_model  # cobra; parent-only

    biomass_id = spec.biomass_id or config.model.biomass_id
    canonical = load_canonical_model(spec.model_path, biomass_id)
    model_id = spec.model_id or canonical.model_id
    reduced = canonical.polytope.reduce()

    resolved = resolve_objective(canonical.polytope, reduced, config.objective)
    lowered = lower_objective(reduced, resolved.objective)
    solution = solve_sparse_objective(reduced, resolved.objective)

    geometry, bootstrap, geometry_meta = _load_or_build_geometry(
        reduced, config, model_id=model_id, cache=cache
    )

    calibration = calibrate(
        geometry,
        reduced,
        bootstrap,
        lowered,
        model_id=model_id,
        optimum=solution.optimum,
        sampler=config.sampler,
        # The proof `_load_or_build_geometry` already checked, handed on rather than recomputed:
        # the pilots are chains stepping in T₀'s frame, so T₀ must be certified before them.
        bootstrap_certificate=ReachabilityCertificate.from_cache(
            geometry_meta["reachability_certificate"]
        ),
        geometry_config=config.geometry,
        # In T₀'s frame; `calibrate` returns it in the final transform's frame. Production only —
        # it never reaches a pilot (`calibration.NeutralPilot`).
        optimum_coordinates=bootstrap.to_coordinates(reduced.to_reduced(solution.optimum.v_full)),
        # The same store the geometry came from. The pilots are the expensive node in the DAG —
        # 19.3 s measured against geometry's 1.17 s — so a `--cache-dir` run that cached the cheap
        # stage and recomputed the expensive one was 16.4× upside-down (§1.6.7, M10.2b).
        cache=cache,
    )
    transform = calibration.transform
    scale = calibration.energy_scale
    optimum_coordinates = calibration.optimum_coordinates

    return ModelPlan(
        model_id=model_id,
        reduced=reduced,
        transform=transform,
        objective=lowered,
        energy_scale_value=scale.value,
        j_star=scale.j_star,
        optimum_coordinates=(
            None
            if optimum_coordinates is None
            else np.ascontiguousarray(optimum_coordinates, dtype=np.float64)
        ),
        movable=movable_reactions(transform),
        storage=SampleStorage.from_config(config.output),
        sampler=config.sampler,
        near_zero_thresholds=config.objective.near_zero_thresholds,
        feasibility_tol=config.geometry.feasibility_tol,
        reports={
            "model_report": canonical.report(),
            "objective": lowered.manifest(),
            "geometry": geometry_meta["geometry_manifest"],
            "energy_scale": scale.manifest(),
            "calibration": calibration.manifest(),
            # The certificate of the transform **this run samples**, in report form. Under
            # re-rounding that is T₁, not T₀ — reporting the bootstrap's certificate here while
            # production steps in T₁'s frame would be this package's signature bug (a manifest
            # describing an artifact that was not used), committed in the milestone about it.
            "reachability_certificate": calibration.certificate_report(),
            "lp_optimum": solution.diagnostics(),
            "config": config.as_dict(),
            "provenance": Provenance.capture().as_dict(),
            # The reaction axis, so cross-model aggregation can label full-length flux columns and
            # pick out exchanges without re-parsing the model (which would need cobra).
            "axes": {
                "reaction_ids": list(canonical.polytope.reaction_ids),
                "exchange_mask": canonical.exchange_mask.astype(bool).tolist(),
                "n_chains": int(config.sampler.n_chains),
                "activity_threshold": float(min(config.objective.near_zero_thresholds)),
            },
        },
    )


def sample_recipe_key(plan: ModelPlan, *, beta_index: int, chain_index: int) -> str:
    """§1.1's **sample artifact key**: ``L2 + L3 + β + chain seed coords + sampler version +
    burn/thin/n_samples``.

    §1.1 has always specified this key. Nothing computed it. Restart skipped a unit on its
    ``COMPLETE`` marker alone, and `output.store_chain` recorded only the ``polytope_key`` — so a
    results directory reused after *any* change that moves the numbers (a new pilot seed, a longer
    schedule, `energy_scale` switched to ``pilot_sd``) resumed the units it still had and sampled
    the rest **from a different law**. The per-strain tree would then hold two experiments, the
    cross-model aggregation would stack them into one table, and every diagnostic would pass:
    each chain is individually correct, which is exactly what makes the mixture invisible.

    M10 forced the issue rather than created it. Before the pilot DAG, ``T`` and ``s_J`` were pure
    functions of the polytope and the config; now they are functions of a *pilot*, so two runs of
    an unchanged config against an unchanged model can legitimately disagree — and the marker could
    not tell.

    **The criterion here is artifact identity, not target identity, and they are different keys.**
    M7 settled that ``optimum_coordinates`` is *deliberately not keyed* on the objective and
    ``s_J``: it is a start hint, it enters only the initial state, and a wrong one cannot move the
    invariant target — only seed a poorer start. That reasoning is correct and it does not belong
    here. This key asks "are these bytes the same artifact?", and by that question the hint is no
    different from `seed` or `chain_index`, which are hashed and define no law either. (Codex,
    M10.2 review round 3 — and it is the identical defect this milestone fixed for the pilots, so
    excluding it here while fixing it there was incoherent.)
    """
    return content_key(
        layer="samples",
        # L2: the objective, and the two numbers `s_J`'s subtraction produced (BUILD_PLAN §1.6.5).
        # `energy_scale_value` is the pilot's fingerprint on production: under `pilot_sd` it is
        # σ̂₀, so a re-run with a different pilot lands here and nowhere else.
        objective_key=plan.objective.objective_key,
        energy_scale_value=float(plan.energy_scale_value),
        j_star=float(plan.j_star),
        # L3: `T` itself, not the settings that were meant to produce it. Under re-rounding this is
        # `T₁`, whose identity no configuration predicts.
        polytope_key=plan.reduced.content_key(),
        transform_key=plan.transform.content_key(),
        # The unit, and the stream that draws it. `model_id` is here because `provenance.
        # stream_seed` keys the RNG on it — a run that overrides `--model-id` against the same file
        # draws different numbers, and only this names that.
        model_id=plan.model_id,
        beta=float(plan.sampler.betas[beta_index]),
        beta_index=int(beta_index),
        chain_index=int(chain_index),
        seed=int(plan.sampler.seed),
        # The start hint — see the note above on artifact vs target identity. It moves the draws
        # (it changes the support hull's cardinality, so the Dirichlet dimension, so RNG
        # consumption on every later transition), and under re-rounding it is expressed in T₁'s
        # frame, so it is not even redundant with the transform key.
        optimum_coordinates=plan.optimum_coordinates,
        # Consumed by start selection and chord validation, so it can move the draws.
        feasibility_tol=float(plan.feasibility_tol),
        # These change the stored `trace_near_zero_counts` arrays — bytes of this artifact.
        near_zero_thresholds=list(plan.near_zero_thresholds),
        # The schedule — and `refresh_interval`, because M5 settled that in float64 the state is
        # `(y, cache error, refresh phase)`, so it moves the draws, not just their bookkeeping.
        burn_in=int(plan.sampler.burn_in),
        n_samples=int(plan.sampler.n_samples),
        thin=int(plan.sampler.thin),
        refresh_interval=int(plan.sampler.refresh_interval),
        # Storage decides the *bytes on disk*: M9 measured that `reduced` round-trips to 1.1e-13 and
        # not bit-exactly (gemv per row vs gemm per block), so byte-identity holds within a mode and
        # not across one — two modes in one directory are two different artifacts.
        store_mode=plan.storage.mode,
        store_flux_dtype=plan.storage.flux_dtype,
        # Two versions, because two different pieces of code decide these bytes:
        # `SAMPLER_IMPL_VERSION` covers the transition kernel that draws them (its docstring says
        # so), `OUTPUT_IMPL_VERSION` the writer that lays them down. An output-only change moves
        # only the second, and used to move neither.
        sampler_impl_version=SAMPLER_IMPL_VERSION,
        output_impl_version=OUTPUT_IMPL_VERSION,
        # `movable` is deliberately absent, and it is the one exclusion that survives the criterion
        # above: `movable_reactions(transform)` is an exact function of a transform already hashed
        # here, so hashing it too would only assert the same fact twice.
    )


def geometry_cache_key(reduced: ReducedPolytope, config: Config, *, model_id: str) -> str:
    """The L3 lookup key: the polytope, the geometry/rounding settings, the seed, and the code.

    ``model_id`` is folded in because it seeds the geometry's RNG (the random-probe pre-pass), so
    two strains that happen to share a polytope still get their *own* geometry bytes — a cached
    artifact must reproduce a fresh build exactly, and a false miss only recomputes (a false hit
    corrupts, §1.1).
    """
    return content_key(
        layer="L3",
        polytope_key=reduced.content_key(),
        model_id=model_id,
        geometry_config=asdict(config.geometry),
        seed=int(config.sampler.seed),
        geometry_impl_version=GEOMETRY_IMPL_VERSION,
        rounding_impl_version=ROUNDING_IMPL_VERSION,
        numpy_version=np.__version__,
    )


def build_l3_bundle(
    reduced: ReducedPolytope, config: Config, *, model_id: str
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Build L3 from scratch and return its cache bundle: **the one writer of this schema.**

    Before M10.2 there were two. This function's body lived in a closure here, while
    `cli._cmd_maxent_build_geometry --cache-dir` assembled its *own* bundle under the *same*
    `geometry_cache_key` — and the two disagreed: the CLI's meta carried no
    ``reachability_certificate``, and it stored the bundle even when the certificate came back
    **REFUSED**. Since `batch` gated only inside its miss path, warming the cache from the CLI and
    then sampling walked M9's gate straight past. Two writers of one schema is the defect; one
    writer is the fix, so the drift has nowhere to live.

    The certificate is computed here rather than inside `build_transform` for the reason M4's span
    certificate is its own step: it costs LPs, and `rounding` stays solver-free so a worker can
    import it (BUILD_PLAN §1.2, §1.4.2).
    """
    geometry = build_geometry(reduced, model_id=model_id, config=config.geometry)
    transform = build_transform(geometry, reduced, config=config.geometry)
    certificate = certify_reachable_mass_balance(transform, reduced)
    require_certified_transform(certificate, transform, reduced)

    geometry_arrays, geometry_meta = geometry.to_bundle()
    transform_arrays, transform_meta = transform.to_bundle()
    return (
        # Namespaced, because both artifacts have a `center` and a flat merge would silently keep
        # one of them. Each is then rebuilt and re-keyed by its own class, so a divergence between
        # the two centres is caught by whichever key it breaks — not hidden by a dedupe.
        {
            **{f"geometry_{name}": array for name, array in geometry_arrays.items()},
            **transform_arrays,
        },
        {
            "geometry": geometry_meta,
            "transform": transform_meta,
            "geometry_manifest": geometry.manifest(),
            "reachability_certificate": certificate.to_cache(),
        },
    )


def _load_or_build_geometry(
    reduced: ReducedPolytope, config: Config, *, model_id: str, cache: ArtifactCache | None
) -> tuple[ReducedGeometry, RoundedTransform, dict[str, Any]]:
    """Return ``(geometry, T₀, meta)``, from the cache if present — certificate enforced either way.

    Returns the **geometry**, not just the transform. That is M10.2's enabling change: `calibration`
    needs ``B`` and ``s`` to re-round from a pilot, and a cache hit used to hand back a transform
    from which they cannot be recovered without inverting ``L₀``.
    """
    if cache is None:
        arrays, meta = build_l3_bundle(reduced, config, model_id=model_id)
    else:
        key = geometry_cache_key(reduced, config, model_id=model_id)
        artifact = cache.get_or_compute(
            "L3", key, lambda: build_l3_bundle(reduced, config, model_id=model_id)
        )
        arrays, meta = artifact.arrays, artifact.meta

    geometry = ReducedGeometry.from_bundle(
        {name: arrays[f"geometry_{name}"] for name in _GEOMETRY_ARRAY_NAMES},
        meta["geometry"],
        reduced,
    )
    transform = RoundedTransform.from_bundle(
        {name: arrays[name] for name in _TRANSFORM_ARRAY_NAMES}, meta["transform"], reduced
    )
    # On **every** path, not just the miss. `build_l3_bundle` already gated what it built, so this
    # line exists for the cache hit — where nothing previously looked at the certificate at all,
    # and a bundle written by any other process was simply believed.
    require_certified_transform(
        ReachabilityCertificate.from_cache(meta["reachability_certificate"]), transform, reduced
    )
    return geometry, transform, meta


# ---- one sampling unit (worker-side: no cobra, no HiGHS) ----------------------------------------


@dataclass(frozen=True)
class SampleJob:
    """The frozen payload for one ``(model, β, chain)`` unit, pickled to a worker."""

    chain_dir: Path
    model_id: str
    beta: float
    beta_index: int
    chain_index: int
    reduced: ReducedPolytope
    transform: RoundedTransform
    objective: ReducedObjective
    energy_scale_value: float
    j_star: float
    optimum_coordinates: NDArray[np.float64] | None
    movable: NDArray[np.intp]
    storage: SampleStorage
    sampler: Any
    near_zero_thresholds: tuple[float, ...]
    feasibility_tol: float
    recipe_key: str = ""
    """`sample_recipe_key` for this unit — written into its manifest, and what restart matches."""


def _already_done(job: SampleJob) -> bool:
    """Is this unit's work already on disk — **from this same experiment**? (§1.1 restart.)

    ``COMPLETE`` alone answers the wrong question. It says "a chain finished here", not "*this*
    chain finished here", and M10 makes the difference material: with re-rounding on, ``T₁`` and
    ``s_J`` come from a pilot, so two runs of one unchanged config against one unchanged model can
    honestly differ. A marker-only check would resume half a ladder from the old law and sample the
    rest from the new one, into the same directory, and the cross-model table would average the two.

    Refuse rather than recompute. A results tree is the user's output, not a cache: an unreadable
    or foreign unit could be silently overwritten, but that destroys a run someone may still want,
    and this package's rule is to refuse where a wrong number would otherwise be reported (M9).
    """
    if not is_complete(job.chain_dir):
        return False

    manifest_path = job.chain_dir / "manifest.json"
    try:
        found = str(read_json(manifest_path).get("recipe_key", ""))
    except (OSError, ValueError) as exc:
        raise BatchError(
            f"{job.chain_dir} is marked COMPLETE but its manifest cannot be read ({exc}); refusing "
            "to treat it as this run's work"
        ) from exc

    if found == job.recipe_key:
        return True
    raise BatchError(
        f"{job.chain_dir} is marked COMPLETE but was sampled by a different recipe "
        f"({found[:16] or '<none>'}… vs {job.recipe_key[:16]}…): its transform, objective, s_J, "
        "schedule, seed or storage mode differs from this run's. Resuming would leave two "
        "experiments in one directory and stack them into one cross-model table, with every "
        "per-chain diagnostic green. Point --out at a new directory, or remove the stale unit."
    )


def run_sample_unit(job: SampleJob) -> dict[str, Any]:
    """Walk one chain, trace ``J`` exactly from the stored fluxes, and write the unit — atomically.

    Runs in a worker process. Touches only the frozen arrays it was handed: `run_chain` at ``β = 0``
    ignores the objective entirely, and at ``β > 0`` reads it from the reduced `L1Objective`. No
    solver is ever imported here — the M8 "zero HiGHS solves in the inner loop" invariant, enforced
    by construction rather than by a counter.
    """
    chain = run_chain(
        job.transform,
        job.reduced,
        config=job.sampler,
        model_id=job.model_id,
        beta=job.beta,
        beta_index=job.beta_index,
        chain_index=job.chain_index,
        objective=job.objective.line,
        energy_scale=job.energy_scale_value,
        optimum_coordinates=job.optimum_coordinates,
        feasibility_tol=job.feasibility_tol,
    )
    trace = trace_objective(
        chain.fluxes,
        job.objective,
        j_star=job.j_star,
        energy_scale=job.energy_scale_value,
        thresholds=job.near_zero_thresholds,
        movable=job.movable,
    )
    from gsmm_compiler.output import store_chain  # local: keeps the worker import surface small

    manifest = store_chain(
        job.chain_dir,
        chain=chain,
        trace=trace,
        reduced=job.reduced,
        model_id=job.model_id,
        beta=job.beta,
        beta_index=job.beta_index,
        chain_index=job.chain_index,
        storage=job.storage,
        recipe_key=job.recipe_key,
    )
    return {
        "model_id": job.model_id,
        "beta": job.beta,
        "beta_index": job.beta_index,
        "chain_index": job.chain_index,
        "diagnostics": manifest["diagnostics"],
        "trace_summary": manifest["trace_summary"],
    }


def _jobs_for(plan: ModelPlan, layout: RunLayout) -> Iterator[SampleJob]:
    for beta_index, beta in enumerate(plan.sampler.betas):
        for chain_index in range(plan.sampler.n_chains):
            yield SampleJob(
                chain_dir=layout.chain_dir(plan.model_id, beta_index, chain_index),
                model_id=plan.model_id,
                beta=float(beta),
                beta_index=beta_index,
                chain_index=chain_index,
                reduced=plan.reduced,
                transform=plan.transform,
                objective=plan.objective,
                energy_scale_value=plan.energy_scale_value,
                j_star=plan.j_star,
                optimum_coordinates=plan.optimum_coordinates,
                movable=plan.movable,
                storage=plan.storage,
                sampler=plan.sampler,
                near_zero_thresholds=plan.near_zero_thresholds,
                feasibility_tol=plan.feasibility_tol,
                recipe_key=sample_recipe_key(
                    plan, beta_index=beta_index, chain_index=chain_index
                ),
            )


# ---- the batch --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOutcome:
    """What became of one strain in a batch run."""

    model_id: str
    status: str  # "complete" | "failed"
    n_units: int
    n_completed: int
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    batch_dir: Path
    outcomes: tuple[ModelOutcome, ...]

    @property
    def completed_model_ids(self) -> tuple[str, ...]:
        return tuple(o.model_id for o in self.outcomes if o.status == "complete")


def run_batch(
    specs: Sequence[ModelSpec],
    config: Config,
    *,
    batch_name: str,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    n_workers: int = 1,
) -> BatchResult:
    """Run every strain in ``specs``, resuming any whose units already carry ``COMPLETE`` markers.

    A model that fails to *prepare* (a bad file, an infeasible objective) is recorded and skipped,
    so the batch still produces valid cross-model tables over the strains that finished (§1.1).
    Sampling units run in one global process pool when ``n_workers > 1``; ``n_workers == 1`` runs
    them in-process, which keeps the deterministic tests free of multiprocessing.
    """
    root = Path(output_root) if output_root is not None else Path(config.output.directory)
    layout = RunLayout(root=root, batch=batch_name)
    cache = ArtifactCache(Path(cache_dir)) if cache_dir is not None else None

    _limit_thread_env()
    executor = None
    if n_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_limit_thread_env,
        )

    outcomes: list[ModelOutcome] = []
    try:
        for spec in specs:
            outcomes.append(
                _run_one_model(spec, config, layout=layout, cache=cache, executor=executor)
            )
    finally:
        if executor is not None:
            executor.shutdown()

    from gsmm_compiler.features import aggregate_cross_model  # local: avoids an import cycle

    aggregate_cross_model(layout, [o.model_id for o in outcomes if o.status == "complete"])

    layout.batch_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        layout.batch_dir / "batch_manifest.json",
        {
            "batch": batch_name,
            "n_models": len(specs),
            "models": [asdict(o) for o in outcomes],
            "provenance": Provenance.capture().as_dict(),
        },
    )
    return BatchResult(batch_dir=layout.batch_dir, outcomes=tuple(outcomes))


def _run_one_model(
    spec: ModelSpec,
    config: Config,
    *,
    layout: RunLayout,
    cache: ArtifactCache | None,
    executor: ProcessPoolExecutor | None,
) -> ModelOutcome:
    try:
        plan = prepare_model(spec, config, cache=cache)
    except Exception as exc:  # a bad strain must not sink the batch
        model_id = spec.model_id or Path(spec.model_path).stem
        return ModelOutcome(
            model_id=model_id,
            status="failed",
            n_units=0,
            n_completed=0,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    jobs = list(_jobs_for(plan, layout))
    try:
        pending = [job for job in jobs if not _already_done(job)]
    except BatchError as exc:
        return ModelOutcome(
            model_id=plan.model_id, status="failed", n_units=len(jobs), n_completed=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    try:
        for result in _execute(pending, executor):
            summaries[(result["beta_index"], result["chain_index"])] = result
    except Exception as exc:
        return ModelOutcome(
            model_id=plan.model_id,
            status="failed",
            n_units=len(jobs),
            n_completed=len(jobs) - len(pending),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    # Fold in the units a previous run already finished (read back for the manifest).
    for job in jobs:
        key = (job.beta_index, job.chain_index)
        if key not in summaries:
            summaries[key] = _read_unit_summary(job.chain_dir)

    _write_model_manifest(plan, layout, summaries)

    from gsmm_compiler.diagnostics import write_run_diagnostics  # local: avoids an import cycle

    write_run_diagnostics(layout, plan.model_id)
    _mark_model_complete(plan.model_id, layout)
    return ModelOutcome(
        model_id=plan.model_id,
        status="complete",
        n_units=len(jobs),
        n_completed=len(jobs),
    )


def _execute(
    jobs: Sequence[SampleJob], executor: ProcessPoolExecutor | None
) -> Iterator[dict[str, Any]]:
    """Run the pending units — in the pool if given one, else in-process (deterministic tests)."""
    if executor is None:
        for job in jobs:
            yield run_sample_unit(job)
        return
    futures = [executor.submit(run_sample_unit, job) for job in jobs]
    for future in futures:
        yield future.result()


def _read_unit_summary(chain_dir: Path) -> dict[str, Any]:
    manifest = read_json(chain_dir / "manifest.json")
    return {
        "model_id": manifest["model_id"],
        "beta": manifest["beta"],
        "beta_index": manifest["beta_index"],
        "chain_index": manifest["chain_index"],
        "diagnostics": manifest["diagnostics"],
        "trace_summary": manifest["trace_summary"],
    }


def _write_model_manifest(
    plan: ModelPlan, layout: RunLayout, summaries: dict[tuple[int, int], dict[str, Any]]
) -> None:
    units = [summaries[key] for key in sorted(summaries)]
    ladder = [
        {
            "beta": float(beta),
            "beta_index": beta_index,
            "mean_j": float(
                np.mean(
                    [
                        u["trace_summary"]["mean_j"]
                        for u in units
                        if u["beta_index"] == beta_index
                    ]
                )
            ),
        }
        for beta_index, beta in enumerate(plan.sampler.betas)
    ]
    layout.model_dir(plan.model_id).mkdir(parents=True, exist_ok=True)
    write_json(
        layout.model_manifest_path(plan.model_id),
        {
            "model_id": plan.model_id,
            "n_units": len(units),
            "betas": [float(b) for b in plan.sampler.betas],
            "store_mode": plan.storage.mode,
            "ladder": ladder,
            "units": units,
            **plan.reports,
        },
    )


def _mark_model_complete(model_id: str, layout: RunLayout) -> None:
    marker = layout.model_complete_marker(model_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "wb") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _limit_thread_env() -> None:
    """Pin BLAS/OpenMP to one thread. Set in the parent *before* the spawn pool starts, so the
    freshly-imported NumPy in each spawned worker inherits it — the real oversubscription risk in
    solver-free workers is nested BLAS threads, not HiGHS (§1.2)."""
    for name in _WORKER_THREAD_ENV:
        os.environ.setdefault(name, "1")
