"""Model parsing layer (cobra) — load a GSMM, validate it, freeze it into the canonical IR.

cobra is a **parser/metadata layer only**: nothing in the numerical core may import this module, and
this is the one place allowed to touch cobra/optlang. Everything downstream sees NumPy arrays and
frozen tuples, never a cobra object.

This is the **L0** layer of the cache DAG (BUILD_PLAN §1.1): the raw file hash alone is not a
sufficient key, because parser semantics decide what the arrays end up containing — so the L0 key
folds in the cobra version and the parser schema version too.

## L0 needs **two** keys, and that is not a hedge  *(M10.2d)*

Every other layer has one. L0 has a chicken-and-egg the others do not: M8 settled that its identity
is **content-addressed** — a fingerprint of the IR the model *actually holds* — and you cannot
fingerprint a model's contents without parsing it, which is the 0.5 s this layer exists to skip. So:

* `model_lookup_key` — a function of the **inputs** (the file's bytes, the biomass override, the
  parser's and cobra's versions). Cheap: `hash_file` measures **1 ms**. Findable before the artifact
  exists, which is what `cache.ArtifactCache` requires of a lookup.
* `CanonicalModel.l0_key` — the **authority**, unchanged from M8, **re-derived on every load** and
  refused on mismatch. A lookup key is a cheap proxy; a content hash is the proof.

**The lookup key must be computable without importing cobra, and that is the whole point.** Measured
on the example model: `load_canonical_model` costs **1.157 s** on the first call and **0.52 s** on
later ones — the gap is cobra's own **0.65 s import**, which is *54% of the prize*. A cache that
skipped the parse and still imported cobra to read `cobra.__version__` would recover barely half of
what it was built for. `provenance._installed_version` reads the version from package **metadata**
instead, and `load_model` imports cobra **lazily, inside itself** — so a hit never touches cobra at
all. That is an architectural property of this module, not an optimisation: `flux_polytope`,
`native_csc` and `provenance` are all cobra-free, so a `CanonicalModel` can be rebuilt from bytes by
code that has never heard of cobra (the same reason §1.2's workers can hold one).

Implemented in **M1**; the store wired in **M10.2d** — see BUILD_PLAN.md.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.flux_polytope import FluxPolytope
from gsmm_compiler.logging_utils import get_logger
from gsmm_compiler.native_csc import VALUE_DTYPE, NativeCSC
from gsmm_compiler.provenance import (
    PARSER_SCHEMA_VERSION,
    Provenance,
    content_key,
    hash_file,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps cobra out of the import graph
    from cobra import Model

_log = get_logger(__name__)


MODEL_CACHE_LAYER = "L0"
"""The parsed model's layer in `cache.ArtifactCache` (BUILD_PLAN §1.1).

**The only new layer M10.2d adds, and the arithmetic is why.** §1.1 names a *four*-layer DAG, and
measured on the example model against a warm cache the other two candidates are not worth a key:

===============================  ==========  ===========================================
`load_canonical_model`           **1.157 s**  **this layer** (0.65 s cobra import + 0.52 s parse)
`.reduce()` → L1                 **0.001 s**  derive it
objective + LP → L2              **0.045 s**  derive it
===============================  ==========  ===========================================

Keying a 1 ms stage is §1.6.7's "16.4× upside-down" mistake with the numbers changed: **cache what
is expensive, derive what is cheap, key everything.** The numbered layers were always a description
of the *dependency* structure, not a shopping list of stores.
"""

MODEL_CACHE_SCHEMA_VERSION = 1
"""Bump when the stored L0 envelope's layout changes — an old artifact must miss, never mis-load."""


class ModelValidationError(ValueError):
    """The parsed model cannot be turned into a well-posed flux polytope."""


class ModelCacheError(RuntimeError):
    """A cached model does not bind to the key it was found under."""


@dataclass(frozen=True)
class ModelSummary:
    """Counts and bound facts reported by ``gsmm-compiler model inspect``."""

    model_id: str
    source_path: Path
    n_reactions: int
    n_metabolites: int
    n_genes: int
    n_exchanges: int
    n_fixed: int
    """Reactions with ``lower_bound == upper_bound`` — eliminated from the sampled state."""
    n_fixed_at_zero: int
    n_reversible: int
    n_infinite_bounds: int
    objective_reaction_ids: tuple[str, ...]

    @property
    def n_free(self) -> int:
        """Reactions that remain variable after fixed-variable elimination."""
        return self.n_reactions - self.n_fixed


@dataclass(frozen=True)
class CanonicalModel:
    """The frozen L0 IR: a validated `FluxPolytope` plus the identity and metadata behind it."""

    model_id: str
    source_path: Path | None
    """Where the model was read from, for the record. `None` for a model assembled in memory."""
    source_sha256: str | None
    """sha256 of the **named source file**, when `load_canonical_model` parsed one — *provenance
    only*. It is deliberately **not** the L0 identity: `build_canonical_model` accepts a model that
    was assembled or mutated in memory, and a file hash cannot prove such a model came from that
    file. `None` when no file was hashed. Identity lives in `l0_key`, which is content-addressed."""
    polytope: FluxPolytope
    exchange_mask: NDArray[np.bool_]
    """Which reactions are exchanges — carried here so `features` never has to import cobra."""
    provenance: Provenance
    l0_key: str
    """The L0 cache identity, **content-addressed** (BUILD_PLAN §1.1). It hashes the frozen IR the
    model actually contains — `model_id`, the polytope, the exchange mask — folded with the parser
    schema and cobra versions, and **never** the source file's bytes. So a model mutated in memory,
    or one handed the wrong `source_path`, gets a *different* key: it can never inherit another
    file's identity, which the old file-hash key allowed (the M8-opening defect)."""

    @property
    def l1_key(self) -> str:
        """Key of the reduced polytope IR derived from this model.

        A *key*, not a cache: `.reduce()` costs **1 ms** measured (M10.2d), so L1 is derived on
        demand and this key exists to name it in manifests, not to look anything up.
        """
        return self.polytope.content_key()

    def to_bundle(self) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Split into ``(arrays, meta)`` for `cache.ArtifactCache` (M10.2d).

        The IDs go in ``meta`` rather than in an array, and that is the one thing worth pausing on:
        `ArtifactCache` hashes every **array** and *trusts the meta*, and M6 settled that the
        reaction IDs **are the coordinate system** — a wrong one makes every index silently address
        a different reaction. They are safe here anyway, and by proof rather than by luck:
        `from_bundle` re-derives `l0_key`, which hashes `polytope.content_key()`, which covers the
        IDs. Tampered IDs reproduce a different key and are refused. That is the same
        lookup-proxy-vs-content-proof split this layer's two keys are built on.
        """
        polytope = self.polytope
        return (
            {
                "stoich_starts": np.ascontiguousarray(polytope.stoichiometry.starts),
                "stoich_indices": np.ascontiguousarray(polytope.stoichiometry.indices),
                "stoich_values": np.ascontiguousarray(polytope.stoichiometry.values),
                "lower_bounds": np.ascontiguousarray(polytope.lower_bounds),
                "upper_bounds": np.ascontiguousarray(polytope.upper_bounds),
                "exchange_mask": np.ascontiguousarray(self.exchange_mask),
            },
            {
                "cache_schema_version": MODEL_CACHE_SCHEMA_VERSION,
                "l0_key": self.l0_key,
                "model_id": self.model_id,
                "source_path": None if self.source_path is None else str(self.source_path),
                "source_sha256": self.source_sha256,
                "reaction_ids": list(polytope.reaction_ids),
                "metabolite_ids": list(polytope.metabolite_ids),
                "biomass_index": int(polytope.biomass_index),
                "n_rows": int(polytope.stoichiometry.n_rows),
                "n_cols": int(polytope.stoichiometry.n_cols),
                "provenance": self.provenance.as_dict(),
            },
        )

    @classmethod
    def from_bundle(cls, arrays: dict[str, NDArray[Any]], meta: dict[str, Any]) -> CanonicalModel:
        """Rebuild a model cached by `to_bundle` — **without importing cobra**, which is the point.

        Nothing here touches the parser: `FluxPolytope`, `NativeCSC` and `Provenance` are all
        cobra-free, so a hit costs ~0 where a miss costs 1.157 s — 0.65 s of it cobra's import.

        The stored `provenance` is **kept, not recaptured**. It describes *the parse that produced
        these arrays* — cobra's version is even folded into `l0_key` — so recapturing it would make
        a hit differ from a miss, reporting this run's environment as the one that did work it did
        not do. The run's own environment is recorded separately, by the run.

        The `l0_key` is **re-derived and compared**, never read back and believed: the lookup key is
        a proxy over the file's bytes, and this is the proof that these arrays are the artifact that
        key names (BUILD_PLAN §1.1's "validates the loaded artifact's content L0 key on load — so a
        false lookup hit is caught, never trusted").
        """
        if int(meta.get("cache_schema_version", -1)) != MODEL_CACHE_SCHEMA_VERSION:
            raise ModelCacheError(
                f"cached model has envelope schema {meta.get('cache_schema_version')!r}, "
                f"expected {MODEL_CACHE_SCHEMA_VERSION}"
            )
        polytope = FluxPolytope(
            reaction_ids=tuple(meta["reaction_ids"]),
            metabolite_ids=tuple(meta["metabolite_ids"]),
            stoichiometry=NativeCSC(
                n_rows=int(meta["n_rows"]),
                n_cols=int(meta["n_cols"]),
                starts=np.asarray(arrays["stoich_starts"], dtype=np.int32),
                indices=np.asarray(arrays["stoich_indices"], dtype=np.int32),
                values=np.asarray(arrays["stoich_values"], dtype=VALUE_DTYPE),
            ),
            lower_bounds=np.asarray(arrays["lower_bounds"], dtype=VALUE_DTYPE),
            upper_bounds=np.asarray(arrays["upper_bounds"], dtype=VALUE_DTYPE),
            biomass_index=int(meta["biomass_index"]),
        )
        exchange_mask = np.asarray(arrays["exchange_mask"], dtype=bool)
        provenance = Provenance(**meta["provenance"])
        model = cls(
            model_id=str(meta["model_id"]),
            source_path=None if meta["source_path"] is None else Path(meta["source_path"]),
            source_sha256=meta["source_sha256"],
            polytope=polytope,
            exchange_mask=exchange_mask,
            provenance=provenance,
            l0_key=str(meta["l0_key"]),
        )
        rederived = _l0_key(polytope, model.model_id, exchange_mask, provenance.cobra_version)
        if rederived != model.l0_key:
            raise ModelCacheError(
                "cached model does not reproduce its own content key "
                f"({model.l0_key[:16]}… stored, {rederived[:16]}… re-derived): these bytes are not "
                "that model's IR"
            )
        return model

    def report(self) -> dict[str, Any]:
        """The contents of ``model_report.json``."""
        polytope = self.polytope
        lower, upper = polytope.lower_bounds, polytope.upper_bounds
        fixed = polytope.fixed_indices

        return {
            "model_id": self.model_id,
            "source_path": None if self.source_path is None else str(self.source_path),
            "source_sha256": self.source_sha256,
            "l0_key": self.l0_key,
            "l1_key": self.l1_key,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "provenance": self.provenance.as_dict(),
            "counts": {
                "reactions": polytope.n_reactions,
                "metabolites": polytope.n_metabolites,
                "exchanges": int(self.exchange_mask.sum()),
                "stoichiometry_nnz": polytope.stoichiometry.nnz,
                "fixed": polytope.n_fixed,
                "fixed_at_zero": int(np.count_nonzero(lower[fixed] == 0.0)),
                "free": polytope.n_free,
                "reversible": int(np.count_nonzero((lower < 0.0) & (upper > 0.0))),
            },
            "biomass": {
                "reaction_id": polytope.biomass_id,
                "index": polytope.biomass_index,
                "is_fixed": bool(polytope.fixed_mask[polytope.biomass_index]),
            },
            "bounds": {
                "min_lower": float(lower.min()),
                "max_upper": float(upper.max()),
                "all_finite": bool(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))),
            },
        }

    def write_report(self, path: str | Path) -> Path:
        """Write ``model_report.json`` and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.report(), indent=2, sort_keys=True) + "\n")
        return destination


def load_model(path: str | Path) -> Model:
    """Load a JSON/SBML GSMM via cobra.

    Raises ``FileNotFoundError`` if the path does not exist and ``ValueError`` for unknown suffixes.
    """
    from cobra.io import load_json_model, read_sbml_model

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".json":
        return load_json_model(str(source))
    if suffix in {".xml", ".sbml"}:
        return read_sbml_model(str(source))
    raise ValueError(f"unsupported model format {suffix!r} (expected .json, .xml or .sbml)")


def _resolve_biomass_index(model: Model, biomass_id: str | None) -> int:
    """Find biomass by ID, else by objective coefficient — insisting on exactly one."""
    reaction_ids = [r.id for r in model.reactions]

    if biomass_id is not None:
        if biomass_id not in reaction_ids:
            raise ModelValidationError(f"biomass reaction {biomass_id!r} is not in the model")
        return reaction_ids.index(biomass_id)

    objective_indices = [i for i, r in enumerate(model.reactions) if r.objective_coefficient != 0.0]
    if not objective_indices:
        raise ModelValidationError(
            "no biomass reaction: the model has no objective, and none was configured"
        )
    if len(objective_indices) > 1:
        named = [reaction_ids[i] for i in objective_indices]
        raise ModelValidationError(
            f"objective spans {len(named)} reactions ({named}); configure a single biomass reaction"
        )
    return objective_indices[0]


def _validate_parsed(model: Model) -> None:
    """Reject anything that would make the polytope ill-posed, naming the offenders."""
    if not model.reactions:
        raise ModelValidationError("model has no reactions")
    if not model.metabolites:
        raise ModelValidationError("model has no metabolites")

    for kind, objects in (("reaction", model.reactions), ("metabolite", model.metabolites)):
        duplicates = [i for i, count in Counter(o.id for o in objects).items() if count > 1]
        if duplicates:
            raise ModelValidationError(f"duplicate {kind} IDs: {sorted(duplicates)[:5]}")

    nonfinite = [
        r.id
        for r in model.reactions
        if not (math.isfinite(r.lower_bound) and math.isfinite(r.upper_bound))
    ]
    if nonfinite:
        raise ModelValidationError(
            f"{len(nonfinite)} reactions have NaN or infinite bounds: {sorted(nonfinite)[:5]}. "
            "The polytope must be bounded — replace infinities with an explicit flux limit."
        )

    inverted = [r.id for r in model.reactions if r.lower_bound > r.upper_bound]
    if inverted:
        raise ModelValidationError(
            f"lower bound exceeds upper bound for: {sorted(inverted)[:5]} (empty polytope)"
        )

    for coefficient in (c for r in model.reactions for c in r.metabolites.values()):
        if not math.isfinite(coefficient):
            raise ModelValidationError(
                "stoichiometric matrix contains NaN or infinite coefficients"
            )


def build_canonical_model(
    model: Model,
    source_path: str | Path | None = None,
    biomass_id: str | None = None,
    *,
    source_sha256: str | None = None,
) -> CanonicalModel:
    """Validate a parsed cobra model and freeze it into the canonical IR.

    **Order is frozen here and never re-derived.** Reaction ``j`` is column ``j`` of ``S`` and entry
    ``j`` of every bound array, for the rest of the pipeline; metabolite ``i`` is row ``i``. Cobra's
    ``DictList`` order is stable for a given file, and every artifact carries the ID tuples, so a
    reordering upstream changes the L1 key rather than silently permuting a saved flux vector.

    ``model`` may be a model cobra just parsed *or* one assembled or mutated in memory. Because this
    function cannot tell which, it **does not hash any file for identity**: the L0 key is derived
    from the model's own frozen content. ``source_path`` is recorded for the record, and
    ``source_sha256`` is stored as-is only when a caller who *did* the parse
    (`load_canonical_model`) supplies it — so a model can never be stamped with a file's cache
    identity it did not come from. See BUILD_PLAN §1.1 and the ``l0_key`` field doc.
    """
    _validate_parsed(model)

    reaction_ids = tuple(r.id for r in model.reactions)
    metabolite_ids = tuple(m.id for m in model.metabolites)
    row_of_metabolite = {metabolite_id: i for i, metabolite_id in enumerate(metabolite_ids)}

    stoichiometry = NativeCSC.from_columns(
        n_rows=len(metabolite_ids),
        columns=[
            {
                row_of_metabolite[m.id]: float(coefficient)
                for m, coefficient in r.metabolites.items()
            }
            for r in model.reactions
        ],
    )

    polytope = FluxPolytope(
        reaction_ids=reaction_ids,
        metabolite_ids=metabolite_ids,
        stoichiometry=stoichiometry,
        lower_bounds=np.array([r.lower_bound for r in model.reactions], dtype=VALUE_DTYPE),
        upper_bounds=np.array([r.upper_bound for r in model.reactions], dtype=VALUE_DTYPE),
        biomass_index=_resolve_biomass_index(model, biomass_id),
    )

    exchange_ids = {r.id for r in model.exchanges}
    exchange_mask = np.array([r in exchange_ids for r in reaction_ids], dtype=bool)
    source = None if source_path is None else Path(source_path)
    model_id = model.id or (source.stem if source is not None else "model")
    provenance = Provenance.capture()

    return CanonicalModel(
        model_id=model_id,
        source_path=source,
        source_sha256=source_sha256,
        polytope=polytope,
        exchange_mask=exchange_mask,
        provenance=provenance,
        # Content-addressed, never file-hash-addressed (§1.1): the key fingerprints the IR this
        # model *actually holds*, so a model mutated in memory — or paired with the wrong
        # `source_path` — gets a distinct key instead of inheriting an unrelated file's identity.
        # `polytope.content_key()` covers reaction/metabolite IDs, the CSC arrays, bounds and the
        # biomass index; `model_id` is folded in because it names the reaction's RNG streams, and
        # the exchange mask because `features`/aggregation read it. cobra + parser versions fold in
        # so a parser change that alters what we extract misses the cache rather than loading stale
        # bytes.
        l0_key=_l0_key(polytope, model_id, exchange_mask, provenance.cobra_version),
    )


def _l0_key(
    polytope: FluxPolytope,
    model_id: str,
    exchange_mask: NDArray[np.bool_],
    cobra_version: str | None,
) -> str:
    """The content-addressed L0 identity — **the one writer**, so a build and a load cannot drift.

    M10.2d split this out of `build_canonical_model` for the reason M10.2a made a rule: a key
    computed in two places is two keys, and `CanonicalModel.from_bundle` has to re-derive exactly
    what the builder derived or the check it performs proves nothing.
    """
    return content_key(
        canonical_ir=polytope.content_key(),
        model_id=model_id,
        exchange_mask=exchange_mask,
        parser_schema_version=PARSER_SCHEMA_VERSION,
        cobra_version=cobra_version,
    )


def model_lookup_key(source: str | Path, biomass_id: str | None = None) -> str:
    """L0's **lookup** key: a function of the inputs, computable **without importing cobra**.

    Not the identity — that is `CanonicalModel.l0_key`, which fingerprints the IR the model actually
    holds and cannot be computed without parsing it. This names the *inputs* that decide those
    bytes, so a run finds the artifact before paying the 1.157 s to build it (`hash_file`: 1 ms).

    Being **over**-specific here is free and being under-specific is not: §1.1's asymmetry says a
    false miss only recomputes while a false hit corrupts. Hence:

    * the **resolved source path**, not only the file's bytes. `build_canonical_model` falls back to
      ``source.stem`` when a model carries no ``id`` of its own — so two identical files under
      different names are two different `model_id`s, and `model_id` **keys the RNG streams**. Two
      copies of one file are then parsed twice, which costs 0.5 s and is the right side of the
      asymmetry to be on.
    * ``biomass_id``, because it selects a different reaction and so a different IR.
    * `cobra_version` **from package metadata** (`provenance._installed_version`), never from
      ``cobra.__version__`` — reading that attribute would import cobra, which is 0.65 s of the
      1.157 s this key exists to avoid.
    * `numpy_version`, `python_version` and `byte_order`, per §1.1's "provenance in every key":
      they decide array semantics, and unlike `l0_key` — which hashes the arrays themselves, so any
      such change shows up in the bytes — a lookup key is computed *before* the bytes exist and must
      predict them.
    """
    resolved = Path(source).resolve()
    environment = Provenance.capture()
    return content_key(
        layer=MODEL_CACHE_LAYER,
        source_path=str(resolved),
        source_sha256=hash_file(resolved),
        biomass_id=biomass_id,
        parser_schema_version=PARSER_SCHEMA_VERSION,
        cache_schema_version=MODEL_CACHE_SCHEMA_VERSION,
        cobra_version=environment.cobra_version,
        numpy_version=environment.numpy_version,
        python_version=environment.python_version,
        byte_order=environment.byte_order,
    )


def load_canonical_model(
    path: str | Path,
    biomass_id: str | None = None,
    *,
    cache: ArtifactCache | None = None,
) -> CanonicalModel:
    """Parse and freeze a model file in one step — the trusted path.

    This is the **only** place a file's hash is bound to a model's identity, and it is honest here
    because the same call both hashes and parses the file: the ``source_sha256`` it records
    provably describes the bytes the ``model`` was built from. That hash is provenance; the cache
    identity is the content-addressed ``l0_key`` computed by `build_canonical_model`.

    **That honesty is exactly what licenses the cache** (M10.2d), and it is why the store lives here
    and not on `build_canonical_model`. §1.1's M8 finding was that a file hash cannot prove an
    *in-memory* model came from the file whose bytes it hashes — so a file-keyed lookup would be a
    lie on that path. Here the correspondence is real because one call does both. `build_canonical_
    model` keeps no ``cache`` parameter: a caller cannot mistakenly file-key a mutated model,
    because there is no parameter through which to try.

    ``cache=None`` re-parses. A false miss only recomputes.
    """
    source = Path(path)
    if cache is None:
        return build_canonical_model(
            load_model(source), source, biomass_id, source_sha256=hash_file(source)
        )

    def compute() -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        _log.info("parsing %s (cobra; not cached)", source.name)
        return build_canonical_model(
            load_model(source), source, biomass_id, source_sha256=hash_file(source)
        ).to_bundle()

    artifact = cache.get_or_compute(
        MODEL_CACHE_LAYER, model_lookup_key(source, biomass_id), compute
    )
    return CanonicalModel.from_bundle(artifact.arrays, artifact.meta)


# ---- reporting ---------------------------------------------------------------------------------


def summarize(model: Model, source_path: str | Path) -> ModelSummary:
    """Compute the counts reported by ``model inspect``, without requiring a valid polytope."""
    fixed = [r for r in model.reactions if r.lower_bound == r.upper_bound]
    infinite = [
        r for r in model.reactions if math.isinf(r.lower_bound) or math.isinf(r.upper_bound)
    ]

    return ModelSummary(
        model_id=model.id or Path(source_path).stem,
        source_path=Path(source_path),
        n_reactions=len(model.reactions),
        n_metabolites=len(model.metabolites),
        n_genes=len(model.genes),
        n_exchanges=len(model.exchanges),
        n_fixed=len(fixed),
        n_fixed_at_zero=sum(1 for r in fixed if r.lower_bound == 0.0),
        n_reversible=sum(1 for r in model.reactions if r.lower_bound < 0.0 < r.upper_bound),
        n_infinite_bounds=len(infinite),
        objective_reaction_ids=tuple(
            r.id for r in model.reactions if r.objective_coefficient != 0.0
        ),
    )


def format_summary(summary: ModelSummary) -> str:
    """Render a `ModelSummary` as the plain-text ``model inspect`` report."""
    lines = [
        f"model_id            {summary.model_id}",
        f"source              {summary.source_path}",
        f"reactions           {summary.n_reactions}",
        f"metabolites         {summary.n_metabolites}",
        f"genes               {summary.n_genes}",
        f"exchanges           {summary.n_exchanges}",
        f"objective           {', '.join(summary.objective_reaction_ids) or '(none)'}",
        f"fixed (l == u)      {summary.n_fixed}  (of which {summary.n_fixed_at_zero} at zero)",
        f"free (l < u)        {summary.n_free}",
        f"reversible          {summary.n_reversible}",
        f"infinite bounds     {summary.n_infinite_bounds}",
    ]
    return "\n".join(lines)
