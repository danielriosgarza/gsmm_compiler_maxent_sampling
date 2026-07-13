"""Model parsing layer (cobra) — load a GSMM, validate it, freeze it into the canonical IR.

cobra is a **parser/metadata layer only**: nothing in the numerical core may import this module, and
this is the one place allowed to touch cobra/optlang. Everything downstream sees NumPy arrays and
frozen tuples, never a cobra object.

This is the **L0** layer of the cache DAG (BUILD_PLAN §1.1): the raw file hash alone is not a
sufficient key, because parser semantics decide what the arrays end up containing — so the L0 key
folds in the cobra version and the parser schema version too.

Implemented in **M1** — see BUILD_PLAN.md.
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

from gsmm_compiler.flux_polytope import FluxPolytope
from gsmm_compiler.native_csc import VALUE_DTYPE, NativeCSC
from gsmm_compiler.provenance import (
    PARSER_SCHEMA_VERSION,
    Provenance,
    content_key,
    hash_file,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps cobra out of the import graph
    from cobra import Model


class ModelValidationError(ValueError):
    """The parsed model cannot be turned into a well-posed flux polytope."""


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
    source_path: Path
    source_sha256: str
    polytope: FluxPolytope
    exchange_mask: NDArray[np.bool_]
    """Which reactions are exchanges — carried here so `features` never has to import cobra."""
    provenance: Provenance
    l0_key: str

    @property
    def l1_key(self) -> str:
        """Key of the reduced polytope IR derived from this model."""
        return self.polytope.content_key()

    def report(self) -> dict[str, Any]:
        """The contents of ``model_report.json``."""
        polytope = self.polytope
        lower, upper = polytope.lower_bounds, polytope.upper_bounds
        fixed = polytope.fixed_indices

        return {
            "model_id": self.model_id,
            "source_path": str(self.source_path),
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
    source_path: str | Path,
    biomass_id: str | None = None,
) -> CanonicalModel:
    """Validate a parsed cobra model and freeze it into the canonical IR.

    **Order is frozen here and never re-derived.** Reaction ``j`` is column ``j`` of ``S`` and entry
    ``j`` of every bound array, for the rest of the pipeline; metabolite ``i`` is row ``i``. Cobra's
    ``DictList`` order is stable for a given file, and every artifact carries the ID tuples, so a
    reordering upstream changes the L1 key rather than silently permuting a saved flux vector.
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
    source = Path(source_path)
    source_sha256 = hash_file(source)
    provenance = Provenance.capture()

    return CanonicalModel(
        model_id=model.id or source.stem,
        source_path=source,
        source_sha256=source_sha256,
        polytope=polytope,
        exchange_mask=np.array([r in exchange_ids for r in reaction_ids], dtype=bool),
        provenance=provenance,
        # The file hash alone would not invalidate on a parser change (§1.1).
        l0_key=content_key(
            source_sha256=source_sha256,
            parser_schema_version=PARSER_SCHEMA_VERSION,
            cobra_version=provenance.cobra_version,
            biomass_id=polytope.biomass_id,
        ),
    )


def load_canonical_model(path: str | Path, biomass_id: str | None = None) -> CanonicalModel:
    """Parse and freeze a model file in one step."""
    source = Path(path)
    return build_canonical_model(load_model(source), source, biomass_id)


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
