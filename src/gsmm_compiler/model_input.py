"""Model parsing layer (cobra) — load a GSMM and describe it.

cobra is a **parser/metadata layer only**: nothing in the numerical core may import this module.
It is the one place allowed to touch cobra/optlang.

M0 provides loading + a summary report. M1 adds order freezing, validation (unique IDs, NaN/inf
bounds, biomass present exactly once, finite bounds) and ``model_report.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps cobra out of the import graph
    from cobra import Model


@dataclass(frozen=True)
class ModelSummary:
    """Counts and bound facts used by ``gsmm-compiler model inspect``."""

    model_id: str
    source_path: Path
    n_reactions: int
    n_metabolites: int
    n_genes: int
    n_exchanges: int
    n_fixed: int
    """Reactions with ``lower_bound == upper_bound`` (eliminated from the sampled state in M1)."""
    n_fixed_at_zero: int
    n_reversible: int
    n_infinite_bounds: int
    objective_reaction_ids: tuple[str, ...]

    @property
    def n_free(self) -> int:
        """Reactions that remain variable after fixed-variable elimination."""
        return self.n_reactions - self.n_fixed


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


def summarize(model: Model, source_path: str | Path) -> ModelSummary:
    """Compute the counts reported by ``model inspect``."""
    import math

    fixed = [r for r in model.reactions if r.lower_bound == r.upper_bound]
    infinite = [
        r for r in model.reactions if math.isinf(r.lower_bound) or math.isinf(r.upper_bound)
    ]
    objective_ids = tuple(r.id for r in model.reactions if r.objective_coefficient != 0.0)

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
        objective_reaction_ids=objective_ids,
    )


def format_summary(summary: ModelSummary) -> str:
    """Render a ``ModelSummary`` as the plain-text ``model inspect`` report."""
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
