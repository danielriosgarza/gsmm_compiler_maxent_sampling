"""Downstream flux features and the cross-model aggregation (M8, BUILD_PLAN §1.1 + §2).

Two jobs:

* **Per-model features** from the stored samples — mean flux, mean ``|flux|``, and an *activity
  fraction* (the share of samples in which a reaction carries ``|v|`` above a declared threshold).
  **Every threshold lives here and only here.** The chain state is never snapped to zero — a
  threshold applied to the walk would move the stationary distribution and can break mass balance
  (spec §3.7, CLAUDE.md). Thresholds touch the *stored* samples, downstream, and nothing else.

* **Cross-model aggregation** into ``results/<batch>/cross_model/`` — the β-summary,
  reaction-activity, and exchange tables stacked across strains that answer §2's question ("do two
  species retain different metabolic flexibility at comparable selection pressure?"). It only ever
  *reads* per-model artifacts, so a **partial batch still yields valid tables** over the strains
  that finished — the M8 gate's "partial batch → valid cross-model tables".

Implemented in **M8** — see BUILD_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.output import RunLayout, load_chain, read_json, write_json

# ---- per-model flux features (thresholds live here, nowhere else) -------------------------------


def active_fraction(fluxes: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Per reaction, the fraction of samples with ``|v| > threshold`` — a *soft* activity signal.

    ``fluxes`` is ``(n_samples, n_reactions)``. Not a snap: the underlying draws keep their exact
    values; this only *counts* how often each reaction is meaningfully on, at a declared cutoff.
    """
    return np.asarray(np.mean(np.abs(fluxes) > threshold, axis=0), dtype=np.float64)


def mean_abs_flux(fluxes: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.mean(np.abs(fluxes), axis=0), dtype=np.float64)


def mean_flux(fluxes: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.mean(fluxes, axis=0), dtype=np.float64)


# ---- reading a model's stored samples -----------------------------------------------------------


def _load_model_fluxes(layout: RunLayout, model_id: str, beta_index: int, n_chains: int) -> (
    NDArray[np.float64] | None
):
    """Stack every chain's full-length fluxes at one ``β``, or ``None`` if they cannot be read.

    Full-flux storage only: a ``reduced``-mode run needs the retained geometry to reconstruct the
    flux, which the aggregator does not carry, so reduced-mode per-reaction features are skipped
    (the β-summary, which reads only the manifest, still covers every mode).
    """
    blocks: list[NDArray[np.float64]] = []
    for chain_index in range(n_chains):
        chain_dir = layout.chain_dir(model_id, beta_index, chain_index)
        manifest = read_json(chain_dir / "manifest.json")
        if manifest["store_mode"] != "full_flux":
            return None
        blocks.append(load_chain(chain_dir).fluxes)
    return np.concatenate(blocks, axis=0) if blocks else None


# ---- cross-model aggregation --------------------------------------------------------------------


def aggregate_cross_model(layout: RunLayout, model_ids: list[str]) -> Path:
    """Write the ``cross_model/`` tables over the models that completed. Returns that directory.

    Reads each model's ``run_manifest.json`` (and, for the activity/exchange tables, its full-flux
    samples). Anything a model does not provide is simply left out of the union tables, so a batch
    where some strains failed still produces a coherent comparison over the ones that did.
    """
    cross = layout.cross_model_dir()
    cross.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        manifest_path = layout.model_manifest_path(model_id)
        if manifest_path.is_file():
            manifests[model_id] = read_json(manifest_path)

    beta_summary = _beta_summary(manifests)
    write_json(cross / "beta_summary.json", beta_summary)
    (cross / "beta_summary.tsv").write_text(_beta_summary_tsv(beta_summary))

    activity, exchange = _activity_and_exchange(layout, manifests)
    write_json(cross / "reaction_activity.json", activity)
    write_json(cross / "exchange_flux.json", exchange)

    return cross


def _beta_summary(manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per ``(model, β)``, averaging the per-chain trace summaries. Manifest-only."""
    rows: list[dict[str, Any]] = []
    for model_id, manifest in manifests.items():
        for beta_index, beta in enumerate(manifest["betas"]):
            units = [u for u in manifest["units"] if u["beta_index"] == beta_index]
            if not units:
                continue
            summaries = [u["trace_summary"] for u in units]
            rows.append(
                {
                    "model_id": model_id,
                    "beta_index": beta_index,
                    "beta": float(beta),
                    "n_chains": len(units),
                    "mean_mu": _avg(summaries, "mean_mu"),
                    "mean_cost": _avg(summaries, "mean_cost"),
                    "mean_j": _avg(summaries, "mean_j"),
                    "mean_std_j": _avg(summaries, "std_j"),
                }
            )
    rows.sort(key=lambda r: (r["model_id"], r["beta_index"]))
    return rows


def _avg(summaries: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([s[key] for s in summaries]))


def _beta_summary_tsv(rows: list[dict[str, Any]]) -> str:
    columns = ["model_id", "beta_index", "beta", "n_chains", "mean_mu", "mean_cost", "mean_j"]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row[column]) for column in columns))
    return "\n".join(lines) + "\n"


def _activity_and_exchange(
    layout: RunLayout, manifests: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-model, per-β reaction-activity and exchange-flux tables, from the full-flux samples.

    Reaction IDs differ across strains, so the tables are keyed by reaction id (a union), not by a
    shared column index: a species that lacks a reaction simply has no entry for it, which is the
    honest representation of "this strain does not have that reaction".
    """
    activity: dict[str, Any] = {"skipped_reduced_mode": [], "models": {}}
    exchange: dict[str, Any] = {"skipped_reduced_mode": [], "models": {}}

    for model_id, manifest in manifests.items():
        axes = manifest.get("axes")
        if axes is None:
            continue
        reaction_ids = axes["reaction_ids"]
        exchange_mask = np.asarray(axes["exchange_mask"], dtype=bool)
        threshold = float(axes.get("activity_threshold", 1e-6))
        n_chains = int(axes.get("n_chains", 1))

        model_activity: dict[str, Any] = {}
        model_exchange: dict[str, Any] = {}
        reduced_seen = False
        for beta_index, beta in enumerate(manifest["betas"]):
            fluxes = _load_model_fluxes(layout, model_id, beta_index, n_chains)
            if fluxes is None:
                reduced_seen = True
                continue
            fractions = active_fraction(fluxes, threshold)
            means = mean_flux(fluxes)
            model_activity[f"{float(beta):g}"] = {
                reaction_ids[i]: float(fractions[i]) for i in range(len(reaction_ids))
            }
            model_exchange[f"{float(beta):g}"] = {
                reaction_ids[i]: float(means[i]) for i in np.flatnonzero(exchange_mask)
            }
        if reduced_seen:
            activity["skipped_reduced_mode"].append(model_id)
            exchange["skipped_reduced_mode"].append(model_id)
        if model_activity:
            activity["models"][model_id] = {"threshold": threshold, "by_beta": model_activity}
            exchange["models"][model_id] = {"by_beta": model_exchange}

    return activity, exchange
