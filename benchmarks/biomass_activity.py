"""M11.5(c) extra — biomass vs metabolic activity across the census.

Requested view: **biomass on x, the number of reactions carrying non-zero flux on average on y.**

What the census data says about the relationship (measured, so the plot is not misread):

* Within a strain, the active-reaction count is nearly **flat across β** while biomass climbs — the
  L1 sparsity penalty here is tiny (≈1e-3), so selection pressure buys growth, not network pruning.
* Across strains the two correlate strongly (r≈0.9), but that is largely a **model-size effect**
  (bigger models have more free reactions *and* more biomass). So this is a cross-strain scatter
  with the within-strain β-effect drawn as short trajectories, and dimension shown as colour — the
  size confound is made visible rather than hidden.

"Non-zero flux on average" is read two ways, both plotted:
* **active** — reactions with |flux| above a threshold *on average over the samples* (mean
  per-sample count = ``n_free − mean(near_zero_count)``, the package's own near-zero accounting).
* **essential / always-on core** — reactions active in ≥ ``CORE_FRAC`` of samples (never idle); the
  closest sample-based proxy to metabolic essentiality.

Threshold defaults to 1e-6 (above solver noise, below any real flux). Reproduce:
    .venv/bin/python benchmarks/biomass_activity.py [OUT_ROOT]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THR_IDX = 1        # near_zero_thresholds index: 0=1e-9, 1=1e-6, 2=1e-3, 3=0.1, 4=1.0
THR_LABEL = "1e-6"
CORE_FRAC = 0.99   # a reaction is "essential/always-on" if active in ≥ this fraction of samples
INK, MUTED, GRID = "#1a1a1a", "#8a8a8a", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def census_model_dirs(root: Path) -> list[str]:
    dirs = []
    for p in glob.glob(str(root / "*" / "out" / "*" / "*" / "samples")):
        label = Path(p).parents[3].name  # <root>/<LABEL>/out/<batch>/<model>/samples
        if label.startswith(("probe", "covstudy", "sj_")):
            continue  # not a census strain
        mdir = p.rsplit("/samples", 1)[0]
        if glob.glob(mdir + "/run_manifest.json"):
            dirs.append(mdir)
    return sorted(dirs)


def strain_stats(mdir: str) -> dict | None:
    m = json.loads(Path(mdir + "/run_manifest.json").read_text())
    rids = m["axes"]["reaction_ids"]
    if "bio1" not in rids:
        return None
    bidx = rids.index("bio1")
    n_free = m["objective"]["n_free"]
    d = m["geometry"]["dimension"]
    rungs = []
    for bdir in sorted(glob.glob(mdir + "/samples/beta_*")):
        chains = sorted(glob.glob(bdir + "/chain_*"))
        bio = np.concatenate([np.load(c + "/flux.npy")[:, bidx] for c in chains])
        nz = np.concatenate(
            [np.load(c + "/trace_near_zero_counts_all_free.npy") for c in chains])  # (N, 5)
        active_mean = float(n_free - nz[:, THR_IDX].mean())  # avg active reactions per sample
        # essential/always-on core: needs per-reaction activity over the movable columns
        flux = np.concatenate([np.load(c + "/flux.npy") for c in chains])  # (N, nrx)
        thr = m["config"]["objective"]["near_zero_thresholds"][THR_IDX]
        active_frac = (np.abs(flux) > thr).mean(axis=0)  # per reaction
        core = int(np.count_nonzero(active_frac >= CORE_FRAC))
        cmani = json.loads(Path(chains[0] + "/manifest.json").read_text())
        beta = float(cmani["diagnostics"]["beta"])
        rungs.append({"beta": beta, "e_biomass": float(bio.mean()),
                      "active_mean": active_mean, "core": core})
    name = m["model_id"].split("_")[3] if len(m["model_id"].split("_")) > 3 else m["model_id"]
    return {"name": name, "dimension": d, "n_free": n_free, "rungs": rungs}


def main(root: Path) -> int:
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    strains = [s for s in (strain_stats(d) for d in census_model_dirs(root)) if s]
    strains.sort(key=lambda s: s["dimension"])
    (root / "biomass_activity.json").write_text(json.dumps(strains, indent=2))
    print(f"{len(strains)} strains with bio1")

    dims = np.array([s["dimension"] for s in strains])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(dims.min(), dims.max())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left: cross-strain — biomass (β=0) vs active count, colour = dimension, β-trajectory drawn.
    b0x, b0y = [], []
    for s in strains:
        r = s["rungs"]
        bx = [q["e_biomass"] for q in r]
        by = [q["active_mean"] for q in r]
        col = cmap(norm(s["dimension"]))
        axL.plot(bx, by, "-", color=col, lw=0.9, alpha=0.7, zorder=2)  # β-trajectory
        axL.scatter([bx[0]], [by[0]], color=col, s=46, zorder=3, edgecolor="white", linewidth=0.6)
        b0x.append(bx[0])
        b0y.append(by[0])
    r_bio = np.corrcoef(b0x, b0y)[0, 1]
    style(axL)
    axL.set_xlabel("mean biomass flux  E[v_bio1]")
    axL.set_ylabel(f"reactions with non-zero flux on average  (|flux|>{THR_LABEL})")
    axL.set_title(f"Biomass vs active reactions across {len(strains)} strains\n"
                  f"dot = β=0, line = β-ladder to 16   (cross-strain r={r_bio:.2f})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axL, label="affine dimension d", fraction=0.046, pad=0.04)

    # Right: the two readings of "essential" at β=0 — total active vs always-on core.
    order = np.argsort(b0x)
    names = [strains[i]["name"][:12] for i in order]
    active0 = [strains[i]["rungs"][0]["active_mean"] for i in order]
    core0 = [strains[i]["rungs"][0]["core"] for i in order]
    y = np.arange(len(order))
    core_label = f"essential core (on in ≥{int(CORE_FRAC * 100)}%)"
    axR.barh(y + 0.2, active0, 0.4, color="#0072B2", label="active (non-zero on avg)")
    axR.barh(y - 0.2, core0, 0.4, color="#E69F00", label=core_label)
    axR.set_yticks(y)
    axR.set_yticklabels(names, fontsize=7)
    style(axR)
    axR.set_xlabel("number of reactions")
    axR.set_title("Two readings of “essential”, at β=0\n(sorted by biomass, low→high)")
    axR.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Biomass vs metabolic activity — requested view "
                 "(active count is size-driven; flat within a strain across β)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig5_biomass_activity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig5_biomass_activity.png'}   (cross-strain r={r_bio:.3f})")
    return 0


if __name__ == "__main__":
    default = Path(os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census"))
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
