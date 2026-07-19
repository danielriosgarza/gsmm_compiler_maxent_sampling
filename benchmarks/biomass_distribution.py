"""M11.5(c) extra — per-model biomass distribution + the bits→range tradeoff.

Two panels for a single strain:

* **A — the actual biomass distribution** per β rung (pooled over chains), as horizontal violins.
  Shows the distribution shifting toward its max and concentrating as selection pressure rises,
  rather than only its mean.
* **B — the information cost of pinning growth.** x = **bits of selection** = KL(π_β‖π₀)/ln2, y =
  biomass range / max biomass. KL is computed by **thermodynamic integration** along the ladder:
  with κ=β/s_J and E_κ[J] the mean of the J-trace at that rung,
  ``KL(π_κ‖π₀) = κ·E_κ[J] − ∫₀^κ E_t[J] dt`` (nats), ``bits = KL/ln2``. As bits rise the biomass
  range collapses toward its max (§1.6.6: KL≈½β² at small β gives the initial slope).

Usage:
    biomass_distribution.py                       # default single model (Pediococcus parvulus)
    biomass_distribution.py <model-substring>     # single model by name
    biomass_distribution.py all                   # ONE figure per completed census strain
Figures: ``benchmarks/plots/fig6_biomass_distribution.png`` (single) or
``benchmarks/plots/per_strain/<model>.png`` (all).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    for p in sorted(glob.glob(str(root / "*" / "out" / "*" / "*" / "samples"))):
        label = Path(p).parents[3].name
        if label.startswith(("probe", "covstudy", "sj_")):
            continue
        mdir = p.rsplit("/samples", 1)[0]
        if glob.glob(mdir + "/run_manifest.json"):
            dirs.append(mdir)
    return sorted(dirs)


def pretty_name(model_id: str) -> str:
    """GCF_964063155_1_Pediococcus_parvulus_IOEB_9646_protein_gapfilled -> Pediococcus parvulus
    IOEB 9646."""
    toks = model_id.split("_")
    drop = {"protein", "gapfilled", "genomic", "complete", "non", "latest", "noO2"}
    body = [t for t in toks[3:] if t not in drop]
    return " ".join(body) if body else model_id


def load_model(mdir: str):
    m = json.loads(Path(mdir + "/run_manifest.json").read_text())
    rids = m["axes"]["reaction_ids"]
    if "bio1" not in rids:
        return None
    bidx = rids.index("bio1")
    s_J = m["energy_scale"]["energy_scale"]
    betas, bio_by_beta, ej = [], [], []
    for bdir in sorted(glob.glob(mdir + "/samples/beta_*")):
        chains = sorted(glob.glob(bdir + "/chain_*"))
        bio = np.concatenate(
            [np.load(c + "/flux.npy", mmap_mode="r")[:, bidx] for c in chains]).astype(float)
        jt = np.concatenate([np.load(c + "/trace_j.npy") for c in chains]).astype(float)
        cmani = json.loads(Path(chains[0] + "/manifest.json").read_text())
        betas.append(float(cmani["diagnostics"]["beta"]))
        bio_by_beta.append(bio)
        ej.append(float(jt.mean()))
    betas = np.array(betas)
    order = np.argsort(betas)
    return {
        "name": pretty_name(m["model_id"]),
        "dimension": int(m["geometry"]["dimension"]),
        "s_J": float(s_J),
        "betas": betas[order],
        "ej": np.array(ej)[order],
        "bio_by_beta": [bio_by_beta[i] for i in order],
    }


def plot_one(data: dict, dest: Path) -> tuple[float, float]:
    betas, ej, bio_by_beta, s_J = data["betas"], data["ej"], data["bio_by_beta"], data["s_J"]
    # bits via thermodynamic integration: KL(κ) = κ·E_κ[J] − ∫₀^κ E_t[J] dt, κ=β/s_J.
    kappa = betas / s_J
    logZ = np.concatenate([[0.0], np.cumsum(0.5 * (ej[1:] + ej[:-1]) * np.diff(kappa))])
    bits = (kappa * ej - logZ) / math.log(2.0)
    max_bio = max(float(b.max()) for b in bio_by_beta)
    rng = np.array([float(np.percentile(b, 97.5) - np.percentile(b, 2.5)) for b in bio_by_beta])
    rng_over_max = rng / max_bio

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(betas.min(), betas.max())

    parts = axA.violinplot(bio_by_beta, positions=np.arange(len(betas)),
                           orientation="horizontal", widths=0.85, showmeans=True,
                           showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(norm(betas[i])))
        body.set_alpha(0.85)
        body.set_edgecolor("white")
    parts["cmeans"].set_color(INK)
    parts["cmeans"].set_linewidth(1.2)
    axA.axvline(max_bio, color=MUTED, ls="--", lw=1, label=f"max biomass ≈ {max_bio:.1f}")
    axA.set_yticks(np.arange(len(betas)))
    axA.set_yticklabels([f"{b:g}" for b in betas])
    style(axA)
    axA.set_xlabel("biomass flux  v_bio1")
    axA.set_ylabel("β (selection pressure)")
    axA.set_title("A — biomass distribution per β")
    axA.legend(frameon=False, fontsize=9, loc="lower right")

    axB.plot(bits, rng_over_max, "-", color="#0072B2", lw=1.6, zorder=2)
    axB.scatter(bits, rng_over_max, c=betas, cmap=cmap, norm=norm, s=60, zorder=3,
                edgecolor="white", linewidth=0.6)
    for x, y, b in zip(bits, rng_over_max, betas, strict=False):
        if b in (0.0, 1.0, 4.0, 16.0):
            axB.annotate(f"β={b:g}", (x, y), textcoords="offset points", xytext=(6, 6),
                         fontsize=8, color=INK)
    style(axB)
    axB.set_xlabel("bits of selection   KL(π_β‖π₀) / ln 2")
    axB.set_ylabel("biomass range (95%) / max biomass")
    axB.set_title("B — information cost of pinning growth")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axB, label="β",
                 fraction=0.046, pad=0.04)

    fig.suptitle(f"{data['name']}  (d={data['dimension']}, s_J={s_J:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    return float(bits[-1]), float(rng_over_max[-1])


def gallery(root: Path, dest: Path) -> int:
    """A 6-wide grid of every strain's biomass distribution (panel-A ridgeline), sorted by d."""
    datas = [d for d in (load_model(m) for m in census_model_dirs(root)) if d]
    datas.sort(key=lambda s: s["dimension"])
    n = len(datas)
    ncol = 6
    nrow = math.ceil(n / ncol)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.9 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, data in enumerate(datas):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        betas, bio = data["betas"], data["bio_by_beta"]
        norm = plt.Normalize(betas.min(), betas.max())
        parts = ax.violinplot(bio, positions=np.arange(len(betas)), orientation="horizontal",
                              widths=0.95, showmeans=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(cmap(norm(betas[i])))
            body.set_alpha(0.9)
            body.set_edgecolor("white")
            body.set_linewidth(0.3)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{data['name'][:22]}\nd={data['dimension']}", fontsize=7, fontweight="bold")
    fig.suptitle("Biomass distribution across the census — one panel per strain, β=0 (bottom) → 16 "
                 "(top), sorted by dimension", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {dest}  ({n} strains)")
    return 0


def find_model(root: Path, substr: str) -> str:
    for mdir in census_model_dirs(root):
        m = json.loads(Path(mdir + "/run_manifest.json").read_text())
        if substr.lower() in m["model_id"].lower():
            return mdir
    raise SystemExit(f"no completed census model matching {substr!r} under {root}")


def main(substr: str, root: Path) -> int:
    plots = Path(__file__).resolve().parent / "plots"
    plots.mkdir(exist_ok=True)
    if substr == "gallery":
        return gallery(root, plots / "fig7_biomass_gallery.png")
    if substr == "all":
        per = plots / "per_strain"
        per.mkdir(exist_ok=True)
        mdirs = census_model_dirs(root)
        print(f"generating one figure per strain — {len(mdirs)} completed census strains")
        done = 0
        for mdir in mdirs:
            data = load_model(mdir)
            if data is None:
                print(f"  skip (no bio1): {mdir}")
                continue
            safe = data["name"].replace(" ", "_")
            b, rm = plot_one(data, per / f"{safe}.png")
            done += 1
            print(f"  [{done:2d}] {data['name']:38s} d={data['dimension']:3d}  "
                  f"β16: {b:6.1f} bits, range/max {rm:.3f}")
        print(f"\nwrote {done} figures to {per}")
        return 0

    mdir = find_model(root, substr)
    data = load_model(mdir)
    if data is None:
        raise SystemExit(f"{substr}: model has no bio1 reaction")
    plot_one(data, plots / "fig6_biomass_distribution.png")
    print(f"  wrote fig6_biomass_distribution.png  ({data['name']}, d={data['dimension']})")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    substr = args[0] if args else "Pediococcus_parvulus_IOEB_9646"
    default_root = os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census")
    root = Path(args[1]) if len(args) > 1 else Path(default_root)
    sys.exit(main(substr, root))
