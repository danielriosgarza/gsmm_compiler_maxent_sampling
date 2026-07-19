"""M11.5(c) census plots — the three components as figures.

Reads the JSON artifacts written by the census + the two studies and renders PNGs to
``benchmarks/plots/``. Reusable: run it on the partial census (per-strain ``census_row.json`` files
assembled on the fly) or on the finished ``census.json``; it plots whatever is present.

Colour: the **Okabe–Ito** palette — the scientific reference colourblind-safe categorical set
(Okabe & Ito 2008), assigned by entity in fixed order, never cycled (dataviz skill non-negotiable).
One measure per axis, recessive grid/axes, legends for ≥2 series, selective direct labels.

Run:  .venv/bin/python benchmarks/plot_census.py [OUT_ROOT]
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

# Okabe–Ito, fixed order. black, orange, sky-blue, bluish-green, yellow, blue, vermillion, purple.
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_census(root: Path) -> list[dict]:
    cj = root / "census.json"
    if cj.exists():
        rows = json.loads(cj.read_text())
    else:  # assemble from per-strain rows (partial run)
        rows = [json.loads(Path(f).read_text())
                for f in glob.glob(str(root / "*" / "census_row.json"))]
    return [r for r in rows if r.get("status") == "OK"]


def p90_ess(rung: dict) -> float:
    t = rung.get("p90_tau_int")
    return rung["n_chains"] * rung["n_samples"] / t if t and np.isfinite(t) else 0.0


# --------------------------------------------------------------------------------------------------
def fig_covariance(root: Path, out: Path) -> None:
    data = json.loads((root / "covariance_study.json").read_text())
    fig, axes = plt.subplots(1, len(data), figsize=(5.4 * len(data), 4.4), squeeze=False)
    for ax, r in zip(axes[0], data, strict=False):
        for tag, colour, marker in (("T0", OI[5], "o"), ("T1", OI[6], "s")):
            c = r[tag]
            eig = np.array(c["full_eigenvalues_desc"], dtype=float)
            rank = np.arange(1, eig.size + 1)
            label = (f"{'T₀ support' if tag == 'T0' else 'T₁ pilot'}  "
                     f"(cond {c['official_condition_number']:.2g})")
            ax.semilogy(rank, eig, marker=marker, ms=3, lw=1.6, color=colour, label=label)
        d = r["T0"]["dimension"]
        style(ax)
        ax.set_title(f"{r['label'].replace('_', ' ')}  (d={d}, C_q full rank {d}/{d})")
        ax.set_xlabel("eigenvalue rank (largest → smallest)")
        ax.set_ylabel("eigenvalue  (variance along axis)")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        # annotate the single near-flat direction that sets the conditioning
        eig0 = np.array(r["T0"]["full_eigenvalues_desc"], dtype=float)
        ax.annotate("one near-flat\ndirection sets cond",
                    xy=(eig0.size, eig0[-1]), xytext=(0.42 * eig0.size, eig0[-1] * 40),
                    fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    fig.suptitle("Component 2 — geometry-pilot covariance spectra: full rank, one hard direction; "
                 "reround helps d=46, hurts d=145", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig1_covariance_spectra.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig1_covariance_spectra.png'}")


# --------------------------------------------------------------------------------------------------
def fig_sj(root: Path, out: Path) -> None:
    data = json.loads((root / "sj_replication" / "sj_replication.json").read_text())
    data = [d for d in data if d.get("n_seeds_ok", 0) >= 2]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: per-seed relative deviation from the strain mean, with the reported-se band.
    for i, d in enumerate(data):
        colour = OI[[5, 3, 1][i % 3]]
        ok = [s for s in d["seeds"] if s["status"] == "OK"]
        mean = d["s_J_mean"]
        rel = [100 * (s["s_J"] - mean) / mean for s in ok]
        se = 100 * d["mean_reported_rel_se"]
        # reported-se band
        axL.add_patch(plt.Rectangle((i - 0.32, -se), 0.64, 2 * se, color=colour, alpha=0.13, lw=0))
        axL.plot([i - 0.32, i + 0.32], [0, 0], color=colour, lw=1, alpha=0.5)
        axL.scatter([i] * len(rel), rel, color=colour, s=36, zorder=3,
                    label=f"{d['label'].split('_')[0]} d={d['seeds'][0].get('dimension', '')}")
        # between-seed SD as an error bar
        sd = 100 * d["s_J_between_seed_rel_sd"]
        axL.errorbar([i + 0.0], [0], yerr=[[sd], [sd]], color=colour, lw=2, capsize=5, zorder=2)
    axL.set_xticks(range(len(data)))
    axL.set_xticklabels([d["label"].split("_")[0].replace("_", " ") for d in data])
    axL.axhline(0, color=MUTED, lw=0.8)
    style(axL)
    axL.set_ylabel("s_J deviation from strain mean  (%)")
    axL.set_title("s_J across 5 pilot seeds\n(shaded = ±reported se, bar = ±between-seed SD)")

    # Right: the agreement — between-seed SD vs reported se, per strain.
    labels = [d["label"].split("_")[0] for d in data]
    between = [100 * d["s_J_between_seed_rel_sd"] for d in data]
    reported = [100 * d["mean_reported_rel_se"] for d in data]
    x = np.arange(len(data))
    axR.bar(x - 0.19, between, 0.36, color=OI[5], label="measured between-seed SD")
    axR.bar(x + 0.19, reported, 0.36, color=OI[1], label="reported within-pilot se")
    for xi, (b, rp) in enumerate(zip(between, reported, strict=False)):
        axR.text(xi - 0.19, b + 0.1, f"{b:.1f}", ha="center", fontsize=8, color=INK)
        axR.text(xi + 0.19, rp + 0.1, f"{rp:.1f}", ha="center", fontsize=8, color=INK)
    axR.set_xticks(x)
    axR.set_xticklabels([f"{lab}\nd={dd['label'].split('_d')[-1]}"
                         for lab, dd in zip(labels, data, strict=False)])
    style(axR)
    axR.set_ylabel("relative variability  (%)")
    axR.set_title("The reported precision is honest\n(between-seed spread ≈ reported se)")
    axR.legend(frameon=False, fontsize=9)
    fig.suptitle("Component 3 — s_J under independent pilot seeds: reproducible in distribution, "
                 "and the precision warning tells the truth", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig2_sJ_replication.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig2_sJ_replication.png'}")


# --------------------------------------------------------------------------------------------------
def fig_census(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda r: r["dimension"])
    d = np.array([r["dimension"] for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

    # (a) schedule scaling — resolved n_samples vs d
    ax = axes[0, 0]
    n_res = np.array([r["resolved_n_samples"] for r in rows], dtype=float)
    caps = np.array([r["cap_hit"] for r in rows])
    ax.scatter(d[~caps], n_res[~caps], color=OI[5], s=34)
    if caps.any():
        ax.scatter(d[caps], n_res[caps], color=OI[6], s=44, marker="x", label="cap hit")
        ax.legend(frameon=False, fontsize=9)  # only a legend when there is a 2nd series to name
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("affine dimension d")
    ax.set_ylabel("resolved n_samples  (sweeps)")
    ax.set_title(f"(a) schedule sizes production per strain  (n={len(rows)})")

    # (b) β=0 target attainment — p90-ESS vs d, target line
    ax = axes[0, 1]
    b0 = [next((x for x in r["rungs"] if x["beta"] == 0.0), None) for r in rows]
    pe = np.array([p90_ess(x) if x else 0 for x in b0])
    ax.scatter(d, pe, color=OI[3], s=34, zorder=3)
    ax.axhline(400, color=OI[6], lw=1.4, ls="--", label="target_ess = 400")
    ax.axhline(360, color=MUTED, lw=0.8, ls=":", label="target − 10%")
    style(ax)
    ax.set_xlabel("affine dimension d")
    ax.set_ylabel("achieved β=0 p90 flux-ESS")
    ax.set_title("(b) the schedule delivers its β=0 target")
    ax.legend(frameon=False, fontsize=9)

    # (c) β-inflation — p90-ESS(β)/p90-ESS(0) per strain + median
    ax = axes[1, 0]
    betas = sorted({x["beta"] for r in rows for x in r["rungs"]})
    curves = []
    for r in rows:
        by_b = {x["beta"]: p90_ess(x) for x in r["rungs"]}
        e0 = by_b.get(0.0, 0.0)
        if e0 <= 0:
            continue
        y = [by_b.get(b, np.nan) / e0 for b in betas]
        curves.append(y)
        ax.plot(betas, y, color=MUTED, lw=0.7, alpha=0.35)
    if curves:
        med = np.nanmedian(np.array(curves), axis=0)
        ax.plot(betas, med, color=OI[6], lw=2.4, marker="o", ms=4, label="median over strains")
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("β (selection pressure)")
    ax.set_ylabel("p90-ESS(β) / p90-ESS(0)")
    ax.set_title("(c) β-inflation is reported, not hidden\n(β=0-sized → high-β under-mixes)")
    ax.plot([], [], color=MUTED, lw=0.7, label="per strain")
    ax.legend(frameon=False, fontsize=9)

    # (d) s_J precision vs d
    ax = axes[1, 1]
    se = np.array([100 * r["s_J_relative_se"] for r in rows])
    ax.scatter(d, se, color=OI[1], s=34, zorder=3)
    ax.axhline(2.0, color=OI[6], lw=1.4, ls="--", label="precision target 2%")
    style(ax)
    ax.set_xlabel("affine dimension d")
    ax.set_ylabel("s_J relative se  (%)")
    ax.set_title("(d) s_J precision degrades with d (a warning, not a gate)")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Component 1 — 40-strain production census at pilot_ess  "
                 f"({len(rows)} strains, seed 0, 8-rung ladder)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig3_census_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig3_census_scaling.png'}")


# --------------------------------------------------------------------------------------------------
def fig_monotonicity(rows: list[dict], out: Path) -> None:
    """Normalized mean-J trajectories: q(β) = (E[J](β) − E₀[J]) / Δ₀ — comparable across strains,
    monotone by the §1.6.2 theorem."""
    rows = sorted(rows, key=lambda r: r["dimension"])
    fig, ax = plt.subplots(figsize=(7.2, 5))
    n_mono = 0
    for r in rows:
        rungs = sorted(r["rungs"], key=lambda x: x["beta"])
        ej = np.array([x["mean_j"] for x in rungs])
        betas = np.array([x["beta"] for x in rungs])
        e0 = ej[0]
        delta0 = r.get("pilot_gap") or (ej[-1] - e0)
        q = (ej - e0) / delta0 if delta0 else ej - e0
        mono = bool(np.all(np.diff(ej) >= -3 * np.hypot(
            np.array([x["mean_j_mcse"] for x in rungs[:-1]]),
            np.array([x["mean_j_mcse"] for x in rungs[1:]]))))
        n_mono += mono
        ax.plot(betas, q, color="#9aa7b5" if mono else OI[6], lw=1.1,
                alpha=0.6 if mono else 1.0)
    style(ax)
    ax.set_xlabel("β (selection pressure)")
    ax.set_ylabel("fractional gap closed  q(β) = (E[J]−E₀[J]) / Δ₀")
    ax.set_title(f"Mean-J monotonicity across the ladder\n"
                 f"{n_mono}/{len(rows)} strains monotone within 3·MCSE (the §1.6.2 theorem)")
    ax.plot([], [], color="#9aa7b5", lw=1.1, label="monotone")
    ax.plot([], [], color=OI[6], lw=1.1, label="violation (>3·MCSE drop)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig4_meanj_monotonicity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig4_meanj_monotonicity.png'}")


def main(root: Path) -> int:
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    rows = load_census(root)
    print(f"census rows: {len(rows)} OK")
    if (root / "covariance_study.json").exists():
        fig_covariance(root, out)
    if (root / "sj_replication" / "sj_replication.json").exists():
        fig_sj(root, out)
    if rows:
        fig_census(rows, out)
        fig_monotonicity(rows, out)
    print(f"plots in {out}")
    return 0


if __name__ == "__main__":
    default = Path(os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census"))
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
