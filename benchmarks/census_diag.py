"""Flux-level convergence + per-chain diagnostics for the M11.4 census.

`maxent diagnose` reports R-hat/ESS for J only. Codex's review: a chain can mix in the objective
while trapped along objective-neutral flux directions, so J-only convergence is not convergence.
This post-processes the stored full-flux arrays for per-REACTION split-R-hat and ESS (mirroring the
package's own diagnostics functions), plus the per-chain manifest diagnostics (refresh drift, etc.).

Usage: census_diag.py <model_run_dir>   (the dir containing samples/beta_XXX/chain_YYY/)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from gsmm_compiler.diagnostics import convergence_report

MOVABLE_STD = 1e-8  # a reaction is "moving" if its pooled flux std exceeds this


def load_beta(beta_dir: str):
    chains = sorted(glob.glob(os.path.join(beta_dir, "chain_*")))
    flux = np.stack([np.load(os.path.join(c, "flux.npy")) for c in chains])  # (nc, ns, nrx)
    diags = []
    for c in chains:
        with open(os.path.join(c, "manifest.json")) as fh:
            diags.append(json.load(fh)["diagnostics"])
    return flux, diags


def summarize_beta(beta_dir: str) -> dict:
    flux, diags = load_beta(beta_dir)
    nc, ns, nrx = flux.shape
    pooled = flux.reshape(nc * ns, nrx)
    std = pooled.std(axis=0)
    movable = np.flatnonzero(std > MOVABLE_STD)
    sub = flux[:, :, movable]  # (nc, ns, n_movable)

    rep = convergence_report(sub)
    rhat, ess = rep.r_hat, rep.ess

    # chain-mean disagreement per movable reaction, standardized by pooled std
    chain_means = sub.mean(axis=1)  # (nc, n_movable)
    pooled_mean = pooled[:, movable].mean(axis=0)
    std_mov = std[movable]
    max_std_chain_gap = float(np.max(np.abs(chain_means - pooled_mean).max(axis=0) / std_mov))

    def agg(key, fn):
        return fn([d[key] for d in diags])

    return {
        "beta": float(diags[0]["beta"]),
        "n_chains": nc,
        "n_samples": ns,
        "n_movable_rx": int(movable.size),
        "worst_rhat": float(rhat.max()),
        "median_rhat": float(np.median(rhat)),
        "p99_rhat": float(np.percentile(rhat, 99)),
        "frac_rhat_gt_1.01": float(np.mean(rhat > 1.01)),
        "min_ess": float(ess.min()),
        "median_ess": float(np.median(ess)),
        "frac_ess_lt_400": float(np.mean(ess < 400)),
        "pooled_min_ess": float(ess.min()),  # per-reaction ESS already pools chains
        "max_std_chain_gap": max_std_chain_gap,
        "max_refresh_drift": agg("max_refresh_drift", max),
        "max_bound_violation": agg("max_bound_violation", max),
        "max_mass_balance_residual": agg("max_mass_balance_residual", max),
        "n_degenerate_steps": agg("n_degenerate_steps", sum),
        "min_start_shrink": agg("start_shrink", min),
        "mean_chord_length": float(np.mean([d["mean_chord_length"] for d in diags])),
    }


def main(run_dir: str) -> int:
    samples = os.path.join(run_dir, "samples")
    beta_dirs = sorted(glob.glob(os.path.join(samples, "beta_*")))
    if not beta_dirs:
        print(f"no beta dirs under {samples}")
        return 1
    print(f"# {os.path.basename(run_dir.rstrip('/'))}")
    hdr = ("beta", "mov_rx", "worstR", "medR", "%R>1.01", "minESS", "medESS", "%ESS<400",
           "chainGap", "refDrift", "massBal", "degen")
    print("  ".join(f"{h:>9s}" for h in hdr))
    rows = [summarize_beta(b) for b in beta_dirs]
    for s in rows:
        print("  ".join([
            f"{s['beta']:9.3g}", f"{s['n_movable_rx']:9d}", f"{s['worst_rhat']:9.4f}",
            f"{s['median_rhat']:9.4f}", f"{s['frac_rhat_gt_1.01']*100:9.1f}",
            f"{s['min_ess']:9.1f}", f"{s['median_ess']:9.1f}", f"{s['frac_ess_lt_400']*100:9.1f}",
            f"{s['max_std_chain_gap']:9.3f}", f"{s['max_refresh_drift']:9.2e}",
            f"{s['max_mass_balance_residual']:9.2e}", f"{s['n_degenerate_steps']:9d}",
        ]))
    # emit JSON for programmatic use
    out = os.path.join(run_dir, "census_flux_diag.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"# wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
