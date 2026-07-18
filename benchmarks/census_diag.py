"""Flux-level convergence + per-chain diagnostics for the M11.4 census; extended to autocorrelation
time in **M11.5**.

`maxent diagnose` reports R-hat/ESS for J only. Codex's review: a chain can mix in the objective
while trapped along objective-neutral flux directions, so J-only convergence is not convergence.
This post-processes the stored full-flux arrays for per-REACTION split-R-hat and ESS (mirroring the
package's own diagnostics functions), plus the per-chain manifest diagnostics (refresh drift, etc.).

## M11.5: the integrated autocorrelation time τ

The M11.5(a) schedule spec asks: given a pilot's τ, how many production sweeps reach a target ESS?
Under the census's own measured law — ESS ∝ sweeps (3.19× ESS for a 4× sweep bump, i.e. a *stable*
autocorrelation time) — the answer is ``n_samples ≈ target_ESS · (n_samples / ESS)``. So the
schedule-driving quantity is **τ_sched = n_samples / ESS_pooled**: sweeps of one chain per effective
sample. Two runs of it are emitted per movable reaction:

* ``tau_sched`` = ``n_samples / ESS`` — the constant in ``n_new = n_pilot·target_ESS / ESS_pilot``.
  It is what a resolver multiplies a target ESS by, so it is the number the schedule needs. It is
  **chain-count dependent** (it is τ_int / n_chains), which is fine — a resolver holds n_chains
  fixed between pilot and production.
* ``tau_int`` = ``n_chains · n_samples / ESS`` — the integrated autocorrelation time in sweeps, the
  frame- and chain-count-independent physical quantity (``ESS_pooled = n_chains·n_samples/τ_int``).
  This is the one to compare *across* runs when characterizing τ(d, β): it does not move if you
  change how many chains you ran.

Both are reported as **worst** (max over movable reactions = the bottleneck coordinate that governs
convergence) and a **high percentile** (p90/p95, robust to a single noisy outlier reaction — min-ESS
is itself noisy at d=145, §M11.4), plus the median. The schedule fork (worst vs percentile vs J) is
decided by `/collab` on this data, so all three are emitted rather than pre-collapsed.

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

    # M11.5: integrated autocorrelation time per movable reaction. ESS = nc·ns / tau_int, so
    # tau_int = nc·ns / ESS and the schedule-driving tau_sched = ns / ESS = tau_int / nc. A reaction
    # with ESS = 0 (a constant coordinate that slipped the movable filter) carries no autocorr
    # information and is dropped rather than reported as an infinite τ.
    ess_pos = ess[ess > 0.0]
    if ess_pos.size:
        tau_sched = ns / ess_pos  # sweeps per effective sample (what the resolver multiplies)
        tau_int = nc * tau_sched  # integrated autocorrelation time in sweeps (frame-independent)
        worst_tau_sched = float(tau_sched.max())
        p90_tau_sched = float(np.percentile(tau_sched, 90))
        p95_tau_sched = float(np.percentile(tau_sched, 95))
        median_tau_sched = float(np.median(tau_sched))
        worst_tau_int = float(tau_int.max())
        p90_tau_int = float(np.percentile(tau_int, 90))
        median_tau_int = float(np.median(tau_int))
    else:  # pragma: no cover - every movable reaction was somehow constant
        worst_tau_sched = p90_tau_sched = p95_tau_sched = median_tau_sched = float("inf")
        worst_tau_int = p90_tau_int = median_tau_int = float("inf")

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
        # M11.5: τ = n/ESS. worst = the bottleneck coordinate; p90/p95 robust to a noisy outlier.
        "worst_tau_sched": worst_tau_sched,
        "p90_tau_sched": p90_tau_sched,
        "p95_tau_sched": p95_tau_sched,
        "median_tau_sched": median_tau_sched,
        "worst_tau_int": worst_tau_int,
        "p90_tau_int": p90_tau_int,
        "median_tau_int": median_tau_int,
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
    hdr = ("beta", "mov_rx", "worstR", "medR", "minESS", "medESS", "wTauInt", "p90TauI",
           "medTauI", "chainGap", "massBal", "degen")
    print("  ".join(f"{h:>9s}" for h in hdr))
    rows = [summarize_beta(b) for b in beta_dirs]
    for s in rows:
        print("  ".join([
            f"{s['beta']:9.3g}", f"{s['n_movable_rx']:9d}", f"{s['worst_rhat']:9.4f}",
            f"{s['median_rhat']:9.4f}",
            f"{s['min_ess']:9.1f}", f"{s['median_ess']:9.1f}",
            f"{s['worst_tau_int']:9.1f}", f"{s['p90_tau_int']:9.1f}", f"{s['median_tau_int']:9.1f}",
            f"{s['max_std_chain_gap']:9.3f}",
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
