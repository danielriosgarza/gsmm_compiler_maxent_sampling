"""M11.5(c) census analysis — turn ``census.json`` into the tables behind ``M11_5_CENSUS.md``.

Reads the aggregated per-strain rows written by ``census_m115.py`` and answers the census's
questions across the batch:

1. **Geometry census** — did build-geometry succeed on all 40 (fail-closed refusals are legitimate
   rows, not crashes)?
2. **Schedule resolution** — did ``pilot_ess`` size sensibly on 40 real strains (monotone-ish in d,
   how many cap hits)?
3. **β=0 target attainment** — did the β=0-sized schedule reach ``target_ess`` at β=0 across the
   batch (the one thing the schedule *predicts*)?
4. **s_J calibration** — s_J, se(σ̂₀), headroom G vs d; how many precision warnings.
5. **Validity** — bound violation / mass-balance / refresh drift / degenerate steps across every
   emitted sample (the M11.4 contract, now on 40).
6. **β-inflation** — achieved ESS decay with β (what the schedule does NOT correct and reports
   instead).
7. **mean-J monotonicity** — E[J] non-decreasing across the 8-rung ladder, in MCSE units.
8. **Reachability** — every production transform T₁ certified.

Run:  .venv/bin/python benchmarks/analyze_census.py [OUT_ROOT]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def p90_ess(rung: dict) -> float:
    t = rung.get("p90_tau_int")
    if not t or not np.isfinite(t):
        return 0.0
    return rung["n_chains"] * rung["n_samples"] / t


def median_ess(rung: dict) -> float:
    return float(rung.get("median_ess", 0.0))


def worst_monotonicity_z(rungs: list[dict]) -> tuple[float, float]:
    """Worst adjacent-rung E[J] drop, in combined-MCSE units (z<0 = a drop). Returns (worst_z,
    worst_delta). E[J] should be non-decreasing across the ladder (mean-J monotonicity, §1.6.2)."""
    worst_z = float("inf")
    worst_delta = 0.0
    ordered = sorted(rungs, key=lambda r: r["beta"])
    for a, b in zip(ordered, ordered[1:], strict=False):
        delta = b["mean_j"] - a["mean_j"]
        mcse = float(np.hypot(a.get("mean_j_mcse", 0.0), b.get("mean_j_mcse", 0.0)))
        z = delta / mcse if mcse > 0 else float("inf")
        if z < worst_z:
            worst_z, worst_delta = z, delta
    return worst_z, worst_delta


def main(root: Path) -> int:
    census = json.loads((root / "census.json").read_text())
    ok = [r for r in census if r.get("status") == "OK"]
    failed = [r for r in census if r.get("status") != "OK"]
    ok.sort(key=lambda r: r["dimension"])

    print(f"# M11.5(c) census — {len(census)} strains, {len(ok)} OK, {len(failed)} not OK\n")

    # ---- 1. geometry census / failures --------------------------------------------------------
    if failed:
        print("## Strains that did NOT complete (fail-closed rows)")
        for r in failed:
            print(f"  {r['label']:40s} {r.get('status')}: {str(r.get('reason'))[:90]}")
        print()

    # ---- 2. schedule resolution ---------------------------------------------------------------
    print("## Schedule resolution (pilot_ess, target_ess=400, q=0.90)")
    print(f"{'strain':32s} {'d':>4s} {'n_resolved':>10s} {'τ_q(sw)':>8s} {'cap':>4s} "
          f"{'s_J':>8s} {'se%':>5s}")
    caps = 0
    for r in ok:
        caps += bool(r["cap_hit"])
        print(f"{r['label'][:32]:32s} {r['dimension']:4d} {r['resolved_n_samples']:10d} "
              f"{r['quantile_tau_int']:8.1f} {str(r['cap_hit']):>4s} "
              f"{r['s_J']:8.3g} {100 * r['s_J_relative_se']:5.1f}")
    print(f"  cap hits: {caps}/{len(ok)}\n")

    # ---- 3. β=0 target attainment -------------------------------------------------------------
    print("## β=0 target attainment (schedule sizes to p90-ESS≈target=400)")
    print(f"{'strain':32s} {'d':>4s} {'p90ESS':>7s} {'medESS':>7s} {'minESS':>7s} {'hit≥360?':>8s}")
    hits = 0
    for r in ok:
        b0 = next((x for x in r["rungs"] if x["beta"] == 0.0), None)
        if not b0:
            continue
        pe = p90_ess(b0)
        hit = pe >= 360  # within 10% of target 400
        hits += hit
        print(f"{r['label'][:32]:32s} {r['dimension']:4d} {pe:7.0f} {median_ess(b0):7.0f} "
              f"{b0['min_ess']:7.0f} {'yes' if hit else 'NO':>8s}")
    print(f"  β=0 p90-ESS ≥ 360 (target 400 − 10%): {hits}/{len(ok)}\n")

    # ---- 4. validity across every rung of every strain ----------------------------------------
    max_boundviol = max_massbal = max_drift = 0.0
    total_degen = 0
    worst_rhat = 0.0
    for r in ok:
        for rung in r["rungs"]:
            max_boundviol = max(max_boundviol, rung["max_bound_violation"])
            max_massbal = max(max_massbal, rung["max_mass_balance_residual"])
            max_drift = max(max_drift, rung["max_refresh_drift"])
            total_degen += rung["n_degenerate_steps"]
            worst_rhat = max(worst_rhat, rung["worst_rhat"])
    print("## Validity — every emitted sample, all strains × all 8 rungs")
    print(f"  max bound violation      : {max_boundviol:.3g}   (contract 0)")
    print(f"  max mass-balance residual: {max_massbal:.3g}   (contract ≤ 1e-9)")
    print(f"  max refresh drift        : {max_drift:.3g}   (must be ~noise)")
    print(f"  total degenerate steps   : {total_degen}")
    print(f"  worst flux R̂ (any rung)  : {worst_rhat:.3f}\n")

    # ---- 5. mean-J monotonicity ---------------------------------------------------------------
    print("## mean-J monotonicity (E[J] non-decreasing across ladder; z = drop / combined MCSE)")
    print(f"{'strain':32s} {'d':>4s} {'worst_z':>8s} {'worst_Δ':>9s} {'verdict':>18s}")
    n_mono = 0
    for r in ok:
        z, delta = worst_monotonicity_z(r["rungs"])
        # a drop within ~3σ of MCSE is noise, not a violation (M11.4 convention)
        verdict = "monotone" if z > -3.0 else "DROP > 3σ"
        n_mono += z > -3.0
        print(f"{r['label'][:32]:32s} {r['dimension']:4d} {z:8.2f} {delta:9.3f} {verdict:>18s}")
    print(f"  monotone within 3σ: {n_mono}/{len(ok)}\n")

    # ---- 6. β-inflation (achieved p90-ESS decay with β) ---------------------------------------
    print("## β-inflation — achieved p90-ESS(β) / p90-ESS(β=0), the schedule does NOT correct this")
    betas = sorted({rung["beta"] for r in ok for rung in r["rungs"]})
    hdr = "  ".join(f"β={b:g}" for b in betas)
    print(f"{'strain':28s} {hdr}")
    infl16 = []
    for r in ok:
        by_b = {rung["beta"]: p90_ess(rung) for rung in r["rungs"]}
        e0 = by_b.get(0.0, 0.0)
        cells = []
        for b in betas:
            e = by_b.get(b, 0.0)
            cells.append(f"{(e / e0 if e0 else 0):.2f}")
        if 16.0 in by_b and e0:
            infl16.append(e0 / by_b[16.0] if by_b[16.0] else float("inf"))
        print(f"{r['label'][:28]:28s} " + "  ".join(f"{c:>5s}" for c in cells))
    if infl16:
        arr = np.array([x for x in infl16 if np.isfinite(x)])
        print(f"  β=16 inflation (ESS0/ESS16): median {np.median(arr):.1f}×, "
              f"max {arr.max():.1f}×, mean {arr.mean():.1f}×\n")

    # ---- 7. reachability + s_J-vs-d ------------------------------------------------------------
    all_cert = all(r["reachable_certified"] for r in ok)
    min_margin = min((r["reachable_margin"] for r in ok), default=float("nan"))
    n_warn = sum(r["s_J_precision_warning"] for r in ok)
    print("## Reachability + calibration")
    print(f"  T₁ certified: {sum(r['reachable_certified'] for r in ok)}/{len(ok)} "
          f"(all={all_cert}); min margin {min_margin:.1f}× inside contract")
    print(f"  s_J precision warnings (se > 2%): {n_warn}/{len(ok)}")
    se_by_d = [(r["dimension"], r["s_J_relative_se"]) for r in ok]
    if se_by_d:
        lo = min(se_by_d, key=lambda t: t[0])
        hi = max(se_by_d, key=lambda t: t[0])
        print(f"  se(σ̂₀): {100 * lo[1]:.1f}% at d={lo[0]} … {100 * hi[1]:.1f}% at d={hi[0]}")

    # emit a machine-readable summary for the writeup
    summary = {
        "n_strains": len(census),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "cap_hits": caps,
        "beta0_target_hits": hits,
        "max_bound_violation": max_boundviol,
        "max_mass_balance_residual": max_massbal,
        "max_refresh_drift": max_drift,
        "total_degenerate_steps": total_degen,
        "worst_flux_rhat": worst_rhat,
        "n_monotone_within_3sigma": n_mono,
        "all_t1_certified": all_cert,
        "min_reachable_margin": min_margin,
        "n_precision_warnings": n_warn,
    }
    (root / "census_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {root / 'census_summary.json'}")
    return 0


if __name__ == "__main__":
    default = Path(os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census"))
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
