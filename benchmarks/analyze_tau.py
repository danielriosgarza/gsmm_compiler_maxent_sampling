"""Analyze the M11.5(a) τ sweep: scaling law τ(d) at β=0, the β-inflation factor, and worst-vs-p90
robustness. Prints the markdown-ready tables + fits behind `benchmarks/M11_5_SCHEDULE_TAU.md`.

Run:  .venv/bin/python benchmarks/analyze_tau.py [tau_sweep.json]
      (defaults to $SWEEP_OUT/tau_sweep.json or /tmp/gsmm_schedule_sweep/tau_sweep.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT = Path(
    os.environ.get("SWEEP_OUT", "/tmp/gsmm_schedule_sweep")
) / "tau_sweep.json"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
rows = json.loads(SRC.read_text())


def by_beta(beta: float) -> list[dict]:
    return sorted([r for r in rows if r["beta"] == beta], key=lambda r: r["dimension"])


def powerfit(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a·d^p; return (p, a) via least squares on logs."""
    mask = (d > 0) & (y > 0)
    p, loga = np.polyfit(np.log(d[mask]), np.log(y[mask]), 1)
    return float(p), float(np.exp(loga))


metric_cols = ["worst_tau_int", "p90_tau_int", "median_tau_int"]

print("=" * 92)
print("TABLE 1 — τ_int (integrated autocorrelation time, sweeps) vs d, per β")
print("=" * 92)
hdr = f"{'strain':<20}{'d':>5}{'β':>5}" + "".join(
    f"{m.replace('_tau_int', ''):>10}" for m in metric_cols
)
hdr += f"{'minESS':>9}{'medESS':>9}{'worstR':>9}{'se_sJ%':>8}"
print(hdr)
for beta in (0.0, 1.0, 8.0, 16.0):
    for r in by_beta(beta):
        se = r.get("se_sJ")
        se_s = f"{se * 100:7.1f}" if isinstance(se, (int, float)) else "     -- "
        print(
            f"{r['strain']:<20}{r['dimension']:>5}{r['beta']:>5g}"
            + "".join(f"{r[m]:>10.1f}" for m in metric_cols)
            + f"{r['min_ess']:>9.0f}{r['median_ess']:>9.0f}{r['worst_rhat']:>9.3f}{se_s}"
        )
    print("-" * 92)

print()
print("=" * 92)
print("TABLE 2 — dimension scaling  τ_int ∝ d^p  (fit across strains, per β and per statistic)")
print("=" * 92)
print(f"{'β':>5}{'stat':>16}{'exponent p':>13}{'τ@d=46':>10}{'τ@d=145':>10}{'ratio':>8}")
for beta in (0.0, 1.0, 8.0, 16.0):
    sub = by_beta(beta)
    d = np.array([r["dimension"] for r in sub], float)
    for m in metric_cols:
        y = np.array([r[m] for r in sub], float)
        p, a = powerfit(d, y)
        t46, t145 = a * 46**p, a * 145**p
        print(
            f"{beta:>5g}{m.replace('_tau_int', ''):>16}{p:>13.2f}"
            f"{t46:>10.1f}{t145:>10.1f}{t145 / t46:>8.2f}"
        )
    print("-" * 60)

print()
print("=" * 92)
print("TABLE 3 — β-inflation  τ_int(β)/τ_int(0)  per strain (does a β=0 pilot under-size high β?)")
print("=" * 92)
print(
    f"{'strain':<20}{'d':>5}"
    + "".join(f"{'β=' + str(int(b)):>12}" for b in (1, 8, 16))
    + "   (statistic: p90_tau_int)"
)
infl: dict[float, list[float]] = {1.0: [], 8.0: [], 16.0: []}
for r0 in by_beta(0.0):
    strain, base = r0["strain"], r0["p90_tau_int"]
    line = f"{strain:<20}{r0['dimension']:>5}"
    for b in (1.0, 8.0, 16.0):
        rb = next((r for r in rows if r["strain"] == strain and r["beta"] == b), None)
        if rb and base > 0:
            infl[b].append(rb["p90_tau_int"] / base)
            line += f"{rb['p90_tau_int'] / base:>12.2f}"
        else:
            line += f"{'--':>12}"
    print(line)
print("-" * 92)
print(f"{'MEAN inflation':<20}{'':>5}" + "".join(f"{np.mean(infl[b]):>12.2f}" for b in infl))
print(f"{'MAX inflation':<20}{'':>5}" + "".join(f"{np.max(infl[b]):>12.2f}" for b in infl))

print()
print("=" * 92)
print("TABLE 4 — worst vs p90: how much of 'worst' is one near-constant outlier coordinate?")
print("=" * 92)
print(f"{'β':>5}{'median(worst/p90)':>22}{'max(worst/p90)':>18}")
for beta in (0.0, 1.0, 8.0, 16.0):
    sub = by_beta(beta)
    ratio = np.array([r["worst_tau_int"] / r["p90_tau_int"] for r in sub if r["p90_tau_int"] > 0])
    print(f"{beta:>5g}{np.median(ratio):>22.2f}{np.max(ratio):>18.2f}")

print()
print("=" * 92)
print("SIZING PREVIEW — sweeps to reach target p90-ESS=400 from a 2000-sweep β=0 pilot")
print("  n_new = 2000 · target_ESS / ESS_pilot(β=0, p90);  ESS_pilot,p90 = n·nc / p90_tau_int(β=0)")
print("=" * 92)
TARGET = 400
_cols = ("strain", "d", "ESSp90(β0)", "n_new(β0)", "×β16 infl", "n_new(β16)")
print(f"{_cols[0]:<20}{_cols[1]:>5}" + "".join(f"{c:>12}" for c in _cols[2:]))
for r0 in by_beta(0.0):
    strain, d_ = r0["strain"], r0["dimension"]
    ess_p90_b0 = r0["n_chains"] * r0["n_samples"] / r0["p90_tau_int"]
    n_new_b0 = 2000 * TARGET / ess_p90_b0
    r16 = next((r for r in rows if r["strain"] == strain and r["beta"] == 16.0), None)
    infl16 = (r16["p90_tau_int"] / r0["p90_tau_int"]) if r16 else float("nan")
    print(
        f"{strain:<20}{d_:>5}{ess_p90_b0:>12.0f}{n_new_b0:>12.0f}"
        f"{infl16:>12.2f}{n_new_b0 * infl16:>12.0f}"
    )
