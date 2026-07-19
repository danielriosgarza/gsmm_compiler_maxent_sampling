"""M11.5(c) component 3 — s_J under an independent pilot seed.

``s_J = σ̂₀`` is estimated from a β=0 pilot, and M10.2e/§1.6.6 claim it is reproducible **in
distribution, not bit-for-bit**: a different pilot seed is "another honest draw from the same pilot
law". Each run reports a *within-pilot* precision, ``se(σ̂₀)/σ̂₀`` (2.6%…5.3% over d — M11.4). That se
is a **prediction** of how much ``s_J`` should move if you reran the pilot. This replicates the
whole pilot DAG at several independent seeds and measures the **actual between-seed spread**, then
asks the one question a single run's se cannot answer itself: *does the reported precision hold?*

If the between-seed relative SD ≈ the mean reported se, the precision warning is honest and β's
label carries exactly the stated uncertainty. If the spread is much larger, the se under-reports and
cross-strain β comparison (§1.1) is noisier than advertised — a calibration finding, never a
correctness one (every individual run samples exactly π_{β/s_J} for its own s_J).

Changing ``sampler.seed`` re-keys both pilot streams (``provenance.stream_seed`` folds it into the
``SeedSequence`` entropy); geometry is keyed on the separate ``geom_seed`` and is cached once and
reused. So one cache dir per strain, pilots recomputed per seed, and ``schedule_mode=fixed`` with a
trivial production (``n_samples=5``) — we want the pilot's ``s_J``, not a production chain.

Run:  .venv/bin/python benchmarks/sj_seed_replication.py [OUT_ROOT]
      (OUT_ROOT defaults to $CENSUS_OUT/sj_replication or /tmp/gsmm_m115_census/sj_replication)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv" / "bin" / "gsmm-compiler"
CURATED = Path(
    os.environ.get(
        "CURATED_MODELS",
        "/home/mcpu/GitHub/metabolicSubcommunities/models/gapfilled/method_3_curated",
    )
)

# Three subjects across the dimension spread: the anaerobe control, a mid sentinel, and d=145.
SUBJECTS = [
    ("bifido_d46", "GCF_000010425_1_ASM1042v1"),
    ("lactis_d51", "GCF_964062975_1_Lactococcus_lactis_CIRM_BIA2553"),
    ("rahnella_d145", "GCA_964063365_1_Rahnella_aquatiliss"),
]
SEEDS = [0, 1, 2, 3, 4]


def resolve(stem: str) -> Path:
    hits = sorted(CURATED.glob(stem + "*.json"))
    if len(hits) != 1:
        raise SystemExit(f"{stem!r} resolved to {len(hits)} files: {hits}")
    return hits[0]


def run_seed(root: Path, label: str, model: Path, seed: int) -> dict:
    out = root / label / f"seed_{seed}" / "out"
    # One cache dir per strain, shared across seeds: geometry (geom_seed) is cached once; the pilots
    # (sampler.seed) miss and recompute per seed, which is exactly the independent draw we want.
    cache = root / label / "cache"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV), "maxent", "sample", str(model),
        "--out", str(out), "--cache-dir", str(cache), "--workers", "8",
        "--set", "sampler.energy_scale=pilot_sd",
        "--set", "sampler.pilot_reround=true",
        "--set", "sampler.pilot_chains=4",
        "--set", "sampler.pilot_burn_in=2000",
        "--set", "sampler.pilot_samples=2000",
        "--set", "sampler.betas=[0.0]",
        "--set", "sampler.n_chains=4",
        "--set", "sampler.burn_in=5",
        "--set", "sampler.n_samples=5",  # trivial production; we want the pilot's s_J
        "--set", "sampler.schedule_mode=fixed",
        "--set", f"sampler.seed={seed}",
    ]
    log = out.parent / "log.txt"
    t0 = time.monotonic()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        return {"seed": seed, "status": "FAIL", "wall_seconds": round(dt, 1),
                "reason": log.read_text()[-300:]}
    mdir = next(p.parent for p in out.glob("*/*/samples") if p.is_dir())
    m = json.loads((mdir / "run_manifest.json").read_text())
    es = m["energy_scale"]
    cal = m["calibration"]
    return {
        "seed": seed,
        "status": "OK",
        "wall_seconds": round(dt, 1),
        "s_J": es["energy_scale"],
        "reported_relative_se": es["pilot_relative_se"],
        "j_star": es["j_star"],
        "pilot_mean_j": es["pilot_mean_j"],
        "pilot_gap": es["pilot_gap"],
        "headroom_g": es["pilot_headroom_g"],
        "pilot_r_hat_j": es["pilot_r_hat_j"],
        "pilot_sd_chain_ratio": es["pilot_sd_chain_ratio"],
        "cond_before": cal["bootstrap_condition_number"],
        "cond_after": cal["final_condition_number"],
    }


def main(root: Path) -> int:
    all_results = []
    for label, stem in SUBJECTS:
        model = resolve(stem)
        print(f"[{label}] {model.name}")
        seeds = []
        for seed in SEEDS:
            row_path = root / label / f"seed_{seed}" / "sj_row.json"
            if row_path.exists():
                r = json.loads(row_path.read_text())
                print(f"    seed {seed}: SKIP (s_J={r.get('s_J')})")
            else:
                r = run_seed(root, label, model, seed)
                row_path.parent.mkdir(parents=True, exist_ok=True)
                row_path.write_text(json.dumps(r, indent=2))
                if r["status"] == "OK":
                    print(f"    seed {seed}: s_J={r['s_J']:.4f} "
                          f"(reported se {100 * r['reported_relative_se']:.1f}%) "
                          f"G={r['headroom_g']:.2f} ({r['wall_seconds']:.0f}s)")
                else:
                    print(f"    seed {seed}: FAIL — {r.get('reason', '')[-120:]}")
            seeds.append(r)

        ok = [r for r in seeds if r["status"] == "OK"]
        summary: dict = {"label": label, "model_file": model.name, "n_seeds_ok": len(ok),
                         "seeds": seeds}
        if len(ok) >= 2:
            sj = np.array([r["s_J"] for r in ok], dtype=float)
            summary.update({
                "s_J_mean": float(sj.mean()),
                "s_J_std": float(sj.std(ddof=1)),
                "s_J_between_seed_rel_sd": float(sj.std(ddof=1) / sj.mean()),
                "mean_reported_rel_se": float(np.mean([r["reported_relative_se"] for r in ok])),
                "s_J_min": float(sj.min()),
                "s_J_max": float(sj.max()),
                "headroom_g_mean": float(np.mean([r["headroom_g"] for r in ok])),
            })
            print(f"  → s_J = {summary['s_J_mean']:.4f} ± {summary['s_J_std']:.4f} "
                  f"(between-seed {100 * summary['s_J_between_seed_rel_sd']:.1f}% vs "
                  f"reported {100 * summary['mean_reported_rel_se']:.1f}%)")
        all_results.append(summary)

    dest = root / "sj_replication.json"
    root.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    default = Path(os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census")) / "sj_replication"
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out_root.mkdir(parents=True, exist_ok=True)
    sys.exit(main(out_root))
