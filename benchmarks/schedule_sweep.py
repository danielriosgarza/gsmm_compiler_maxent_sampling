"""M11.5(a) MEASURE-FIRST driver: sweep the production sampler across a wider strain set at several
β, and aggregate the integrated autocorrelation time τ vs (d, β). Produces the data behind
`benchmarks/M11_5_SCHEDULE_TAU.md`.

The spec forbids fitting a scaling rule to the 4 census sentinels. This runs 9 strains spanning
d = 34…145 (three of them the census sentinels, so the numbers cross-check against M11.4) at
β ∈ {0, 1, 8, 16}, under the *production* config (energy_scale=pilot_sd, pilot_reround=true — so the
chain steps in T₁, the frame a resolver's scale pilot would measure in), and reports τ per (d, β).

Run:  .venv/bin/python benchmarks/schedule_sweep.py [OUTPUT_ROOT]
      (OUTPUT_ROOT defaults to $SWEEP_OUT or /tmp/gsmm_schedule_sweep)
It shells out to the CLI per strain (clean cache isolation), then imports the census_diag summarizer
(sibling module) for τ, writing OUTPUT_ROOT/tau_sweep.json. Feed that to analyze_tau.py.
Deterministic: seed 0 everywhere.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_diag import summarize_beta  # noqa: E402  (sibling module in benchmarks/)

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv" / "bin" / "gsmm-compiler"
# The curated batch this package exists to run (see CLAUDE.md / DEVELOPMENT_STATUS verify commands).
CURATED = Path(
    os.environ.get(
        "CURATED_MODELS",
        "/home/mcpu/GitHub/metabolicSubcommunities/models/gapfilled/method_3_curated",
    )
)

BETAS = "[0.0,1.0,8.0,16.0]"
N_CHAINS = 4
BURN_IN = 2000
N_SAMPLES = 2000

# (label, filename-glob-stem). d is read from the run manifest, not hardcoded. Census sentinels are
# marked "_S" so the writeup can cross-check them against M11.4's numbers.
STRAINS = [
    ("parvulus8801_d34", "GCF_964063445_1_Pediococcus_parvulus_IOEB_8801"),
    ("ethanolidurans_d42", "GCF_964062635_1_Pediococcus_ethanolidurans"),
    ("bifido_d46_ctrl", "GCF_000010425_1_ASM1042v1"),  # census control (anaerobe)
    ("lactis2553_d51_S", "GCF_964062975_1_Lactococcus_lactis_CIRM_BIA2553"),  # census sentinel
    ("kefiri_d58", "GCF_964062605_1_Lentilactobacillus_kefiri"),
    ("gilvus_d70", "GCF_964062525_1_Enterococcus_gilvus"),
    ("pentosus_d71_S", "GCF_964063425_1_Lactiplantibacillus_pentosus"),  # census sentinel
    ("pseudomonas_d109", "GCF_964063345_1_Pseudomonas_lini"),
    ("rahnella_d145_S", "GCA_964063365_1_Rahnella_aquatiliss"),  # census sentinel
]


def resolve(stem: str) -> Path:
    hits = sorted(CURATED.glob(stem + "*.json"))
    if len(hits) != 1:
        raise SystemExit(f"{stem!r} resolved to {len(hits)} files under {CURATED}: {hits}")
    return hits[0]


def run_one(root: Path, label: str, model: Path) -> Path:
    out = root / label / "out"
    (root / label).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV), "maxent", "sample", str(model),
        "--out", str(out), "--cache-dir", str(root / label / "cache"), "--workers", "8",
        "--set", "sampler.energy_scale=pilot_sd",
        "--set", "sampler.pilot_reround=true",
        "--set", f"sampler.betas={BETAS}",
        "--set", f"sampler.n_chains={N_CHAINS}",
        "--set", f"sampler.burn_in={BURN_IN}",
        "--set", f"sampler.n_samples={N_SAMPLES}",
    ]
    log = root / label / "log.txt"
    t0 = time.monotonic()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"{label}: sample failed (rc={proc.returncode}); see {log}")
    print(f"  {label}: sampled in {time.monotonic() - t0:.0f}s")
    return out


def model_dir(out: Path) -> Path:
    # Layout is <out>/<batch>/<model_id>/samples/... — descend to the dir with a samples/ child.
    hits = [p.parent for p in out.glob("*/*/samples") if p.is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"expected one model dir under {out}, found {hits}")
    return hits[0]


def sweep(root: Path) -> None:
    import glob

    rows = []
    for label, stem in STRAINS:
        model = resolve(stem)
        print(f"[{label}] {model.name}")
        out = run_one(root, label, model)
        mdir = model_dir(out)
        manifest = json.loads((mdir / "run_manifest.json").read_text())
        d = int(manifest["geometry"]["dimension"])
        es = manifest.get("energy_scale", {})
        se_sj = es.get("relative_se") or es.get("pilot_relative_se")
        for bdir in sorted(glob.glob(os.path.join(mdir, "samples", "beta_*"))):
            s = summarize_beta(bdir)
            s["strain"] = label
            s["dimension"] = d
            s["se_sJ"] = se_sj
            rows.append(s)
            print(
                f"    β={s['beta']:>5g}  d={d:>3d}  τint(worst/p90/med)="
                f"{s['worst_tau_int']:7.1f}/{s['p90_tau_int']:6.1f}/{s['median_tau_int']:6.1f}  "
                f"minESS={s['min_ess']:.0f}  medESS={s['median_ess']:.0f}  "
                f"worstR={s['worst_rhat']:.3f}"
            )

    dest = root / "tau_sweep.json"
    dest.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {dest}  ({len(rows)} rows) — feed to benchmarks/analyze_tau.py")


if __name__ == "__main__":
    out_root = Path(
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("SWEEP_OUT", "/tmp/gsmm_schedule_sweep")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sweep(out_root)
