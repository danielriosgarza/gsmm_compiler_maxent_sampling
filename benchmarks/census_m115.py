"""M11.5(c) production census driver — the full 40-strain sweep through the ``pilot_ess`` sampler.

M11.4 ran the pipeline on **4 sentinels** and found no correctness defect; M11.5(a) built the
dimension-scaled schedule (``schedule_mode="pilot_ess"``) and measured τ on **9 strains**. This is
the production census the tracker's M11.5(c) names: run **all 40 curated strains** through the full
production ladder under the adaptive schedule, and record — per strain and per β rung —

* the **schedule resolution** (``resolved_n_samples``, ``quantile_tau_int``, ``cap_hit``): the
  schedule is exercised on 40 real strains, not the 9 the rule was measured on;
* the **s_J calibration** (``energy_scale``, ``se(σ̂₀)``, headroom ``G``, R̂(J)) across the batch;
* the **validity** contract on every emitted sample (bound violation, mass-balance residual, refresh
  drift, degenerate steps) — the M11.4 guarantee, now on 40;
* the **achieved per-rung flux-ESS** (via ``census_diag.summarize_beta`` — flux-level, not J-only),
  so the census can say where the β=0-sized schedule *met* its target and where the β-inflation left
  it short (which the schedule reports rather than hides);
* per-rung **E[J] ± MCSE** for the mean-J monotonicity oracle (checked in ``analyze_census.py``).

Config is the M11.4 census config + the M11.5 schedule: ``energy_scale=pilot_sd``,
``pilot_reround=true``, 4 chains, 2000+2000 pilot, the 8-rung production ladder, seed 0,
``schedule_mode=pilot_ess``, ``target_ess=400``, ``schedule_ess_quantile=0.90``. Deterministic.

**Robust + resumable by design** — a 40-strain, multi-hour campaign must not die on one
basis-marginal strain (the 2 Hafnia / pumilus / Liquorilactobacillus are recorded refusals, not
crashes — a census reports what happens, including a fail-closed refusal). A strain whose
``census_row.json`` already exists is skipped, so an interrupted run resumes.

Run:  .venv/bin/python benchmarks/census_m115.py [OUTPUT_ROOT]
      (OUTPUT_ROOT defaults to $CENSUS_OUT or /tmp/gsmm_m115_census)
Then: .venv/bin/python benchmarks/analyze_census.py [OUTPUT_ROOT]
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_diag import summarize_beta  # noqa: E402  (sibling module in benchmarks/)

from gsmm_compiler.diagnostics import effective_sample_size  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv" / "bin" / "gsmm-compiler"
CURATED = Path(
    os.environ.get(
        "CURATED_MODELS",
        "/home/mcpu/GitHub/metabolicSubcommunities/models/gapfilled/method_3_curated",
    )
)

# The 8-rung production ladder (M11.4 census ladder). pilot_ess sizes n_samples once from the β=0
# scale pilot and applies it to every rung — so β>0 rungs run at the β=0-sized budget and are
# expected to fall short of target_ess by the measured β-inflation, which the census documents.
BETAS = "[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0]"
N_CHAINS = 4
PILOT_BURN_IN = 2000
PILOT_SAMPLES = 2000
BURN_IN = 2000
TARGET_ESS = 400
ESS_QUANTILE = 0.90
WORKERS = 8


def strain_label(path: Path) -> str:
    """A compact, stable label: the accession + genus_species from the filename."""
    stem = path.stem
    parts = stem.split("_")
    # GCF_964062525_1_Enterococcus_gilvus_CIRM_... -> GCF964062525_Enterococcus_gilvus
    acc = "".join(parts[0:2])
    # the two tokens after the assembly index are genus, species in these filenames
    genus_species = "_".join(parts[3:5]) if len(parts) >= 5 else stem
    return f"{acc}_{genus_species}"


def run_one(root: Path, label: str, model: Path) -> tuple[str, str]:
    """Run the sampler on one strain. Returns (status, detail). Never raises."""
    out = root / label / "out"
    cache = root / label / "cache"
    (root / label).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV), "maxent", "sample", str(model),
        "--out", str(out), "--cache-dir", str(cache), "--workers", str(WORKERS),
        "--set", "sampler.energy_scale=pilot_sd",
        "--set", "sampler.pilot_reround=true",
        "--set", f"sampler.pilot_chains={N_CHAINS}",
        "--set", f"sampler.pilot_burn_in={PILOT_BURN_IN}",
        "--set", f"sampler.pilot_samples={PILOT_SAMPLES}",
        "--set", f"sampler.betas={BETAS}",
        "--set", f"sampler.n_chains={N_CHAINS}",
        "--set", f"sampler.burn_in={BURN_IN}",
        "--set", "sampler.n_samples=5",  # placeholder floor; resolved by the schedule
        "--set", "sampler.schedule_mode=pilot_ess",
        "--set", f"sampler.target_ess={TARGET_ESS}",
        "--set", f"sampler.schedule_ess_quantile={ESS_QUANTILE}",
    ]
    log = root / label / "log.txt"
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        # Extract a one-line reason from the log — a fail-closed refusal is a legitimate census row.
        text = log.read_text()
        reason = ""
        for pat in ("kUnknown@", "does not resolve", "cannot be resolved", "Error:", "error:",
                    "raise", "Refus", "not exhaustive", "degenerate"):
            for line in text.splitlines():
                if pat in line:
                    reason = line.strip()[:200]
                    break
            if reason:
                break
        return "FAIL", reason or f"rc={proc.returncode}; see {log}"
    return "OK", ""


def model_dir(out: Path) -> Path:
    hits = [p.parent for p in out.glob("*/*/samples") if p.is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"expected one model dir under {out}, found {hits}")
    return hits[0]


def per_beta_j(beta_dir: str) -> tuple[float, float, float]:
    """Pooled E[J], its MCSE from ESS(J), and ESS(J) for one rung — for the monotonicity oracle."""
    chains = sorted(glob.glob(os.path.join(beta_dir, "chain_*")))
    traces = np.stack([np.load(os.path.join(c, "trace_j.npy")) for c in chains])  # (nc, ns)
    pooled_mean = float(traces.mean())
    ess_j = float(effective_sample_size(traces[:, :, None])[0])
    sd_j = float(traces.reshape(-1).std())
    mcse = sd_j / np.sqrt(ess_j) if ess_j > 0 else float("inf")
    return pooled_mean, float(mcse), ess_j


def extract_row(label: str, model: Path, mdir: Path) -> dict:
    m = json.loads((mdir / "run_manifest.json").read_text())
    geo = m.get("geometry", {})
    cal = m.get("calibration", {})
    es = m.get("energy_scale", {})
    sched = m.get("schedule", {})
    cert = m.get("reachability_certificate", {})

    rungs = []
    for bdir in sorted(glob.glob(os.path.join(mdir, "samples", "beta_*"))):
        s = summarize_beta(bdir)
        ej, mcse, ess_j = per_beta_j(bdir)
        s["mean_j"] = ej
        s["mean_j_mcse"] = mcse
        s["ess_j"] = ess_j
        rungs.append(s)

    return {
        "label": label,
        "model_file": model.name,
        "status": "OK",
        "model_id": m.get("model_id"),
        "dimension": int(geo.get("dimension", -1)),
        "n_free": int(geo.get("n_free", -1)),
        "n_blocked": int(geo.get("n_blocked", -1)),
        "n_blocked_escalated": int(geo.get("n_blocked_escalated", 0)),
        "n_cold_solves": int(geo.get("n_cold_solves", 0)),
        "span_certificate_exhaustive": bool(geo.get("span_certificate_exhaustive", False)),
        "span_resolution": geo.get("span_resolution"),
        "geometry_mass_balance_error": geo.get("mass_balance_error"),
        "max_width": geo.get("max_width"),
        # calibration / conditioning
        "cond_c_q_before": cal.get("bootstrap_condition_number"),
        "cond_c_q_after": cal.get("final_condition_number"),
        "rerounded": bool(cal.get("rerounded", False)),
        # s_J calibration
        "s_J": es.get("energy_scale"),
        "s_J_relative_se": es.get("pilot_relative_se"),
        "s_J_precision_warning": bool(es.get("pilot_precision_warning", False)),
        "j_star": es.get("j_star"),
        "pilot_mean_j": es.get("pilot_mean_j"),
        "pilot_gap": es.get("pilot_gap"),
        "headroom_g": es.get("pilot_headroom_g"),
        "pilot_r_hat_j": es.get("pilot_r_hat_j"),
        "pilot_sd_chain_ratio": es.get("pilot_sd_chain_ratio"),
        "pilot_skewness": es.get("pilot_skewness"),
        "pilot_excess_kurtosis": es.get("pilot_excess_kurtosis"),
        "energy_scale_fell_back": bool(es.get("energy_scale_fell_back", False)),
        # schedule resolution (THE deliverable being exercised)
        "resolved_n_samples": sched.get("resolved_n_samples"),
        "requested_n_samples": sched.get("requested_n_samples"),
        "quantile_tau_int": sched.get("quantile_tau_int"),
        "pilot_ess_at_quantile": sched.get("pilot_ess_at_quantile"),
        "cap_hit": bool(sched.get("cap_hit", False)),
        "target_ess": sched.get("target_ess"),
        "n_movable_schedule": sched.get("n_movable"),
        # reachability of the production transform (T₁)
        "reachable_certified": bool(cert.get("reachable_is_certified", False)),
        "reachable_margin": cert.get("reachable_margin"),
        "reachable_worst_absolute": cert.get("reachable_worst_absolute"),
        "reachable_n_unknown_witnesses": cert.get("reachable_n_unknown_witnesses"),
        "rungs": rungs,
    }


def census(root: Path) -> None:
    models = sorted(CURATED.glob("*.json"))
    if len(models) != 40:
        print(f"WARNING: expected 40 curated models, found {len(models)} under {CURATED}")
    print(f"# M11.5(c) production census — {len(models)} strains, ladder {BETAS}, "
          f"target_ess={TARGET_ESS} q={ESS_QUANTILE}")
    rows = []
    campaign_t0 = time.monotonic()
    for i, model in enumerate(models, 1):
        label = strain_label(model)
        row_path = root / label / "census_row.json"
        if row_path.exists():  # resume: skip a strain already recorded
            rows.append(json.loads(row_path.read_text()))
            print(f"[{i:2d}/{len(models)}] {label}: SKIP (already done)")
            continue
        print(f"[{i:2d}/{len(models)}] {label}: running…", flush=True)
        t0 = time.monotonic()
        status, detail = run_one(root, label, model)
        dt = time.monotonic() - t0
        if status != "OK":
            row = {"label": label, "model_file": model.name, "status": "FAIL",
                   "reason": detail, "wall_seconds": round(dt, 1)}
            print(f"    FAIL ({dt:.0f}s): {detail}")
        else:
            try:
                mdir = model_dir(root / label / "out")
                row = extract_row(label, model, mdir)
                row["wall_seconds"] = round(dt, 1)
                r0 = row["rungs"][0]
                print(f"    OK ({dt:.0f}s) d={row['dimension']} "
                      f"n→{row['resolved_n_samples']} τq={row['quantile_tau_int']:.1f} "
                      f"s_J={row['s_J']:.3g}(±{100 * row['s_J_relative_se']:.1f}%) "
                      f"β0:medESS={r0['median_ess']:.0f} p90ESS={_p90_ess(r0):.0f} "
                      f"cap={row['cap_hit']} cert={row['reachable_certified']}")
            except Exception:  # pragma: no cover - defensive: extraction must not lose the run
                row = {"label": label, "model_file": model.name, "status": "EXTRACT_FAIL",
                       "reason": traceback.format_exc()[-500:], "wall_seconds": round(dt, 1)}
                print(f"    EXTRACT_FAIL: {row['reason'][-200:]}")
        row_path.write_text(json.dumps(row, indent=2))
        rows.append(row)

    dest = root / "census.json"
    dest.write_text(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if r.get("status") == "OK")
    total_dt = time.monotonic() - campaign_t0
    print(f"\nwrote {dest} — {ok}/{len(rows)} OK, campaign {total_dt / 60:.1f} min")
    print("feed to benchmarks/analyze_census.py")


def _p90_ess(rung: dict) -> float:
    # p90-ESS = ESS at the schedule-protected coordinate (10th-percentile ESS); τ_int p90 ↔ this.
    p90_tau = rung.get("p90_tau_int")
    if not p90_tau or not np.isfinite(p90_tau):
        return 0.0
    return rung["n_chains"] * rung["n_samples"] / p90_tau


if __name__ == "__main__":
    out_root = Path(
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    census(out_root)
