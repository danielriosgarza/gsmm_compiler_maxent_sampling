"""M11.5(c) component 2 — the geometry-pilot covariance / eigenvalue-rank study at d=145.

M11.4 reported one covariance fact off the anaerobe and one off Rahnella: re-rounding improves
``cond(C_q)`` 2.57× at d=46 but makes it **0.60× (worse)** at d=145. That is a summary (two
condition numbers); it does not say *why*, and "why" is a property of the **eigenvalue spectrum** of
``C_q``, which no manifest stores. This study reconstructs the full spectrum of both covariances —
``T₀`` (the M4 support-vertex rounding) and ``T₁`` (the pilot re-rounding, spec §17.4) — at d=145
(Rahnella) and, as a contrast, d=46 (the Bifido anaerobe where re-rounding *helps*).

**Method — capture, do not reconstruct.** Both transform builders route through
``rounding._transform_from_coordinates`` → ``rounding._support_covariance``. We spy on the former
(it carries the ``source`` label) during a **cold** ``batch.prepare_model``, so ``build_transform``
(T₀, support points) and ``reround_transform`` (T₁, pilot draws) both fire and we hold the exact
coordinate sets the code rounded. The spectra are then the code's own ``np.linalg.eigvalsh`` on the
code's own covariance — no independent re-derivation to drift. Validated: the recomputed ridged
condition numbers must equal the manifest's ``bootstrap_condition_number`` (T₀) and
``final_condition_number`` (T₁).

Run:  .venv/bin/python benchmarks/covariance_study_d145.py [OUT]
      (OUT defaults to $CENSUS_OUT/covariance_study.json, i.e. /tmp/gsmm_m115_census/…)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from gsmm_compiler import rounding
from gsmm_compiler.batch import ModelSpec, prepare_model
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.config import load as load_config

CURATED = Path(
    os.environ.get(
        "CURATED_MODELS",
        "/home/mcpu/GitHub/metabolicSubcommunities/models/gapfilled/method_3_curated",
    )
)
# (label, filename stem). Rahnella is the d=145 subject; Bifido the d=46 anaerobe contrast.
SUBJECTS = [
    ("rahnella_d145", "GCA_964063365_1_Rahnella_aquatiliss"),
    ("bifido_d46", "GCF_000010425_1_ASM1042v1"),
]
OVERRIDES = [
    "sampler.energy_scale=pilot_sd",
    "sampler.pilot_reround=true",
    "sampler.pilot_chains=4",
    "sampler.pilot_burn_in=2000",
    "sampler.pilot_samples=2000",
    "sampler.betas=[0.0]",
    "sampler.n_chains=4",
    "sampler.burn_in=200",
    "sampler.n_samples=5",
]


def resolve(stem: str) -> Path:
    hits = sorted(CURATED.glob(stem + "*.json"))
    if len(hits) != 1:
        raise SystemExit(f"{stem!r} resolved to {len(hits)} files: {hits}")
    return hits[0]


def spectrum_profile(eig: np.ndarray) -> dict:
    """A compact, log-scale description of a positive eigenvalue spectrum."""
    e = np.sort(eig)[::-1]  # descending
    logs = np.log10(np.clip(e, 1e-300, None))
    # count of eigenvalues within decades of the top — a spectrum's "shape" without 145 numbers
    top = e[0]
    return {
        "lambda_max": float(e[0]),
        "lambda_min": float(e[-1]),
        "lambda_p50": float(np.median(e)),
        "log10_range": float(logs[0] - logs[-1]),
        "n_within_1_decade_of_max": int(np.count_nonzero(e > top * 1e-1)),
        "n_within_3_decades": int(np.count_nonzero(e > top * 1e-3)),
        "n_within_6_decades": int(np.count_nonzero(e > top * 1e-6)),
        "n_below_9_decades": int(np.count_nonzero(e < top * 1e-9)),
        # decile spectrum: log10(eigenvalue) at 0,10,..100% — the full shape in 11 numbers
        "log10_deciles": [float(np.percentile(logs, p)) for p in range(0, 101, 10)],
    }


def study_one(label: str, model: Path, cache_dir: Path) -> dict:
    captured: list[dict] = []
    orig = rounding._transform_from_coordinates

    def spy(geometry, reduced, coordinates, *, config, source):  # noqa: ANN001
        transform = orig(geometry, reduced, coordinates, config=config, source=source)
        cov, trace, mean_norm = rounding._support_covariance(coordinates)
        eig = np.linalg.eigvalsh(cov)
        eig = eig[eig > 0.0]  # eigvalsh can emit tiny negatives on a PSD matrix
        d = int(geometry.dimension)
        mean_var = trace / d
        raw_rank = int(np.count_nonzero(eig > mean_var * config.covariance_rank_tol))
        diag = transform.diagnostics
        ridge = diag.ridge
        ridged = np.sort(eig)[::-1] + ridge
        captured.append({
            "source": source,
            "dimension": d,
            "n_points": int(coordinates.shape[0]),
            "covariance_trace": float(trace),
            "mean_variance": float(mean_var),
            "coordinate_mean_norm": float(mean_norm),
            "raw_covariance_rank": raw_rank,
            "raw_rank_deficit": d - raw_rank,
            "ridge": float(ridge),
            "ridge_relative": float(diag.ridge_relative),
            "official_covariance_rank": int(diag.covariance_rank),
            "official_condition_number": float(diag.condition_number),
            "official_step_scale_ratio": float(diag.step_scale_ratio),
            "recomputed_cond_ridged": float(ridged.max() / ridged.min()),
            "raw_spectrum": spectrum_profile(eig),
            "full_eigenvalues_desc": np.sort(eig)[::-1].tolist(),
        })
        return transform

    rounding._transform_from_coordinates = spy  # type: ignore[assignment]
    try:
        config = load_config(None, OVERRIDES)
        spec = ModelSpec(model_path=str(model))
        cache = ArtifactCache(cache_dir)
        prepare_model(spec, config, cache=cache)
    finally:
        rounding._transform_from_coordinates = orig  # type: ignore[assignment]

    by_source = {c["source"]: c for c in captured}
    t0 = by_source.get("support_points")
    t1 = by_source.get("pilot")
    return {"label": label, "model_file": model.name, "T0": t0, "T1": t1}


def main(out_path: Path) -> int:
    results = []
    for label, stem in SUBJECTS:
        model = resolve(stem)
        print(f"[{label}] {model.name}")
        cache_dir = out_path.parent / f"covstudy_{label}_cache"
        r = study_one(label, model, cache_dir)
        results.append(r)
        for tag in ("T0", "T1"):
            c = r[tag]
            if c is None:
                print(f"    {tag}: (not captured)")
                continue
            sp = c["raw_spectrum"]
            print(
                f"    {tag} ({c['source']}) d={c['dimension']} K={c['n_points']}  "
                f"rank(C_q)={c['raw_covariance_rank']}/{c['dimension']} "
                f"(deficit {c['raw_rank_deficit']})  cond={c['official_condition_number']:.3g}  "
                f"stepσ={c['official_step_scale_ratio']:.2e}  "
                f"λ log-range={sp['log10_range']:.1f}"
            )
        if r["T0"] and r["T1"]:
            c0 = r["T0"]["official_condition_number"]
            c1 = r["T1"]["official_condition_number"]
            print(f"    reround cond change: {c0:.3g} → {c1:.3g} ({c0 / c1:.2f}×)")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    default = Path(os.environ.get("CENSUS_OUT", "/tmp/gsmm_m115_census")) / "covariance_study.json"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
