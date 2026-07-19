"""Build a `gsmm-compiler maxent batch` manifest from a `metabolicSubcommunities` strains.tsv.

`batch.load_models_manifest` accepts a TSV whose header names a **`model_path`** column (plus
optional `biomass_id` / `model_id`). The curated `strains.tsv` does not have that column — it names
the curated model under **`gapfilled_method_3_curated_model_path`**, relative to the
metabolicSubcommunities repo root. This script bridges the two: it reads strains.tsv, resolves each
curated model to an absolute path, and writes a manifest the batch loader consumes directly.
Metadata columns (organism, strain_label, oxygen_class, genome_accession) are carried through for
readability and cross-model aggregation; the loader ignores unknown columns.

`biomass_id` / `model_id` are intentionally left unset — the models declare their own biomass and
the loader derives a distinct `model_id` from each file, which is exactly what the M11.5(c) census
ran (so a batch over this manifest reproduces the census).

Usage:
    python manifests/build_manifest.py <strains.tsv> [-o OUT.tsv] [--models-root DIR]
    python manifests/build_manifest.py <strains.tsv> --relative-to DIR   # write portable rel paths

Then:  gsmm-compiler maxent batch OUT.tsv --out results/curated --cache-dir cache \\
           --set sampler.energy_scale=pilot_sd --set sampler.pilot_reround=true \\
           --set sampler.schedule_mode=pilot_ess --set sampler.target_ess=400 \\
           --set 'sampler.betas=[0,0.25,0.5,1,2,4,8,16]'
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

MODEL_COL = "gapfilled_method_3_curated_model_path"
META_COLS = ("genome_accession", "organism", "strain_label", "oxygen_class")


def build(strains_tsv: Path, models_root: Path, relative_to: Path | None) -> list[dict[str, str]]:
    with open(strains_tsv, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or MODEL_COL not in rows[0]:
        raise SystemExit(f"{strains_tsv} has no {MODEL_COL!r} column — not a curated strains.tsv?")
    out: list[dict[str, str]] = []
    missing = 0
    for r in rows:
        rel = (r.get(MODEL_COL) or "").strip()
        if not rel:
            continue
        abs_path = (models_root / rel).resolve()
        if not abs_path.is_file():
            print(f"  WARNING: model file not found, skipping: {abs_path}", file=sys.stderr)
            missing += 1
            continue
        model_path = str(abs_path) if relative_to is None else _relpath(abs_path, relative_to)
        row = {"model_path": model_path}
        row.update({c: (r.get(c) or "").strip() for c in META_COLS})
        out.append(row)
    if missing:
        print(f"  {missing} model file(s) missing (see warnings above)", file=sys.stderr)
    return out


def _relpath(path: Path, base: Path) -> str:
    import os
    return os.path.relpath(path, base.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strains_tsv", type=Path, help="path to metabolicSubcommunities strains.tsv")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output manifest TSV (default: stdout)")
    ap.add_argument("--models-root", type=Path, default=None,
                    help="repo root the strains.tsv paths are relative to "
                         "(default: the strains.tsv's grandparent, i.e. metadata/../)")
    ap.add_argument("--relative-to", type=Path, default=None,
                    help="write model_path relative to this dir (portable) instead of absolute")
    args = ap.parse_args()

    models_root = args.models_root or args.strains_tsv.resolve().parent.parent
    rows = build(args.strains_tsv, models_root, args.relative_to)
    if not rows:
        raise SystemExit("no models resolved — check --models-root")

    header = ["model_path", *META_COLS]
    lines = ["\t".join(header)]
    lines += ["\t".join(r.get(c, "") for c in header) for r in rows]
    text = "\n".join(lines) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text)
        print(f"wrote {args.out}  ({len(rows)} models)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
