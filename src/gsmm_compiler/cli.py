"""``gsmm-compiler`` command-line entry point.

M0/M1 ship ``model inspect`` and ``config show``. The
``maxent solve-lp | build-geometry | sample | diagnose`` commands arrive with M8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gsmm_compiler import __version__
from gsmm_compiler.logging_utils import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsmm-compiler",
        description="Sparse-objective maximum-entropy flux sampler for genome-scale models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subcommands = parser.add_subparsers(dest="command", required=True)

    model = subcommands.add_parser("model", help="inspect and validate metabolic models")
    model_actions = model.add_subparsers(dest="model_command", required=True)

    inspect = model_actions.add_parser("inspect", help="print a summary report for a model file")
    inspect.add_argument("path", help="path to a model file (.json, .xml, .sbml)")
    inspect.add_argument(
        "--biomass-id",
        default=None,
        help="biomass reaction ID (default: the model's objective, which must name exactly one)",
    )
    inspect.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="also write the full model_report.json here",
    )
    inspect.set_defaults(func=_cmd_model_inspect)

    config = subcommands.add_parser("config", help="inspect the resolved configuration")
    config_actions = config.add_subparsers(dest="config_command", required=True)

    show = config_actions.add_parser(
        "show",
        help="echo the resolved config: defaults + file + overrides, as the run would see it",
    )
    show.add_argument("--config", default=None, metavar="PATH", help="TOML config file")
    show.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help="override a config key (repeatable)",
    )
    show.set_defaults(func=_cmd_config_show)

    _add_maxent_commands(subcommands)
    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """The ``--config`` file + repeatable ``--set`` overrides every maxent command shares."""
    parser.add_argument("--config", default=None, metavar="PATH", help="TOML config file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help="override a config key (repeatable)",
    )


def _add_maxent_commands(subcommands: Any) -> None:
    """The M8 pipeline commands: ``solve-lp | build-geometry | sample | batch | diagnose``."""
    maxent = subcommands.add_parser("maxent", help="the sampling pipeline (M8)")
    actions = maxent.add_subparsers(dest="maxent_command", required=True)

    solve = actions.add_parser("solve-lp", help="resolve λ and solve the sparse-objective LP (L2)")
    solve.add_argument("path", help="path to a model file")
    solve.add_argument("--biomass-id", default=None, help="biomass reaction ID")
    _add_config_args(solve)
    solve.set_defaults(func=_cmd_maxent_solve_lp)

    geometry = actions.add_parser("build-geometry", help="build + certify the geometry (L3)")
    geometry.add_argument("path", help="path to a model file")
    geometry.add_argument("--biomass-id", default=None, help="biomass reaction ID")
    geometry.add_argument("--cache-dir", default=None, help="content-addressed cache directory")
    _add_config_args(geometry)
    geometry.set_defaults(func=_cmd_maxent_build_geometry)

    sample = actions.add_parser("sample", help="sample the β-ladder for one model")
    sample.add_argument("path", help="path to a model file")
    sample.add_argument("--biomass-id", default=None, help="biomass reaction ID")
    sample.add_argument("--model-id", default=None, help="result dir id (default: model's own)")
    _add_run_args(sample)
    _add_config_args(sample)
    sample.set_defaults(func=_cmd_maxent_sample)

    batch = actions.add_parser("batch", help="sample every strain in a models manifest")
    batch.add_argument("manifest", help="a .json or .tsv models manifest")
    _add_run_args(batch)
    _add_config_args(batch)
    batch.set_defaults(func=_cmd_maxent_batch)

    diagnose = actions.add_parser("diagnose", help="assemble a completed model's diagnostics JSON")
    diagnose.add_argument("out", help="the batch output root (the directory holding <batch>/)")
    diagnose.add_argument("--batch", required=True, help="batch name")
    diagnose.add_argument("--model-id", required=True, help="which strain to diagnose")
    diagnose.set_defaults(func=_cmd_maxent_diagnose)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, metavar="DIR", help="output root for results")
    parser.add_argument("--batch", default="run", help="batch name (subdirectory of --out)")
    parser.add_argument("--cache-dir", default=None, help="content-addressed cache directory")
    parser.add_argument(
        "--workers", type=int, default=1, help="worker processes (1 = in-process, deterministic)"
    )


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    from gsmm_compiler.model_input import (
        format_summary,
        load_canonical_model,
        load_model,
        summarize,
    )

    model = load_model(args.path)
    print(format_summary(summarize(model, args.path)))

    # Freezing into the canonical IR is what validates the model; the reduction is what the sampler
    # will actually walk, so report its size here rather than making the user infer it.
    canonical = load_canonical_model(args.path, args.biomass_id)
    reduced = canonical.polytope.reduce()
    affine_rhs = "nonzero" if reduced.rhs.any() else "zero (all fixed fluxes are 0)"
    print(
        f"\nreduced polytope\n"
        f"  free reactions    {reduced.n_free}\n"
        f"  eliminated (l==u) {reduced.n_fixed}\n"
        f"  affine RHS        {affine_rhs}\n"
        f"  L0 key            {canonical.l0_key[:16]}…\n"
        f"  L1 key            {canonical.l1_key[:16]}…"
    )

    if args.report:
        print(f"\nwrote {canonical.write_report(args.report)}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    from gsmm_compiler.config import load

    print(load(args.config, args.overrides).echo(), end="")
    return 0


def _cmd_maxent_solve_lp(args: argparse.Namespace) -> int:
    from gsmm_compiler.config import load
    from gsmm_compiler.model_input import load_canonical_model
    from gsmm_compiler.sparse_objective import resolve_objective, solve_sparse_objective

    config = load(args.config, args.overrides)
    canonical = load_canonical_model(args.path, args.biomass_id or config.model.biomass_id)
    reduced = canonical.polytope.reduce()
    resolved = resolve_objective(canonical.polytope, reduced, config.objective)
    solution = solve_sparse_objective(reduced, resolved.objective)
    scale = resolved.scale
    d = solution.diagnostics()

    print(
        f"model             {canonical.model_id}\n"
        f"λ̃ (scaled)        {scale.l1_penalty_scaled:g}\n"
        f"λ* (cliff)        {scale.critical_l1_penalty:g}\n"
        f"λ  (raw)          {scale.l1_penalty:g}\n"
        f"origin feasible   {scale.origin_is_feasible}\n"
        f"J*                {d['j_star']:g}\n"
        f"μ(v*)             {d['mu_at_optimum']:g}\n"
        f"C(v*)             {d['cost_at_optimum']:g}\n"
        f"biomass retention {d['biomass_retention']:g}\n"
        f"sparsity-dominated {solution.is_sparsity_dominated}"
    )
    return 0


def _cmd_maxent_build_geometry(args: argparse.Namespace) -> int:
    from gsmm_compiler.affine_geometry import build_geometry
    from gsmm_compiler.config import load
    from gsmm_compiler.model_input import load_canonical_model
    from gsmm_compiler.rounding import build_transform

    config = load(args.config, args.overrides)
    canonical = load_canonical_model(args.path, args.biomass_id or config.model.biomass_id)
    reduced = canonical.polytope.reduce()
    geometry = build_geometry(reduced, model_id=canonical.model_id, config=config.geometry)
    transform = build_transform(geometry, reduced, config=config.geometry)

    gm = geometry.manifest()
    rd = transform.diagnostics
    print(
        f"model             {canonical.model_id}\n"
        f"dimension d       {gm['dimension']}\n"
        f"span certificate  {'exhaustive' if gm['span_certificate_exhaustive'] else 'partial'}"
        f" (resolution {gm['span_resolution']:g})\n"
        f"blocked reactions {gm['n_blocked']}\n"
        f"step_scale_ratio  {rd.step_scale_ratio:g}\n"
        f"cond(Cε)          {rd.condition_number:g}\n"
        f"min chord @center {rd.min_chord_at_center:g}"
    )
    if args.cache_dir is not None:
        from gsmm_compiler.batch import geometry_cache_key
        from gsmm_compiler.cache import ArtifactCache

        cache = ArtifactCache(Path(args.cache_dir))
        arrays, meta = transform.to_bundle()
        arrays = {**arrays, "support_points": geometry.support_points}
        meta = {**meta, "geometry_manifest": gm}
        key = geometry_cache_key(reduced, config, model_id=canonical.model_id)
        cache.store("L3", key, arrays=arrays, meta=meta)
        print(f"cached L3          {key[:16]}…")
    return 0


def _cmd_maxent_sample(args: argparse.Namespace) -> int:
    from gsmm_compiler.batch import ModelSpec, run_batch
    from gsmm_compiler.config import load

    config = load(args.config, args.overrides)
    spec = ModelSpec(model_path=args.path, biomass_id=args.biomass_id, model_id=args.model_id)
    result = run_batch(
        [spec],
        config,
        batch_name=args.batch,
        output_root=args.out,
        cache_dir=args.cache_dir,
        n_workers=args.workers,
    )
    return _report_batch(result)


def _cmd_maxent_batch(args: argparse.Namespace) -> int:
    from gsmm_compiler.batch import load_models_manifest, run_batch
    from gsmm_compiler.config import load

    config = load(args.config, args.overrides)
    specs = load_models_manifest(args.manifest)
    result = run_batch(
        specs,
        config,
        batch_name=args.batch,
        output_root=args.out,
        cache_dir=args.cache_dir,
        n_workers=args.workers,
    )
    return _report_batch(result)


def _report_batch(result: object) -> int:
    """Print each strain's outcome; exit non-zero if any strain failed."""
    outcomes = result.outcomes  # type: ignore[attr-defined]
    for outcome in outcomes:
        detail = "" if outcome.error is None else f"  ({outcome.error.splitlines()[0]})"
        print(
            f"{outcome.status:9s} {outcome.model_id}  "
            f"({outcome.n_completed}/{outcome.n_units} units){detail}"
        )
    print(f"\nresults in {result.batch_dir}")  # type: ignore[attr-defined]
    return 0 if all(o.status == "complete" for o in outcomes) else 1


def _cmd_maxent_diagnose(args: argparse.Namespace) -> int:
    from gsmm_compiler.diagnostics import write_run_diagnostics
    from gsmm_compiler.output import RunLayout, read_json

    layout = RunLayout(root=args.out, batch=args.batch)
    destination = write_run_diagnostics(layout, args.model_id)
    report = read_json(destination)

    feas = report["feasibility"]
    mono = report["mcmc"]["mean_j_monotonicity"]
    print(
        f"model             {report['model_id']}\n"
        f"max bound viol.   {feas['max_bound_violation']:g}\n"
        f"max mass-bal res. {feas['max_mass_balance_residual']:g}\n"
        f"max R̂(J)          {report['mcmc']['max_r_hat_j']:g}\n"
        f"min ESS(J)        {report['mcmc']['min_ess_j']:g}\n"
        f"E[J] monotone     {mono['is_monotone']} "
        f"(worst drop {mono['worst_drop_sigma']:g}σ)\n"
        f"wrote {destination}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
