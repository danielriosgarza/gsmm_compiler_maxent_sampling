"""``gsmm-compiler`` command-line entry point.

M0/M1 ship ``model inspect`` and ``config show``. The
``maxent solve-lp | build-geometry | sample | diagnose`` commands arrive with M8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

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

    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
