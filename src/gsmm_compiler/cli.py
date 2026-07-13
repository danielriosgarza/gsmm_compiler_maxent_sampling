"""``gsmm-compiler`` command-line entry point.

M0 ships ``model inspect``. The ``maxent solve-lp | build-geometry | sample | diagnose`` commands
arrive with M8.
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    model = subcommands.add_parser("model", help="inspect and validate metabolic models")
    model_actions = model.add_subparsers(dest="model_command", required=True)

    inspect = model_actions.add_parser("inspect", help="print a summary report for a model file")
    inspect.add_argument("path", help="path to a model file (.json, .xml, .sbml)")
    inspect.set_defaults(func=_cmd_model_inspect)

    return parser


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    from gsmm_compiler.model_input import format_summary, load_model, summarize

    model = load_model(args.path)
    print(format_summary(summarize(model, args.path)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
