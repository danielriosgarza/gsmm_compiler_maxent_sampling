"""Logging setup shared by the CLI and the library.

Kept dependency-free (stdlib only) so worker processes can use it without importing cobra/HiGHS.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stderr handler on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    for existing in root.handlers:
        if getattr(existing, "_gsmm_handler", False):
            existing.setLevel(level)
            return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler._gsmm_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return the package-scoped logger for ``name``."""
    return logging.getLogger(name)
