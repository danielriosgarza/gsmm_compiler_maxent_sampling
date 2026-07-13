"""Coordinate hit-and-run sampler for pi_beta, in reduced coordinates.

No HiGHS solve occurs here: the transform and objective are frozen before sampling starts.

Implemented in **M5/M6** — see BUILD_PLAN.md.
"""

from __future__ import annotations
