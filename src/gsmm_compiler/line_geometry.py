"""Feasible chord ``[t_lo, t_hi]`` along a direction.

Keeps **every nonzero** direction component (BUILD_PLAN §1.6.1: dropping small components samples
outside the bounds); ``nextafter`` inward; redraw on a zero-length chord.

Implemented in **M2** — see BUILD_PLAN.md.
"""

from __future__ import annotations
