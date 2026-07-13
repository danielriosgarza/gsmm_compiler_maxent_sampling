"""`FluxPolytope` and the **reduced polytope IR**: eliminate ``l == u`` reactions,
store the affine reconstruction ``v_full = R @ v_reduced + c``, and carry the affine mass-balance
RHS ``S_F v_F = -S_fixed v_fixed``.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations
