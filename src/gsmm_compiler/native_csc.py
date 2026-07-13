"""Native CSC sparse matrix (`NativeCSC`) built column-wise from NumPy arrays.

No ``scipy.sparse`` — these arrays are passed straight to ``highspy.Highs.passModel``.

Implemented in **M1** — see BUILD_PLAN.md.
"""

from __future__ import annotations
