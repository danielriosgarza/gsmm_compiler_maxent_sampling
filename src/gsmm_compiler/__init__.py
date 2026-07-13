"""GSMM-Compiler: sparse-objective maximum-entropy flux sampler for genome-scale models.

Importing this package must **not** pull in the cobra/optlang parser stack or HiGHS: MCMC worker
processes import only the numerical core (see BUILD_PLAN.md §1.2). Submodules are therefore never
re-exported here; import them explicitly.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
