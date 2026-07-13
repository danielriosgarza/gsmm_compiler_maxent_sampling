# GSMM-Compiler — MaxEnt Flux Sampler

Sparse-objective **maximum-entropy flux sampler** for genome-scale metabolic models (GSMMs).

Given a metabolic model and a sparse objective `J(v)`, the sampler draws fluxes from

    π_β(v) ∝ exp(β · (J(v) − J*) / s_J)   on the flux polytope   {v : S v = 0, l ≤ v ≤ u}

by coordinate hit-and-run in a rounded, reduced affine coordinate system. At `β = 0` this is the
uniform distribution on the polytope; increasing `β` concentrates mass near the sparse-objective
optimum, tracing how much metabolic flexibility a strain retains under selection pressure.

## Design constraints

- **No SciPy in the numerical path.** LPs are built from native NumPy CSC arrays and passed straight
  to `highspy.Highs.passModel`. cobra is a parser/metadata layer only.
- **No HiGHS solve inside the MCMC inner loop.** All solver work happens up front (objective LP,
  geometry discovery); a solve counter asserts this.
- **float64 throughout computation**; float32 only as an explicit storage option.
- **Fixed reactions (`l == u`) are eliminated** from the sampled state, but every saved sample is a
  full-length flux vector with reaction IDs and fixed-status metadata.

See [BUILD_PLAN.md](BUILD_PLAN.md) for the design and milestones,
[DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for live progress, and
[GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md](GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md)
for the mathematics.

## Install (Python 3.11, `uv`)

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Usage

```bash
gsmm-compiler model inspect models/GCF_000010425_1_ASM1042v1_protein_non_gapfilled_latest_gapfilled_noO2.json
```
