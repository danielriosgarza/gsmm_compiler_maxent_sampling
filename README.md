# GSMM-Compiler — MaxEnt Flux Sampler

Sparse-objective **maximum-entropy flux sampler** for genome-scale metabolic models (GSMMs).

Given a metabolic model and a sparse objective `J(v)`, the sampler draws fluxes from

    π_β(v) ∝ exp(β · (J(v) − J*) / s_J)   on the flux polytope   {v : S v = 0, l ≤ v ≤ u}

by coordinate hit-and-run in a rounded, reduced affine coordinate system. At `β = 0` this is the
uniform distribution on the polytope; increasing `β` concentrates mass near the sparse-objective
optimum, tracing how much metabolic flexibility a strain retains under selection pressure.

## Status

The pipeline is validated end-to-end on the **40 curated strains** the package exists for
(`metabolicSubcommunities` method_3_curated). The full production census (M11.5(c),
[`benchmarks/M11_5_CENSUS.md`](benchmarks/M11_5_CENSUS.md)) found:

- **36 of 40 strains sample correctly** — validity absolute (mass balance ≤ 4e-11, zero bound
  violations, zero degenerate steps), mean-J monotone across the β-ladder on all 36, every
  production transform reachability-certified, at up to **d = 145** affine dimensions.
- **4 strains refuse — fail-closed.** Two *Hafnia alvei* (a blocked/moving classifier floor at the
  numerical resolution) and *Bacillus pumilus* + *Liquorilactobacillus satsumensis* (a span
  certificate √k-resolution floor) raise a `GeometryError` **before any sampling** — no incorrect
  numbers are ever produced. These four have polytope geometry sitting right at float64's resolution
  limit; the package refuses rather than sample a coordinate system it cannot certify complete. A
  tighter span certificate and blocked-floor guidance are tracked as future work.

See [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) for live progress and
[`BUILD_PLAN.md`](BUILD_PLAN.md) for the design and milestones.

## Install (Python 3.11, [`uv`](https://docs.astral.sh/uv/))

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Wheel-only; no source builds. Run the tests with `.venv/bin/python -m pytest -q`.

## Usage

### Inspect a model

```bash
gsmm-compiler model inspect path/to/model.json
```

### Sample one model (production configuration)

The production settings calibrate the β-axis to each strain's own neutral fluctuation
(`energy_scale=pilot_sd`) and size the chain length to a target effective sample size from a pilot
(`schedule_mode=pilot_ess`), so `β` and the sampling quality mean the same thing across strains:

```bash
gsmm-compiler maxent sample path/to/model.json \
  --out results/one --cache-dir cache --workers 8 \
  --set sampler.energy_scale=pilot_sd \
  --set sampler.pilot_reround=true \
  --set 'sampler.betas=[0,0.25,0.5,1,2,4,8,16]' \
  --set sampler.schedule_mode=pilot_ess \
  --set sampler.target_ess=400
```

`build-geometry` (the first stage) can be run alone to check a model is samplable:

```bash
gsmm-compiler maxent build-geometry path/to/model.json
```

### Batch over the curated strains

The batch runner takes a **manifest TSV** with a `model_path` column (plus optional `biomass_id` /
`model_id`). Build one from a `metabolicSubcommunities` `strains.tsv` with the bundled bridge:

```bash
python manifests/build_manifest.py /path/to/metabolicSubcommunities/metadata/strains.tsv \
  -o curated.tsv                                    # 40 curated models → curated.tsv

gsmm-compiler maxent batch curated.tsv \
  --out results/curated --cache-dir cache --workers 8 \
  --set sampler.energy_scale=pilot_sd \
  --set sampler.pilot_reround=true \
  --set 'sampler.betas=[0,0.25,0.5,1,2,4,8,16]' \
  --set sampler.schedule_mode=pilot_ess \
  --set sampler.target_ess=400
```

Geometry is computed and cached **per model**; one shared worker pool spans all `(model, β, chain)`
units. A strain that fails closed is reported and skipped — a partial batch still yields valid
per-strain results and cross-model tables over the strains that finished (so the curated 40 yields
36 completed runs plus 4 documented refusals).

### Output layout

```
results/<batch>/<model_id>/
    run_manifest.json                     # geometry, calibration, schedule, reachability certificate
    samples/beta_XXX/chain_YYY/
        flux.npy            # full-length flux vectors (float64 default; every reaction, with IDs)
        trace_j.npy, trace_mu.npy, ...    # objective / activity traces
        manifest.json                     # per-chain diagnostics (drift, degeneracy, chord length)
results/<batch>/cross_model/              # aggregated tables across strains
```

Sample storage is configurable (`output.store_flux_dtype`, `output.store_mode`): full-flux float64
(default), float32 (half the disk), or reduced `y`-states + a flux/exchange summary. Fixed reactions
(`l == u`) are eliminated from the sampled state, but every saved sample is a full-length flux vector
with reaction IDs and fixed-status metadata.

## Design constraints (non-negotiable — see `BUILD_PLAN.md` §1)

- **No SciPy in the numerical path.** LPs are built from native NumPy CSC arrays and passed straight
  to `highspy.Highs.passModel`. cobra is a parser/metadata layer only.
- **No HiGHS solve inside the MCMC inner loop.** All solver work is up front (objective LP, geometry
  discovery); a solve counter asserts this.
- **float64 throughout computation**; float32 only as an explicit storage option.
- **Reproducibility** is keyed on stable semantic coordinates `(model_id, stage, β, chain)` via
  `numpy.random.SeedSequence`, within a declared numerical-runtime profile (BLAS pinned for geometry
  determinism). `s_J` is reproducible **in distribution** across machines, not bit-for-bit.

## Documentation

| File | What it holds |
|---|---|
| [`GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md`](GSMM_Compiler_MaxEnt_Sampling_Implementation_Spec.md) | The mathematics (source of truth for the method) |
| [`BUILD_PLAN.md`](BUILD_PLAN.md) | Design, milestones, acceptance gates, cross-cutting decisions |
| [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) | Live progress tracker |
| [`benchmarks/`](benchmarks/) | Measurement reports (M9 speed, M11.4/M11.5 census) + reproducible scripts |

## License

[MIT](LICENSE) © 2026 Daniel Rios Garza.
