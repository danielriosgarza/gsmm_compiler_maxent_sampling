# GSMM-Compiler: sparse-objective maximum-entropy flux sampling

## Implementation specification for a coding agent

**Status:** design specification for a new repository  
**Primary input:** a COBRApy JSON genome-scale metabolic model  
**Primary output:** flux samples from a maximum-entropy distribution that progressively favours high biomass production and sparse, low-throughput flux states  
**Core numerical tools:** COBRApy for model parsing, native `highspy` for linear programming, NumPy for numerical arrays and reduced-coordinate geometry  

---

## 1. What we are building

We are building the first computational component of **GSMM-Compiler**, whose broader purpose is to compile a genome-scale metabolic model into a smaller set of representative metabolic subprograms.

The immediate task is not yet to define the final metabolic modes. It is to construct a principled sampler that reveals how the feasible metabolic state space changes as stronger selection is placed on a combined objective:

\[
J(v)=\mu(v)-\lambda\sum_{r\in\mathcal R_p}w_r|v_r|.
\]

Here:

- \(v\in\mathbb R^n\) is a steady-state reaction-flux vector.
- \(\mu(v)=v_b\) is the biomass-reaction flux.
- \(\mathcal R_p\) is the set of reactions included in the flux penalty.
- \(w_r>0\) is a fixed reaction weight.
- \(\lambda>0\) controls the strength of the pressure against flux.

The biomass term favours productive states. The weighted \(L_1\) term favours parsimonious states and pushes many fluxes to zero at the linear-program optimum.

We first solve a linear program to find the maximum possible value \(J^*\). We then sample the full feasible flux space from the maximum-entropy family

\[
\pi_\beta(v)
=\frac{1}{Z(\beta)}
\exp\!\left[\beta J(v)\right]
\mathbf 1_{\mathcal P}(v),
\]

where \(\mathcal P\) is the feasible steady-state flux polytope and \(\beta\geq 0\) controls how strongly the sampler favours the sparse objective.

At \(\beta=0\), the distribution is uniform over the feasible polytope, with respect to affine flux-space volume. As \(\beta\) increases, probability shifts towards states with higher biomass and lower total weighted absolute flux. In the limit of large \(\beta\), the distribution concentrates near the maximizers of the LP objective.

The scientific output is therefore a sequence of sampled metabolic populations, ranging from broadly feasible metabolism to strongly selected, sparse, high-performance metabolism.

---

## 2. Why this is useful for metabolic flexibility

A single FBA solution gives one flux vector. Even when the optimum is a face containing many equivalent solutions, a conventional LP solver returns one point on that face.

The biological question is broader:

> Which distinct metabolic strategies remain available as stronger selection is placed on biomass production and sparse flux use?

The maximum-entropy family answers this without forcing all states onto an exact objective slice.

For each \(\beta\), the samples can show:

- which reactions are reliably active;
- which reactions become inactive as selection increases;
- which alternative pathways coexist at similar objective values;
- whether the high-performance region contains one narrow strategy or several alternatives;
- how uptake and secretion patterns change with selection;
- how rapidly metabolic diversity collapses near the optimum;
- whether two species retain different amounts or types of metabolic flexibility at comparable selection pressures.

The eventual metabolic modes can be inferred from these samples in a biologically chosen feature space, such as exchange conversions, biomass-normalized yields, pathway-level fluxes, redox usage, or reaction-activity patterns.

This repository should first produce correct, reproducible samples and the metadata needed for that downstream analysis. It should not prematurely bake one clustering method into the sampler.

---

## 3. Mathematical formulation

### 3.1 Feasible flux polytope

For a stoichiometric matrix \(S\in\mathbb R^{m\times n}\), lower bounds \(\ell\), and upper bounds \(u\), define

\[
\mathcal P
=
\left\{
 v\in\mathbb R^n:
 Sv=0,\quad
 \ell\leq v\leq u
\right\}.
\]

The first implementation should support exactly this form: steady-state equality constraints plus finite reaction bounds.

General linear inequalities can be added later, but they should not complicate the first implementation.

Sampling requires a bounded polytope. If any reaction has a genuinely infinite bound, the program must either:

1. receive an explicit finite replacement bound from the user, or
2. stop with a clear error.

The program must never silently replace infinite bounds with arbitrary constants.

### 3.2 Sparse biomass objective

Let \(b\) denote the biomass-reaction index. Define

\[
\mu(v)=v_b
\]

and

\[
C(v)=\sum_{r\in\mathcal R_p}w_r|v_r|.
\]

The scalar objective is

\[
J(v)=\mu(v)-\lambda C(v).
\]

The implementation must always report \(\mu(v)\), \(C(v)\), and \(J(v)\) separately. The scalar objective defines the chosen trade-off, but downstream interpretation should retain the two components.

The default penalty set should be all reactions except the biomass reaction. Exchange, demand, and sink reactions should remain included by default, because excluding them changes the meaning of the cost. The user must be able to override the set explicitly.

The exact penalty set and weight vector must be written to the output manifest.

### 3.3 Linear-program representation of absolute flux

For every penalized reaction \(r\), introduce an auxiliary variable \(z_r\geq 0\) with

\[
z_r\geq v_r,
\qquad
z_r\geq -v_r.
\]

The LP is

\[
\begin{aligned}
\max_{v,z}\quad
& v_b-\lambda\sum_{r\in\mathcal R_p}w_r z_r\\
\text{subject to}\quad
& Sv=0,\\
& \ell\leq v\leq u,\\
& v_r-z_r\leq 0,
&& r\in\mathcal R_p,\\
& -v_r-z_r\leq 0,
&& r\in\mathcal R_p,\\
& z_r\geq 0.
\end{aligned}
\]

Because each \(z_r\) has a negative objective coefficient, an optimum satisfies

\[
z_r=|v_r|.
\]

The solver returns

\[
J^*=\max_{v\in\mathcal P}J(v)
\]

and at least one maximizing flux vector \(v^*\).

### 3.4 A critical distinction: auxiliary variables are LP-only

The maximum-entropy sampler must sample only the biological flux vector \(v\).

It must **not** sample the auxiliary \(z\) variables.

The variables \(z\) are a device for linearizing the LP. If they were included in the sampled state space, every flux vector would acquire extra artificial volume from the possible values of \(z\). This would change the intended distribution over fluxes. At \(\beta=0\), unbounded \(z\) variables would also make the expanded sampling space unbounded.

During sampling, evaluate

\[
J(v)=v_b-\lambda\sum_rw_r|v_r|
\]

directly from \(v\).

### 3.5 Maximum-entropy distribution

The maximum-entropy problem is

\[
\max_p
\left[-\int_{\mathcal P}p(v)\log p(v)\,dv\right]
\]

subject to

\[
\int_{\mathcal P}p(v)\,dv=1
\]

and a chosen expected objective

\[
\int_{\mathcal P}J(v)p(v)\,dv=\bar J.
\]

The resulting exponential family is

\[
\pi_\beta(v)
=
\frac{1}{Z(\beta)}
\exp\!\left[\beta J(v)\right]
\mathbf 1_{\mathcal P}(v),
\]

with partition function

\[
Z(\beta)=\int_{\mathcal P}\exp\!\left[\beta J(v)\right]dv.
\]

Because the weighted absolute-flux cost is convex, \(J(v)\) is concave. For \(\beta\geq 0\), the resulting density is log-concave on the convex feasible polytope. This structure is central to the line sampler developed below.

Useful identities are

\[
\frac{d\log Z}{d\beta}
=\mathbb E_\beta[J]
\]

and

\[
\frac{d\mathbb E_\beta[J]}{d\beta}
=\operatorname{Var}_\beta(J)\geq 0.
\]

Thus the expected objective is nondecreasing in \(\beta\). This property can later be used to calibrate \(\beta\) values to desired mean-performance levels.

### 3.6 Numerical form of the log density

For numerical stability, use

\[
\log \widetilde\pi_\beta(v)
=
\beta\frac{J(v)-J^*}{s_J},
\]

where \(s_J>0\) is an explicitly recorded objective scale.

Subtracting \(J^*\) does not change the distribution. Dividing by \(s_J\) only reparameterizes \(\beta\).

Support two modes:

- `energy_scale = 1.0`: \(\beta\) is in reciprocal raw-objective units.
- `energy_scale = "warmup_range"`: set \(s_J\) from a robust range of objective values among warm-up points, then report both \(s_J\) and the corresponding raw \(\beta/s_J\).

No hidden scaling is permitted.

### 3.7 What the zero-promoting term does

The \(L_1\) penalty produces exact zeros at many LP optima because an LP optimum lies on a boundary or vertex of the feasible region.

At finite \(\beta\), the sampled distribution is continuous. A continuous distribution gives any exact hyperplane \(v_r=0\) probability zero unless the feasible polytope itself fixes that reaction to zero.

Therefore:

- the LP solution can contain exact zeros;
- finite-\(\beta\) samples should be analysed using a declared near-zero threshold;
- the sampler must not round or snap small fluxes to zero during MCMC;
- snapping would alter the stationary distribution and can violate mass balance.

A stronger zero pressure can later be obtained with reweighted \(L_1\), described in Section 12. The final weights must be frozen before sampling.

---

## 4. Scope and non-goals

### Required in the first complete version

1. Load a COBRA JSON model.
2. Extract a deterministic reaction and metabolite order.
3. Construct the stoichiometric matrix directly in native compressed-column arrays.
4. Build and solve the sparse-objective LP with native `highspy`.
5. Discover a reduced affine coordinate system without using SciPy sparse matrices.
6. Round the reduced polytope.
7. Sample \(\pi_\beta\) with coordinate hit-and-run and an exact one-dimensional conditional sampler.
8. Run a configurable \(\beta\)-ladder and multiple chains.
9. Save samples, objective components, metadata, and diagnostics.
10. Test mathematical correctness on small analytic models.
11. Benchmark solver construction, warm starts, and MCMC throughput.

### Explicit non-goals for the first version

- No MILP.
- No enumeration of elementary flux modes.
- No loopless MILP constraints.
- No automatic clustering into final metabolic modes.
- No dynamic FBA.
- No enzyme-constrained model unless its constraints are already represented as supported linear bounds and equalities.
- No use of COBRApy's sampling implementation in the production path.
- No use of `scipy.optimize.linprog`.
- No construction of the computational LP through optlang.
- No SciPy sparse matrix in the LP construction or MCMC inner loop.

COBRApy is a parser and metadata layer here. HiGHS and the custom geometry code form the numerical core.

---

## 5. High-level algorithm

The complete workflow is:

1. Load and validate the model.
2. Freeze model ordering.
3. Build native CSC arrays for \(S\).
4. Build a flux-only HiGHS model for feasibility and geometry LPs.
5. Build the \((v,z)\) sparse-objective HiGHS model.
6. Solve for \(J^*\) and \(v^*\).
7. Discover the affine direction space of \(\mathcal P\) using warm-started support LPs.
8. Construct a feasible centre from the discovered support points.
9. Estimate a rounding transform in reduced coordinates.
10. Precompute the reduced-to-flux coordinate matrix.
11. For each \(\beta\), run coordinate hit-and-run chains.
12. At every MCMC step, sample exactly along one feasible chord from the piecewise-exponential conditional density.
13. Save flux samples and scalar summaries.
14. Check feasibility, chain mixing, and reproducibility.

The expensive solver work occurs before production MCMC. Once the reduced coordinates are constructed, no HiGHS solve should occur inside a sampling step.

---

## 6. Repository structure

Use a standard `src` layout:

```text
GSMM-Compiler/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── gsmm_compiler/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── model_input.py
│       ├── native_csc.py
│       ├── flux_polytope.py
│       ├── highs_backend.py
│       ├── sparse_objective.py
│       ├── affine_geometry.py
│       ├── rounding.py
│       ├── line_geometry.py
│       ├── line_distribution.py
│       ├── maxent_sampler.py
│       ├── diagnostics.py
│       ├── output.py
│       └── features.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── statistical/
│   └── performance/
├── examples/
│   ├── toy_network.json
│   ├── toy_config.toml
│   └── run_toy.py
└── docs/
    ├── mathematics.md
    ├── numerical_design.md
    └── output_format.md
```

The code should use type hints throughout. Prefer small typed data classes to dictionaries passed across the numerical core.

Suggested primary data classes:

```python
@dataclass(frozen=True)
class NativeCSC:
    n_rows: int
    n_cols: int
    starts: NDArray[np.int64]
    indices: NDArray[np.int64]
    values: NDArray[np.float64]

@dataclass(frozen=True)
class FluxPolytope:
    reaction_ids: tuple[str, ...]
    metabolite_ids: tuple[str, ...]
    stoichiometry: NativeCSC
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    biomass_index: int

@dataclass(frozen=True)
class SparseFluxObjective:
    biomass_index: int
    penalized_indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    l1_penalty: float

@dataclass(frozen=True)
class ReducedGeometry:
    center_flux: NDArray[np.float64]
    transform: NDArray[np.float64]
    dimension: int
    scaling: NDArray[np.float64]
```

The names may change, but the separation of responsibilities should remain.

---

## 7. Dependencies and numerical policy

### Required runtime dependencies

- Python 3.11 or newer.
- COBRApy.
- `highspy`.
- NumPy.

### Development dependencies

- `pytest`.
- `pytest-cov`.
- `ruff`.
- `mypy` or `pyright`.

Additional plotting or analysis packages should be optional and kept out of the sampler core.

### No-SciPy requirement

The first implementation should not depend on SciPy.

In particular, do not use:

- `scipy.optimize.linprog`;
- `scipy.sparse.csc_matrix` as an intermediate object for HiGHS;
- COBRApy helpers that construct SciPy matrices in the production numerical path;
- SciPy linear algebra in the MCMC inner loop.

Build NumPy arrays matching the native HiGHS compressed-column representation and pass them directly through `highspy.HighsLp` and `Highs.passModel`.

NumPy dense linear algebra is permitted for reduced-coordinate rounding, with explicit memory checks.

### Version policy

Pin the exact tested COBRApy and `highspy` versions in the lock file or reproducible environment definition.

At program start, write the following to the run manifest:

- Python version;
- NumPy version;
- COBRApy version;
- `highspy` version;
- operating system;
- CPU information when available;
- Git commit hash;
- model-file SHA-256 hash.

---

## 8. Configuration and command-line interface

Use a TOML configuration file as the canonical configuration. CLI arguments may override fields.

Example:

```toml
[model]
path = "model.json"
biomass_reaction = "BIOMASS_Ecoli_core_w_GAM"

[objective]
l1_penalty = 0.01
penalty_set = "all_non_biomass"
weights_file = ""
reweighted_l1_iterations = 0
reweighted_l1_epsilon = 1e-6

[sampling]
betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
energy_scale = "warmup_range"
chains = 4
samples_per_chain = 5000
burn_in = 5000
thin = 10
seed = 42
refresh_interval = 1000

[geometry]
stall_probes = 24
validation_probes = 32
rank_tolerance = 1e-10
width_tolerance = 1e-9
rounding_ridge = 1e-8
pilot_steps = 20000
pilot_thin = 20
max_geometry_memory_gb = 8.0

[solver]
threads = 1
output_flag = false
primal_feasibility_tolerance = 1e-8
dual_feasibility_tolerance = 1e-8

[output]
directory = "results/run_001"
store_flux_dtype = "float64"
store_geometry = true
```

Suggested CLI:

```bash
gsmm-compiler maxent sample --config run.toml
```

Useful inspection commands:

```bash
gsmm-compiler model inspect model.json
gsmm-compiler maxent solve-lp --config run.toml
gsmm-compiler maxent build-geometry --config run.toml
gsmm-compiler maxent sample --config run.toml
gsmm-compiler maxent diagnose results/run_001
```

Every command should be restartable and should produce atomic outputs. Do not leave apparently complete result directories after a failed run.

---

## 9. Loading and validating the COBRA model

Load the model with:

```python
from cobra.io import load_json_model
model = load_json_model(path)
```

After loading:

1. Freeze the current metabolite order.
2. Freeze the current reaction order.
3. Build `metabolite_id -> row_index` and `reaction_id -> column_index` maps.
4. Check that all IDs are unique.
5. Check every reaction bound for NaN.
6. Check every stoichiometric coefficient for NaN or infinity.
7. Confirm that the biomass reaction exists exactly once.
8. Confirm that the biomass reaction is capable of carrying nonnegative flux.
9. Check that all bounds are finite.
10. Record the original COBRA objective, but do not silently use it in place of the configured biomass reaction.

The program must not reorder reactions or metabolites later. Every output array is interpreted using this frozen order.

### Model validation report

Write a model report containing:

- model ID;
- number of reactions;
- number of metabolites;
- number of genes;
- number of exchange, demand, and sink reactions according to COBRApy metadata;
- biomass reaction and bounds;
- number of reversible reactions;
- number of fixed-bound reactions;
- minimum and maximum finite bounds;
- original model objective;
- any warnings.

Warnings should not silently modify the model.

---

## 10. Constructing the stoichiometric matrix without SciPy

The matrix is naturally assembled by reaction columns.

For every reaction \(j\):

1. Read `reaction.metabolites`.
2. Convert each metabolite object to its frozen row index.
3. Sort entries by row index for deterministic output.
4. Append row indices and coefficients to flat arrays.
5. Append the new nonzero count to the `starts` array.

Pseudocode:

```python
starts = [0]
indices = []
values = []

for reaction in reactions:
    entries = sorted(
        (metabolite_index[m.id], float(coeff))
        for m, coeff in reaction.metabolites.items()
        if coeff != 0.0
    )
    for row, value in entries:
        indices.append(row)
        values.append(value)
    starts.append(len(values))
```

Convert once:

```python
starts_array = np.asarray(starts, dtype=np.int64)
indices_array = np.asarray(indices, dtype=np.int64)
values_array = np.asarray(values, dtype=np.float64)
```

Before passing arrays to HiGHS, convert integer types to the integer width accepted by the installed `highspy` build. Add a unit test that verifies this on the pinned version.

The `NativeCSC` class should provide occasional validation operations:

- shape checking;
- monotonic `starts` checking;
- index-range checking;
- duplicate-entry checking;
- a reference `matvec` for tests and feasibility diagnostics;
- a reference `rmatvec` for geometry diagnostics.

These reference multiplications do not need to be the MCMC bottleneck. The MCMC inner loop should not multiply by \(S\) at every step.

---

## 11. Native HiGHS backend

Create a thin adapter around `highspy`. Do not scatter raw HiGHS calls through the scientific modules.

The adapter should:

- create a `highspy.HighsLp`;
- populate column costs and bounds from NumPy arrays;
- populate row bounds from NumPy arrays;
- set `a_matrix_.format_` to column-wise storage;
- populate `start_`, `index_`, and `value_` directly;
- pass the model with `Highs.passModel`;
- set solver options;
- run the solver;
- check return status and model status;
- extract the full solution vector in one conversion;
- expose objective modification for repeated warm-started solves;
- expose the solver basis for optional explicit reuse;
- never return an unchecked solution.

The official HiGHS Python documentation notes that repeated element-by-element access to arrays returned by `highspy` can be slow. Convert the returned solution vector once to a NumPy array or Python list, then slice it.

Suggested interface:

```python
class HighsLinearProgram:
    def solve(self) -> LPSolution: ...
    def set_objective(self, costs: NDArray[np.float64]) -> None: ...
    def set_maximize(self) -> None: ...
    def get_basis(self) -> object: ...
    def set_basis(self, basis: object) -> None: ...
```

`LPSolution` should contain:

- status;
- model status;
- objective value;
- primal vector;
- maximum primal infeasibility;
- simplex iteration count;
- elapsed time.

### Solver reuse

Build each HiGHS model once.

For affine-basis discovery, repeatedly change only the linear objective coefficients. Reuse the same solver object. Prefer the simplex solver for this phase because basis warm starts can make repeated related LPs much cheaper.

Do not assume that warm starts are effective. Record simplex iteration counts and benchmark the selected solver options.

Set HiGHS threads to one during geometry construction unless a benchmark establishes a reproducible benefit from more threads. This also avoids oversubscription when several independent jobs are run in parallel.

---

## 12. Building and solving the sparse-objective LP

Let \(p=|\mathcal R_p|\). The LP has \(n+p\) columns:

```text
[v_0, ..., v_(n-1), z_0, ..., z_(p-1)]
```

It has \(m+2p\) rows:

```text
0 .. m-1          mass-balance equalities
m + 2k            v_r - z_k <= 0
m + 2k + 1       -v_r - z_k <= 0
```

### Column bounds

For flux columns:

\[
\ell_j\leq v_j\leq u_j.
\]

For each auxiliary column:

\[
0\leq z_k\leq \max(|\ell_r|,|u_r|).
\]

The finite upper bound is not mathematically required when the objective penalizes \(z\), but it improves model boundedness and diagnostics.

### Row bounds

For mass-balance rows:

\[
0\leq (Sv)_i\leq 0.
\]

For absolute-value rows:

\[
-\infty < v_r-z_k\leq 0
\]

and

\[
-\infty < -v_r-z_k\leq 0.
\]

### Objective coefficients

Flux columns receive zero cost except biomass:

\[
c_b=1.
\]

Auxiliary columns receive

\[
c_{z_k}=-\lambda w_r.
\]

Set objective sense to maximize.

### Direct CSC assembly

Construct the expanded LP matrix directly by columns. Do not first construct \(S\) as a SciPy matrix and concatenate blocks.

For a penalized flux column \(v_r\), append two additional nonzeros after its stoichiometric entries:

- `+1` in row `m + 2k`;
- `-1` in row `m + 2k + 1`.

For auxiliary column \(z_k\), append:

- `-1` in row `m + 2k`;
- `-1` in row `m + 2k + 1`.

### LP result checks

After solving:

1. Require an optimal HiGHS model status.
2. Extract \(v^*\) and \(z^*\).
3. Check \(Sv^*\) within tolerance.
4. Check all flux bounds.
5. Check \(z_r^*\geq |v_r^*|-\varepsilon\).
6. Check \(|z_r^*-|v_r^*||\) within objective tolerance.
7. Recompute \(J(v^*)\) directly.
8. Compare the recomputed value with the HiGHS objective.
9. Store \(J^*\), \(v^*\), \(\mu(v^*)\), and \(C(v^*)\).

Also solve a biomass-only LP once and report

\[
\mu_{\max}=\max_{v\in\mathcal P}v_b.
\]

This is a diagnostic. It reveals how much biomass the chosen sparse objective retains at its optimum without changing the formulation.

---

## 13. Optional stronger zero pressure through reweighted L1

Implement this only after the fixed-weight sampler is correct.

Iterative reweighted \(L_1\) can approximate a stronger penalty on the number of active reactions while retaining LP subproblems.

Starting from base weights \(w_r^{(0)}\), iterate:

1. Solve the sparse-objective LP with weights \(w^{(k)}\).
2. Update

\[
w_r^{(k+1)}
=
\frac{w_r^{\mathrm{base}}}
{|v_r^{(k)}|+\varepsilon}.
\]

3. Clip extreme weights to configured limits.
4. Renormalize the positive weights so their median is one.
5. Stop when the active set and solution change less than configured tolerances.

Important rules:

- save every weight vector and LP solution;
- label the procedure experimental;
- do not claim that it solves exact cardinality minimization;
- freeze the final weights before maximum-entropy sampling;
- never update weights from the current MCMC state;
- updating weights during sampling would change the target distribution and invalidate the stationary-law argument.

The fixed final weights preserve the same piecewise-linear, concave objective structure required by the sampler.

---

## 14. Flux-only HiGHS model for geometry

Build a second LP containing only the biological flux variables:

\[
Sv=0,
\qquad
\ell\leq v\leq u.
\]

Its objective coefficients will be changed repeatedly.

This model serves four purposes:

1. find an initial feasible point;
2. discover the affine direction space;
3. generate support points for rounding;
4. validate that the discovered reduced coordinates span the feasible polytope.

The MCMC sampler must not call this LP after production sampling begins.

---

## 15. Discovering the affine direction space with LPs

### 15.1 Why this step is needed

The flux polytope is lower-dimensional in \(\mathbb R^n\) because every feasible direction \(d\) must satisfy

\[
Sd=0.
\]

Directly drawing a random vector in reaction space almost never satisfies this equality.

A conventional implementation would compute an explicit null-space basis with sparse QR or SVD. Here, we want a route that:

- uses the existing HiGHS LP;
- avoids SciPy sparse matrices;
- automatically accounts for reactions that are effectively fixed by the feasible region;
- returns feasible directions as differences of feasible flux vectors.

### 15.2 Scaled coordinates

Define a positive reaction scale \(s_i\). A practical initial choice is

\[
s_i=
\begin{cases}
 u_i-\ell_i,&u_i>\ell_i,\\
 1,&u_i=\ell_i.
\end{cases}
\]

Apply a configurable lower floor when a nonzero range is extremely small.

For a flux difference \(\Delta v\), define scaled coordinates

\[
\Delta x_i=\frac{\Delta v_i}{s_i}.
\]

The affine basis will be orthonormal in this scaled coordinate system.

### 15.3 Support-LP basis discovery

Maintain an orthonormal matrix

\[
B\in\mathbb R^{n\times d}
\]

whose columns span the discovered feasible directions in scaled coordinates.

Start with no columns.

At each iteration:

1. Draw \(g\sim\mathcal N(0,I_n)\).
2. Project it orthogonally away from the current basis:

\[
p=g-B(B^Tg).
\]

3. Reorthogonalize once more for numerical stability.
4. Normalize \(p\).
5. Convert the scaled-coordinate direction into a flux-space LP objective:

\[
c_i=\frac{p_i}{s_i}.
\]

6. Solve

\[
v^+=\arg\max_{v\in\mathcal P}c^Tv.
\]

7. Solve

\[
v^-=\arg\min_{v\in\mathcal P}c^Tv
\]

by maximizing \(-c^Tv\).

8. Compute

\[
\Delta x
=
\frac{v^+-v^-}{s}
\]

elementwise.

9. Remove existing basis components from \(\Delta x\), using two-pass modified Gram-Schmidt.
10. If the residual norm exceeds tolerance, normalize it and append it to \(B\).
11. Store \(v^+\) and \(v^-\) as support points.
12. Reset the stall counter when a basis vector is added. Otherwise increment it.

Why this works:

- both \(v^+\) and \(v^-\) are feasible;
- their difference satisfies \(S(v^+-v^-)=0\);
- \(p\) is orthogonal to the current basis;
- if the optimized width \(c^T(v^+-v^-)\) is positive, then \(\Delta x\) contains a feasible direction outside the current span;
- therefore every successful iteration adds one affine dimension.

In exact arithmetic, if the current basis misses any feasible direction, a random projected objective detects a missing component with probability one. Floating-point tolerances require several unsuccessful probes before stopping.

### 15.4 Stopping and validation

Stop basis discovery after `stall_probes` consecutive projected objectives have width below tolerance and add no basis vector.

Then run a fresh validation phase with `validation_probes` new random objectives projected outside \(B\).

For every validation probe, require the maximum-minus-minimum objective width to be below the configured width tolerance.

Also check:

\[
\|B^TB-I\|_{\max}
\]

and

\[
\|S\,\operatorname{diag}(s)B\|_{\max}.
\]

The latter can be evaluated once with the reference CSC multiplication routines.

If validation fails, continue basis discovery rather than silently sampling a lower-dimensional subset.

### 15.5 Numerical implementation details

- Use float64.
- Use two-pass modified Gram-Schmidt.
- Store basis columns in a Fortran-contiguous NumPy array.
- Allocate basis storage in blocks rather than reallocating on every appended column.
- Reject candidate vectors whose norm is below tolerance.
- Record every support-LP width and simplex iteration count.
- Preserve the HiGHS basis between objective changes when possible.
- Place an explicit memory guard before allocating \(B\).

The basis requires approximately

\[
8nd\text{ bytes}
\]

in float64.

If the estimated allocation exceeds `max_geometry_memory_gb`, stop with a clear message or use a separately implemented low-memory backend. Do not allow the operating system to discover the limit by crashing.

### 15.6 Pseudocode

```python
def discover_affine_basis(polytope, highs_lp, scales, rng, cfg):
    basis = OrthonormalBasis(n=polytope.n_reactions)
    initial_feasible = highs_lp.solve_feasibility()
    support_points = [initial_feasible]
    stall = 0

    while stall < cfg.stall_probes:
        g = rng.normal(size=polytope.n_reactions)
        p = basis.remove_components(g, reorthogonalize=True)
        p_norm = np.linalg.norm(p)
        if p_norm <= cfg.rank_tolerance:
            continue
        p /= p_norm

        objective = p / scales
        v_plus = highs_lp.maximize(objective)
        v_minus = highs_lp.maximize(-objective)

        delta_x = (v_plus - v_minus) / scales
        residual = basis.remove_components(delta_x, reorthogonalize=True)
        width = float(objective @ (v_plus - v_minus))

        if (
            width > cfg.width_tolerance
            and np.linalg.norm(residual) > cfg.rank_tolerance
        ):
            basis.append_normalized(residual)
            support_points.extend([v_plus, v_minus])
            stall = 0
        else:
            stall += 1

    validate_affine_basis(...)
    return basis.matrix, np.asarray(support_points)
```

---

## 16. Constructing a feasible centre

Let \(W=\{w_1,\ldots,w_K\}\) contain the initial feasible point and the feasible support points obtained during basis discovery.

Use

\[
v_c=\frac{1}{K}\sum_{k=1}^K w_k.
\]

A convex average of feasible points is feasible. With sufficiently diverse support points, the average is usually far from the sharpest corners of the polytope.

Check:

- mass-balance residual;
- bound violations;
- finite values.

Do not clip individual fluxes to repair the centre, because independent clipping can violate \(Sv=0\).

If the averaged centre exceeds tolerance, rerun the involved LPs at tighter feasibility tolerance or solve a dedicated repair LP. Never silently project with an operation that does not preserve mass balance.

If the discovered affine dimension is zero, the feasible set contains one point within tolerance. Return that point as every sample and skip MCMC.

---

## 17. Rounding the reduced polytope

### 17.1 Reduced coordinates

The scaled affine basis gives

\[
v=v_c+\operatorname{diag}(s)Bq.
\]

Here \(q\in\mathbb R^d\) are unrounded reduced coordinates.

Flux polytopes are commonly highly anisotropic. A direction that is long in one reduced coordinate may be extremely short in another. Coordinate hit-and-run mixes poorly without preconditioning.

### 17.2 Warm-up covariance

Map every support point into reduced coordinates:

\[
q_k
=
B^T\operatorname{diag}(s)^{-1}(w_k-v_c).
\]

Compute the empirical covariance

\[
C_q
=
\frac{1}{K-1}\sum_k(q_k-\bar q)(q_k-\bar q)^T.
\]

Because \(v_c\) is the mean of the support points, \(\bar q\) should be close to zero.

Add a ridge:

\[
C_\varepsilon=C_q+\varepsilon I.
\]

Choose \(\varepsilon\) relative to `trace(C_q)/d`, not as an unexplained absolute constant.

Compute a Cholesky factor

\[
C_\varepsilon=LL^T.
\]

If Cholesky fails, increase the ridge geometrically and record the final value.

### 17.3 Rounded coordinates

Define

\[
q=Ly.
\]

Then

\[
v=v_c+Ty,
\]

where

\[
T=\operatorname{diag}(s)BL.
\]

Precompute \(T\in\mathbb R^{n\times d}\).

The production sampler stores the reduced state \(y\), while maintaining a synchronized flux vector \(v\).

Check once:

\[
\|ST\|_{\max}
\]

and require it to be below the configured geometry tolerance.

### 17.4 Optional pilot rounding

Support-LP vertices are not uniform samples, so their covariance is only an initial approximation.

A better production workflow is:

1. build an initial transform from support points;
2. run a pilot \(\beta=0\) coordinate hit-and-run chain;
3. collect thinned reduced-coordinate states;
4. estimate a new covariance from the pilot;
5. construct a final transform;
6. discard the pilot samples;
7. restart all production chains with the final frozen transform.

Do not adapt the rounding matrix during production. Freezing it before saved samples keeps the transition kernel stationary and the reasoning simple.

### 17.5 Why coordinate directions are efficient

At each production step, choose a reduced coordinate \(k\) and use flux direction

\[
d=T_{:,k}.
\]

This costs \(O(n)\) to inspect and does not require multiplying a dense \(n\times d\) matrix by a new random vector at every step.

The coordinate axes span the reduced space. Randomly choosing an axis and resampling the position along its feasible chord is a random-scan Gibbs or coordinate hit-and-run transition.

Rounding makes these coordinate directions useful rather than needle-like.

---

## 18. Maximum-entropy coordinate hit-and-run

### 18.1 Target in reduced coordinates

The linear transformation from \(y\) to \(v\) has a constant Jacobian. Therefore the target in reduced coordinates is, up to a constant,

\[
\pi_\beta(y)
\propto
\exp\!\left[
\beta\frac{J(v_c+Ty)-J^*}{s_J}
\right]
\]

for all \(y\) whose mapped flux vector satisfies its bounds.

### 18.2 One MCMC step

Given current \((y,v)\):

1. choose coordinate \(k\) uniformly from \(\{1,\ldots,d\}\);
2. set \(d=T_{:,k}\);
3. calculate the feasible interval \([t_-,t_+]\) such that

\[
\ell\leq v+td\leq u;
\]

4. sample \(t\) from the exact one-dimensional conditional density

\[
p(t\mid v,d)
\propto
\exp\!\left[
\beta\frac{J(v+td)-J^*}{s_J}
\right]
\mathbf 1_{[t_-,t_+]}(t);
\]

5. update

\[
y_k\leftarrow y_k+t;
\]

and

\[
v\leftarrow v+td.
\]

There is no Metropolis rejection when the conditional sample is exact.

At \(\beta=0\), sample \(t\) uniformly on the chord.

### 18.3 Stationarity

For a fixed coordinate \(k\), this step replaces one reduced coordinate by a draw from its exact conditional distribution while leaving the others fixed. It is a Gibbs transition and preserves \(\pi_\beta\).

A uniform mixture over coordinates also preserves \(\pi_\beta\).

The practical requirements are:

- the transform columns span the affine direction space;
- the coordinate-selection law is independent of the current state;
- the transform is frozen during production;
- the line conditional is sampled correctly;
- finite chains are checked for convergence.

---

## 19. Computing the feasible chord

For every reaction \(i\) with \(|d_i|\) above a direction tolerance, solve

\[
\ell_i\leq v_i+td_i\leq u_i.
\]

This gives a reaction-specific interval.

If \(d_i>0\):

\[
\frac{\ell_i-v_i}{d_i}
\leq t\leq
\frac{u_i-v_i}{d_i}.
\]

If \(d_i<0\), the two ratios swap order.

The full chord is

\[
t_-=
\max_i
\min\left(
\frac{\ell_i-v_i}{d_i},
\frac{u_i-v_i}{d_i}
\right)
\]

and

\[
t_+=
\min_i
\max\left(
\frac{\ell_i-v_i}{d_i},
\frac{u_i-v_i}{d_i}
\right).
\]

Implementation requirements:

- vectorize the bound calculations with NumPy masks;
- ignore components with \(|d_i|\) below tolerance;
- assert that \(t_-\leq 0\leq t_+\) within tolerance;
- reject and redraw a direction if the chord length is numerically zero;
- use `np.nextafter` or a small relative inward adjustment to avoid sampling outside a bound due only to floating-point rounding;
- do not call HiGHS to calculate a chord.

At \(\beta=0\), this chord calculation is almost the entire inner-loop cost.

---

## 20. Exact conditional sampling for the L1 objective

### 20.1 Objective along a line

Along a chord,

\[
J(t)
=
(v_b+td_b)
-
\lambda
\sum_{r\in\mathcal R_p}
 w_r|v_r+td_r|.
\]

Each absolute-value term changes slope only when

\[
v_r+td_r=0.
\]

The breakpoint is

\[
\tau_r=-\frac{v_r}{d_r}
\]

when \(d_r\neq 0\) and \(\tau_r\in(t_-,t_+)\).

Between consecutive breakpoints, every sign is fixed and \(J(t)\) is linear.

### 20.2 Concavity and slope updates

The function \(-|v_r+td_r|\) is concave in \(t\). Therefore \(J(t)\) is concave and piecewise linear.

Within a segment, its slope is

\[
m
=
d_b
-
\lambda
\sum_{r\in\mathcal R_p}
 w_r\operatorname{sgn}(v_r+td_r)d_r.
\]

When \(t\) crosses \(\tau_r\), the slope decreases by

\[
2\lambda w_r|d_r|.
\]

This monotone slope structure makes segment construction efficient.

### 20.3 Segment construction

For \(\beta>0\):

1. find all internal breakpoints for penalized reactions;
2. sort them;
3. merge breakpoints that are equal within tolerance;
4. construct segment endpoints

\[
t_-=a_0<a_1<\cdots<a_K=t_+;
\]

5. evaluate \(J(a_0)\) directly;
6. calculate the first segment slope from a midpoint;
7. update slopes using the known decreases at each breakpoint;
8. propagate objective values continuously from segment to segment.

### 20.4 Segment probability mass

On segment \([a,b]\), let

\[
J(t)=J(a)+m(t-a).
\]

Define

\[
\kappa=\frac{\beta m}{s_J},
\qquad
L=b-a,
\qquad
h_a=\frac{\beta[J(a)-J^*]}{s_J}.
\]

The unnormalized segment mass is

\[
M
=
\int_a^b
\exp\!\left[
\frac{\beta[J(t)-J^*]}{s_J}
\right]dt.
\]

If \(\kappa=0\):

\[
M=e^{h_a}L.
\]

If \(\kappa\neq 0\):

\[
M=e^{h_a}\frac{e^{\kappa L}-1}{\kappa}.
\]

Compute log masses with stable `log1p` and `expm1` formulas. Never evaluate a large positive exponential directly.

Normalize the segment log masses with a custom log-sum-exp implementation and choose one segment categorically.

### 20.5 Sampling within a segment

Within the selected segment, set \(x=t-a\in[0,L]\). The density is proportional to

\[
e^{\kappa x}.
\]

For \(|\kappa L|\) below a small threshold, use

\[
x=UL,
\qquad U\sim\operatorname{Uniform}(0,1).
\]

For \(\kappa<0\), let \(r=-\kappa>0\):

\[
x
=
-\frac{1}{r}
\log\left[
1-U(1-e^{-rL})
\right].
\]

For \(\kappa>0\), sample distance \(y=L-x\) from a decreasing truncated exponential:

\[
y
=
-\frac{1}{\kappa}
\log\left[
1-U(1-e^{-\kappa L})
\right],
\]

then set

\[
x=L-y.
\]

These forms avoid overflow for large \(|\kappa L|\).

### 20.6 Complexity

At \(\beta=0\), one step is \(O(n)\).

At \(\beta>0\), finding breakpoints is \(O(p)\), and sorting the internal breakpoints is \(O(q\log q)\), where \(q\leq p\) is the number of penalized fluxes that cross zero on the chord.

The first implementation should prioritize correctness and vectorized NumPy operations. Benchmark breakpoint sorting before attempting a more complicated line sampler.

A later alternative is a one-dimensional slice sampler. Because \(J(t)\) is concave, every slice is an interval and can be located by root finding without sorting all breakpoints. That kernel can target the same distribution, but it should be added only after the exact conditional implementation is fully tested.

### 20.7 Line-sampler pseudocode

```python
def sample_line_step(v, direction, beta, objective, bounds, energy_scale, rng):
    t_lo, t_hi = feasible_chord(v, direction, bounds)

    if beta == 0.0:
        return rng.uniform(t_lo, t_hi)

    cuts, slope_drops = objective.zero_crossings_and_drops(
        v=v,
        direction=direction,
        lower=t_lo,
        upper=t_hi,
    )

    segments = build_piecewise_linear_objective(
        v=v,
        direction=direction,
        lower=t_lo,
        upper=t_hi,
        cuts=cuts,
        slope_drops=slope_drops,
        objective=objective,
    )

    segment_index = choose_segment_by_log_mass(
        segments=segments,
        beta=beta,
        energy_scale=energy_scale,
        rng=rng,
    )

    return sample_truncated_exponential_on_segment(
        segment=segments[segment_index],
        beta=beta,
        energy_scale=energy_scale,
        rng=rng,
    )
```

---

## 21. State updates and numerical refresh

Maintain both:

- reduced state \(y\);
- flux state \(v=v_c+Ty\).

A coordinate update changes only one entry of \(y\):

```python
y[k] += t
v += t * transform[:, k]
```

Repeated incremental updates can accumulate roundoff. Every `refresh_interval` steps, reconstruct

```python
v = center_flux + transform @ y
```

and check:

- bounds;
- objective consistency;
- mass-balance residual on a configurable diagnostic schedule.

Do not project or clip the chain state unless a separately validated repair operation preserves the target. A feasibility violation larger than tolerance should stop the chain and write a diagnostic snapshot.

---

## 22. Choosing the beta ladder

### 22.1 Basic user-specified ladder

The first implementation should accept an explicit nondecreasing list

\[
0=\beta_0<\beta_1<\cdots<\beta_K.
\]

Including \(\beta=0\) is strongly recommended because it provides a uniform baseline and tests the geometry independently of the objective.

A useful initial dimensionless ladder is

```text
0, 0.25, 0.5, 1, 2, 4, 8, 16
```

but no universal ladder should be hard-coded as scientifically correct.

### 22.2 Objective-scale calibration

When using a warm-up range \(s_J\), define it from the objective values of support or pilot points. A robust choice is

\[
s_J
=
J^*-Q_{0.05}(J(W)).
\]

If this value is too small, fall back to a declared positive scale and issue a warning.

This makes \(\beta\) describe selection relative to the observed objective range rather than the raw numerical magnitude of one model.

### 22.3 Mapping beta to expected performance

A later calibration routine may target desired expected-objective fractions.

Let \(\bar J_0\) be the mean objective under \(\beta=0\). Define

\[
q(\beta)
=
\frac{
\mathbb E_\beta[J]-\bar J_0
}{
J^*-\bar J_0
}.
\]

Because \(d\mathbb E_\beta[J]/d\beta\geq 0\), \(q(\beta)\) is nondecreasing. Stochastic bisection can find \(\beta\) values corresponding approximately to desired levels such as 0.05, 0.10, ..., 0.95.

This is the maximum-entropy analogue of scanning fixed objective fractions.

Do not implement this calibration before the fixed-ladder sampler is validated.

---

## 23. Chains, initialization, burn-in, and thinning

### Multiple chains

Run at least four chains for diagnostic use. Use NumPy `SeedSequence` to derive independent child random-number streams from one recorded master seed.

### Starting states

Use feasible convex combinations rather than arbitrary perturbations.

For \(\beta=0\), the rounded centre is a natural starting state.

For high \(\beta\), a useful interior starting point is

\[
v_{\mathrm{start}}
=(1-\epsilon)v^*+\epsilon v_c
\]

with a small \(\epsilon>0\). This remains feasible by convexity and avoids starting exactly at a sharp LP vertex.

Map the starting flux into reduced coordinates by solving

\[
q=B^T\operatorname{diag}(s)^{-1}(v_{\mathrm{start}}-v_c)
\]

and then

\[
y=L^{-1}q.
\]

Use `numpy.linalg.solve`, not an explicit matrix inverse.

### Burn-in

Treat burn-in as configurable. Do not claim that a fixed number guarantees convergence for every model.

Store objective traces during burn-in even when flux vectors are discarded. They are needed for diagnostics.

### Thinning

Thinning may reduce storage and can be useful for coordinate hit-and-run, but it does not magically create independent samples.

Record:

- total MCMC steps;
- burn-in steps;
- thinning interval;
- saved samples;
- effective sample-size estimates.

### Sequential beta warm starts

A chain at \(\beta_{k+1}\) may start from a final sample at \(\beta_k\). This is computationally useful, but saved samples at the new \(\beta\) still require burn-in.

### Optional parallel tempering

A later extension may run the \(\beta\)-ladder simultaneously and propose swaps between adjacent replicas.

For states \(v_i\) at \(\beta_i\) and \(v_j\) at \(\beta_j\), accept a swap with probability

\[
\min\left[
1,
\exp\left(
\frac{(\beta_i-\beta_j)(J(v_j)-J(v_i))}{s_J}
\right)
\right].
\]

This can improve movement between broad low-\(\beta\) regions and narrow high-\(\beta\) regions. It is a second-phase feature.

---

## 24. Diagnostics

### 24.1 Feasibility diagnostics

For every saved sample, or for a configurable random subset when output is very large, check:

\[
\max_i |(Sv)_i|
\]

and bound violations.

Track the maximum values per chain and per \(\beta\).

### 24.2 Objective diagnostics

Store for every saved sample:

- biomass \(\mu(v)\);
- weighted absolute-flux cost \(C(v)\);
- raw objective \(J(v)\);
- normalized log-energy \((J(v)-J^*)/s_J\);
- number of reactions with \(|v_r|\) below each declared analysis threshold.

Check empirically that mean \(J\) is nondecreasing with \(\beta\), allowing for Monte Carlo uncertainty.

### 24.3 MCMC diagnostics

Compute diagnostics for:

- objective;
- biomass;
- flux cost;
- selected exchange fluxes;
- selected central reactions;
- fixed random projections of the full flux vector.

At minimum implement:

- trace summaries;
- lag autocorrelation;
- integrated autocorrelation-time estimate;
- effective sample size;
- split \(\hat R\) across chains for scalar summaries.

Do not compute \(\hat R\) independently for thousands of reactions and then present only the best values. Summarize the distribution and report the worst selected diagnostics.

### 24.4 Geometry diagnostics

Write:

- discovered affine dimension;
- number of support LPs;
- number of stalled probes;
- validation-probe widths;
- \(\|B^TB-I\|\);
- \(\|S\operatorname{diag}(s)B\|\);
- \(\|ST\|\);
- condition estimate of the rounding covariance;
- ridge added before Cholesky;
- support-point objective distribution.

### 24.5 Solver diagnostics

Write:

- objective LP solve time;
- biomass-only LP solve time;
- geometry LP count;
- geometry LP total time;
- simplex iterations per support solve;
- evidence of warm-start reuse;
- HiGHS statuses.

### 24.6 No-solver-in-inner-loop diagnostic

Instrument the HiGHS adapter with a solve counter.

After production MCMC starts, the solve count must remain unchanged.

Add an integration test for this requirement.

---

## 25. Output format

Suggested run directory:

```text
results/run_001/
├── COMPLETE
├── config.resolved.toml
├── manifest.json
├── model_report.json
├── reaction_index.tsv
├── metabolite_index.tsv
├── objective_weights.tsv
├── lp_optimum.npz
├── geometry.npz
├── support_points.npy
├── samples/
│   ├── beta_0000_chain_00.npy
│   ├── beta_0000_chain_01.npy
│   ├── beta_0025_chain_00.npy
│   └── ...
├── scalar_traces/
│   ├── beta_0000_chain_00.npz
│   └── ...
├── summaries/
│   ├── beta_summary.tsv
│   ├── reaction_activity.tsv
│   └── exchange_summary.tsv
└── diagnostics/
    ├── geometry.json
    ├── solver.json
    ├── mcmc.json
    └── feasibility.json
```

Create the `COMPLETE` marker only after all requested work and validation finish successfully.

### Flux storage

Use NumPy `.npy` arrays or memory-mapped arrays for the first implementation. Store samples in reaction order.

Each sample file should have shape

```text
(number_of_saved_samples, number_of_reactions)
```

Store float64 by default. Permit float32 output only as an explicit storage option, while all calculations remain float64.

### Geometry storage

When enabled, store:

- centre flux;
- reaction scaling;
- affine basis;
- Cholesky factor;
- final transform;
- support points;
- tolerances.

For very large models, geometry storage may be disabled to save disk space, but all diagnostics and reconstruction metadata must remain available.

---

## 26. Features for later metabolic-mode discovery

The sampler should expose a separate feature-extraction layer without deciding the final clustering method.

Useful feature families include:

### Raw flux features

\[
f(v)=v.
\]

These retain all internal detail but are sensitive to model representation and reaction duplication.

### Reaction-activity features

For threshold \(\varepsilon_r\):

\[
a_r(v)=\mathbf 1\{|v_r|>\varepsilon_r\}.
\]

Thresholds must be declared and sensitivity-tested.

### External conversion features

Select exchange reactions and form

\[
f_{\mathrm{ex}}(v)=v_{\mathrm{exchange}}.
\]

For growing samples, one may also use biomass-normalized exchange yields:

\[
\bar f_{\mathrm{ex}}(v)
=
\frac{v_{\mathrm{exchange}}}{\mu(v)}
\]

when \(\mu(v)\) exceeds a safe threshold.

### Pathway-level features

Aggregate reactions into curated pathways, subsystems, electron-carrier usage, ATP production, or other biological modules.

### Selection trajectories

For reaction \(r\), estimate

\[
P_\beta(|v_r|>\varepsilon).
\]

For a proposed mode \(k\), estimate its prevalence as a function of \(\beta\).

The way these probabilities appear, disappear, or split across \(\beta\) is a candidate description of metabolic flexibility.

Mode discovery should be a downstream command that reads stored samples. It must not modify the sampling distribution.

---

## 27. Unit tests

### 27.1 Native CSC tests

- construct a tiny known matrix;
- compare starts, indices, and values with hand calculations;
- verify deterministic ordering;
- verify `matvec` and `rmatvec`;
- reject malformed starts and out-of-range indices.

### 27.2 COBRA adapter tests

- load a toy JSON model;
- preserve reaction and metabolite order;
- identify biomass by exact ID;
- detect missing biomass;
- detect duplicate IDs;
- detect NaN and infinite bounds.

### 27.3 Sparse-objective LP tests

Use a toy network with an analytically known optimum.

Check:

- HiGHS optimum status;
- direct objective equals solver objective;
- \(z=|v|\) within tolerance;
- increasing \(\lambda\) changes the expected optimum in the toy case;
- unpenalized reactions do not receive auxiliary variables;
- biomass is excluded from the penalty by default;
- custom weights are applied correctly.

### 27.4 Reweighted-L1 tests

When implemented:

- weight update formula;
- clipping and normalization;
- deterministic output for a fixed solver and seed;
- weights frozen before MCMC.

### 27.5 Affine-basis tests

For polytopes with known affine dimensions:

- discover the correct dimension;
- verify orthonormality;
- verify mass-balanced basis directions;
- verify validation widths are zero after completion;
- detect an intentionally truncated basis;
- return dimension zero for a singleton feasible set.

### 27.6 Rounding tests

- map support points into reduced coordinates and back;
- verify reconstruction error;
- verify positive-definite regularized covariance;
- verify \(ST\approx 0\);
- verify centre maps to zero reduced coordinates.

### 27.7 Chord tests

For hand-defined boxes and directions:

- compute exact lower and upper step limits;
- handle positive and negative direction components;
- ignore zero components;
- include \(t=0\);
- reject zero-length numerical chords.

### 27.8 Piecewise objective tests

For random \(v\), \(d\), and weights:

- compare reconstructed piecewise-linear \(J(t)\) with direct evaluation on many points;
- check continuity at breakpoints;
- check that slopes are nonincreasing;
- check grouped duplicate breakpoints;
- check no-breakpoint lines.

### 27.9 One-dimensional distribution tests

Use analytic targets.

#### Uniform interval

At \(\beta=0\), sampled means and variances should match a uniform distribution.

#### Linear objective

For

\[
p(t)\propto e^{\kappa t},
\qquad t\in[a,b],
\]

compare sampled moments and quantiles with analytic values.

#### Absolute-value objective

For

\[
p(t)\propto e^{-\alpha|t|},
\qquad t\in[-1,1],
\]

check symmetry and analytic mean absolute value.

Use fixed seeds and confidence intervals designed to avoid flaky tests.

### 27.10 Full-chain statistical tests

Construct a two-dimensional box with

\[
J(x,y)=-\lambda(|x|+|y|).
\]

The target factorizes into two truncated Laplace distributions. Compare sampled marginal moments with analytic values.

Construct a simple equality-constrained polygon and compare the custom sampler with numerical quadrature in its one-dimensional reduced coordinate.

### 27.11 Reproducibility tests

- same seed and configuration gives byte-identical scalar traces;
- different chain child seeds differ;
- thread count one produces reproducible LP outputs on the supported platform;
- resolved configuration is stored.

### 27.12 No-SciPy and no-inner-loop-LP tests

- run the test suite in an environment without SciPy;
- scan core imports for forbidden SciPy imports;
- count HiGHS solves before and after MCMC;
- require zero solver calls per production sampling step.

---

## 28. Integration tests

### Toy COBRA model

Run the complete CLI on a bundled toy JSON model and verify every expected output file.

### COBRApy textbook model

Use a small standard COBRApy model converted to JSON during test setup.

Verify:

- model loading;
- objective LP;
- affine geometry;
- \(\beta=0\) sampling;
- positive-\(\beta\) concentration;
- feasibility of all saved samples.

### Medium genome-scale model

Mark this test as slow. It should run in continuous integration only on a scheduled or dedicated runner.

Measure:

- native CSC construction time;
- LP build time;
- sparse-objective solve time;
- geometry solve count and time;
- sampling steps per second;
- peak resident memory.

---

## 29. Performance requirements and benchmarks

The main performance goal is to pay the LP cost during setup and keep production MCMC in NumPy.

Benchmark separately:

1. JSON parsing.
2. Native CSC construction.
3. `HighsLp` construction and `passModel`.
4. First flux-only LP solve.
5. Repeated objective changes with warm starts.
6. Sparse-objective LP solve.
7. Affine-basis discovery.
8. Rounding.
9. \(\beta=0\) steps per second.
10. Positive-\(\beta\) steps per second.
11. Breakpoint count distribution.
12. Output-writing throughput.

### Required performance assertions

- no row-by-row or column-by-column `highspy` model construction in the production path;
- no SciPy conversion;
- no HiGHS call per MCMC step;
- no Python loop over reactions inside the chord calculation;
- no element-by-element extraction from a `highspy` solution vector;
- no repeated allocation of the full transform matrix;
- no full flux reconstruction every step;
- objective and breakpoint calculations use cached NumPy index arrays.

### Profiling order

Do not optimize speculative bottlenecks.

Profile in this order:

1. native model construction;
2. affine-basis LP solves;
3. dense basis orthogonalization;
4. breakpoint sorting;
5. output writing.

The likely bottleneck may change with model size and \(\beta\).

---

## 30. Milestones for the coding agent

### Milestone 1: repository and model adapter

Deliver:

- package skeleton;
- configuration parser;
- JSON loading;
- frozen model ordering;
- native CSC builder;
- model inspection CLI;
- unit tests.

Acceptance condition: the toy model is represented exactly without SciPy.

### Milestone 2: native HiGHS LPs

Deliver:

- flux-only LP;
- sparse-objective LP;
- biomass-only diagnostic LP;
- status checking;
- solution extraction;
- LP tests.

Acceptance condition: direct evaluation reproduces the solver objective and all feasibility tests pass.

### Milestone 3: one-dimensional mathematics

Deliver:

- objective evaluator;
- chord calculation;
- zero-crossing detection;
- piecewise-linear segment builder;
- stable segment-mass calculations;
- truncated-exponential sampler;
- analytic statistical tests.

Acceptance condition: one-dimensional tests pass before any genome-scale MCMC is attempted.

### Milestone 4: affine geometry

Deliver:

- support-LP basis discovery;
- validation probes;
- feasible centre;
- reduced coordinates;
- geometry diagnostics.

Acceptance condition: known toy affine dimensions are recovered and \(S\operatorname{diag}(s)B\approx0\).

### Milestone 5: rounding and beta-zero sampler

Deliver:

- covariance and Cholesky rounding;
- coordinate hit-and-run at \(\beta=0\);
- multiple chains;
- feasibility and convergence diagnostics.

Acceptance condition: analytic uniform targets are reproduced and no HiGHS calls occur inside MCMC.

### Milestone 6: positive-beta maximum-entropy sampler

Deliver:

- exact piecewise-exponential line conditionals;
- explicit beta ladder;
- objective traces;
- concentration tests.

Acceptance condition: analytic log-concave targets are reproduced and mean objective rises with \(\beta\) within uncertainty.

### Milestone 7: production output and benchmarks

Deliver:

- complete run directory;
- restartable stages;
- model and software manifest;
- memory guards;
- performance report;
- user-facing README.

Acceptance condition: the complete workflow runs on a selected genome-scale JSON model with documented resource use.

### Milestone 8: optional extensions

Only after all previous milestones:

- reweighted \(L_1\);
- pilot rerounding;
- beta calibration to expected-performance fractions;
- parallel tempering;
- slice-based line kernel;
- downstream mode-feature extraction.

---

## 31. Definition of done

The first complete implementation is done when all of the following are true:

1. A user can provide a COBRA JSON model and biomass reaction through a configuration file.
2. The program constructs \(S\) without SciPy.
3. The program builds both HiGHS LPs through native arrays and `passModel`.
4. The sparse objective is solved correctly.
5. The sampler operates only on \(v\), never on auxiliary \(z\).
6. The affine direction space is discovered and validated.
7. The rounded coordinate transform preserves mass balance within tolerance.
8. \(\beta=0\) produces correct uniform samples on analytic test polytopes.
9. \(\beta>0\) produces correct exponential-family samples on analytic tests.
10. All saved GSMM samples satisfy mass balance and flux bounds within declared tolerance.
11. Multiple-chain diagnostics are produced.
12. The complete run is reproducible from the stored configuration, versions, hashes, and seed.
13. There are no HiGHS solves in the MCMC inner loop.
14. The project runs in an environment without SciPy.
15. A benchmark documents setup time, sampling speed, and memory use.

---

## 32. Common implementation mistakes to avoid

### Sampling auxiliary absolute-value variables

Do not do this. It changes the marginal distribution over fluxes.

### Splitting reversible reactions into forward and reverse variables

The signed-flux plus auxiliary-absolute-value formulation is cleaner here. Splitting can introduce simultaneous forward and reverse flow and expands the sampled state space.

### Snapping small sampled fluxes to zero

Do not alter chain states. Apply thresholds only during analysis.

### Updating L1 weights during MCMC

Do not do this. Freeze weights before defining \(J\).

### Recomputing the LP for each sampling step

Do not do this. The chord is calculated from bounds in reduced coordinates.

### Using LP vertices as if they were samples

Support LPs construct geometry. They are not distributed according to \(\pi_\beta\).

### Adapting the rounding matrix while saving samples

Run adaptation only in a discarded pilot phase, then freeze and restart.

### Ignoring affine-span validation

A sampler restricted to an incomplete basis can look numerically stable while exploring only one metabolic sheet. Validation probes are mandatory.

### Treating MCMC samples as independent

Report effective sample size and autocorrelation.

### Hiding the penalty set or weights

They define the objective and must be exported.

### Silently changing infinite bounds

Stop and request explicit finite bounds.

### Claiming finite-beta exact sparsity

Finite-beta samples are continuous. Report near-zero probabilities with explicit thresholds.

---

## 33. Scientific caveats that must appear in the README

1. The distribution at \(\beta=0\) is uniform with respect to affine volume in the model's flux coordinates. Duplicating reactions or changing model granularity can change that volume measure.
2. Flux bounds define the feasible polytope and therefore influence the distribution.
3. The \(L_1\) penalty discourages large internal cycles but does not guarantee thermodynamic looplessness.
4. The choice of \(\lambda\), reaction weights, and penalty set is part of the biological model.
5. Maximum-entropy sampling describes feasible metabolic states under the chosen objective pressure. It is not automatically a model of population frequencies without an additional biological interpretation.
6. MCMC output is approximate and correlated. Convergence must be evaluated.
7. The final definition of a metabolic mode depends on the downstream feature space.

These caveats do not weaken the method. They state exactly what object is being sampled.

---

## 34. Compact end-to-end pseudocode

```python
def run_maxent_sampling(config_path: Path) -> None:
    cfg = load_and_resolve_config(config_path)
    rng_master = np.random.SeedSequence(cfg.sampling.seed)

    cobra_model = load_json_model(cfg.model.path)
    polytope = build_flux_polytope(cobra_model, cfg)
    objective = build_sparse_flux_objective(polytope, cobra_model, cfg)

    flux_lp = build_native_flux_lp(polytope, cfg.solver)
    sparse_lp = build_native_sparse_objective_lp(
        polytope,
        objective,
        cfg.solver,
    )

    biomass_solution = solve_biomass_only(flux_lp, polytope.biomass_index)
    sparse_solution = solve_sparse_objective(sparse_lp, objective)
    j_star = objective.evaluate(sparse_solution.fluxes)

    if cfg.objective.reweighted_l1_iterations > 0:
        objective = run_reweighted_l1(...)
        sparse_lp = build_native_sparse_objective_lp(...)
        sparse_solution = solve_sparse_objective(...)
        j_star = objective.evaluate(sparse_solution.fluxes)

    basis, support_points, geometry_report = discover_affine_basis(
        polytope=polytope,
        highs_lp=flux_lp,
        rng=np.random.default_rng(rng_master.spawn(1)[0]),
        cfg=cfg.geometry,
    )

    center = support_points.mean(axis=0)
    geometry = build_rounded_geometry(
        polytope=polytope,
        basis=basis,
        support_points=support_points,
        center=center,
        cfg=cfg.geometry,
    )

    energy_scale = choose_energy_scale(
        objective=objective,
        support_points=support_points,
        j_star=j_star,
        cfg=cfg.sampling,
    )

    child_seeds = rng_master.spawn(
        len(cfg.sampling.betas) * cfg.sampling.chains
    )

    solve_count_before_sampling = flux_lp.solve_count + sparse_lp.solve_count

    for beta in cfg.sampling.betas:
        for chain_id in range(cfg.sampling.chains):
            rng = np.random.default_rng(next(child_seeds))
            start_flux = choose_feasible_start(
                beta=beta,
                center=center,
                optimum=sparse_solution.fluxes,
            )

            sampler = MaxEntCoordinateHitAndRun(
                polytope=polytope,
                objective=objective,
                geometry=geometry,
                beta=beta,
                j_star=j_star,
                energy_scale=energy_scale,
                rng=rng,
            )

            sampler.initialize(start_flux)
            sampler.run_burn_in(cfg.sampling.burn_in)
            samples, traces = sampler.draw(
                n_samples=cfg.sampling.samples_per_chain,
                thin=cfg.sampling.thin,
            )
            write_chain_output(beta, chain_id, samples, traces, cfg)

    solve_count_after_sampling = flux_lp.solve_count + sparse_lp.solve_count
    assert solve_count_after_sampling == solve_count_before_sampling

    run_diagnostics(...)
    write_complete_marker(...)
```

---

## 35. References and implementation sources

### Maximum-entropy metabolism

De Martino, D., Andersson, A. M. C., Bergmiller, T., Guet, C. C., and Tkačik, G. (2018). *Statistical mechanics for metabolic networks during steady state growth*. Nature Communications 9, 2988.  
https://doi.org/10.1038/s41467-018-05417-9

De Martino, D. and De Martino, A. (2018). *An introduction to the maximum entropy approach and its application to inference problems in biology*. Heliyon 4, e00596.  
https://doi.org/10.1016/j.heliyon.2018.e00596

### Flux-space sampling and rounding

Haraldsdóttir, H. S., Cousins, B., Thiele, I., Fleming, R. M. T., and Vempala, S. (2017). *CHRR: coordinate hit-and-run with rounding for uniform sampling of constraint-based models*. Bioinformatics 33, 1741-1743.  
https://doi.org/10.1093/bioinformatics/btx052

Theorell, A., Jadebeck, J. F., Nöh, K., and Stelling, J. (2022). *PolyRound: polytope rounding for random sampling in metabolic networks*. Bioinformatics 38, 566-567.  
https://doi.org/10.1093/bioinformatics/btab552

Lovász, L. and Vempala, S. (2006). *Hit-and-run from a corner*. SIAM Journal on Computing 35, 985-1005.  
https://doi.org/10.1137/S009753970544727X

### COBRApy

COBRApy documentation, reading and writing JSON models:  
https://cobrapy.readthedocs.io/en/latest/io.html

COBRApy JSON API:  
https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/io/json/index.html

### HiGHS and highspy

HiGHS Python interface and examples:  
https://ergo-code.github.io/HiGHS/dev/interfaces/python/

HiGHS model-passing examples:  
https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/

HiGHS `HighsLp` data structure:  
https://ergo-code.github.io/HiGHS/dev/structures/classes/HighsLp/

HiGHS `HighsSparseMatrix` data structure:  
https://ergo-code.github.io/HiGHS/dev/structures/classes/HighsSparseMatrix/

---

## 36. Final instruction to the coding agent

Build the method from the mathematics outward.

First make the sparse objective and one-dimensional conditional sampler correct on tiny models. Then construct the reduced geometry. Only after those tests pass should the code touch a genome-scale model.

The central invariant is simple:

> Every production sample must be a feasible biological flux vector drawn by a transition that preserves the declared maximum-entropy target.

Do not trade that invariant for a convenient solver wrapper, an unvalidated heuristic direction set, or an attractive-looking flux histogram.
