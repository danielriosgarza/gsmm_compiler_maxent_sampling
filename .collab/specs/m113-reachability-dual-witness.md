# M11.3 — reachability's caller-specific dual-witness path

**Milestone class:** M9 math-critical (the reachable-state mass-balance certificate, BUILD_PLAN §1.4.2).
Requires a `/collab` adversarial review as a gate step, before building and again to close.

## The defect (measured, not asserted)

`certify_reachable_mass_balance` (rounding.py) proves every reachable state `c + T·y` satisfies the
mass-balance contract via two LPs per metabolite: `R_i = max(|r_c,i + min_Y E_i·y|, |r_c,i + max_Y
E_i·y|)`, certified `⟺ max_i R_i ≤ η = 1e-9`. Each extreme is a **rigorous upper bound from weak
duality** computed in `_reachable_extreme`, which reads **only `solution.row_duals`** — no primal,
no objective_value. The bound is sound for *any* finite duals (it assumes nothing about optimality
or even feasibility of the returned point).

But `_reachable_extreme` obtains the solution via `program.maximize(unit)` → `HighsLinearProgram.solve()`,
and `solve()` raises `LPNotOptimalError` **before** calling `getSolution()` on any non-`kOptimal`
model status. So on a `kUnknown` solve the certificate refuses on the one output it never reads.

`kUnknown` here is warm-start degradation, the M11 root cause: a persistent warm-started HiGHS
instance re-solving hundreds of single-objective LPs eventually cannot *certify* optimality for one
objective, though its duals stay sound. build-geometry fails for the whole strain as a result.

## The premise, measured on all 6 failing strains (this machine, ambient threads)

Sweep of the 40 curated strains: **34 OK, 6 FAIL, all 6 on `reachable_mass_balance`**. (The tracker's
older "30 OK / 10 fail" included 2 Hafnia + 1 pumilus + 1 Liquorilactobacillus that pass here — they
are basis/RNG-marginal per §1.6.10/§1.6.11 and are not this milestone.)

A faithful dry-run of the fix (reads the duals HiGHS returns on `kUnknown` via `program._highs`,
normalizing the ~1e-13 objective exactly as `_reachable_extreme` does, on the same warm program):

| strain | d | kUnknown solves | max_dual_infeas | warm-vs-cold bound agreement | worst_absolute | CERTIFIED margin |
|---|---|---|---|---|---|---|
| Rahnella aquatilis | 145 | 1 | 0.0 | 8.3 digits | 7.60e-11 | 13.2x |
| Lentilactobacillus kefiri | 58 | 1 | 0.0 | 8.4 digits | 5.95e-11 | 16.8x |
| Lactococcus lactis BIA2553 | 51 | 1 | 0.0 | 9.0 digits | 3.75e-11 | 26.6x |
| Lactiplantibacillus pentosus | 71 | 1 | 0.0 | 8.2 digits | 6.04e-11 | 16.6x |
| Lentilactobacillus buchneri | 67 | 2 | 0.0 | 7.8 digits | 5.61e-11 | 17.8x |
| Lactiplantibacillus plantarum | 97 | 1 | 0.0 | 8.4 digits | 8.52e-11 | 11.7x |

Uniform, across ~2838 solves:
1. The only non-optimal status ever seen is `kUnknown`. Never `kUnbounded`, never `kInfeasible`.
2. `max_dual_infeasibility == 0.0` on every `kUnknown` solve: the discarded duals are dual-feasible.
3. The bound from the discarded `kUnknown` duals agrees with a **cold optimal** re-solve of the same
   LP to 7.8-9.0 digits: the duals are tight, not merely sound.
4. The binding row (worst_absolute) is always a cleanly-solved `kOptimal` row. Accepting the
   `kUnknown` duals does not change the verdict; it lets the loop *finish* instead of raising.

## The proposed fix (REVISED after /collab round 1 — Codex DISAGREE, 3 points, all engaged)

A **caller-specific dual-witness path** that reads the duals whatever the status, WITHOUT loosening
`HighsLinearProgram.solve()` (`sparse_objective.critical_l1_penalty` needs `LPNotOptimalError` to
detect an unbounded `J*`; the FVA escalation needs it too).

**Backend (highs_backend.py).** A **narrow** witness type — NOT `LPSolution`, whose docstring
guarantees "holding one proves kOptimal" (Codex round-1 point 2):

    @dataclass(frozen=True)
    class LPDualWitness:
        """Row multipliers from a solve that may be non-optimal. Holding one proves NOTHING about
        primal feasibility or optimality — only a weak-duality bound, valid for any finite duals."""
        model_status: str
        run_status: str
        row_duals: NDArray[np.float64]
        elapsed_seconds: float

Refactor `solve()` so its observable behaviour is byte-identical (still raises before `getSolution()`
on non-optimal) by extracting `_run_solver()` (frozen-check, run, counter, kError → status). Add:

    def solve_dual_witness(self, *, accept: frozenset[str]) -> LPDualWitness:
        # accept validated at call: every token must be a real HighsModelStatus member name,
        # else a typo silently never-matches and refuses everything.
        status, run_status, elapsed = self._run_solver()
        if status.name not in accept:
            raise LPNotOptimalError(str(status), self.name, accepted=accept)  # accurate message
        sol = self._highs.getSolution()
        row_duals = np.asarray(sol.row_dual, dtype=VALUE_DTYPE)
        if row_duals.shape != (self._n_rows,):   # validate ONLY the duals, not the primal
            raise HighsBackendError(...)
        return LPDualWitness(str(status), str(run_status), row_duals, elapsed)

    def maximize_dual_witness(self, costs, *, accept) -> LPDualWitness:  # mirrors maximize()
        self.set_objective(costs); self.set_maximize()
        return self.solve_dual_witness(accept=accept)

The whitelist is checked BEFORE `getSolution()` — same safety ordering as `solve()`.
`solve_dual_witness` does NOT inspect `max_dual_infeasibility` or any quality signal: gating a
dual-based bound on a dual-quality signal is the exact M11 anti-pattern. Match key is `status.name`
(exact) — NOT the substring `"Unbounded" in str(status)` precedent, which also matches
`kUnboundedOrInfeasible` (Codex). `LPNotOptimalError` gains an optional `accepted=` so its message
says "expected one of {…}", not the now-false "expected kOptimal".

**rounding.py.** A module constant declares the provably-reachable statuses for *this* LP:

    _REACHABLE_WITNESS_STATUSES: Final = frozenset({"kOptimal", "kUnknown"})

`_reachable_extreme` swaps `program.maximize(unit)` for
`program.maximize_dual_witness(unit, accept=_REACHABLE_WITNESS_STATUSES)`. The whitelist is
`{kOptimal, kUnknown}` — NOT tighter (`kOptimal` is the common case) and NOT wider (limit/error
statuses are NOT ruled out by boundedness/feasibility, so they stay fail-closed; Codex). The reasons
`kUnbounded`/`kInfeasible` cannot arise are properties of *this* LP:
- **not `kUnbounded`:** every column is bounded, `|y_k| ≤ Ω_k` with `Ω` finite.
- **not `kInfeasible`:** `y = 0 ∈ Y` (centre feasible ⇒ `T·0 = 0 ∈ [l−c, u−c]`, `|0| ≤ Ω`).
`kUnknown` is accepted because it is the **measured** degradation mode — not because boundedness and
feasibility prove it the *only* possible non-optimal status.

**Telemetry (Codex point 3).** `certify_reachable_mass_balance` counts witnesses that came back
`kUnknown` and records `n_unknown_witnesses` on `ReachabilityCertificate` (a durable, cached field —
"visibility beside elimination", the M10.2e manifest lesson), so the exceptional path is visible in a
successful artifact, not just at build time.

**Cache identity (Codex point 3).** Adding the field changes the certificate schema (its `from_cache`
requires every field), so **bump `ROUNDING_IMPL_VERSION` 2 → 3** (folded into `geometry_cache_key`):
stale bundles MISS and recompute, never hit-then-error. No bound *value* changes (see point 1 below),
so this is a schema/telemetry bump, not a numerical one.

## Why this is sound (the load-bearing argument)

The weak-duality bound is a rigorous UPPER bound for **any finite π** — exactly (no assumption of
optimality, dual feasibility, or even primal feasibility of the returned point). Looser duals can
only make each extreme, and thus `worst_absolute`, **larger** → accepting `kUnknown` duals errs
toward **refusing** a transform, never toward a false CERTIFY. Measurement shows the duals are in
fact tight (8-digit agreement); soundness does not depend on that. `solve()` is untouched.

## Codex round-1 point 1 (residual charge): CONCEDED as real, DEFERRED as out of scope — measured

Codex: `_reachable_extreme` proves the bound for `norm·unit`, not `objective`; the residual
`r = objective − norm·unit` term `Σ|r_k|·Ω_k` is uncharged, and the final `*norm` / `+offset` / `max`
round inward. **Conceded: the gap is real.** Two facts decide scope:
1. **It is dual-independent.** `r` involves only the objective normalization, never `π`. It is
   *identical* for a kOptimal and a kUnknown solve. So it is orthogonal to the dual-witness change —
   accepting kUnknown duals does not touch it — and predates this fix (M9).
2. **Measured:** the uncharged term is **2.7e-25 (Rahnella d145), 4.2e-26 (lactis d51)** in objective
   units — ~16 orders below the 1e-9 contract, ~8e-17 relative (sub-ULP). It cannot change a verdict
   at any contract the package declares.

**Decision: defer, do not bundle.** M11.3 changes zero bound *values* (the witness path only lets a
solve's duals be *read*); bundling a ~1e-25 dual-independent arithmetic change would perturb every
bound for reasons unrelated to the milestone. Recorded as a measured follow-up (harden
`_reachable_extreme`: charge `Σ|objective − norm·unit|·Ω` and round the final scale/offset/max
outward), to be done as its own small step. The spec's soundness claim is narrowed to the precise
true statement above (exact for any finite π), which is what the dual-witness change actually relies
on.
