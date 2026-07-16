# M7 review round 2 — you reviewed BLIND last time (your sandbox bwrap failed, you saw the pre-M7
# public branch). Here is the ACTUAL code. Verify your round-1 claims against it. I have already
# implemented fixes for points 5 and 6; confirm they close the issue or find what still leaks.

## Round-1 resolution summary (my assessment)
- Point 1 (λ re-resolve): you AGREED it holds. Settled.
- Point 2 (median no-op, clip order): you agreed the invariance is exact for a fixed clipped vector.
  Your residual concern — frozen weights can exceed the nominal clip after normalization — is COSMETIC:
  the manifest reports the actual final weights (SparseFluxObjective.manifest lists weights[penalized]),
  and the λw *dynamic range* (a ratio) is preserved by uniform scaling. No distribution effect. Do you
  still contest, given the manifest is honest?
- Point 3 (freeze): your concrete attack `with_weights(view)` is REFUTED — with_weights does
  np.asarray(weights).copy(), from_polytope does .copy(), lower_objective indexes with integer arrays
  (fancy index → copy). I verified at runtime: owndata=True, writeable=False, in-place mutate raises,
  owner-mutation does not propagate. The residual "adversary flips writeable back to True" is accepted
  as OUT OF SCOPE per the M5 precedent (rounding._freeze docstring: "accident-proof, not
  adversary-proof"). Do you accept M5's precedent, or argue M7 needs more?
- Point 4 (objective_key): reduces to point 3 — it needs an in-place mutation of a frozen buffer, which
  raises. The production path (lower_objective) always computes objective_key = objective.content_key()
  from the same weights. Do you still see a NON-mutation path?
- Point 5 (convergence blind on small coords): CONFIRMED and FIXED. See code below.
- Point 6 (n_shed net-only): CONFIRMED as reporting gap and FIXED. See code below.

## The actual current code (post-fix)

### reweighting.py — convergence + weight metric (the point-5 fix)
```python
def _relative_weight_change(new, old, penalty_mask):
    both = penalty_mask & ((new > 0.0) | (old > 0.0))
    if not np.any(both):
        return 0.0
    denominator = np.maximum(new[both], old[both])
    return float(np.max(np.abs(new[both] - old[both]) / denominator))

# inside the loop, each iteration:
    v = solution.optimum.v_full
    active = set(flatnonzero(|v| > active_tol))
    if iteration == 0: active_base = active
    updated, at_low, at_high = update_weights(base, v, mask, epsilon, clip)  # w_{k+1} = F(v_k)
    weight_change = _relative_weight_change(updated, weights, mask)          # F(v_k) vs w_k=F(v_{k-1})
    if previous_v is None:
        flux_change, set_changes = inf, -1
    else:
        set_changes = len(active ^ previous_active)
        converged = set_changes == 0 and weight_change <= solution_tol      # WEIGHTS, not fluxes
    history.append(ReweightingStep(..., max_weight_change=weight_change, max_flux_change=flux_change, ...))
    if converged: break
    previous_v, previous_active = v.copy(), active
    weights = updated
# after loop: frozen objective = resolved.objective (w_k), solution = final.solution (v_k = solve(w_k)).
# convergence guarantees ||F(v_k) - w_k||_rel <= tol, so w_k IS a fixed point (standard stopping rule).
```

### reweighting.py — point-6 reporting fix
```python
# ReweightingReport now stores active_base, active_final: frozenset[int]
n_turned_off = len(active_base - active_final)   # L1-on that reweighting shut
n_turned_on  = len(active_final - active_base)   # L1-off that reweighting lit (support rearranged)
n_shed       = n_active_base - n_active_final    # net (headline)
support_unchanged = (active_base == active_final)  # the TRUE "did nothing to support" signal
# warning now fires on support_unchanged (symmetric diff empty), NOT net n_shed==0 (which warns
# spuriously under a straight swap). Warning explicitly notes the frozen weights may still differ
# from uniform, so the sampled law is not necessarily plain L1's — only the sparsity claim is empty.
```

## Attack the FIXES specifically
A. Is converging on `weight_change = _relative_weight_change(F(v_k), w_k)` sound as a fixed-point test?
   Consider: LP degeneracy (v_k jumps between tied vertices with the SAME active set and SAME weights
   — could weight_change be ~0 while the chain later samples a non-unique optimum region? No, sampling
   is at β<∞ over the whole polytope, J is fixed). Any case where weight_change ≤ tol but the frozen
   objective is NOT a fixed point of w↦F(solve(w))?
B. Does _relative_weight_change divide by a small number that is noise (the M4 rule)? denominator =
   max(new,old) ≥ clip_min = 1e-3 on the penalty set. Is that safe?
C. The frozen `active_final = active` (the converged iteration's active set). Is that the active set of
   the SAME v that final.solution holds, and that objective.content_key() covers? Trace for an
   off-by-one.
D. Anything ELSE that corrupts the SAMPLED distribution while every test passes — the codebase's
   signature failure. Especially: does the frozen objective handed to the sampler exactly match the
   objective whose J* and s_J were computed (choose_energy_scale takes optimum: LPOptimum keyed on
   objective_key; run_ladder checks energy_scale.objective_key == objective.objective_key)?

End with exactly:
VERDICT: AGREE | DISAGREE
CONTESTED: <numbered list of points you still dispute, or 'none'>
