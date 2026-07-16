# M7 adversarial review — reweighted-L1 with frozen weights

You are reviewing a maximum-entropy flux sampler for genome-scale metabolic models. The objective is
J(v) = μ(v) − λ·Σ w_r|v_r| (μ = biomass flux). A coordinate hit-and-run MCMC samples π_β ∝ exp(β·(J−J*)/s_J)
over a polytope. M7 adds iterative reweighted-L1 (spec §13): iterate w_r ← w_base/(|v_r|+ε), clip,
renormalize to unit median, stop at a fixed point, FREEZE the weights, then sample. A weight that moves
mid-chain retargets every 1-D conditional and destroys stationarity — that is the cardinal sin.

This codebase's signature bug (found 12× in the M6 review): "two artifacts never computed against each
other, silently joined" — e.g. an objective's J* subtracted from a DIFFERENT objective's J(W) to form s_J.
It computes fine and every diagnostic agrees while describing the wrong distribution.

## Read these
- src/gsmm_compiler/reweighting.py         (NEW: the loop, λ policy, freezing)
- src/gsmm_compiler/sparse_objective.py     (changed near: SparseFluxObjective.content_key/_frozen,
                                             LPOptimum.objective_key, ReducedObjective.objective_key,
                                             choose_energy_scale, lower_objective)
- src/gsmm_compiler/maxent_sampler.py       (changed: _check_bindings — the objective_key guard ~line 1187)
- src/gsmm_compiler/config.py               (ObjectiveConfig.reweighting_* fields ~line 97)

## Claims to attack (find a concrete failure, name file:line, give the input that breaks it)

1. **λ policy.** λ is RE-RESOLVED each iteration: λ_k = λ̃·λ*(w_k) via resolve_objective. NOT frozen at the
   base value. Measured justification: a frozen λ collapses effective pressure λ/λ*(w) from 0.5 to ~4e-6
   and by iteration 2 crashes M3's z==|v| LP gate (deviation 25 at default clip [1e-3,1e3], 1e3 at wider).
   Attack: does re-resolving λ each iteration make "converged" ill-defined or create a target that never
   settles? Does the fixed point depend on the clip in a way that isn't reported?

2. **Median renorm is a no-op.** Claimed: because λ=λ̃·λ*(w) and λ*(cw)=λ*(w)/c, scaling w by c leaves
   λ·w invariant, so step-4 normalization cannot move the target. Attack: is this EXACT? Clipping is
   applied to the RAW update BEFORE normalization (update_weights). Does the clip→normalize order break
   the invariance in a way that changes the frozen fixed point vs a normalize→clip order? Is the claim
   as-implemented (not just in theory) sound?

3. **Freeze completeness.** ReducedObjective.weights and L1Objective.weights are made read-only via
   _frozen (np flags.writeable=False) in lower_objective. SparseFluxObjective.weights also frozen in
   from_polytope/with_weights. reweighting.py cannot import maxent_sampler. Attack: any alias or path by
   which a frozen weight buffer could still be mutated after freezing, or by which the sampler could
   receive unfrozen weights? Check np.ascontiguousarray/asarray returning the SAME array (M5 found this).

4. **objective_key join.** M6 keyed EnergyScale only on polytope_key. M7 makes TWO objectives per polytope
   (base vs reweighted), s_J differs 100× on the toy. Added objective_key to ReducedObjective, LPOptimum,
   EnergyScale; choose_energy_scale(objective, warmup, *, optimum: LPOptimum, ...) now raises if
   optimum.objective_key != objective.objective_key; run_ladder checks energy_scale.objective_key ==
   objective.objective_key. Attack: is there STILL a path where a base-objective s_J or J* reaches a
   reweighted ladder and passes every check? Are objective_key and content_key computed from the same
   inputs so they actually match when they should?

5. **Frozen == solved.** The returned frozen ReducedObjective is the SAME object the converging iteration's
   LP solved against (variable `resolved` reused, not rebuilt), so solution.optimum.objective_key ==
   objective.content_key(). Attack: is the fixed-point identification right? Could the loop return a
   `solution` from iteration k but weights from iteration k (they must match) — trace the loop's last
   iteration and the break condition. Off-by-one?

6. **n_shed guard.** A clip ceiling below the "nearly off" band (|v|~1e-3) merges it with "off" and
   reweighted-L1 silently becomes plain L1. n_shed = n_active_base − n_active_final is reported + a warning.
   Attack: is n_shed the right signal, or a deeper silent degeneration (e.g. reweighting that MOVES the
   active set without changing its size, so n_shed=0 but the distribution still shifted vs plain L1)?

Also: convergence criterion is (active-set symmetric-difference == 0) AND (relative max|Δv| ≤ solution_tol),
both against the PREVIOUS iterate; unconverged RAISES unless allow_unconverged. Attack the criterion.

End with exactly:
VERDICT: AGREE | DISAGREE
CONTESTED: <numbered list of points you dispute, or 'none'>
