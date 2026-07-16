# M7 review round 3 — both round-2 contested points FIXED. Confirm closure.

Round-2 scoreboard: you CONCEDED points 1,2,3,5,6 (and confirmed fixes A/C). Two contested remained:

## Contested 1 — Point 4/D: LPOptimum not bound to its polytope. FIXED.
- Added `polytope_key: str` to `LPOptimum`, stamped in `SparseObjectiveLP._verify` from
  `self.reduced.content_key()`.
- `choose_energy_scale` now checks BOTH:
    if optimum.objective_key != objective.objective_key: raise   # same-polytope, wrong-objective
    if optimum.polytope_key  != objective.polytope_key:  raise   # same-objective-key, wrong-polytope
- Verified at runtime with your exact counterexample: two polytopes, identical reaction ids/biomass/
  order/λ/weights but different bounds → same objective_key (True), different polytope_key (True) →
  cross-join `s_J = J*_A − Q(J_B(W))` is now REFUSED ("different polytope"). Regression test added.
- run_ladder already checks energy_scale.polytope_key == reduced.content_key() and
  energy_scale.objective_key == objective.objective_key. So the chain end is covered too.

## Contested 2 — Fix B: clip-ratio underflow. FIXED.
- Root cause you named: after median normalization the smallest weight is bounded by clip_min/clip_max
  (the RATIO), not clip_min. Config validation previously allowed any finite 0<min<max.
- Added ObjectiveConfig validation: reject weight_clip_max/weight_clip_min > 1e9 (default [1e-3,1e3]
  is 1e6; measured LP breakage at 1e12). Test: [1e-20,1e20] now raises "clip ratio".
- Belt-and-suspenders: `_relative_weight_change` returns inf if any denominator is non-finite or ≤0,
  so a degenerate metric fails LOUD (never converges) rather than silently NaN-ing to "converged".

## Your task
Confirm these two closures are sound, OR name what still leaks. In particular:
- Is checking BOTH objective_key AND polytope_key on the optimum sufficient, or is there a THIRD
  identity (e.g. the warmup_fluxes provenance — they come from geometry.support_points; are they
  bound to the polytope)? Note choose_energy_scale evaluates warmup via `objective.evaluate_many`,
  and objective.polytope_key is checked against optimum.polytope_key, but the warmup ARRAY itself
  carries no key — is that a hole (someone passes polytope B's objective+optimum but polytope C's
  raw warmup array)?
- Any remaining distribution-corrupting join in the M7 path.

End with exactly:
VERDICT: AGREE | DISAGREE
CONTESTED: <numbered list, or 'none'>
