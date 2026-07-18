# M11.2 — span gate + build-wide solve session  (Claude × Codex, design review before build)

`/collab` think, 2026-07-18. Verdict DISAGREE-as-written; the **substance is agreed**, the factoring
corrected. This records the decisions the round changed, per the CLAUDE.md protocol.

## Measurement that licensed the build (Codex required it first)
Wrapping `program.maximize` so a `kUnknown` solve is re-solved on a **fresh** instance, over the
strains that fail geometry after M11.1:
- **3 of 3 `kUnknown@flux_only` strains tested**: the failing solve is `kUnknown` WARM, `kOptimal`
  COLD (50–83 iters, finite obj). Same warm-start history as M11.1's `blocked_reactions`. *(5 of 8
  such strains untested — Codex flagged this; the shared cause is established for 3, assumed for 5.)*
- **span refusals are RNG-marginal**: a strain `SPAN-CERT-REFUSED` (n_inconclusive>0) under its own
  `model_id` builds with n_inconclusive=0 under another — the refusal tracks the seed via
  `model_id → stream_seed → discovery draws → basis → which complement directions get probed`.
- **decisive for the gate**: `lactis` has `exhaustive=False, n_inconclusive=2`, **but
  `resolution = 5.361e-11 ≤ span_tol = 1e-9`**. The old gate refuses it; `resolution ≤ span_tol`
  passes it. Others measured 4.28e-11, 7.64e-11, 4.93e-10 — all under 1e-9, though the last with only
  ~2× margin (Codex: thin, watch it).

## (A) Span gate — SOUND (Codex confirmed)
Replace the exhaustiveness gate's `n_inconclusive == 0` with `resolution ≤ span_tol`:
`exhaustive = failing is None and complete and not capped and resolution ≤ span_tol`.
- Every completed complement probe contributes a **rigorous** `width_upper` (weak duality);
  `is_conclusive` is about whether the *primal/residual* discovery signals are usable, and is **not**
  required for that upper bound. So an inconclusive probe may pass and a conclusive-but-loose one may
  fail — the new predicate is **not** a subset of the old, and that is the point.
- "complete" means complete **at resolution `span_tol`**, not exact dimension below it.
- **Extract one `_span_resolution(...)`** used by both `SpanCertificate.resolution` and the gate —
  never duplicate the formula. Keep `n_inconclusive`, `max_width_floor`, `worst_dual_error` as
  diagnostics. Update the `SpanCertificate.exhaustive` docstring (it still says "every probe
  conclusive"). Cached diagnostic semantics change → **bump `GEOMETRY_IMPL_VERSION`**.

## (B) Escalation — per-probe cold re-solve is sound, BUT the lifecycle must be build-wide
- A probe is **externally atomic**: if its `max` succeeds and its `min` returns `kUnknown`, no basis /
  support / certificate accumulator has been touched, so re-solving only the failed objective cold is
  sound (the two endpoints need not share an instance; their weak-duality bounds stay independently
  valid; accumulators update only after the complete `SupportProbe` returns). Same for `maximize(0)`.
- 🔴 **Current-state leak (Codex found it — a hole in M11.1)**: `blocked_reactions` sets its **local**
  `warm = None`, but `build_geometry` reuses the **same** `program` for `maximize(0)`, discovery and
  the sweep. The abandonment does **not cross stage boundaries** — almost certainly why a strain
  recovers inside `blocked_reactions` yet still dies `kUnknown@flux_only` downstream. The fix: a
  **build-wide solve session** owns the instance lifecycle, not a stateless helper.
- A `kUnknown` does **not** prove later warm solves are wrong (a later `kOptimal` still passes primal
  checks, dual bound sound-if-loose). Two valid policies: **(1)** cold-only for the rest of the build
  (conservative, M11.1-consistent) or **(2)** a new warm epoch from the fresh retry (cheaper, permits
  new history). Chosen: **(1)**, and **record** when the transition happened and how many cold solves
  followed. Catch **only `kUnknown`** — infeasible/unbounded/limit keep their hard-failure meaning.
- Before a **final `resolution > span_tol` refusal**, do a **bounded fully-cold confirmation** (the
  same discipline that separated *Hafnia* from the warm artifact) — do not claim model-intrinsic on a
  possibly-loose warm dual.

## (C) Factoring (Codex)
- low-level `solve_fresh_once(reduced, costs)` — one fresh LP, one solve, collect diagnostics.
- a build-wide **session** for the initial/support/span objectives: warm attempt; on `kUnknown`,
  discard the active instance and one fresh retry, then cold-only.
- `blocked_reactions` **keeps** its bracket + cold-**pair** decision (routing each leg through an
  automatic cold-on-`kUnknown` primitive would silently expand M11.1's measured 1-pair budget), uses
  `solve_fresh_once` per leg, and **notifies** the session that the persistent instance degraded.
- **Fix solve accounting**: `program.solve_count` omits fresh-instance solves; the session counts all
  attempts (warm, failed-warm, cold retries, M11.1 cold pairs).

## Not established by the measurement (carry as open)
Recovery for the other 5 of 8 `kUnknown@flux_only`; that *all* span/seed/machine cases clear the new
gate; that a post-`kUnknown` instance actually corrupts later `kOptimal` solves; the `4.93e-10` ~2×
margin; anything downstream of geometry (rounding/pilots/sampler never run on an aerobe).

## Closing review (Codex, DISAGREE-then-fixed) — 4 real discrepancies + a finding the sweep refined
Built both parts; census: build-geometry OK **4 → 20 → 30 of 40**, `kUnknown@flux_only` **8 → 0**,
the §1.6.10 noise-swamped span refusals (strain_1/strain_11) gone. Codex read the diff and found the
shipped code did not match the approved factoring on four points, all fixed:
1. `blocked_reactions` caught **every** `LPNotOptimalError`, not just `kUnknown` — an infeasible/
   unbounded status (a model verdict) would have been silently cold-retried. Now `_reraise_unless_
   kunknown` re-raises everything but `kUnknown`, in the warm bracket and the session alike.
2. The cold pair bypassed `solve_fresh_once`. Now `_cold_width_bracket` calls the one primitive twice.
3. `degraded_at`/`n_cold_solves` described different solve populations than `n_lp_solves` without
   saying so. `degraded_at` (a session solve index that read 0 when degradation happened in
   `blocked_reactions`, whose FVA bypasses the session) → **`degraded: bool`**; `n_lp_solves` now
   documents that it is the process-global total (exact for a serialized build; the batch runs
   geometry one strain at a time) and `n_cold_solves` that it is the session's own cold retries.
4. **One cold observation per strain does not establish a model-intrinsic floor** — the cold re-sweep
   keeps the RNG-discovered basis, so a different `model_id` gives a different resolution. Codex made
   this a required measurement, and it split the two span refusals:
   - **pumilus: 8/8 seeds refuse** (resolution 2.35e-9…5.20e-9, d=88). A **genuine √k floor**: the
     axis-wise complement certificate cannot certify it to 1e-9 under any seed.
   - **Liquorilactobacillus: 3/8 seeds *pass*** (resolution 7.67e-10…4.57e-9, d=56). **RNG-marginal,
     not a floor** — `max_width` varies with the discovered basis, so the resolution straddles
     span_tol. This is §1.6.10 at the *resolution* level, and it needs a **tighter (basis-independent)
     span certificate**, not the gate change M11.2 made. Deferred.

**Ruling adopted**: M11.2 closes on its stated scope (the `kUnknown` crashes and the noise-swamped
span refusals are fixed) with Option A (strict `resolution ≤ span_tol` gate). The distribution
"valid to resolution R" contract (Option B) is explicitly a **separate** milestone — a small
orthogonal-width bound does not by itself bound the sampled distribution's error (a thin polytope can
have cross-sectional volume varying strongly along retained directions; in TV the slice law can even
be singular vs the full-dimensional target). Two span residues deferred: pumilus (genuine floor) and
Liquorilactobacillus (basis-dependent √k looseness). Also flagged and added: an end-to-end test that
forces a warm resolution failure and proves the fully-cold re-sweep decides it.
