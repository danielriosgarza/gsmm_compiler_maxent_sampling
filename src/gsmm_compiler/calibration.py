"""The pilot DAG: bootstrap geometry → β=0 pilot → {final ``T``, ``s_J``}  (M10, spec §17.4/§22.2).

Two of v1's recorded findings converge on one structural change, and it is the same change:

- **M5**: this model mixes slowly (`step_scale_ratio = 0.008`). Re-rounding from a β=0 pilot's own
  covariance instead of M4's support-LP vertices improves ``cond(C_q)`` **1.54e4 → 5.97e3 (2.57×)**
  and ``step_scale_ratio`` 0.0081 → 0.0129 (spec §17.4). *(Measured M10.2b at the shipped default
  config. The 5.11e3/5.36e3 this line and BUILD_PLAN §1.6.6 used to record are **pre-M10.2a**: that
  milestone removed the objective's start hint from the pilots, which — as its own
  `CALIBRATION_IMPL_VERSION` = 2 note says — **changes every pilot's draws**, and nothing
  re-measured the numbers those draws produce. Re-running the old path with the hint reproduces
  5304, confirming the cause. The finding survives; only its magnitude was stale. A derived
  quantity's recorded value is a claim with a premise, and M10.2a moved the premise.)*
- **M6**: the β axis is uncalibrated. ``s_J`` is read off the support *vertices*, where the L1
  cost is enormous, while the chain lives in the interior — so ``s_J = 32.5`` calibrates β against
  a range 13× wider than the one actually explored (spec §22.2, BUILD_PLAN §2.1).

Both want the same object: **a frozen β=0 pilot chain**. This module runs it, and hands its
covariance to `rounding.reround_transform` and its ``J`` spread to
`sparse_objective.pilot_energy_scale`.

## The pipeline, and why it is sequential

```
1. geometry pilot at β=0 under T₀          (OBJECTIVE-INDEPENDENT)
2. freeze its covariance → build T₁
3. INDEPENDENT scale pilot at β=0 under T₁  (better mixing → better ESS for σ̂₀)
4. freeze σ̂₀ → production chains on independent streams
```

The two pilots are **separate streams on purpose**. One shared pilot would be perfectly valid — the
transform cannot move the stationary law and both artifacts are frozen before production — but it
would make pilot-seed sensitivity **unattributable**: geometry quality and the selected target
would move together, so a run that looked odd could not be diagnosed into "the rounding got
unlucky" versus "the scale got unlucky". Separating them separates *random efficiency calibration*
from *random target calibration*. (Codex, M10 review round 2.)

⚠️ **Say that precisely: what is independent is each pilot's RNG stream *given its inputs*, not the
two artifacts.** ``T₁`` is derived from the geometry pilot and the scale pilot runs under ``T₁``, so
the scale pilot depends on the geometry pilot through its own frame — unconditional independence is
not on offer and never was. The evidence for the property that *is* claimed is each chain's
**spawn key**, recomputed from the recipe and compared (`PilotRecipe.expected_spawn_keys`). It is
not ``not np.allclose(a, b)`` over two pilots' arrays, which proves **non-identity** and was what
M10.2a's tests actually checked while their names claimed independence. (Codex, M10.2b review
round 1 — the argument that killed a payload design built to preserve those tests.)

The scale pilot runs under ``T₁`` deliberately, and the compounding worry is answered by noticing
what can and cannot propagate: a poor ``T₀`` **cannot deform the neutral target** — only the
efficiency of estimating σ̂₀ from it. So the errors do not compound as target deformation; they
compound as imprecision, which is measured and reported.

## What this module may and may not do

`calibration` imports `maxent_sampler`; **the sampler must never import `calibration`**. That is not
style, it is the invariant: an adaptive ``T`` or a ``s_J`` that could be re-derived mid-chain makes
the transition kernel depend on the chain's own history, and the samples are then not from ``π_β``
at
all (spec §17.4, §18.3). The dependency runs one way so a production chain *structurally cannot*
re-calibrate itself — the same guard shape M7 uses to keep reweighting out of the sampler, pinned by
`tests/unit/test_calibration_cannot_be_imported_by_sampler`.

## What the DAG guarantees, stated precisely

Freezing ``T₁`` and ``σ̂₀`` before production gives a **time-homogeneous kernel with a fixed
conditional invariant law**. It does *not* give stationarity from iteration zero — burn-in provides
convergence, not stationarity, unless the initial state is drawn from the law. And conditional on
the
pilot artifact the invariant target is ``π_{β/σ̂₀}``, **not** the ideal ``π_{β/σ₀}``; marginalising
over pilot randomness gives a *mixture* of calibrated targets. That is **calibration uncertainty,
not
an MCMC invariance failure** — a real thing, reported in `PilotScaleReport`, and different in kind
from a bug. (Codex, M10 review round 3; recorded rather than blurred.)

The transform's own invariance is a theorem (`range(diag(s)·B·L)` is ``L``-invariant, §1.6.1), but
range-invariance alone is *not* the clean condition: ``T₁`` must be a nonsingular affine coordinate
change **on the affine hull**. The real risks are implementation ones — rank loss, feasibility
tolerance, state carry-over, residual adaptation — which is what `rounding`'s SVD rank check and
this
module's freezing exist to close.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
from numpy.typing import NDArray

from gsmm_compiler.affine_geometry import ReducedGeometry
from gsmm_compiler.cache import ArtifactCache
from gsmm_compiler.config import GeometryConfig, SamplerConfig
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.logging_utils import get_logger
from gsmm_compiler.maxent_sampler import SAMPLER_IMPL_VERSION, SamplerResult, run_chains
from gsmm_compiler.provenance import content_key, stream_seed
from gsmm_compiler.rounding import (
    ReachabilityCertificate,
    RoundedTransform,
    certify_reachable_mass_balance,
    require_certified_transform,
    reround_transform,
)
from gsmm_compiler.sparse_objective import (
    EnergyScale,
    LPOptimum,
    ReducedObjective,
    pilot_energy_scale,
)

_log = get_logger(__name__)

VALUE_DTYPE = np.float64

CALIBRATION_IMPL_VERSION = 3
"""Bump when the pilot's arithmetic or schedule semantics change — invalidates cached pilots.

3 (M10.2b): the pilots now honour ``geometry.feasibility_tol`` instead of silently taking
`maxent_sampler.run_chains`' 1e-9 default, so a run that overrides it draws different numbers.

**This bump is bookkeeping, and the rationale first written here was wrong.** It claimed the bump
was needed to stop a v2 pilot colliding with a v3 one. It cannot be: `PilotRecipe.content_key`
passes `feasibility_tol` as a **named component**, so every v3 key already differs from every v2 key
that lacked it — and no v2 pilot was ever stored anyway: M10.2b is the milestone that wires the
store. The bump says "the semantics of this artifact changed", which is true and worth recording; it
prevents nothing that the key does not already prevent. (Codex, M10.2b review round 3. Kept, with an
honest reason — a version constant defended by a false argument is worse than one defended by none.)

2 (M10.2): the β=0 pilots no longer receive the objective's ``optimum_coordinates`` start hint, so
every pilot's draws change; and the pilot key gained `seed`, `refresh_interval` and
`sampler_impl_version`. The bump would be redundant with the added key components alone — but a
*removed* input leaves no trace in a key, so a v1 pilot and an M10.2 pilot of the same recipe would
otherwise hash identically while holding different draws.
"""

PILOT_CACHE_LAYER = "pilot"
"""The pilot's layer in `cache.ArtifactCache` (BUILD_PLAN §1.1).

Not a new *layer* of the DAG in §1.1's numbered sense — §1.6.7 settled that the pilot and ``T₁`` do
not become L4. It is a named immutable node keyed on `PilotRecipe`, which is what §1.1's rule
("cache what is expensive, derive what is cheap, key everything") actually asks for: the pilots cost
**19.3 s** measured against ``build_geometry``'s 1.17 s and ``reround_transform``'s 9 ms.
"""

GEOMETRY_PILOT_STAGE = "geometry_pilot"
SCALE_PILOT_STAGE = "scale_pilot"
"""The two `provenance.stream_seed` stages. Distinct names are what make the pilots' streams
independent *given their inputs*: the RNG is keyed on ``(model_id, stage, β_index, chain_index)``,
so two stages that share a name share their draws — which is exactly the coupling this DAG
separates. They are **not** the artifacts' independence: ``T₁`` is derived from the geometry pilot,
so the scale pilot depends on it through its own frame (Codex, M10.2b review round 1)."""

PILOT_BETA_INDEX = 0
"""A pilot is always β=0, so its stream's rung coordinate is always 0. Named rather than inlined
because `PilotRecipe.expected_spawn_keys` has to reproduce exactly what `run_chains` consumed."""

SPAWN_KEY_WIDTH = 4
"""``(model_id, stage, β_index, chain_index)`` — `provenance.stream_seed`'s four coordinates."""


class CalibrationError(ValueError):
    """A pilot could not be run, or its artifacts do not bind to each other."""


@dataclass(frozen=True)
class PilotRecipe:
    """Everything that decides a pilot's bytes — and **nothing that a pilot's bytes decide**.

    Split out from the artifact in M10.2b because a content-addressed store needs the key *before*
    the artifact exists: `cache.ArtifactCache` looks an entry up by a function of its **inputs** and
    validates it by a content hash. M10.1 wrote the key as a method on the built pilot, which is
    unusable for a lookup — so the alternative was a second key function taking the inputs, and
    **two writers of one key is the defect M10.2a fixed for the L3 bundle**. One object, computed
    once, hashed once, carried by the artifact it names.

    Every field here is read by `maxent_sampler.run_chain`, so every one of them moves the draws.
    That claim is not a comment: `tests/unit/test_m10_2b_pilot_cache` asserts that perturbing any
    single field changes `content_key`, and the audit behind it walked every ``config.*`` the kernel
    touches. The objective is deliberately **absent** — see `NeutralPilot`.
    """

    model_id: str

    stage: str
    """`GEOMETRY_PILOT_STAGE` or `SCALE_PILOT_STAGE` — also the RNG stream's name.

    A field here, a `ClassVar` on the artifact, and `NeutralPilot.__post_init__` refuses them if
    they disagree. The recipe needs it as data (the key must be computable before the pilot runs);
    the artifact must not, or `GeometryPilot(stage=SCALE_PILOT_STAGE)` would hash as a scale pilot
    while holding coordinates — the artifact would stop being a function of its key, in the
    milestone about that (Codex, M10.2b review round 2).
    """

    polytope_key: str

    transform_key: str
    """`RoundedTransform.content_key` of the transform this pilot **ran under**.

    A pilot's coordinates are only meaningful in the frame that produced them. `reround_transform`
    maps them back with ``q = L₀·y``, which is the right change of coordinates for *that* ``L₀`` and
    a silent corruption for any other — same shape, different geometry, every check green.
    """

    n_chains: int
    n_draws: int
    burn_in: int
    thin: int

    seed: int
    """``SamplerConfig.seed``. `provenance.stream_seed` passes it as the `SeedSequence` *entropy*,
    so it changes every draw — and it was missing from the M10.1 key, which is a false hit."""

    refresh_interval: int
    """Sweeps between exact rebuilds of ``v`` from ``y``. M5 settled that in float64 this chain's
    state is ``(y, cache error, refresh phase)``, not ``y`` alone — so this changes the bytes."""

    feasibility_tol: float
    """The tolerance the pilot's chains are actually run at.

    🔴 **M10.2b: the pilots did not have one.** `run_neutral_pilot` called `run_chains` without it
    and so silently took the **1e-9 default**, while production chains use
    ``config.geometry.feasibility_tol`` (`batch.prepare_model` passes it). A run overriding the
    tolerance calibrated its axis on chains the run never asked for. It is not a reporting
    threshold: it reaches start selection, chord construction and refresh validation, so it moves
    both the draws and the RNG consumption.

    The two halves of that fix are inseparable, which is why it lands in the milestone that wires
    the store: while the tolerance was a hardcoded constant the key was **complete without it**, and
    the instant the pilot honours the config an unhashed tolerance becomes a false-hit generator.
    (Codex, M10.2b review round 2: in scope, because M10.2b is creating the cache identity that
    would otherwise preserve the inconsistency indefinitely.)
    """

    def content_key(self) -> str:
        """Everything that can change this pilot's bytes (BUILD_PLAN §1.1).

        Not merely polytope+stream: the **input transform** and the **schedule** change the draws
        too, and a key that omits them lets a pilot run under one ``T`` be reused under another.
        (Codex, M10 review round 3.)

        **An incomplete key is worse than none** — no key means no cache; an incomplete one is a
        false-hit generator, and §1.1's rule is that a false miss only recomputes while a false hit
        corrupts. Until M10.2b nothing had wired this key to a store, so nothing had yet been *able*
        to hit it; that is exactly why M10.2a could find it incomplete and M10.2b must not.
        """
        return content_key(
            model_id=self.model_id,
            stage=self.stage,
            polytope_key=self.polytope_key,
            transform_key=self.transform_key,
            n_chains=self.n_chains,
            n_draws=self.n_draws,
            burn_in=self.burn_in,
            thin=self.thin,
            seed=self.seed,
            refresh_interval=self.refresh_interval,
            feasibility_tol=self.feasibility_tol,
            sampler_impl_version=SAMPLER_IMPL_VERSION,
            calibration_impl_version=CALIBRATION_IMPL_VERSION,
            numpy_version=np.__version__,
        )

    def expected_spawn_keys(self) -> NDArray[np.int64]:
        """``(n_chains, 4)`` — the **semantic spawn coordinates** this recipe names, recomputed.

        The point is that this is **re-derived, never read back**. A pilot stores the spawn keys its
        chains actually consumed; this recomputes what they should have been; `NeutralPilot` refuses
        the artifact if they differ. So M10.2a's defect — `run_chain` keying every stream on a
        hardcoded ``"sample"``, which made the two pilots draw *identical numbers* — is caught by
        construction, at the point of construction, with no statistical test involved.

        ⚠️ **What it does not prove, stated because the first draft of this docstring overclaimed
        it.** `provenance.stream_seed` puts `seed` in the `SeedSequence`'s **entropy**, and only
        ``(model_id, stage, β_index, chain_index)`` in its `spawn_key` — so a regression that
        ignored or hardcoded `config.seed` would pass this guard untouched. This checks the four
        semantic coordinates, **not the whole stream**. `run_chain` does not record the entropy, so
        the guard cannot honestly reach further; `seed` is covered by `content_key` (a wrong seed
        is a different artifact) and by `test_the_premise_that_moving_those_fields_moves_the_draws`,
        which proves reseeding moves the draws. (Codex, round 3: *narrow the guard's claim*.)

        This replaces the stored flux fingerprint proposed in review round 1 and withdrawn in round
        2: a digest of the geometry pilot's *discarded* fluxes would be an unrederivable assertion
        living in trusted metadata, and comparing two digests proves only **non-identity** — the
        weak proxy this milestone rejected. Evidence you recompute is evidence; evidence you store
        and read back is a claim. (Codex, M10.2b review round 2.)
        """
        return np.array(
            [
                stream_seed(
                    model_id=self.model_id,
                    stage=self.stage,
                    seed=self.seed,
                    beta_index=PILOT_BETA_INDEX,
                    chain_index=chain_index,
                ).spawn_key
                for chain_index in range(self.n_chains)
            ],
            dtype=np.int64,
        )

    def to_cache(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "stage": self.stage,
            "polytope_key": self.polytope_key,
            "transform_key": self.transform_key,
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "burn_in": self.burn_in,
            "thin": self.thin,
            "seed": self.seed,
            "refresh_interval": self.refresh_interval,
            "feasibility_tol": self.feasibility_tol,
        }

    @classmethod
    def from_cache(cls, payload: dict[str, Any]) -> PilotRecipe:
        try:
            return cls(
                model_id=str(payload["model_id"]),
                stage=str(payload["stage"]),
                polytope_key=str(payload["polytope_key"]),
                transform_key=str(payload["transform_key"]),
                n_chains=int(payload["n_chains"]),
                n_draws=int(payload["n_draws"]),
                burn_in=int(payload["burn_in"]),
                thin=int(payload["thin"]),
                seed=int(payload["seed"]),
                refresh_interval=int(payload["refresh_interval"]),
                feasibility_tol=float(payload["feasibility_tol"]),
            )
        except KeyError as exc:
            raise CalibrationError(f"cached pilot recipe is missing {exc}") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "pilot_content_key": self.content_key(),
            "pilot_stage": self.stage,
            "pilot_model_id": self.model_id,
            "pilot_polytope_key": self.polytope_key,
            "pilot_transform_key": self.transform_key,
            "pilot_n_chains": self.n_chains,
            "pilot_n_draws": self.n_draws,
            "pilot_burn_in": self.burn_in,
            "pilot_thin": self.thin,
            "pilot_seed": self.seed,
            "pilot_refresh_interval": self.refresh_interval,
            "pilot_feasibility_tol": self.feasibility_tol,
            "calibration_impl_version": CALIBRATION_IMPL_VERSION,
        }


@dataclass(frozen=True)
class NeutralPilot(ABC):
    """A frozen β=0 pilot chain — **objective-independent**, and that is load-bearing.

    The β=0 law does not see ``J``: the target is uniform on the polytope and the kernel draws
    uniformly on each chord. So **one neutral pilot serves every objective sharing this polytope,
    transform and pilot recipe**, which is not a micro-optimisation — M7 puts a base *and* a
    reweighted objective on one polytope, and they must be calibrated against the *same* neutral
    ensemble or their β axes are not comparable with each other, let alone across strains.

    Hence the split: this artifact carries **no objective key**, and the derived `EnergyScale` does.
    (Codex, M10 review round 2.) M10.2 made that structural rather than aspirational — the builders
    have no ``objective`` and no ``optimum_coordinates`` parameter, so objective state cannot reach
    a neutral pilot by a caller forgetting to pass ``None``.

    **Abstract.** A pilot's payload is fixed by its stage, so the concrete classes are the stages:
    `GeometryPilot` holds coordinates because `rounding.reround_transform` reads coordinates;
    `ScalePilot` holds reduced fluxes because `sparse_objective.pilot_energy_scale` reads fluxes.
    Neither holds the other's array, which halves the cached bytes (39.2 MB → 19.6 MB per model,
    measured) — and that matters most exactly where it was designed to: in ``reduced`` storage mode
    a model's whole production output is **11.8 MB**, so store-both pilots would be 3.3× the
    deliverable they calibrate.

    **The payload belongs to the builder, never to the store.** If `run_*_pilot` returned both
    arrays and the cache kept one, a second run would hand back an object missing a field the first
    run had — a crash reachable only on re-run. Hit and miss must be the same object, so the
    projection happens where both paths share it.

    The fluxes are stored raw rather than as ``J``, and that is what makes the objective-
    independence claim *usable*: a second objective on this polytope evaluates its own ``J`` from
    the same neutral ensemble. Nor are they reconstructed from coordinates — M9 measured
    ``c + T·y`` per-row (gemv) against per-block (gemm) at 1.1e-13, so a reconstructing hit would
    differ from a miss.
    """

    recipe: PilotRecipe
    spawn_keys: NDArray[np.int64]
    """``(n_chains, 4)`` — the ``(model_id, stage, β_index, chain_index)`` coordinates each chain's
    RNG stream *actually* consumed, read off `maxent_sampler.ChainDiagnostics`.

    BUILD_PLAN §1.2 has always said to "derive streams from stable semantic coordinates … **and
    store the spawn keys**". The pilot discarded `ChainResult.diagnostics` wholesale, so a cached
    pilot could not say which streams produced it: plan/code drift of the M10.2a shape, found while
    fixing another instance of it. Stored as a **cache-hashed array**, not metadata, because
    `ArtifactCache` hashes arrays and trusts meta.
    """

    STAGE: ClassVar[str] = ""
    """The concrete stage. A `ClassVar`, so it cannot be passed, defaulted, or disagreed with."""

    PAYLOAD: ClassVar[str] = ""
    """The name of the one array this stage keeps — the field, and its `.npy` in the cache."""

    def __post_init__(self) -> None:
        """Normalize the arrays and refuse any pilot that contradicts its own recipe.

        **Normalize, not merely check.** `_frozen` is called *here*, so read-only float64 C-order
        arrays are a class invariant rather than a convention two well-behaved constructors happen
        to share: `run_*_pilot` and `from_bundle` both went through `_frozen` and *public dataclass
        construction did not*, so a caller could hold a `GeometryPilot` whose coordinates a later
        line mutates — and spec §17.4/§18.3 turns entirely on a pilot being unable to move once
        production starts. Passing through `_frozen` on every path makes the claim true of the
        **class**. (Codex, M10.2b review round 3.)
        """
        if not self.STAGE or not self.PAYLOAD:
            raise CalibrationError(
                "NeutralPilot is abstract — a pilot's payload is fixed by its stage, so construct "
                "a GeometryPilot or a ScalePilot"
            )
        if self.recipe.stage != self.STAGE:
            raise CalibrationError(
                f"a {type(self).__name__} carries {self.PAYLOAD} but its recipe names stage "
                f"{self.recipe.stage!r}, so it would be keyed as a {self.recipe.stage!r} artifact "
                f"while holding {self.STAGE!r} bytes"
            )

        object.__setattr__(self, "spawn_keys", _frozen(self.spawn_keys, dtype=np.int64))
        object.__setattr__(self, self.PAYLOAD, _frozen(self.payload, dtype=VALUE_DTYPE))

        # The recipe says how many chains drew how many samples; the payload must be that array.
        # `from_bundle` validated neither, so a bundle of the right dtype and the wrong shape
        # reconstructed happily and failed later, somewhere else. (Codex, round 3.)
        shape = self.payload.shape
        if len(shape) != 3 or shape[:2] != (self.recipe.n_chains, self.recipe.n_draws):
            raise CalibrationError(
                f"this pilot's {self.PAYLOAD} are {shape}, but its recipe names "
                f"{self.recipe.n_chains} chains × {self.recipe.n_draws} draws"
            )

        expected = self.recipe.expected_spawn_keys()
        actual = np.asarray(self.spawn_keys, dtype=np.int64)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise CalibrationError(
                f"this pilot's chains drew from streams its recipe does not name: expected spawn "
                f"keys {expected.tolist()}, got {actual.tolist()}. Either the stage did not reach "
                "`run_chain` (M10.2a: it was hardcoded to 'sample', so both pilots drew identical "
                "numbers) or these bytes are not this recipe's pilot."
            )

    @property
    def payload(self) -> NDArray[np.float64]:
        """The one array this pilot kept, whichever it is."""
        array: NDArray[np.float64] = getattr(self, self.PAYLOAD)
        return array

    @classmethod
    @abstractmethod
    def project(
        cls: type[P], recipe: PilotRecipe, result: SamplerResult, spawn_keys: NDArray[np.int64]
    ) -> P:
        """Keep this stage's array from a finished run and drop the other.

        Each class owns its own projection so the stage→payload map exists in exactly one place per
        stage — the alternative, a dispatch table beside the classes, is a second writer of the
        mapping, which is the shape of defect M10.2a spent a milestone removing.

        **Projection is strictly post-run**: `_run_pilot_chains` walks the identical chains whatever
        the payload, so what a pilot keeps provably cannot reach `maxent_sampler.run_chains` and
        change what it draws.

        `abstractmethod`, not a `NotImplementedError` body: the sentinel version failed only *after*
        `_build_pilot` had walked every chain, so an incomplete subclass burned the full 19.3 s
        before reporting that it could not keep the result. `ABC` refuses it at instantiation
        instead. (Codex, M10.2b review round 3: runtime-abstract is not type-abstract.)
        """

    def to_bundle(self) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Split into ``(arrays, meta)`` for `cache.ArtifactCache`."""
        return (
            {
                self.PAYLOAD: np.ascontiguousarray(self.payload),
                "spawn_keys": np.ascontiguousarray(self.spawn_keys),
            },
            {
                "content_key": self.recipe.content_key(),
                "recipe": self.recipe.to_cache(),
                "pilot_kind": self.STAGE,
            },
        )

    @classmethod
    def from_bundle(
        cls: type[P], arrays: dict[str, NDArray[Any]], meta: dict[str, Any], recipe: PilotRecipe
    ) -> P:
        """Rebuild a pilot cached by `to_bundle` — **indistinguishable from a fresh one**.

        That is the whole contract, and it is the one thing a cache can get wrong in a way no test
        of the numbers would notice: `_build_pilot` freezes its arrays read-only, so this does too.
        A reconstruction that differed in dtype, contiguity or writeability would make the second
        run of a pipeline behave unlike the first.

        Three guards, each refusing a different lie. ``pilot_kind`` is checked **against the class
        the caller asked for and never used to choose it** — a cache that picks the type from the
        bytes it loaded lets the bytes decide what they are. The recipe must be the one requested,
        and the stored key must **re-derive** to the requested key: the store is content-addressed,
        so an artifact that does not reproduce its own key is not the artifact the key names,
        whatever its arrays look like (the `affine_geometry.ReducedGeometry.from_bundle` pattern).
        `__post_init__` then re-derives the spawn keys, so a pilot that ran on streams this recipe
        does not name cannot be loaded either.
        """
        kind = str(meta.get("pilot_kind", ""))
        if kind != cls.STAGE:
            raise CalibrationError(
                f"cached artifact is a {kind or '<unknown>'!r} pilot, not the {cls.STAGE!r} pilot "
                "it was looked up as"
            )
        stored = PilotRecipe.from_cache(meta["recipe"])
        if stored != recipe:
            raise CalibrationError(
                f"cached pilot was built from a different recipe than the one requested "
                f"({stored.content_key()[:16]}… vs {recipe.content_key()[:16]}…)"
            )
        if str(meta.get("content_key", "")) != recipe.content_key():
            raise CalibrationError(
                "cached pilot does not reproduce the key it is stored under; the store is "
                "content-addressed, so these are not that key's bytes"
            )
        payload = _frozen(arrays[cls.PAYLOAD], dtype=VALUE_DTYPE)
        spawn_keys = _frozen(arrays["spawn_keys"], dtype=np.int64)
        return cls(recipe=recipe, spawn_keys=spawn_keys, **{cls.PAYLOAD: payload})

    def manifest(self) -> dict[str, Any]:
        return {**self.recipe.manifest(), "pilot_spawn_keys": self.spawn_keys.tolist()}


P = TypeVar("P", bound=NeutralPilot)


@dataclass(frozen=True)
class GeometryPilot(NeutralPilot):
    """The pilot whose covariance re-rounds ``T₀`` into ``T₁`` (spec §17.4)."""

    coordinates: NDArray[np.float64]
    """``(n_chains, n_draws, d)`` — the chains' rounded ``y`` under ``recipe.transform_key``."""

    STAGE = GEOMETRY_PILOT_STAGE
    PAYLOAD = "coordinates"

    @classmethod
    def project(
        cls, recipe: PilotRecipe, result: SamplerResult, spawn_keys: NDArray[np.int64]
    ) -> GeometryPilot:
        return cls(
            recipe=recipe,
            spawn_keys=spawn_keys,
            coordinates=_frozen(
                np.stack([chain.coordinates for chain in result.chains]), dtype=VALUE_DTYPE
            ),
        )

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[2])

    def pooled_coordinates(self) -> NDArray[np.float64]:
        """``(n_chains·n_draws, d)`` — the draws as one point set, for a covariance."""
        return self.coordinates.reshape(-1, self.dimension)


@dataclass(frozen=True)
class ScalePilot(NeutralPilot):
    """The pilot whose ``J`` spread sets ``s_J = σ̂₀`` (spec §22.2, BUILD_PLAN §1.6.6)."""

    fluxes: NDArray[np.float64]
    """``(n_chains, n_draws, n_free)`` — **reduced fluxes, not** ``J``. Storing the fluxes is what
    lets a second objective on this polytope reuse this neutral ensemble (see `NeutralPilot`)."""

    STAGE = SCALE_PILOT_STAGE
    PAYLOAD = "fluxes"

    @classmethod
    def project(
        cls, recipe: PilotRecipe, result: SamplerResult, spawn_keys: NDArray[np.int64]
    ) -> ScalePilot:
        return cls(
            recipe=recipe,
            spawn_keys=spawn_keys,
            fluxes=_frozen(
                np.stack([chain.fluxes for chain in result.chains]), dtype=VALUE_DTYPE
            ),
        )


@dataclass(frozen=True)
class CalibrationResult:
    """What the DAG froze, plus the evidence for it."""

    transform: RoundedTransform
    """``T₁`` — re-rounded from the geometry pilot, or ``T₀`` unchanged if re-rounding was off."""

    energy_scale: EnergyScale
    """``s_J``, in whichever mode the config asked for."""

    geometry_pilot: GeometryPilot | None
    scale_pilot: ScalePilot | None

    bootstrap_condition_number: float
    """``cond(C_q)`` of ``T₀`` — the support-vertex rounding, kept so the improvement is *shown*
    rather than asserted."""

    certificate: ReachabilityCertificate
    """M9's reachable mass-balance proof for **`transform`** — always, never ``None``.

    It was ``None`` when re-rounding was off, on the reasoning that ``T₀``'s builder already held
    its certificate. Codex (M10.2 review round 3) was right that this is a hole: it makes the
    *caller* remember which of two objects to consult, and a reader of the run manifest then gets
    ``T₀``'s certificate reported as the run's while production samples ``T₁``. So this field is
    the proof for the transform this result actually ships, whichever branch produced it — one
    object, one question, no bookkeeping.
    """

    optimum_coordinates: NDArray[np.float64] | None
    """The production start hint, re-expressed in `transform`'s frame (BUILD_PLAN §1.6.5).

    A start hint and nothing more: it enters only a production chain's initial state, never the
    kernel, the objective or ``s_J``. It is **not** given to the pilots — that is the M10.2 defect
    (`NeutralPilot`). But it is expressed in ``T₀``'s coordinates, so once ``T₁`` exists a caller
    handing the old vector to a chain stepping in the new frame would aim it at an arbitrary
    interior point. `calibrate` owns both frames, so it does the change of coordinates here rather
    than leaving a trap for the caller.
    """

    def manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bootstrap_condition_number": self.bootstrap_condition_number,
            "final_condition_number": self.transform.diagnostics.condition_number,
            "rerounded": self.geometry_pilot is not None,
            **self.energy_scale.manifest(),
        }
        if self.geometry_pilot is not None:
            out["geometry_pilot"] = self.geometry_pilot.manifest()
        if self.scale_pilot is not None:
            out["scale_pilot"] = self.scale_pilot.manifest()
        return out

    def certificate_report(self) -> dict[str, Any]:
        """The reachable mass-balance certificate **of the transform this run samples**.

        `as_dict`, not `to_cache`: this is the human-facing manifest, where the renamed keys and the
        derived ``reachable_is_certified`` / ``reachable_margin`` are the point. The cache stores
        the fields and re-derives the verdict; a reader wants the verdict.
        """
        return self.certificate.as_dict()


def _frozen(array: NDArray[Any], *, dtype: Any) -> NDArray[Any]:
    """One contiguous, correctly-typed, read-only array — the pilots' only array constructor.

    Both the builder and `NeutralPilot.from_bundle` go through it, because "a cache hit is
    indistinguishable from a miss" is a claim about dtype, contiguity and writeability as much as
    about the numbers, and two constructors are how those drift apart.
    """
    out = np.ascontiguousarray(array, dtype=dtype)
    out.flags.writeable = False
    return out


def require_pilot_inputs(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    certificate: ReachabilityCertificate,
) -> None:
    """Every precondition a pilot's inputs must meet — **the one place both paths ask.**

    🔴 **The two checks below were split, and the split was a hit/miss asymmetry.** M10.2b hoisted
    `require_certified_transform` out of the ``compute()`` closure (M10.2a's defect) and left the
    polytope relation behind in `_run_pilot_chains`, i.e. **on the miss path only**. The recipe
    looks like it mitigates that — it hashes both `polytope_key` and `transform_key` — and it does
    not: `rounding.RoundedTransform.content_key` hashes ``geometry_key``, ``transform``, ``center``
    and ``ridge``, **not the transform's own `polytope_key`**. So a transform whose `polytope_key`
    is a lie keys *identically*, the certificate gate passes (it compares the certificate's keys to
    the reduced polytope and to the transform's content key, both unchanged), and the pilot is
    served. Measured: with an empty cache the call **refuses**; with a warm one it **returns a
    pilot**.
    (Codex, M10.2b review round 3, executed and confirmed — "the hole behind each repair", and this
    one was in the repair itself.)

    So the preconditions live in one function that both `_run_pilot_chains` and `_load_or_run_pilot`
    call. A precondition a cache lookup can skip is not a precondition, and two call sites checking
    *different* subsets is how one of them ends up checking none.
    """
    if transform.polytope_key != reduced.content_key():
        raise CalibrationError(
            "the transform was not built from this polytope; the pilot would step along one "
            "model's directions and be bounds-checked against another's"
        )
    require_certified_transform(certificate, transform, reduced)


def pilot_recipe(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    stage: str,
    feasibility_tol: float,
) -> PilotRecipe:
    """The recipe for a pilot that has not run yet — **the cache's lookup key**.

    The one writer of a `PilotRecipe`. `_build_pilot` and `_load_or_run_pilot` both come here, so
    the key a pilot is stored under and the key it is looked up by cannot drift apart.
    """
    return PilotRecipe(
        model_id=model_id,
        stage=stage,
        polytope_key=reduced.content_key(),
        transform_key=transform.content_key(),
        n_chains=int(config.n_chains),
        n_draws=int(config.n_samples),
        burn_in=int(config.burn_in),
        thin=int(config.thin),
        seed=int(config.seed),
        refresh_interval=int(config.refresh_interval),
        feasibility_tol=float(feasibility_tol),
    )


def _run_pilot_chains(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    stage: str,
    certificate: ReachabilityCertificate,
    feasibility_tol: float,
) -> tuple[PilotRecipe, SamplerResult, NDArray[np.int64]]:
    """Walk one pilot's chains at β=0 and hand back the draws with the streams that made them.

    **No ``objective`` and no ``optimum_coordinates``**, and neither is an omission — together they
    are what makes `NeutralPilot`'s objective-independence a fact about the code rather than a
    sentence in a docstring. At β=0 the target is flat and ``J`` provably never enters the draw, so
    an objective here could only be decoration a later reader would mistake for a dependency. The
    start hint is subtler and it is what M10.2 had to fix: it carries the objective's LP optimum and
    it changes the draws. Absent parameters cannot be passed by mistake; a defaulted ``None`` can.
    """
    require_pilot_inputs(transform, reduced, certificate)

    result = run_chains(
        transform, reduced,
        config=config,
        model_id=model_id,
        beta=0.0,
        beta_index=PILOT_BETA_INDEX,
        objective=None,
        stage=stage,
        feasibility_tol=feasibility_tol,
    )
    recipe = pilot_recipe(
        transform, reduced,
        config=config, model_id=model_id, stage=stage, feasibility_tol=feasibility_tol,
    )
    spawn_keys = np.array(
        [chain.diagnostics.spawn_key for chain in result.chains], dtype=np.int64
    )
    return recipe, result, _frozen(spawn_keys, dtype=np.int64)


def _build_pilot(
    pilot_type: type[P],
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    certificate: ReachabilityCertificate,
    feasibility_tol: float,
) -> P:
    """Run ``pilot_type``'s chains and project them — the **one** path from a config to a pilot.

    The stage comes off the class, never off a parameter, so the artifact's payload and the key's
    stage are decided by the same fact.
    """
    recipe, result, spawn_keys = _run_pilot_chains(
        transform, reduced,
        config=config, model_id=model_id, stage=pilot_type.STAGE,
        certificate=certificate, feasibility_tol=feasibility_tol,
    )
    return pilot_type.project(recipe, result, spawn_keys)


def run_geometry_pilot(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    certificate: ReachabilityCertificate,
    feasibility_tol: float = 1e-9,
) -> GeometryPilot:
    """Run the β=0 pilot that re-rounds ``T₀``, and freeze its coordinates.

    There is **no ``stage`` parameter**: the stage is `GeometryPilot.STAGE`. M10.2a's public
    `run_neutral_pilot` took one, which is what let a class and its key disagree; this package's
    rule is that an absent parameter cannot be passed by mistake while a defaulted one can. It is
    also why the payload needs no parameter — a geometry pilot keeps coordinates because that is
    what a geometry pilot *is*.

    ``certificate`` must prove **this** ``transform``. A pilot is not a lesser chain: every artifact
    the DAG freezes descends from its draws, so it earns the same gate production gets.
    """
    return _build_pilot(
        GeometryPilot, transform, reduced,
        config=config, model_id=model_id, certificate=certificate,
        feasibility_tol=feasibility_tol,
    )


def run_scale_pilot(
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    certificate: ReachabilityCertificate,
    feasibility_tol: float = 1e-9,
) -> ScalePilot:
    """Run the β=0 pilot that sets ``s_J = σ̂₀``, and freeze its **reduced fluxes**.

    See `run_geometry_pilot` on the absent ``stage`` parameter, and `NeutralPilot` on why the
    fluxes rather than ``J`` are what is kept.
    """
    return _build_pilot(
        ScalePilot, transform, reduced,
        config=config, model_id=model_id, certificate=certificate,
        feasibility_tol=feasibility_tol,
    )


def _load_or_run_pilot(
    pilot_type: type[P],
    transform: RoundedTransform,
    reduced: ReducedPolytope,
    *,
    config: SamplerConfig,
    model_id: str,
    certificate: ReachabilityCertificate,
    feasibility_tol: float,
    cache: ArtifactCache | None,
) -> P:
    """Return this recipe's pilot, from `PILOT_CACHE_LAYER` if it is there — **gated either way**.

    🔴 **The gate is here, before the dispatch, and that placement is the point.** Wiring it as
    ``get_or_compute(layer, key, lambda: run_pilot(...))`` would leave `require_pilot_inputs`
    inside the ``compute()`` closure — which runs **only on a miss**. That is M10.2a's defect
    verbatim: M9's mass-balance gate lived in the ``compute()`` closure of
    `batch._load_or_build_geometry`, so warming the cache and then sampling walked straight past it.
    A cache lookup must not change a precondition; hit and miss must have identical acceptance
    semantics (Codex, M10.2b review round 2, agreeing this was the diff's highest-risk line).

    It calls `require_pilot_inputs` — **all** the preconditions — rather than the certificate gate
    alone. Hoisting one of two checks is how the first version of this function shipped an
    asymmetry while claiming to have closed one; see that function.

    The builders keep their own gate for direct callers, so this is defence in depth rather than a
    relocation — the M10.2a lesson that gating an orchestrator does not gate its primitive.

    The certificate is deliberately **not** stored in the pilot artifact: it certifies the
    ``(polytope, transform, policy)`` edge, not the pilot. ``T₀``'s belongs to L3 and ``T₁``'s is
    recomputed by `calibrate` in 0.5 s before the scale pilot is ever asked for.
    """
    require_pilot_inputs(transform, reduced, certificate)

    def build() -> P:
        return _build_pilot(
            pilot_type, transform, reduced,
            config=config, model_id=model_id,
            certificate=certificate, feasibility_tol=feasibility_tol,
        )

    if cache is None:
        _log.info(
            "%s: running %d chains × (%d burn-in + %d) sweeps at β=0 (no cache)",
            pilot_type.STAGE, config.n_chains, config.burn_in, config.n_samples,
        )
        return build()

    recipe = pilot_recipe(
        transform, reduced,
        config=config, model_id=model_id, stage=pilot_type.STAGE,
        feasibility_tol=feasibility_tol,
    )

    # Reported after the fact, from whether `build` actually ran — never from an `is_cached` probe
    # before the dispatch, which another process can invalidate between the check and the claim.
    # A log that announces a 19.3 s pilot it then loads in 40 ms is this package's signature bug in
    # miniature: a manifest describing work that did not happen. (Found by reading the M10.2b CLI
    # run's own output, where the pre-dispatch line duly announced a pilot on a cache hit.)
    ran = False

    def compute() -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        nonlocal ran
        ran = True
        _log.info(
            "%s: running %d chains × (%d burn-in + %d) sweeps at β=0",
            pilot_type.STAGE, config.n_chains, config.burn_in, config.n_samples,
        )
        return build().to_bundle()

    artifact = cache.get_or_compute(PILOT_CACHE_LAYER, recipe.content_key(), compute)
    if not ran:
        _log.info(
            "%s: loaded from cache (%s…)", pilot_type.STAGE, recipe.content_key()[:16]
        )
    return pilot_type.from_bundle(artifact.arrays, artifact.meta, recipe)


def calibrate(
    geometry: ReducedGeometry,
    reduced: ReducedPolytope,
    bootstrap: RoundedTransform,
    objective: ReducedObjective,
    *,
    model_id: str,
    optimum: LPOptimum,
    sampler: SamplerConfig,
    bootstrap_certificate: ReachabilityCertificate,
    geometry_config: GeometryConfig | None = None,
    optimum_coordinates: NDArray[np.float64] | None = None,
    cache: ArtifactCache | None = None,
) -> CalibrationResult:
    """Run the pilot DAG and freeze ``T₁`` and ``s_J`` before any production chain exists.

    Both stages are opt-in via `SamplerConfig`: ``pilot_reround`` re-rounds the transform, and
    ``energy_scale = "pilot_sd"`` calibrates β against the neutral ensemble's own spread. They are
    independent switches — either alone is coherent — but they share the machinery, so this runs
    whichever the config asked for and no pilot the config did not.

    Returns ``T₀`` and the support-vertex scale unchanged when neither is enabled, so a caller can
    route every run through here without a branch and without changing v1's numbers.

    ``optimum_coordinates`` is **for production only** and never reaches a pilot: it comes back on
    `CalibrationResult.optimum_coordinates`, mapped into the final transform's frame. `calibrate`
    is the only place that holds both frames, so it is the only place that can do that mapping —
    and `run_neutral_pilot` has no parameter to receive it through (`NeutralPilot`).

    ``bootstrap_certificate`` is **required, and it is an argument rather than a computation** on
    purpose. A pilot is a chain: it steps in ``T₀``'s frame long before production exists, so an
    uncertified ``T₀`` walks off the steady-state manifold *here*, and every artifact this function
    freezes descends from those draws. Certifying it internally would re-run 334 LPs that
    `batch.build_l3_bundle` has already run for the same matrix; demanding the proof instead costs
    nothing, cannot be forgotten, and makes an uncertified transform unable to enter the pilot DAG
    at all. (Codex, M10.2 review round 3: `certificate=None` was safe behind `prepare_model` and a
    hole in the public API.)

    ``cache`` is where the DAG's real cost lives (M10.2b). The pilots are **19.3 s** of serial
    parent work per model, measured, against `build_geometry`'s 1.17 s — so an uncached pilot means
    a restart re-runs 19.3 s before it can resume a single chain, and M9's Amdahl ceiling of 24.9×
    falls to ~3.65×. Passing ``None`` recomputes them, which is the honest default for a caller with
    nowhere to put them: a false miss only recomputes.
    """
    geometry_config = geometry_config or GeometryConfig()

    if bootstrap.polytope_key != reduced.content_key():
        raise CalibrationError("the bootstrap transform was not built from this polytope")
    require_certified_transform(bootstrap_certificate, bootstrap, reduced)
    if objective.polytope_key != reduced.content_key():
        raise CalibrationError(
            "the objective was not lowered from this polytope — the M6 'two artifacts that never "
            "met' join (BUILD_PLAN §1.6.3)"
        )

    wants_reround = sampler.pilot_reround
    wants_pilot_scale = sampler.energy_scale == "pilot_sd"
    bootstrap_condition = bootstrap.diagnostics.condition_number

    transform = bootstrap
    geometry_pilot: GeometryPilot | None = None
    certificate = bootstrap_certificate
    if wants_reround:
        pilot_config = _pilot_schedule(sampler)
        # `_load_or_run_pilot` logs what it *did*; announcing the pilot here would announce one on a
        # cache hit too.
        geometry_pilot = _load_or_run_pilot(
            GeometryPilot,
            bootstrap, reduced,
            config=pilot_config,
            model_id=model_id,
            certificate=bootstrap_certificate,  # this pilot steps in T₀'s frame
            feasibility_tol=geometry_config.feasibility_tol,
            cache=cache,
        )
        transform = reround_transform(
            geometry, reduced, bootstrap,
            pilot_coordinates=geometry_pilot.pooled_coordinates(),
            config=geometry_config,
        )
        # M9's gate, applied to T₁ — **here, before the scale pilot**, not before production.
        # The scale pilot is itself a chain stepping in T₁'s frame: an uncertified T₁ would let it
        # walk off the steady-state manifold, and σ̂₀ would then be computed from off-manifold
        # fluxes — the β axis calibrated against states the model forbids.
        #
        # The exact-arithmetic theorem does *not* transfer T₀'s certificate. range(T₁) = range(T₀)
        # exactly (§1.6.1), so the reachable *flux set* is identical and the true worst residual is
        # the same number. But the certificate is a **numerical** bound: it recomputes E = S·T₁ and
        # Ω from a fresh T₁⁺, and fl(B·L₀) and fl(B·L₁) need not share a floating-point column
        # space just because their exact formulas do. 334 LPs / ~0.5 s against a ~19 s pilot.
        certificate = certify_reachable_mass_balance(transform, reduced)
        require_certified_transform(certificate, transform, reduced)
        _log.info(
            "re-rounded: cond(C_q) %.3g → %.3g (%.2f×), step_scale_ratio %.3g → %.3g; "
            "T₁ reachable ‖Sv−b‖ %.3g (certified, %.0f× inside contract)",
            bootstrap_condition,
            transform.diagnostics.condition_number,
            bootstrap_condition / transform.diagnostics.condition_number,
            bootstrap.diagnostics.step_scale_ratio,
            transform.diagnostics.step_scale_ratio,
            certificate.worst_absolute,
            certificate.margin,
        )

    scale_pilot: ScalePilot | None = None
    if wants_pilot_scale:
        pilot_config = _pilot_schedule(sampler)
        scale_pilot = _load_or_run_pilot(
            ScalePilot,
            transform, reduced,
            config=pilot_config,
            model_id=model_id,
            # `certificate` tracks `transform`: T₁'s proof when re-rounding ran, T₀'s otherwise.
            # This is the pairing the ordering exists for — the scale pilot steps in *this* frame.
            certificate=certificate,
            feasibility_tol=geometry_config.feasibility_tol,
            cache=cache,
        )
        energy_scale = pilot_energy_scale(
            objective, scale_pilot.fluxes,
            optimum=optimum,
            pilot_polytope_key=scale_pilot.recipe.polytope_key,
        )
        assert energy_scale.pilot is not None
        _log.info(
            "s_J = σ̂₀ = %.4g (±%.1f%%), E₀[J] = %.4g, Δ₀ = %.4g, G = %.2f, R̂(J) = %.3f",
            energy_scale.value,
            100.0 * energy_scale.pilot.relative_se,
            energy_scale.pilot.mean_j, energy_scale.pilot.gap,
            energy_scale.pilot.headroom,
            energy_scale.pilot.r_hat_j,
        )
    else:
        from gsmm_compiler.sparse_objective import choose_energy_scale

        energy_scale = choose_energy_scale(
            objective, geometry.support_points,
            optimum=optimum,
            warmup_polytope_key=reduced.content_key(),
            mode=sampler.energy_scale,
            quantile=sampler.energy_scale_quantile,
            fallback=sampler.energy_scale_fallback,
        )

    return CalibrationResult(
        transform=transform,
        energy_scale=energy_scale,
        geometry_pilot=geometry_pilot,
        scale_pilot=scale_pilot,
        bootstrap_condition_number=bootstrap_condition,
        certificate=certificate,
        optimum_coordinates=_recoordinate(bootstrap, transform, optimum_coordinates),
    )


def _pilot_schedule(sampler: SamplerConfig) -> SamplerConfig:
    """The pilot's own schedule — β=0 only, and its own lengths.

    A pilot is not the production run and must not inherit its ladder: `run_chains` is called at
    β=0, so a `betas` tuple here would be silently ignored, which is worse than being absent.
    """
    from dataclasses import replace

    return replace(
        sampler,
        betas=(0.0,),
        n_chains=sampler.pilot_chains,
        burn_in=sampler.pilot_burn_in,
        n_samples=sampler.pilot_samples,
    )


def _recoordinate(
    bootstrap: RoundedTransform,
    transform: RoundedTransform,
    optimum_coordinates: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Re-express a start hint in the new transform's frame, if the transform changed.

    ``optimum_coordinates`` is a *start hint* and nothing else (BUILD_PLAN §1.6.5) — it enters only
    the initial state, never the kernel, the objective or ``s_J``, so a wrong one cannot change the
    invariant target, only seed a poorer start. But it is expressed in ``T₀``'s coordinates, and
    handing those to a chain stepping in ``T₁``'s frame would point it at an arbitrary interior
    point. Cheaper and clearer to lift through the flux space, which both frames share.
    """
    if optimum_coordinates is None or transform is bootstrap:
        return optimum_coordinates
    flux = bootstrap.to_flux(np.asarray(optimum_coordinates, dtype=VALUE_DTYPE))
    return np.ascontiguousarray(transform.to_coordinates(flux), dtype=VALUE_DTYPE)
