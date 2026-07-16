"""Parsing, validation, and freezing into the canonical (L0) IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gsmm_compiler.model_input import (
    CanonicalModel,
    ModelValidationError,
    build_canonical_model,
    load_canonical_model,
    load_model,
)


def _toy_dict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(tmp_path: Path, model: dict[str, Any]) -> Path:
    destination = tmp_path / "model.json"
    destination.write_text(json.dumps(model))
    return destination


class TestLoading:
    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "absent.json")

    def test_rejects_an_unknown_format(self, tmp_path: Path) -> None:
        bogus = tmp_path / "model.csv"
        bogus.write_text("nope")
        with pytest.raises(ValueError, match="unsupported model format"):
            load_model(bogus)


class TestOrderPreservation:
    """Reaction j is column j of S, for the whole pipeline. Nothing may re-derive this order."""

    def test_reaction_and_metabolite_order_match_the_file(
        self, toy_canonical: CanonicalModel, toy_path: Path
    ) -> None:
        source = _toy_dict(toy_path)

        assert list(toy_canonical.polytope.reaction_ids) == [r["id"] for r in source["reactions"]]
        assert list(toy_canonical.polytope.metabolite_ids) == [
            m["id"] for m in source["metabolites"]
        ]

    def test_each_column_holds_that_reaction_s_own_stoichiometry(
        self, toy_canonical: CanonicalModel, toy_path: Path
    ) -> None:
        polytope = toy_canonical.polytope
        dense = polytope.stoichiometry.to_dense()

        for reaction in _toy_dict(toy_path)["reactions"]:
            column = polytope.reaction_ids.index(reaction["id"])
            for metabolite_id, coefficient in reaction["metabolites"].items():
                row = polytope.metabolite_ids.index(metabolite_id)
                assert dense[row, column] == coefficient

    def test_bounds_line_up_with_the_reaction_order(
        self, toy_canonical: CanonicalModel, toy_path: Path
    ) -> None:
        polytope = toy_canonical.polytope
        for j, reaction in enumerate(_toy_dict(toy_path)["reactions"]):
            assert polytope.lower_bounds[j] == reaction["lower_bound"]
            assert polytope.upper_bounds[j] == reaction["upper_bound"]


class TestBiomassResolution:
    def test_falls_back_to_the_objective(self, toy_canonical: CanonicalModel) -> None:
        assert toy_canonical.polytope.biomass_id == "BIO"

    def test_explicit_id_overrides_the_objective(self, toy_path: Path) -> None:
        canonical = load_canonical_model(toy_path, biomass_id="R1")
        assert canonical.polytope.biomass_id == "R1"

    def test_unknown_biomass_id_is_rejected(self, toy_path: Path) -> None:
        with pytest.raises(ModelValidationError, match="not in the model"):
            load_canonical_model(toy_path, biomass_id="NOPE")

    def test_absent_objective_is_rejected(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        for reaction in model["reactions"]:
            reaction["objective_coefficient"] = 0.0

        with pytest.raises(ModelValidationError, match="no biomass reaction"):
            load_canonical_model(_write(tmp_path, model))

    def test_an_objective_spanning_several_reactions_is_rejected(
        self, tmp_path: Path, toy_path: Path
    ) -> None:
        """ "Biomass once" — an ambiguous objective must be resolved by the user, not guessed."""
        model = _toy_dict(toy_path)
        model["reactions"][0]["objective_coefficient"] = 1.0  # now EX_A and BIO both score

        with pytest.raises(ModelValidationError, match="objective spans 2 reactions"):
            load_canonical_model(_write(tmp_path, model))

    def test_an_ambiguous_objective_can_be_disambiguated_by_id(
        self, tmp_path: Path, toy_path: Path
    ) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][0]["objective_coefficient"] = 1.0

        canonical = load_canonical_model(_write(tmp_path, model), biomass_id="BIO")
        assert canonical.polytope.biomass_id == "BIO"


class TestInfiniteBounds:
    """The one malformation cobra is happy to hand us.

    Unbounded reactions are legal, routine COBRA — but an unbounded flux polytope has infinite
    volume and nothing to sample, so we must reject them ourselves and say which reactions are at
    fault. (NaN, inverted, and duplicate-ID models are caught by cobra first; see below.)
    """

    def test_infinite_upper_bound_is_rejected_naming_the_reaction(
        self, tmp_path: Path, toy_path: Path
    ) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["upper_bound"] = float("inf")

        with pytest.raises(ModelValidationError, match="NaN or infinite bounds") as error:
            load_canonical_model(_write(tmp_path, model))
        assert "R1" in str(error.value)

    def test_the_message_says_how_to_fix_it(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["lower_bound"] = float("-inf")

        with pytest.raises(ModelValidationError, match="replace infinities with an explicit"):
            load_canonical_model(_write(tmp_path, model))


class TestCobraRejectsFirst:
    """Malformations cobra's own parser refuses. Pinned so we notice if that ever stops being true.

    ``ModelValidationError`` subclasses ``ValueError``, so these assertions hold whichever layer
    does the rejecting — what matters is that no such model is ever *silently* loaded.
    """

    def test_nan_bounds(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["lower_bound"] = float("nan")

        with pytest.raises((ValueError, Exception)):
            load_canonical_model(_write(tmp_path, model))

    def test_inverted_bounds(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["lower_bound"] = 5.0
        model["reactions"][1]["upper_bound"] = 1.0

        with pytest.raises(ValueError, match="lower bound must be less than"):
            load_canonical_model(_write(tmp_path, model))

    def test_duplicate_reaction_ids(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["id"] = model["reactions"][2]["id"]

        with pytest.raises(ValueError, match="non-unique"):
            load_canonical_model(_write(tmp_path, model))


class TestOurGuardsOnProgrammaticModels:
    """`build_canonical_model` takes any cobra ``Model``, not only one cobra just parsed.

    A model assembled or mutated in memory can carry bounds cobra's setters would have refused, so
    the guards have to hold there too — this is where they are actually reachable.
    """

    def _minimal_model(self) -> Any:
        from cobra import Metabolite, Model, Reaction

        model = Model("programmatic")
        metabolite = Metabolite("A", compartment="c")
        reaction = Reaction("R1")
        reaction.add_metabolites({metabolite: -1.0})
        reaction.bounds = (0.0, 10.0)
        model.add_reactions([reaction])
        model.objective = reaction  # only assignable once the reaction has a model
        return model

    def test_nan_bound_poked_past_cobra_s_setter_is_still_caught(self, tmp_path: Path) -> None:
        model = self._minimal_model()
        model.reactions[0]._lower_bound = float("nan")  # cobra's setter would have refused this

        with pytest.raises(ModelValidationError, match="NaN or infinite bounds"):
            build_canonical_model(model, tmp_path / "source.json")

    def test_inverted_bounds_poked_past_cobra_s_setter_are_still_caught(
        self, tmp_path: Path
    ) -> None:
        model = self._minimal_model()
        model.reactions[0]._lower_bound = 5.0
        model.reactions[0]._upper_bound = 1.0

        with pytest.raises(ModelValidationError, match="empty polytope"):
            build_canonical_model(model, tmp_path / "source.json")

    def test_a_model_with_no_reactions_is_rejected(self, tmp_path: Path) -> None:
        from cobra import Model

        with pytest.raises(ModelValidationError, match="no reactions"):
            build_canonical_model(Model("empty"), tmp_path / "source.json")


class TestCanonicalModel:
    def test_exchange_mask_is_carried_so_the_core_never_needs_cobra(
        self, example_canonical: CanonicalModel
    ) -> None:
        assert example_canonical.exchange_mask.sum() == 63
        assert example_canonical.exchange_mask.dtype == bool

    def test_l0_key_is_deterministic(self, toy_path: Path) -> None:
        assert load_canonical_model(toy_path).l0_key == load_canonical_model(toy_path).l0_key

    def test_l0_key_changes_when_the_file_changes(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["upper_bound"] = 999.0

        assert (
            load_canonical_model(toy_path).l0_key
            != load_canonical_model(_write(tmp_path, model)).l0_key
        )

    def test_l0_key_changes_when_the_biomass_choice_changes(self, toy_path: Path) -> None:
        """Same bytes, different question — so a different key, and a cache miss."""
        assert (
            load_canonical_model(toy_path, biomass_id="BIO").l0_key
            != load_canonical_model(toy_path, biomass_id="R1").l0_key
        )

    def test_l1_key_ignores_the_source_path(self, tmp_path: Path, toy_path: Path) -> None:
        """L1 keys the *polytope*: copying a file to a new path must not invalidate the geometry."""
        copied = _write(tmp_path, _toy_dict(toy_path))
        assert load_canonical_model(toy_path).l1_key == load_canonical_model(copied).l1_key

    def test_l1_key_changes_when_a_bound_changes(self, tmp_path: Path, toy_path: Path) -> None:
        model = _toy_dict(toy_path)
        model["reactions"][1]["upper_bound"] = 999.0
        assert (
            load_canonical_model(toy_path).l1_key
            != load_canonical_model(_write(tmp_path, model)).l1_key
        )


def _valid_model(*, upper_bound: float = 10.0, model_id: str = "m") -> Any:
    """A minimal mass-balanced cobra model, assembled in memory (no file behind it)."""
    from cobra import Metabolite, Model, Reaction

    model = Model(model_id)
    a, b = Metabolite("A", compartment="c"), Metabolite("B", compartment="c")
    r_in = Reaction("R_in")
    r_in.add_metabolites({a: 1.0})
    r_in.bounds = (0.0, upper_bound)
    r_out = Reaction("R_out")
    r_out.add_metabolites({a: -1.0, b: 1.0})
    r_out.bounds = (0.0, 10.0)
    r_sink = Reaction("R_sink")
    r_sink.add_metabolites({b: -1.0})
    r_sink.bounds = (0.0, 10.0)
    model.add_reactions([r_in, r_out, r_sink])
    model.objective = r_out
    return model


class TestL0KeyIsContentAddressedNotFileAddressed:
    """The M8-opening defect: `build_canonical_model` used to hash `source_path` while freezing a
    *separately supplied* model, so a model could inherit an unrelated file's cache identity. The
    L0 key is now derived from the model's own frozen content, which closes every variant of it.
    """

    def test_mutating_a_model_changes_its_l0_key_under_the_same_source_path(
        self, toy_path: Path
    ) -> None:
        """The exact defect: same file named, different content ⇒ the key must move.

        The old file-hash key would have returned the *same* ``l0_key`` here (both calls hash
        ``toy_path``), silently stamping the mutated model with the pristine file's identity.
        """
        model = load_model(toy_path)
        before = build_canonical_model(model, toy_path).l0_key

        model.reactions.get_by_id("R1").upper_bound = 500.0  # mutate in memory, keep the same path
        after = build_canonical_model(model, toy_path).l0_key

        assert before != after

    def test_two_unrelated_models_do_not_collide_via_a_shared_source_path(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "shared.json"  # a name both calls point at; neither is hashed for id
        first = build_canonical_model(_valid_model(upper_bound=10.0), source)
        second = build_canonical_model(_valid_model(upper_bound=20.0), source)

        assert first.l0_key != second.l0_key

    def test_l0_key_ignores_reformatting_that_leaves_the_content_identical(
        self, tmp_path: Path, toy_path: Path
    ) -> None:
        """Content-addressing gives a beneficial cache *hit* a file hash would have missed.

        Two files with identical parsed content but different bytes (whitespace) produce the same
        ``l0_key`` — the downstream artifacts would be byte-for-byte identical, so a miss here would
        only waste work. The recorded ``source_sha256`` still differs; it is provenance, not id.
        """
        compact = _write(tmp_path, _toy_dict(toy_path))
        pretty = tmp_path / "pretty.json"
        pretty.write_text(json.dumps(_toy_dict(toy_path), indent=4))

        compact_model = load_canonical_model(compact)
        pretty_model = load_canonical_model(pretty)

        assert compact_model.l0_key == pretty_model.l0_key
        assert compact_model.source_sha256 != pretty_model.source_sha256  # bytes differ

    def test_l0_key_depends_on_model_id_because_it_names_the_rng_streams(self) -> None:
        a = build_canonical_model(_valid_model(model_id="strain_a"))
        b = build_canonical_model(_valid_model(model_id="strain_b"))

        assert a.l0_key != b.l0_key

    def test_source_sha256_is_recorded_only_on_the_trusted_load_path(
        self, toy_path: Path
    ) -> None:
        """A directly built model has no proven file behind it, so no file hash is stored."""
        built = build_canonical_model(load_model(toy_path), toy_path)
        assert built.source_sha256 is None

        loaded = load_canonical_model(toy_path)
        assert loaded.source_sha256 is not None and len(loaded.source_sha256) == 64

    def test_a_model_assembled_with_no_source_path_still_gets_an_identity(self) -> None:
        canonical = build_canonical_model(_valid_model(model_id="anon"))
        assert canonical.source_path is None
        assert canonical.source_sha256 is None
        assert len(canonical.l0_key) == 64
        assert canonical.report()["source_path"] is None


class TestModelReport:
    def test_report_counts_the_toy_network(self, toy_canonical: CanonicalModel) -> None:
        report = toy_canonical.report()

        assert report["counts"]["reactions"] == 7
        assert report["counts"]["metabolites"] == 3
        assert report["counts"]["fixed"] == 2
        assert report["counts"]["fixed_at_zero"] == 1  # BLK only; FIX sits at 2.0
        assert report["counts"]["free"] == 5
        assert report["biomass"]["reaction_id"] == "BIO"
        assert report["biomass"]["is_fixed"] is False
        assert report["bounds"]["all_finite"] is True

    def test_report_counts_the_example_model(self, example_canonical: CanonicalModel) -> None:
        report = example_canonical.report()

        assert report["counts"]["reactions"] == 773
        assert report["counts"]["fixed"] == 513
        assert report["counts"]["fixed_at_zero"] == 513  # every blocked reaction sits at zero
        assert report["counts"]["free"] == 260
        assert report["biomass"]["reaction_id"] == "bio1"

    def test_report_is_written_as_json(self, toy_canonical: CanonicalModel, tmp_path: Path) -> None:
        written = toy_canonical.write_report(tmp_path / "nested" / "model_report.json")

        reloaded = json.loads(written.read_text())
        assert reloaded == toy_canonical.report()
        assert reloaded["provenance"]["numpy_version"] == np.__version__

    def test_report_records_both_cache_keys(self, toy_canonical: CanonicalModel) -> None:
        report = toy_canonical.report()
        assert report["l0_key"] == toy_canonical.l0_key
        assert report["l1_key"] == toy_canonical.l1_key


def test_build_canonical_model_accepts_a_cobra_model_directly(toy_path: Path) -> None:
    canonical = build_canonical_model(load_model(toy_path), toy_path)
    assert canonical.model_id == "toy_network"
