"""The models manifest and per-model preparation (M8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsmm_compiler.batch import (
    BatchError,
    ModelSpec,
    geometry_cache_key,
    load_models_manifest,
    prepare_model,
)
from gsmm_compiler.config import Config, SamplerConfig


class TestLoadModelsManifest:
    def test_json_list(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps([{"model_path": "a.json"}, {"model_path": "b.json"}]))
        specs = load_models_manifest(path)
        assert [Path(s.model_path).name for s in specs] == ["a.json", "b.json"]

    def test_json_object_with_models_key(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps({"models": [{"model_path": "a.json", "biomass_id": "bio", "model_id": "A"}]})
        )
        (spec,) = load_models_manifest(path)
        assert spec.biomass_id == "bio"
        assert spec.model_id == "A"

    def test_tsv_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "m.tsv"
        path.write_text("model_path\tmodel_id\tbiomass_id\nx.json\tX\tbioX\ny.json\tY\t\n")
        specs = load_models_manifest(path)
        assert [s.model_id for s in specs] == ["X", "Y"]
        assert specs[0].biomass_id == "bioX"
        assert specs[1].biomass_id is None  # an empty TSV cell is an absence, not ""

    def test_relative_paths_resolve_against_the_manifest_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        path = nested / "m.tsv"
        path.write_text("model_path\nmodel.json\n")
        (spec,) = load_models_manifest(path)
        assert spec.model_path == str(nested / "model.json")

    def test_absolute_paths_are_left_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "m.tsv"
        path.write_text("model_path\n/abs/model.json\n")
        (spec,) = load_models_manifest(path)
        assert spec.model_path == "/abs/model.json"

    def test_duplicate_ids_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps([{"model_path": "a.json"}, {"model_path": "a.json"}]))
        with pytest.raises(BatchError, match="resolve to model_id"):
            load_models_manifest(path)

    def test_missing_model_path_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps([{"biomass_id": "bio"}]))
        with pytest.raises(BatchError, match="missing 'model_path'"):
            load_models_manifest(path)

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BatchError, match="not found"):
            load_models_manifest(tmp_path / "absent.tsv")

    def test_unknown_format_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.csv"
        path.write_text("model_path\na.json")
        with pytest.raises(BatchError, match="unsupported manifest format"):
            load_models_manifest(path)


class TestPrepareModel:
    def test_prepares_the_toy_with_and_without_a_cache(
        self, toy_path: Path, tmp_path: Path
    ) -> None:
        from gsmm_compiler.cache import ArtifactCache

        config = Config(sampler=SamplerConfig(betas=(0.0,), n_chains=1, n_samples=10, burn_in=10))
        spec = ModelSpec(model_path=str(toy_path), model_id="toy")

        plan_no_cache = prepare_model(spec, config)
        cache = ArtifactCache(tmp_path / "cache")
        plan_cached = prepare_model(spec, config, cache=cache)

        # The cache path must reconstruct the *same* transform the direct build produced.
        assert plan_cached.transform.content_key() == plan_no_cache.transform.content_key()
        key = geometry_cache_key(plan_cached.reduced, config, model_id="toy")
        assert cache.is_cached("L3", key)

    def test_a_second_prepare_hits_the_geometry_cache(self, toy_path: Path, tmp_path: Path) -> None:
        from gsmm_compiler.cache import ArtifactCache

        config = Config(sampler=SamplerConfig(betas=(0.0,), n_chains=1, n_samples=10, burn_in=10))
        spec = ModelSpec(model_path=str(toy_path), model_id="toy")
        cache = ArtifactCache(tmp_path / "cache")

        first = prepare_model(spec, config, cache=cache)
        second = prepare_model(spec, config, cache=cache)  # should load L3, not rebuild
        assert second.transform.content_key() == first.transform.content_key()
