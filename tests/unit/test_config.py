"""Configuration loading, overrides, and the resolved-config echo."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from gsmm_compiler import config as config_module
from gsmm_compiler.config import (
    Config,
    ConfigError,
    ObjectiveConfig,
    SamplerConfig,
    apply_overrides,
    from_dict,
    load,
)


def test_defaults_are_usable_with_no_file() -> None:
    resolved = load()
    assert resolved.sampler.betas == (0.0,)
    assert resolved.sampler.n_chains == 4
    assert resolved.output.store_flux_dtype == "float64"
    assert resolved.runtime.solver_threads == 1


def test_loads_the_toy_config(toy_config_path: Path) -> None:
    resolved = load(toy_config_path)
    assert resolved.model.biomass_id == "BIO"
    assert resolved.sampler.betas == (0.0, 1.0, 4.0)
    assert resolved.sampler.n_samples == 2000
    assert resolved.output.directory == "results/toy"


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load(tmp_path / "absent.toml")


class TestUnknownKeys:
    """A typo must fail loudly. Silently defaulting produces plausible numbers from a wrong run."""

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"unknown key\(s\) in \[sampler\]: \['n_chian'\]"):
            from_dict({"sampler": {"n_chian": 8}})

    def test_unknown_section_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"unknown config section\(s\): \['smapler'\]"):
            from_dict({"smapler": {"n_chains": 8}})

    def test_a_section_must_be_a_table(self) -> None:
        with pytest.raises(ConfigError, match="must be a table"):
            from_dict({"sampler": 4})


class TestOverrides:
    def test_override_types_are_inferred_the_way_toml_would(self) -> None:
        merged = apply_overrides(
            {},
            [
                "sampler.n_chains=8",
                "sampler.betas=[0.0, 2.0]",
                "objective.l1_penalty_scaled=0.5",
                "objective.exclude_biomass_from_penalty=false",
                "model.path=examples/toy_network.json",
            ],
        )
        resolved = from_dict(merged)

        assert resolved.sampler.n_chains == 8
        assert resolved.sampler.betas == (0.0, 2.0)
        assert resolved.objective.l1_penalty_scaled == 0.5
        assert resolved.objective.exclude_biomass_from_penalty is False
        assert resolved.model.path == "examples/toy_network.json"  # bare string, unquoted

    def test_override_wins_over_the_file(self, toy_config_path: Path) -> None:
        resolved = load(toy_config_path, ["sampler.n_samples=17"])
        assert resolved.sampler.n_samples == 17
        assert resolved.sampler.n_chains == 4  # untouched keys keep the file's values

    def test_override_into_a_section_the_file_omits(self, toy_config_path: Path) -> None:
        assert load(toy_config_path, ["runtime.n_workers=7"]).runtime.n_workers == 7

    @pytest.mark.parametrize("malformed", ["n_chains=8", "sampler.n_chains", "=8", "sampler.=8"])
    def test_malformed_overrides_are_rejected(self, malformed: str) -> None:
        with pytest.raises(ConfigError, match="section.key=value"):
            apply_overrides({}, [malformed])


class TestConstraints:
    """Constraints the science depends on, not merely input hygiene."""

    def test_negative_beta_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="betas must all be >= 0"):
            from_dict({"sampler": {"betas": [0.0, -1.0]}})

    def test_empty_beta_ladder_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="at least one"):
            from_dict({"sampler": {"betas": []}})

    def test_negative_penalty_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="l1_penalty_scaled must be >= 0"):
            from_dict({"objective": {"l1_penalty_scaled": -1.0}})

    def test_multithreaded_highs_is_rejected(self) -> None:
        """Determinism of the geometry depends on this (BUILD_PLAN §1.2)."""
        with pytest.raises(ConfigError, match="nondeterministic"):
            from_dict({"runtime": {"solver_threads": 4}})

    def test_unknown_store_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="store_mode"):
            from_dict({"output": {"store_mode": "parquet"}})

    def test_float16_storage_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="store_flux_dtype"):
            from_dict({"output": {"store_flux_dtype": "float16"}})

    def test_inverted_weight_clips_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="0 < min < max"):
            from_dict({"objective": {"weight_clip_min": 10.0, "weight_clip_max": 1.0}})


class TestEcho:
    def test_echo_is_valid_toml_that_reloads_to_the_same_config(
        self, toy_config_path: Path
    ) -> None:
        """The manifest records the *resolved* config, so it must round-trip."""
        resolved = load(toy_config_path, ["sampler.n_chains=6"])

        reloaded = from_dict(tomllib.loads(resolved.echo()))

        assert reloaded.sampler.n_chains == 6
        assert reloaded.sampler.betas == resolved.sampler.betas
        assert reloaded.output.directory == resolved.output.directory

    def test_echo_reports_defaults_the_file_never_mentioned(self, toy_config_path: Path) -> None:
        echoed = load(toy_config_path).echo()
        assert "refresh_interval" in echoed  # never set in toy_config.toml
        assert "[runtime]" in echoed

    def test_an_unset_optional_stays_unset_across_a_round_trip(self) -> None:
        """``biomass_id`` unset means "use the model's own objective". TOML has no null, so it must
        not be echoed as ``""`` — that would reload as a literal empty reaction ID, which is a
        value rather than an absence, and would then fail to match any reaction."""
        echoed = Config().echo()
        assert 'biomass_id = ""' not in echoed
        assert "# biomass_id" in echoed

        reloaded = from_dict(tomllib.loads(echoed))
        assert reloaded.model.biomass_id is None
        assert reloaded.geometry.max_span_probes is None


def test_config_module_imports_no_cobra() -> None:
    """A worker reads its config without the parser stack."""
    assert not hasattr(config_module, "cobra")


class TestTheEnergyScaleSetting:
    """``sampler.energy_scale`` (spec §3.6) — ``"warmup_range"`` or a declared positive number."""

    def test_the_default_is_the_warmup_range(self) -> None:
        """Because only ``warmup_range`` makes ``β`` mean the same thing in two different strains,
        which is what the batch comparison this package exists for depends on."""
        assert SamplerConfig().energy_scale == "warmup_range"

    def test_a_declared_positive_number_is_accepted(self) -> None:
        assert SamplerConfig(energy_scale=2.5).energy_scale == 2.5

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_nonpositive_or_nonfinite_scale_is_refused(self, bad: float) -> None:
        """A zero or negative ``s_J`` does not rescale the tilt — it annihilates or **flips** it, so
        the chain would sample ``exp(−βJ/|s_J|)`` while faithfully reporting the β it was asked for.
        The failure is caught here rather than several thousand sweeps in."""
        with pytest.raises(ConfigError, match="energy_scale"):
            SamplerConfig(energy_scale=bad)

    def test_an_unknown_mode_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="warmup_range"):
            SamplerConfig(energy_scale="pilot_range")  # M10's, not v1's

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_the_quantile_must_lie_strictly_inside_the_unit_interval(self, bad: float) -> None:
        with pytest.raises(ConfigError, match="quantile"):
            SamplerConfig(energy_scale_quantile=bad)

    def test_the_fallback_must_be_a_positive_scale(self) -> None:
        with pytest.raises(ConfigError, match="fallback"):
            SamplerConfig(energy_scale_fallback=0.0)

    @pytest.mark.parametrize("value", ["warmup_range", 3.5])
    def test_it_round_trips_through_the_echo(self, value: str | float) -> None:
        """`Config.echo` must reload to the same config, or the manifest does not describe the run.

        The string and the float take different TOML representations (quoted vs bare), so both forms
        have to survive — an ``energy_scale`` echoed as ``3.5`` and read back as the *string*
        ``"3.5"`` would reach `choose_energy_scale` as an unknown mode and kill the run at the far
        end of a geometry build.
        """
        echoed = from_dict({"sampler": {"energy_scale": value}}).echo()

        assert from_dict(tomllib.loads(echoed)).sampler.energy_scale == value


class TestTheNearZeroThresholds:
    """``objective.near_zero_thresholds`` (spec §3.7) — *analysis* only; the chain never sees
    them."""

    def test_several_are_declared_by_default(self) -> None:
        """One threshold would hide the fact that the answer depends on which one you pick. At
        finite β the law is continuous, so "how many reactions are off" has no threshold-free
        answer — the L1 term promotes small fluxes, it does not produce exact zeros."""
        assert len(ObjectiveConfig().near_zero_thresholds) >= 2

    def test_they_must_be_positive_and_finite(self) -> None:
        for bad in ((0.0,), (-1e-9,), (1e-9, float("inf"))):
            with pytest.raises(ConfigError, match="near_zero_thresholds"):
                ObjectiveConfig(near_zero_thresholds=bad)

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="at least one"):
            ObjectiveConfig(near_zero_thresholds=())
