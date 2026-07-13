"""CLI surface (M0): ``gsmm-compiler model inspect``."""

from __future__ import annotations

from pathlib import Path

import pytest

from gsmm_compiler import __version__
from gsmm_compiler.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_model_inspect_reports_the_example_model(
    example_model_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["model", "inspect", str(example_model_path)]) == 0

    report = capsys.readouterr().out
    assert "reactions           773" in report
    assert "metabolites         894" in report
    assert "fixed (l == u)      513" in report
    assert "free (l < u)        260" in report
    assert "bio1" in report

    assert "free reactions    260" in report
    assert "eliminated (l==u) 513" in report
    assert "zero (all fixed fluxes are 0)" in report


def test_model_inspect_reports_the_toy_s_nonzero_affine_rhs(
    toy_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["model", "inspect", str(toy_path)]) == 0

    report = capsys.readouterr().out
    assert "free reactions    5" in report
    assert "eliminated (l==u) 2" in report
    assert "affine RHS        nonzero" in report


def test_model_inspect_writes_the_report(
    toy_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    destination = tmp_path / "model_report.json"
    assert main(["model", "inspect", str(toy_path), "--report", str(destination)]) == 0

    capsys.readouterr()
    report = json.loads(destination.read_text())
    assert report["counts"]["free"] == 5
    assert report["biomass"]["reaction_id"] == "BIO"


def test_config_show_echoes_the_resolved_config(
    toy_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["config", "show", "--config", str(toy_config_path), "--set", "sampler.n_chains=9"])
        == 0
    )

    echoed = capsys.readouterr().out
    assert "n_chains = 9" in echoed  # the override
    assert "betas = [0.0, 1.0, 4.0]" in echoed  # the file
    assert "refresh_interval = 1000" in echoed  # a default the file never mentioned


def test_config_show_works_with_no_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show"]) == 0
    assert "[sampler]" in capsys.readouterr().out


def test_model_inspect_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["model", "inspect", str(tmp_path / "nope.json")])


def test_model_inspect_unknown_format(tmp_path: Path) -> None:
    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    with pytest.raises(ValueError, match="unsupported model format"):
        main(["model", "inspect", str(bogus)])
