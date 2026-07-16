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


class TestMaxentCommands:
    """The M8 pipeline commands, exercised on the toy network."""

    def test_solve_lp_reports_the_resolved_penalty(
        self, toy_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["maxent", "solve-lp", str(toy_path)]) == 0
        out = capsys.readouterr().out
        assert "λ* (cliff)" in out
        assert "J*" in out
        assert "sparsity-dominated False" in out

    def test_build_geometry_reports_the_dimension(
        self, toy_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["maxent", "build-geometry", str(toy_path)]) == 0
        out = capsys.readouterr().out
        assert "dimension d       2" in out
        assert "span certificate  exhaustive" in out

    def test_sample_then_diagnose(
        self, toy_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "maxent",
                "sample",
                str(toy_path),
                "--model-id",
                "toy",
                "--out",
                str(tmp_path / "r"),
                "--batch",
                "b",
                "--set",
                "sampler.betas=[0.0,2.0]",
                "--set",
                "sampler.n_samples=40",
                "--set",
                "sampler.burn_in=40",
                "--set",
                "sampler.n_chains=2",
                "--set",
                "sampler.refresh_interval=20",
            ]
        )
        assert code == 0
        assert "complete  toy" in capsys.readouterr().out

        diagnose = ["maxent", "diagnose", str(tmp_path / "r"), "--batch", "b", "--model-id", "toy"]
        assert main(diagnose) == 0
        diag = capsys.readouterr().out
        assert "E[J] monotone" in diag
        assert (tmp_path / "r" / "b" / "toy" / "diagnostics" / "diagnostics.json").is_file()

    def test_batch_over_a_manifest_with_a_bad_strain_exits_nonzero(
        self, toy_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = tmp_path / "m.tsv"
        manifest.write_text(
            f"model_path\tmodel_id\n{toy_path}\tgood\n/nope.json\tbroken\n"
        )
        code = main(
            [
                "maxent",
                "batch",
                str(manifest),
                "--out",
                str(tmp_path / "r"),
                "--batch",
                "b",
                "--set",
                "sampler.betas=[0.0]",
                "--set",
                "sampler.n_samples=30",
                "--set",
                "sampler.burn_in=30",
                "--set",
                "sampler.n_chains=2",
                "--set",
                "sampler.refresh_interval=15",
            ]
        )
        assert code == 1  # one strain failed
        out = capsys.readouterr().out
        assert "complete  good" in out
        assert "failed    broken" in out


def test_model_inspect_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["model", "inspect", str(tmp_path / "nope.json")])


def test_model_inspect_unknown_format(tmp_path: Path) -> None:
    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    with pytest.raises(ValueError, match="unsupported model format"):
        main(["model", "inspect", str(bogus)])
