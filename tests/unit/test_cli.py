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


def test_model_inspect_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["model", "inspect", str(tmp_path / "nope.json")])


def test_model_inspect_unknown_format(tmp_path: Path) -> None:
    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    with pytest.raises(ValueError, match="unsupported model format"):
        main(["model", "inspect", str(bogus)])
