import sys

from typer.testing import CliRunner

from easyupscaler.cli.main import app

runner = CliRunner()


def test_help_does_not_import_torch(without_torch) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scale" in result.stdout
    assert "denoise" in result.stdout
    assert "easyupscaler 0.1.0" in result.stdout
    assert "torch" not in sys.modules


def test_version_does_not_import_torch(without_torch) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
    assert "torch" not in sys.modules
