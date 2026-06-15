import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from easyupscaler.cli.main import app, main_entry
from easyupscaler.cli.models import models_app
from easyupscaler.config.settings import ConfigService
from easyupscaler.models.registry import ModelEntry, ModelRegistry

runner = CliRunner()


def _add_model(isolated_paths, name: str = "RealESRGAN_x4plus", scale: int = 4) -> None:
    models_dir = isolated_paths / "data" / "easyupscaler" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weight_path = models_dir / f"{name}.pth"
    weight_path.write_bytes(b"weights")
    ModelRegistry().add(
        ModelEntry(
            name=name,
            filename=f"{name}.pth",
            path=weight_path,
            scale=scale,
            imported_at=datetime(2025, 6, 15, tzinfo=UTC),
        )
    )


def test_models_without_subcommand_shows_help(without_torch) -> None:
    result = runner.invoke(models_app, [], prog_name="easyupscaler models")
    assert result.exit_code == 0
    assert "Missing command" not in result.output
    assert "Manage installed upscaling models" in result.output
    assert "list" in result.output
    assert "install-completion" not in result.output
    assert "show-completion" not in result.output
    assert "torch" not in sys.modules


def test_models_help_has_no_completion_options(without_torch) -> None:
    result = runner.invoke(models_app, ["--help"], prog_name="easyupscaler models")
    assert result.exit_code == 0
    assert "install-completion" not in result.output
    assert "show-completion" not in result.output


def test_unknown_subcommand_shows_suggestion(without_torch) -> None:
    result = runner.invoke(models_app, ["lsit"])
    assert result.exit_code == 2
    assert "No such command 'lsit'" in result.stderr
    assert "Did you mean 'list'?" in result.stderr
    assert "torch" not in sys.modules


def test_unknown_subcommand_via_main_entry(without_torch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["easyupscaler", "models", "lsit"])
    with pytest.raises(SystemExit) as exc_info:
        main_entry()
    assert exc_info.value.code == 2


def test_help_does_not_import_torch(without_torch) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "torch" not in sys.modules


def test_version_does_not_import_torch(without_torch) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "easyupscaler 0.1.0" in result.stdout
    assert "torch" not in sys.modules


def test_list_empty(isolated_paths, without_torch) -> None:
    result = runner.invoke(models_app, ["list"])
    assert result.exit_code == 0
    assert "No models installed" in result.stdout
    assert "torch" not in sys.modules


def test_list_with_models(isolated_paths, without_torch) -> None:
    _add_model(isolated_paths)
    weight_path = isolated_paths / "data" / "easyupscaler" / "models" / "RealESRGAN_x4plus.pth"
    result = runner.invoke(models_app, ["list"])
    assert result.exit_code == 0
    assert "Name" in result.stdout
    assert "Path" in result.stdout
    assert "RealESRGAN_x4plus" in result.stdout
    assert str(weight_path) in result.stdout
    assert "4×" in result.stdout
    assert "torch" not in sys.modules


def test_default_success(isolated_paths, without_torch) -> None:
    _add_model(isolated_paths)
    result = runner.invoke(models_app, ["default", "RealESRGAN_x4plus"])
    assert result.exit_code == 0
    assert "Default model set to RealESRGAN_x4plus" in result.stdout
    assert ConfigService().get_default_model() == "RealESRGAN_x4plus"
    assert "torch" not in sys.modules


def test_default_unknown_lists_models(isolated_paths, without_torch) -> None:
    _add_model(isolated_paths)
    result = runner.invoke(models_app, ["default", "missing"])
    assert result.exit_code == 1
    assert "Error: model 'missing' not found" in result.stderr
    assert "RealESRGAN_x4plus" in result.stderr
    assert "torch" not in sys.modules


def test_remove_with_yes(isolated_paths, without_torch) -> None:
    _add_model(isolated_paths)
    ConfigService().set_default_model("RealESRGAN_x4plus")
    result = runner.invoke(models_app, ["remove", "RealESRGAN_x4plus", "--yes"])
    assert result.exit_code == 0
    assert ModelRegistry().list() == []
    assert ConfigService().get_default_model() is None
    assert "Warning: RealESRGAN_x4plus was the default model" in result.stderr
    assert "torch" not in sys.modules


def test_remove_unknown(isolated_paths) -> None:
    result = runner.invoke(models_app, ["remove", "missing"])
    assert result.exit_code == 1
    assert "Error: model 'missing' not found." in result.stderr


def test_import_command_success(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "demo.pth"
    source.write_bytes(b"weights")

    fake_entry = ModelEntry(
        name="demo",
        filename="demo.pth",
        path=source,
        scale=4,
        imported_at=datetime(2025, 6, 15, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "easyupscaler.models.import_model.import_model",
        lambda path, force=False: fake_entry,
    )

    result = runner.invoke(models_app, ["import", str(source)])
    assert result.exit_code == 0
