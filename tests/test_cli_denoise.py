from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from easyupscaler.cli.main import app
from easyupscaler.denoise.pipeline import DenoiseResult

runner = CliRunner()


def test_empty_denoise_exit_one(without_torch) -> None:
    result = runner.invoke(app, ["denoise", "photo"])
    assert result.exit_code == 1
    assert "Error: no input images" in result.stderr


def test_invalid_mode_exit_one(without_torch, tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    result = runner.invoke(app, ["denoise", "invalid", str(image)])
    assert result.exit_code == 1
    assert "invalid mode" in result.stderr


def test_invalid_strength_exit_one(without_torch, tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    result = runner.invoke(app, ["denoise", "photo", str(image), "--strength", "medium"])
    assert result.exit_code == 1
    assert "invalid strength" in result.stderr


def test_non_tty_denoise_output(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "input-denoised.png"

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        result = DenoiseResult(path=image, output=output, error=None)
        if on_progress:
            on_progress(result)
        return [result]

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )
    monkeypatch.setattr("easyupscaler.cli.denoise.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, ["denoise", "photo", str(image)])
    assert result.exit_code == 0
    assert f"{image} → {output}" in result.stdout


def test_partial_failure_exit_one(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    good.write_bytes(b"image")

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        results = [
            DenoiseResult(path=bad, output=None, error="file not found"),
            DenoiseResult(path=good, output=tmp_path / "good-denoised.png", error=None),
        ]
        if on_progress:
            for item in results:
                on_progress(item)
        return results

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )
    monkeypatch.setattr("easyupscaler.cli.denoise.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, ["denoise", "art", str(bad), str(good)])
    assert result.exit_code == 1
    assert "FAILED: file not found" in result.stdout


def test_download_failure_exit_one(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        raise ValueError(
            "Error: could not download scunet_color_real_psnr.pth. Check your network."
        )

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "photo", str(image)])
    assert result.exit_code == 1
    assert "could not download" in result.stderr


def test_output_flag_forwards_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output_dir = tmp_path / "results"
    captured: dict[str, Path | None] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["output_dir"] = kwargs.get("output_dir")
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "photo", str(image), "--output", str(output_dir)])
    assert result.exit_code == 0
    assert captured["output_dir"] == output_dir
    assert output_dir.is_dir()


def test_short_output_flag_forwards_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output_dir = tmp_path / "results"
    captured: dict[str, Path | None] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["output_dir"] = kwargs.get("output_dir")
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "photo", str(image), "-o", str(output_dir)])
    assert result.exit_code == 0
    assert captured["output_dir"] == output_dir


def test_output_path_is_file_exit_one_before_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output_file = tmp_path / "results"
    output_file.write_bytes(b"not-a-directory")
    service_called = False

    def fake_run(*args, **kwargs):
        nonlocal service_called
        service_called = True
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "photo", str(image), "--output", str(output_file)])
    assert result.exit_code == 1
    assert "output path is not a directory" in result.stderr
    assert service_called is False
