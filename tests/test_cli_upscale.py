import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from easyupscaler.cli.main import app
from easyupscaler.cli.upscale import run_upscale
from easyupscaler.upscaling.service import UpscaleResult

runner = CliRunner()


def test_empty_paths_exit_one_before_torch(without_torch) -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 1
    assert "Error: no input images" in result.stderr
    assert "torch" not in sys.modules


def test_missing_default_exit_one(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    result = runner.invoke(app, [str(image)])
    assert result.exit_code == 1
    assert "no default model set" in result.stderr


def test_non_tty_output(isolated_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "input-upscaled.jpg"

    def fake_run(paths, model_name, on_progress=None):
        result = UpscaleResult(path=image, output=output, error=None)
        if on_progress:
            on_progress(result)
        return [result]

    monkeypatch.setattr("easyupscaler.cli.upscale.UpscaleService", lambda: MagicMock(run=fake_run))
    monkeypatch.setattr("easyupscaler.cli.upscale.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, [str(image)])
    assert result.exit_code == 0
    assert f"{image} → {output}" in result.stdout


def test_tty_completed_includes_elapsed(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "input-upscaled.jpg"
    perf_counter_values = iter([100.0, 145.0])
    echoed: list[str] = []

    def fake_run(paths, model_name, on_progress=None):
        result = UpscaleResult(path=image, output=output, error=None)
        if on_progress:
            on_progress(result)
        return [result]

    class FakeProgress:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def add_task(self, *args, **kwargs) -> int:
            return 0

        def advance(self, *args, **kwargs) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("easyupscaler.cli.upscale.UpscaleService", lambda: MagicMock(run=fake_run))
    monkeypatch.setattr("easyupscaler.cli.upscale.Progress", FakeProgress)
    monkeypatch.setattr("easyupscaler.cli.upscale.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        "easyupscaler.cli.upscale.typer.echo",
        lambda message, **kwargs: echoed.append(message),
    )
    monkeypatch.setattr(
        "easyupscaler.cli.upscale.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    run_upscale([str(image)], model="test-model")

    assert "Completed: 1 succeeded, 0 failed in 0:45." in echoed


def test_partial_failure_exit_one(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    good.write_bytes(b"image")

    def fake_run(paths, model_name, on_progress=None):
        results = [
            UpscaleResult(path=bad, output=None, error="file not found"),
            UpscaleResult(path=good, output=tmp_path / "good-upscaled.jpg", error=None),
        ]
        if on_progress:
            for item in results:
                on_progress(item)
        return results

    monkeypatch.setattr("easyupscaler.cli.upscale.UpscaleService", lambda: MagicMock(run=fake_run))
    monkeypatch.setattr("easyupscaler.cli.upscale.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, [str(bad), str(good)])
    assert result.exit_code == 1
    assert "FAILED: file not found" in result.stdout
