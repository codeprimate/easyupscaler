from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from easyupscaler.cli.main import app
from easyupscaler.denoise.pipeline import DenoiseResult
from easyupscaler.progress import PhaseKind
from tests.conftest import emit_phase_done

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
    assert "document" in result.stderr


def test_document_mode_routes_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    captured: dict[str, str] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["mode"] = mode
        captured["strength"] = strength
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "document", str(image)])
    assert result.exit_code == 0
    assert captured["mode"] == "document"
    assert captured["strength"] == "low"


def test_document_mode_high_strength_routes_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    captured: dict[str, str] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["mode"] = mode
        captured["strength"] = strength
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "document", str(image), "--strength", "high"])
    assert result.exit_code == 0
    assert captured["mode"] == "document"
    assert captured["strength"] == "high"


def test_no_text_flag_forwards_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    captured: dict[str, bool] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["extract_text"] = kwargs.get("extract_text")
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "document", str(image), "--no-text"])
    assert result.exit_code == 0
    assert captured["extract_text"] is False


def test_ocrai_flag_forwards_to_service(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    captured: dict[str, bool] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["use_ocrai"] = kwargs.get("use_ocrai")
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(app, ["denoise", "document", str(image), "--ocrai"])
    assert result.exit_code == 0
    assert captured["use_ocrai"] is True


def test_ocrai_with_no_text_still_forwards_both_flags(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    captured: dict[str, bool] = {}

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        captured["extract_text"] = kwargs.get("extract_text")
        captured["use_ocrai"] = kwargs.get("use_ocrai")
        return []

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )

    result = runner.invoke(
        app,
        ["denoise", "document", str(image), "--ocrai", "--no-text"],
    )
    assert result.exit_code == 0
    assert captured["extract_text"] is False
    assert captured["use_ocrai"] is True


def test_document_mode_stdout_shows_png_txt_and_md(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    png_output = tmp_path / "scan-denoised.png"
    txt_output = tmp_path / "scan.txt"
    md_output = tmp_path / "scan.md"

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        result = DenoiseResult(
            path=image,
            output=png_output,
            error=None,
            text_output=txt_output,
            markdown_output=md_output,
        )
        on_phase = kwargs.get("on_phase")
        emit_phase_done(on_phase, path=image, phase=PhaseKind.IMAGE, output_path=png_output)
        emit_phase_done(on_phase, path=image, phase=PhaseKind.TEXT, output_path=txt_output)
        emit_phase_done(
            on_phase,
            path=image,
            phase=PhaseKind.MARKDOWN,
            output_path=md_output,
        )
        if on_progress:
            on_progress(result)
        return [result]

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )
    monkeypatch.setattr("easyupscaler.cli.job_progress.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, ["denoise", "document", str(image), "--ocrai"])
    assert result.exit_code == 0
    assert "scan.heic: PNG →" in result.stdout or "scan.jpg: PNG →" in result.stdout
    assert "scan.txt" in result.stdout
    assert "scan.md" in result.stdout


def test_document_mode_stdout_shows_png_and_txt(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image")
    png_output = tmp_path / "scan-denoised.png"
    txt_output = tmp_path / "scan.txt"

    def fake_run(paths, mode, strength, on_progress=None, on_download_progress=None, **kwargs):
        result = DenoiseResult(
            path=image,
            output=png_output,
            error=None,
            text_output=txt_output,
        )
        on_phase = kwargs.get("on_phase")
        emit_phase_done(on_phase, path=image, phase=PhaseKind.IMAGE, output_path=png_output)
        emit_phase_done(on_phase, path=image, phase=PhaseKind.TEXT, output_path=txt_output)
        if on_progress:
            on_progress(result)
        return [result]

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )
    monkeypatch.setattr("easyupscaler.cli.job_progress.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, ["denoise", "document", str(image)])
    assert result.exit_code == 0
    assert "PNG →" in result.stdout
    assert "Text →" in result.stdout


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
        on_phase = kwargs.get("on_phase")
        emit_phase_done(on_phase, path=image, phase=PhaseKind.IMAGE, output_path=output)
        if on_progress:
            on_progress(result)
        return [result]

    monkeypatch.setattr(
        "easyupscaler.cli.denoise.DenoiseService",
        lambda: MagicMock(run=fake_run),
    )
    monkeypatch.setattr("easyupscaler.cli.job_progress.sys.stdout.isatty", lambda: False)

    result = runner.invoke(app, ["denoise", "photo", str(image)])
    assert result.exit_code == 0
    assert "Denoise →" in result.stdout
    assert str(output) in result.stdout


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
    monkeypatch.setattr("easyupscaler.cli.job_progress.sys.stdout.isatty", lambda: False)

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
