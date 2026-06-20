
import pytest

from easyupscaler.denoise.catalog import path_for
from easyupscaler.denoise.downloader import ensure_models
from easyupscaler.errors import DenoiseDownloadError


def test_skips_existing_model(isolated_paths) -> None:
    target = path_for("archivist_medium")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * 2048)

    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True

    import easyupscaler.denoise.downloader as downloader_module

    original = downloader_module._download_model
    downloader_module._download_model = fake_download
    try:
        ensure_models(["archivist_medium"])
    finally:
        downloader_module._download_model = original

    assert called is False


def test_downloads_missing_model(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_download(filename, url, target, on_download_progress):
        captured.append(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 2048)

    import easyupscaler.denoise.downloader as downloader_module

    monkeypatch.setattr(downloader_module, "_download_model", fake_download)
    ensure_models(["archivist_medium"])
    assert captured == ["1x-Archivist_Medium.pth"]


def test_network_error_raises(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(filename, url, target, on_download_progress):
        raise DenoiseDownloadError("Error: could not download test.pth.")

    import easyupscaler.denoise.downloader as downloader_module

    monkeypatch.setattr(downloader_module, "_download_model", fake_download)
    with pytest.raises(DenoiseDownloadError, match="could not download"):
        ensure_models(["archivist_medium"])
