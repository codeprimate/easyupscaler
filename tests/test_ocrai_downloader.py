import urllib.error
from pathlib import Path

import pytest

from easyupscaler.config.paths import ensure_models_dir
from easyupscaler.denoise.ocrai_catalog import OCRAI_BACKBONE_FILENAME, OCRAI_MMPROJ_FILENAME
from easyupscaler.denoise.ocrai_downloader import ensure_ocrai_models
from easyupscaler.errors import DenoiseDownloadError


def test_skips_existing_ocrai_models(isolated_paths) -> None:
    models_dir = ensure_models_dir()
    target = models_dir / OCRAI_BACKBONE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"GGUF" + b"\x00" * 2048)
    mmproj = models_dir / OCRAI_MMPROJ_FILENAME
    mmproj.write_bytes(b"GGUF" + b"\x00" * 2048)

    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True

    import easyupscaler.denoise.ocrai_downloader as downloader_module

    original = downloader_module._download_ocrai_file
    downloader_module._download_ocrai_file = fake_download
    try:
        ensure_ocrai_models()
    finally:
        downloader_module._download_ocrai_file = original

    assert called is False


def test_downloads_missing_ocrai_models(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_download(source, target, on_download_progress):
        captured.append(source.local_filename)
        target_path = ensure_models_dir() / source.local_filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"GGUF" + b"\x00" * 2048)

    import easyupscaler.denoise.ocrai_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "_download_ocrai_file", fake_download)
    ensure_ocrai_models()
    assert captured == [OCRAI_BACKBONE_FILENAME, OCRAI_MMPROJ_FILENAME]


def test_ocrai_download_failure_raises(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(source, target, on_download_progress):
        raise DenoiseDownloadError(f"Error: could not download {source.local_filename}.")

    import easyupscaler.denoise.ocrai_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "_download_ocrai_file", fake_download)
    with pytest.raises(DenoiseDownloadError, match="could not download"):
        ensure_ocrai_models()


def test_stream_download_tries_next_repo_on_http_error(
    isolated_paths,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import easyupscaler.denoise.ocrai_downloader as downloader_module
    from easyupscaler.denoise.ocrai_catalog import OCRAI_MODEL_SOURCES

    source = OCRAI_MODEL_SOURCES[0]
    target = tmp_path / source.local_filename
    attempts: list[str] = []

    def fake_urlopen(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

        class FakeResponse:
            headers = {"Content-Length": "2048"}

            def read(self, size=-1):
                if not hasattr(self, "_done"):
                    self._done = True
                    return b"GGUF" + b"\x00" * 2044
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    downloader_module._download_ocrai_file(source, target, None)
    assert len(attempts) == 2
    assert target.exists()
    assert target.read_bytes()[:4] == b"GGUF"
