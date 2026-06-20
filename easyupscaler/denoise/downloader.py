import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from easyupscaler.config.paths import ensure_models_dir
from easyupscaler.denoise.catalog import DENOISE_MODEL_CATALOG, CatalogKey, path_for
from easyupscaler.errors import DenoiseDownloadError, DenoiseModelCorruptError

MIN_VALID_MODEL_BYTES = 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

DownloadProgressCallback = Callable[[str, int, int | None], None]


def ensure_models(
    keys: list[CatalogKey],
    *,
    on_download_progress: DownloadProgressCallback | None = None,
) -> None:
    ensure_models_dir()
    for key in keys:
        entry = DENOISE_MODEL_CATALOG[key]
        target = path_for(key)
        if target.exists() and target.stat().st_size >= MIN_VALID_MODEL_BYTES:
            continue
        if target.exists():
            target.unlink()
        _download_model(entry.filename, entry.url, target, on_download_progress)


def _download_model(
    filename: str,
    url: str,
    target: Path,
    on_download_progress: DownloadProgressCallback | None,
) -> None:
    models_dir = ensure_models_dir()
    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else None
            with tempfile.NamedTemporaryFile(
                dir=models_dir,
                delete=False,
                prefix=f".{filename}.",
            ) as temp_file:
                temp_path = Path(temp_file.name)
                downloaded = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    if on_download_progress is not None:
                        on_download_progress(filename, downloaded, total_bytes)
        if downloaded < MIN_VALID_MODEL_BYTES:
            temp_path.unlink(missing_ok=True)
            raise DenoiseModelCorruptError(_corrupt_message(filename))
        os.replace(temp_path, target)
    except DenoiseModelCorruptError:
        raise DenoiseModelCorruptError(
            _corrupt_message(filename),
        ) from None
    except urllib.error.URLError as exc:
        raise DenoiseDownloadError(
            _download_error_message(filename, url, models_dir),
        ) from exc
    except OSError as exc:
        raise DenoiseDownloadError(
            _download_error_message(filename, url, models_dir),
        ) from exc


def delete_corrupt_model(key: CatalogKey) -> None:
    path = path_for(key)
    if path.exists():
        path.unlink()


def _download_error_message(filename: str, url: str, models_dir: Path) -> str:
    return (
        f"Error: could not download {filename}. Check your network or download "
        f"manually from {url} and place in {models_dir}."
    )


def _corrupt_message(filename: str) -> str:
    return f"Error: downloaded {filename} appears corrupt. Delete it and retry."
