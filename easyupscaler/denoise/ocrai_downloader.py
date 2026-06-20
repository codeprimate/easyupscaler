import os
import tempfile
import urllib.error
import urllib.request

from easyupscaler.config.paths import ensure_models_dir
from easyupscaler.denoise.downloader import (
    DOWNLOAD_CHUNK_SIZE,
    MIN_VALID_MODEL_BYTES,
    DownloadProgressCallback,
)
from easyupscaler.denoise.ocrai_catalog import (
    GGUF_MAGIC,
    OCRAI_MODEL_SOURCES,
    OCRAI_REPO_FALLBACK_ORDER,
    OcraiModelSource,
    build_ocrai_download_url,
)
from easyupscaler.errors import DenoiseDownloadError, DenoiseModelCorruptError


def ensure_ocrai_models(
    *,
    on_download_progress: DownloadProgressCallback | None = None,
) -> None:
    ensure_models_dir()
    for source in OCRAI_MODEL_SOURCES:
        target = ensure_models_dir() / source.local_filename
        if target.exists() and _is_valid_gguf(target):
            continue
        if target.exists():
            target.unlink()
        _download_ocrai_file(source, target, on_download_progress)


def _is_valid_gguf(path: os.PathLike[str] | str) -> bool:
    model_path = os.fspath(path)
    try:
        size = os.path.getsize(model_path)
    except OSError:
        return False
    if size < MIN_VALID_MODEL_BYTES:
        return False
    with open(model_path, "rb") as model_file:
        header = model_file.read(len(GGUF_MAGIC))
    return header == GGUF_MAGIC


def _download_ocrai_file(
    source: OcraiModelSource,
    target: os.PathLike[str] | str,
    on_download_progress: DownloadProgressCallback | None,
) -> None:
    target_path = os.fspath(target)
    models_dir = ensure_models_dir()
    last_error: DenoiseDownloadError | None = None

    for repo in OCRAI_REPO_FALLBACK_ORDER:
        remote_filename = source.remote_by_repo.get(repo)
        if remote_filename is None:
            continue
        url = build_ocrai_download_url(repo, remote_filename)
        try:
            _stream_download(
                source.local_filename,
                url,
                target_path,
                models_dir,
                on_download_progress,
            )
        except DenoiseDownloadError as exc:
            last_error = exc
            continue
        else:
            return

    if last_error is not None:
        raise last_error
    fallback_url = build_ocrai_download_url(
        OCRAI_REPO_FALLBACK_ORDER[0],
        source.remote_by_repo[OCRAI_REPO_FALLBACK_ORDER[0]],
    )
    raise DenoiseDownloadError(
        _download_error_message(source.local_filename, fallback_url, models_dir),
    )


def _stream_download(
    local_filename: str,
    url: str,
    target: str,
    models_dir: os.PathLike[str],
    on_download_progress: DownloadProgressCallback | None,
) -> None:
    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else None
            with tempfile.NamedTemporaryFile(
                dir=models_dir,
                delete=False,
                prefix=f".{local_filename}.",
            ) as temp_file:
                temp_path = temp_file.name
                downloaded = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    if on_download_progress is not None:
                        on_download_progress(local_filename, downloaded, total_bytes)
        if downloaded < MIN_VALID_MODEL_BYTES:
            os.unlink(temp_path)
            raise DenoiseModelCorruptError(_corrupt_message(local_filename))
        if not _is_valid_gguf(temp_path):
            os.unlink(temp_path)
            raise DenoiseModelCorruptError(_corrupt_message(local_filename))
        os.replace(temp_path, target)
    except DenoiseModelCorruptError:
        raise DenoiseModelCorruptError(_corrupt_message(local_filename)) from None
    except urllib.error.HTTPError as exc:
        raise DenoiseDownloadError(
            _download_error_message(local_filename, url, models_dir),
        ) from exc
    except urllib.error.URLError as exc:
        raise DenoiseDownloadError(
            _download_error_message(local_filename, url, models_dir),
        ) from exc
    except OSError as exc:
        raise DenoiseDownloadError(
            _download_error_message(local_filename, url, models_dir),
        ) from exc


def _download_error_message(
    filename: str,
    url: str,
    models_dir: os.PathLike[str],
) -> str:
    return (
        f"Error: could not download {filename}. Check your network or download "
        f"manually from {url} and place in {models_dir}."
    )


def _corrupt_message(filename: str) -> str:
    return f"Error: downloaded {filename} appears corrupt. Delete it and retry."
