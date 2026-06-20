from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from easyupscaler.denoise.pipeline import DenoiseService
from easyupscaler.io.images import ImageIO


class FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def denoise(self, image: np.ndarray, *, qf_override=None) -> np.ndarray:
        self.calls += 1
        return image


def test_single_pass_photo_jpeg(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_test_jpeg(source)
    backends: dict = {}

    def factory(key):
        backend = FakeBackend(key)
        backends[key] = backend
        return backend

    service = DenoiseService(
        backend_factory=factory,
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "photo", "low")
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].output is not None
    assert results[0].output.name == "photo-denoised.png"
    assert backends["scunet_psnr"].calls == 1


def test_run_writes_to_custom_output_dir(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    output_dir = tmp_path / "results"
    _write_test_jpeg(source)

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "photo", "low", output_dir=output_dir)
    assert results[0].error is None
    assert results[0].output == output_dir / "photo-denoised.png"
    assert results[0].output.exists()


def test_two_pass_photo_heic(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "photo.heic"
    source.write_bytes(b"placeholder")
    backends: dict = {}

    def factory(key):
        backend = FakeBackend(key)
        backends[key] = backend
        return backend

    image_io = MagicMock(spec=ImageIO)
    image_io.read_preserving_grayscale_info.return_value = (
        np.zeros((4, 4, 3), dtype=np.float32),
        False,
    )
    image_io.write_denoised.return_value = tmp_path / "photo-denoised.png"

    service = DenoiseService(
        image_io=image_io,
        backend_factory=factory,
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "photo", "low")
    assert results[0].error is None
    assert backends["scunet_psnr"].calls == 1
    assert backends["fbcnn_color"].calls == 1
    assert results[0].pass_description == "SCUNet PSNR + FBCNN"


def test_manga_grayscale_preserved(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    _write_test_jpeg(source)
    image_io = MagicMock(spec=ImageIO)
    image_io.read_preserving_grayscale_info.return_value = (
        np.zeros((4, 4, 3), dtype=np.float32),
        True,
    )
    image_io.write_denoised.return_value = tmp_path / "page-denoised.png"

    service = DenoiseService(
        image_io=image_io,
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    service.run([source], "manga", "low")
    image_io.write_denoised.assert_called_once()
    kwargs = image_io.write_denoised.call_args.kwargs
    assert kwargs["preserve_grayscale"] is True
    assert kwargs["was_grayscale"] is True


def test_batch_partial_failure(isolated_paths, tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    bad = tmp_path / "missing.jpg"
    _write_test_jpeg(good)

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([bad, good], "art", "low")
    assert results[0].error == "file not found"
    assert results[1].error is None


def test_download_failure_aborts_job(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_test_jpeg(source)

    def fail_download(keys, **kwargs):
        raise ValueError("Error: could not download model.")

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=fail_download,
    )
    with pytest.raises(ValueError, match="could not download"):
        service.run([source], "photo", "low")


def test_document_mode_writes_grayscale_png(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "scan.jpg"
    _write_test_jpeg(source, size=100)
    call_order: list[str] = []

    def factory(key):
        backend = FakeBackend(key)
        call_order.append(key)
        return backend

    service = DenoiseService(
        backend_factory=factory,
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "document", "low")
    assert results[0].error is None
    assert results[0].output is not None
    assert results[0].output.name == "scan-denoised.png"
    assert call_order == ["archivist_medium"]

    from PIL import Image

    with Image.open(results[0].output) as output_image:
        assert output_image.mode == "L"


def test_document_mode_high_strength(isolated_paths, tmp_path: Path) -> None:
    source = tmp_path / "scan.jpg"
    _write_test_jpeg(source, size=100)

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "document", "high")
    assert results[0].error is None
    assert results[0].output is not None


def test_document_mode_too_small_image_fails_per_file(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.jpg"
    _write_test_jpeg(source, size=20)

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([source], "document", "high")
    assert results[0].error is not None
    assert "image too small for document mode" in results[0].error


def test_document_mode_batch_continues_after_too_small(
    isolated_paths,
    tmp_path: Path,
) -> None:
    tiny = tmp_path / "tiny.jpg"
    good = tmp_path / "good.jpg"
    _write_test_jpeg(tiny, size=20)
    _write_test_jpeg(good, size=100)

    service = DenoiseService(
        backend_factory=lambda key: FakeBackend(key),
        download_models=lambda keys, **kwargs: None,
    )
    results = service.run([tiny, good], "document", "high")
    assert results[0].error is not None
    assert results[1].error is None


def _write_test_jpeg(path: Path, size: int = 4) -> None:
    from PIL import Image

    Image.new("RGB", (size, size), color=(128, 64, 32)).save(path, format="JPEG")
