from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from easyupscaler.config.settings import ConfigService
from easyupscaler.errors import ImageReadError
from easyupscaler.io.images import ImageIO
from easyupscaler.models.registry import ModelEntry, ModelRegistry
from easyupscaler.upscaling.service import UpscaleService


class FakeBackend:
    scale = 2

    def upscale(self, image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        return np.ones((height * self.scale, width * self.scale, 3), dtype=np.float32)


def _seed_registry(registry: ModelRegistry, models_dir: Path) -> None:
    weight_path = models_dir / "demo.pth"
    weight_path.write_bytes(b"weights")
    registry.add(
        ModelEntry(
            name="demo",
            filename="demo.pth",
            path=weight_path,
            scale=2,
            imported_at=datetime(2025, 6, 15, tzinfo=UTC),
        )
    )


def test_run_scale_one_preserves_output_dimensions(isolated_paths, tmp_path: Path) -> None:
    models_dir = isolated_paths / "data" / "easyupscaler" / "models"
    models_dir.mkdir(parents=True)
    registry = ModelRegistry()
    weight_path = models_dir / "1xDetail.pth"
    weight_path.write_bytes(b"weights")
    registry.add(
        ModelEntry(
            name="1xDetail",
            filename="1xDetail.pth",
            path=weight_path,
            scale=1,
            imported_at=datetime(2025, 6, 15, tzinfo=UTC),
        )
    )
    ConfigService().set_default_model("1xDetail")

    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (8, 6), color=(128, 128, 128)).save(input_path, format="JPEG")

    class ScaleOneBackend:
        scale = 1

        def upscale(self, image: np.ndarray) -> np.ndarray:
            return image.copy()

    service = UpscaleService(registry=registry, backend_factory=lambda _: ScaleOneBackend())
    results = service.run([input_path], None)
    assert results[0].error is None
    assert results[0].output is not None
    with Image.open(results[0].output) as output_image:
        assert output_image.size == (8, 6)


def test_run_all_succeed(isolated_paths, tmp_path: Path) -> None:
    models_dir = isolated_paths / "data" / "easyupscaler" / "models"
    models_dir.mkdir(parents=True)
    registry = ModelRegistry()
    _seed_registry(registry, models_dir)
    ConfigService().set_default_model("demo")

    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(input_path, format="JPEG")

    service = UpscaleService(registry=registry, backend_factory=lambda _: FakeBackend())
    results = service.run([input_path], None)
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].output is not None


def test_missing_default_fails_before_backend(isolated_paths, tmp_path: Path) -> None:
    service = UpscaleService(registry=ModelRegistry(), backend_factory=lambda _: FakeBackend())
    with pytest.raises(ValueError, match="no default model set"):
        service.run([tmp_path / "input.jpg"], None)


def test_missing_model_fails_before_backend(isolated_paths, tmp_path: Path) -> None:
    service = UpscaleService(registry=ModelRegistry(), backend_factory=lambda _: FakeBackend())
    with pytest.raises(ValueError, match="not found"):
        service.run([tmp_path / "input.jpg"], "missing")


def test_batch_continues_on_failure(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = isolated_paths / "data" / "easyupscaler" / "models"
    models_dir.mkdir(parents=True)
    registry = ModelRegistry()
    _seed_registry(registry, models_dir)
    ConfigService().set_default_model("demo")

    good = tmp_path / "good.jpg"
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(good, format="JPEG")
    bad = tmp_path / "missing.jpg"

    service = UpscaleService(registry=registry, backend_factory=lambda _: FakeBackend())
    results = service.run([bad, good], None)
    assert results[0].error == "file not found"
    assert results[1].error is None


def test_corrupt_image_records_error(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = isolated_paths / "data" / "easyupscaler" / "models"
    models_dir.mkdir(parents=True)
    registry = ModelRegistry()
    _seed_registry(registry, models_dir)
    ConfigService().set_default_model("demo")

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"bad")

    def failing_read(path: Path) -> np.ndarray:
        raise ImageReadError("cannot read image")

    image_io = ImageIO()
    monkeypatch.setattr(image_io, "read", failing_read)

    service = UpscaleService(
        registry=registry,
        image_io=image_io,
        backend_factory=lambda _: FakeBackend(),
    )
    results = service.run([corrupt], None)
    assert results[0].error == "cannot read image"
