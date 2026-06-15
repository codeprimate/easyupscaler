from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from spandrel import ModelTiling

from easyupscaler.upscaling.backends.spandrel_backend import SpandrelBackend


@dataclass
class FakeLoadedModel:
    scale: int = 4
    tiling: ModelTiling = ModelTiling.SUPPORTED
    calls: int = field(default=0, init=False)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        _, channels, height, width = tensor.shape
        return torch.full(
            (1, channels, height * self.scale, width * self.scale),
            0.75,
            dtype=tensor.dtype,
        )


@pytest.fixture
def weights_path(tmp_path: Path) -> Path:
    path = tmp_path / "RealESRGAN_x4plus.pth"
    path.write_bytes(b"weights")
    return path


def _build_backend(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_model: FakeLoadedModel | None = None,
    mps_available: bool = False,
) -> tuple[SpandrelBackend, FakeLoadedModel]:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    model = fake_model or FakeLoadedModel()
    with patch("spandrel_extra_arches.install") as install_mock:
        with patch("spandrel.ModelLoader") as loader_cls:
            loader_cls.return_value.load_from_file.return_value = model
            backend = SpandrelBackend(weights_path)
    install_mock.assert_called_once()
    loader_cls.return_value.load_from_file.assert_called_once_with(weights_path)
    return backend, model


def test_init_sets_scale_from_loaded_model(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _ = _build_backend(weights_path, monkeypatch, fake_model=FakeLoadedModel(scale=2))
    assert backend.scale == 2


def test_select_device_warns_when_mps_unavailable(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _build_backend(weights_path, monkeypatch, mps_available=False)
    captured = capsys.readouterr()
    assert "Warning: MPS not available, using CPU" in captured.err


def test_select_device_uses_mps_when_available(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with patch("spandrel_extra_arches.install"):
        with patch("spandrel.ModelLoader") as loader_cls:
            loader_cls.return_value.load_from_file.return_value = FakeLoadedModel()
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            backend = SpandrelBackend(weights_path)
    assert backend._device.type == "mps"


def test_upscale_returns_scaled_clipped_ndarray(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, model = _build_backend(weights_path, monkeypatch, fake_model=FakeLoadedModel(scale=2))
    image = np.ones((8, 8, 3), dtype=np.float32) * 0.5

    output = backend.upscale(image)

    assert output.shape == (16, 16, 3)
    assert output.dtype == np.float32
    assert output.max() <= 1.0
    assert model.calls == 1


def test_upscale_clips_values_above_one(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _ = _build_backend(weights_path, monkeypatch)

    def bright_tiled_upscale(model, tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = tensor.shape
        return torch.full((1, channels, height * 4, width * 4), 1.5)

    monkeypatch.setattr(
        "easyupscaler.upscaling.backends.spandrel_backend.tiled_upscale",
        bright_tiled_upscale,
    )
    image = np.ones((4, 4, 3), dtype=np.float32)

    output = backend.upscale(image)

    assert output.max() == pytest.approx(1.0)


def _build_mps_backend_for_retry_tests(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SpandrelBackend, MagicMock]:
    loader_cls = MagicMock()
    loader_cls.return_value.load_from_file.return_value = FakeLoadedModel()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr("spandrel_extra_arches.install", lambda: None)
    monkeypatch.setattr("spandrel.ModelLoader", loader_cls)

    backend = SpandrelBackend(weights_path)
    backend._device = torch.device("mps")

    def cpu_ndarray_to_tensor(image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    monkeypatch.setattr(backend, "_ndarray_to_tensor", cpu_ndarray_to_tensor)
    return backend, loader_cls


def test_upscale_retries_on_mps_runtime_error(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempts = {"count": 0}

    def flaky_tiled_upscale(model, tensor: torch.Tensor) -> torch.Tensor:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("MPS backend out of memory")
        _, channels, height, width = tensor.shape
        return torch.ones((1, channels, height * 4, width * 4), dtype=torch.float32)

    monkeypatch.setattr(
        "easyupscaler.upscaling.backends.spandrel_backend.tiled_upscale",
        flaky_tiled_upscale,
    )
    backend, loader_cls = _build_mps_backend_for_retry_tests(weights_path, monkeypatch)

    image = np.ones((4, 4, 3), dtype=np.float32)
    output = backend.upscale(image)

    assert output.shape == (16, 16, 3)
    assert attempts["count"] == 2
    assert loader_cls.return_value.load_from_file.call_count == 2
    assert backend._device.type == "cpu"
    captured = capsys.readouterr()
    assert f"Warning: MPS error on {weights_path.name}" in captured.err


def test_upscale_reraises_non_mps_runtime_error(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_tiled_upscale(model, tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        "easyupscaler.upscaling.backends.spandrel_backend.tiled_upscale",
        failing_tiled_upscale,
    )
    backend, _ = _build_mps_backend_for_retry_tests(weights_path, monkeypatch)

    image = np.ones((4, 4, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        backend.upscale(image)


def test_upscale_retries_on_not_implemented_error(
    weights_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}

    def flaky_tiled_upscale(model, tensor: torch.Tensor) -> torch.Tensor:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("not implemented on this device")
        _, channels, height, width = tensor.shape
        return torch.ones((1, channels, height * 4, width * 4), dtype=torch.float32)

    monkeypatch.setattr(
        "easyupscaler.upscaling.backends.spandrel_backend.tiled_upscale",
        flaky_tiled_upscale,
    )
    backend, _ = _build_mps_backend_for_retry_tests(weights_path, monkeypatch)

    output = backend.upscale(np.ones((4, 4, 3), dtype=np.float32))

    assert output.shape == (16, 16, 3)
    assert attempts["count"] == 2
