from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from easyupscaler.denoise.backends.fbcnn_backend import FBCNN_HIGH_STRENGTH_QF, FBCNNBackend
from easyupscaler.denoise.backends.spandrel_common import SpandrelDenoiseBackend


def test_fbcnn_auto_qf_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "fbcnn.pth"
    weights.write_bytes(b"x")

    backend = FBCNNBackend.__new__(FBCNNBackend)
    backend._weights_path = weights
    backend._device = torch.device("cpu")
    mock_model = MagicMock()
    mock_model.model.return_value = (torch.zeros(1, 3, 4, 4), torch.tensor([[0.5]]))
    backend._model = mock_model

    def fake_stream(image, forward_tile, device, **kwargs):
        forward_tile(torch.zeros(1, 3, 4, 4))
        return np.zeros((4, 4, 3), dtype=np.float32)

    monkeypatch.setattr(
        "easyupscaler.denoise.backends.fbcnn_backend.stream_tiled_ndarray",
        fake_stream,
    )

    output = backend.denoise(np.zeros((4, 4, 3), dtype=np.float32))
    assert output.shape == (4, 4, 3)
    mock_model.model.assert_called_once()


def test_fbcnn_qf_override_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "fbcnn.pth"
    weights.write_bytes(b"x")

    backend = FBCNNBackend.__new__(FBCNNBackend)
    backend._weights_path = weights
    backend._device = torch.device("cpu")
    mock_model = MagicMock()
    mock_model.model.return_value = (torch.zeros(1, 3, 4, 4), torch.tensor([[0.5]]))
    backend._model = mock_model

    def fake_stream(image, forward_tile, device, **kwargs):
        forward_tile(torch.zeros(1, 3, 4, 4))
        return np.zeros((4, 4, 3), dtype=np.float32)

    monkeypatch.setattr(
        "easyupscaler.denoise.backends.fbcnn_backend.stream_tiled_ndarray",
        fake_stream,
    )

    backend.denoise(np.zeros((4, 4, 3), dtype=np.float32), qf_override=FBCNN_HIGH_STRENGTH_QF)
    args, _kwargs = mock_model.model.call_args
    qf_input = args[1]
    expected = 1.0 - FBCNN_HIGH_STRENGTH_QF / 100
    assert float(qf_input.item()) == pytest.approx(expected)


def test_spandrel_denoise_backend_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")

    mock_descriptor = MagicMock()
    mock_descriptor.scale = 1
    mock_descriptor.return_value = torch.ones(1, 3, 8, 8)

    monkeypatch.setattr(
        "easyupscaler.denoise.backends.spandrel_common.stream_tiled_ndarray",
        lambda image, forward, device, **kwargs: np.ones(image.shape, dtype=np.float32) * 0.5,
    )

    with patch.object(SpandrelDenoiseBackend, "_load_model", return_value=mock_descriptor):
        with patch.object(
            SpandrelDenoiseBackend,
            "_select_device",
            return_value=torch.device("cpu"),
        ):
            with patch("spandrel_extra_arches.install"):
                backend = SpandrelDenoiseBackend(weights)

    image = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    output = backend.denoise(image)
    assert output.shape == (8, 8, 3)
