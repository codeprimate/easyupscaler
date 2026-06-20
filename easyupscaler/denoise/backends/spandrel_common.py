import sys
from pathlib import Path

import numpy as np
import torch

from easyupscaler.upscaling.tiling import stream_tiled_ndarray

_extra_arches_installed = False


def _ensure_extra_arches() -> None:
    global _extra_arches_installed
    if _extra_arches_installed:
        return
    import spandrel_extra_arches

    spandrel_extra_arches.install()
    _extra_arches_installed = True


class SpandrelDenoiseBackend:
    """Shared Spandrel load/device/MPS logic for 1× denoise models."""

    def __init__(self, weights_path: Path) -> None:
        _ensure_extra_arches()

        self._weights_path = weights_path
        self._device = self._select_device()
        self._model = self._load_model(self._device)

    def _forward(self, image: np.ndarray) -> np.ndarray:
        def forward_tile(tile: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return self._model(tile.to(self._device))

        try:
            return stream_tiled_ndarray(
                np.asarray(image, dtype=np.float32),
                forward_tile,
                self._device,
            )
        except RuntimeError as exc:
            if self._device.type != "mps" or not self._is_mps_error(exc):
                raise
            print(
                f"Warning: MPS error on {self._weights_path.name}: {exc}. Retrying on CPU.",
                file=sys.stderr,
            )
            self._retry_on_cpu()
            return stream_tiled_ndarray(
                np.asarray(image, dtype=np.float32),
                forward_tile,
                self._device,
            )

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return self._forward(image)

    def _retry_on_cpu(self) -> None:
        cpu_device = torch.device("cpu")
        self._device = cpu_device
        self._model = self._load_model(cpu_device)

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print(
            "Warning: MPS not available, using CPU (inference will be slower).",
            file=sys.stderr,
        )
        return torch.device("cpu")

    def _load_model(self, device: torch.device):
        from spandrel import ModelLoader

        loader = ModelLoader(device=device)
        return loader.load_from_file(self._weights_path)

    def _is_mps_error(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "mps" in message or "not implemented" in message

    @property
    def model(self):
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device
