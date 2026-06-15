import sys
from pathlib import Path

import numpy as np
import torch

from easyupscaler.upscaling.tiling import tiled_upscale


class SpandrelBackend:
    def __init__(self, weights_path: Path) -> None:
        import spandrel_extra_arches

        spandrel_extra_arches.install()


        self._weights_path = weights_path
        self._device = self._select_device()
        self._model = self._load_model(self._device)
        self._scale = int(self._model.scale)

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, image: np.ndarray) -> np.ndarray:
        tensor = self._ndarray_to_tensor(image)
        try:
            output = tiled_upscale(self._model, tensor)
        except RuntimeError as exc:
            if self._device.type != "mps" or not self._is_mps_error(exc):
                raise
            print(
                f"Warning: MPS error on {self._weights_path.name}: {exc}. Retrying on CPU.",
                file=sys.stderr,
            )
            cpu_device = torch.device("cpu")
            self._device = cpu_device
            self._model = self._load_model(cpu_device)
            tensor = tensor.to(cpu_device)
            output = tiled_upscale(self._model, tensor)

        return self._tensor_to_ndarray(output)

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

    def _ndarray_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=self._device, dtype=torch.float32)

    def _tensor_to_ndarray(self, tensor: torch.Tensor) -> np.ndarray:
        array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(array, 0.0, 1.0).astype(np.float32)

    def _is_mps_error(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "mps" in message or "not implemented" in message
