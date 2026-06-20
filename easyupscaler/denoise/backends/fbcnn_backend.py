from pathlib import Path

import numpy as np
import torch

from easyupscaler.denoise.backends.spandrel_common import SpandrelDenoiseBackend
from easyupscaler.denoise.memory import FBCNN_TILE_OVERLAP, FBCNN_TILE_SIZE
from easyupscaler.upscaling.tiling import stream_tiled_ndarray

FBCNN_HIGH_STRENGTH_QF = 20


class FBCNNBackend(SpandrelDenoiseBackend):
    def denoise(self, image: np.ndarray, *, qf_override: float | None = None) -> np.ndarray:
        def forward_tile(tile: torch.Tensor) -> torch.Tensor:
            return self._forward_tensor(tile, qf_override=qf_override)

        return stream_tiled_ndarray(
            np.asarray(image, dtype=np.float32),
            forward_tile,
            self._device,
            tile_size=FBCNN_TILE_SIZE,
            overlap=FBCNN_TILE_OVERLAP,
        )

    def _forward_tensor(
        self,
        tensor: torch.Tensor,
        *,
        qf_override: float | None,
    ) -> torch.Tensor:
        if qf_override is None:
            output_tensor, _ = self._model.model(tensor)
        else:
            qf_input = torch.tensor([[1.0 - qf_override / 100]], device=self._device)
            output_tensor, _ = self._model.model(tensor, qf_input)
        return output_tensor

    def __init__(self, weights_path: Path) -> None:
        super().__init__(weights_path)
