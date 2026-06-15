from typing import Protocol

import numpy as np


class UpscalerBackend(Protocol):
    @property
    def scale(self) -> int: ...

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Accept and return float32 RGB ndarray in [0, 1], shape (H, W, 3)."""
        ...
