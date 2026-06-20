from typing import Protocol

import numpy as np


class DenoiseBackend(Protocol):
    def denoise(self, image: np.ndarray) -> np.ndarray: ...


class FBCNNDenoiseBackend(Protocol):
    def denoise(self, image: np.ndarray, *, qf_override: float | None = None) -> np.ndarray: ...
