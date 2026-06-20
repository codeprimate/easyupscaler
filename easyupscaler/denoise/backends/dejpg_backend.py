from pathlib import Path

from easyupscaler.denoise.backends.spandrel_common import SpandrelDenoiseBackend


class DeJPGBackend(SpandrelDenoiseBackend):
    def __init__(self, weights_path: Path) -> None:
        super().__init__(weights_path)
