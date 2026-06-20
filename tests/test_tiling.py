from dataclasses import dataclass

import numpy as np
import pytest
import torch
from spandrel import ModelTiling

from easyupscaler.upscaling.tiling import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    MIN_TILE_SIZE,
    stream_tiled_ndarray,
    tiled_same_size,
    tiled_upscale,
)


@dataclass
class FakeModel:
    scale: int = 2
    tiling: ModelTiling = ModelTiling.SUPPORTED
    purpose: str = "SR"
    oom_attempts: int = 0
    calls: int = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.oom_attempts > 0:
            self.oom_attempts -= 1
            raise RuntimeError("MPS backend out of memory")
        _, channels, height, width = tensor.shape
        return torch.ones((1, channels, height * self.scale, width * self.scale))


def test_small_image_single_pass() -> None:
    model = FakeModel(scale=2)
    tensor = torch.rand(1, 3, 256, 256)
    output = tiled_upscale(model, tensor, tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_TILE_OVERLAP)
    assert output.shape == (1, 3, 512, 512)
    assert model.calls == 1


def test_scale_one_preserves_dimensions() -> None:
    model = FakeModel(scale=1)
    tensor = torch.rand(1, 3, 256, 256)
    output = tiled_upscale(model, tensor, tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_TILE_OVERLAP)
    assert output.shape == (1, 3, 256, 256)
    assert model.calls == 1


def test_large_image_uses_multiple_tiles() -> None:
    model = FakeModel(scale=2)
    tensor = torch.rand(1, 3, 1024, 1024)
    output = tiled_upscale(model, tensor, tile_size=512, overlap=32)
    assert output.shape == (1, 3, 2048, 2048)
    assert model.calls > 1


def test_oom_halves_tile_size_and_retries() -> None:
    model = FakeModel(scale=2, oom_attempts=1)
    tensor = torch.rand(1, 3, 1024, 1024)
    output = tiled_upscale(model, tensor, tile_size=512, overlap=32)
    assert output.shape == (1, 3, 2048, 2048)


def test_oom_at_minimum_tile_size_raises() -> None:
    model = FakeModel(scale=2, oom_attempts=10)
    tensor = torch.rand(1, 3, 1024, 1024)
    with pytest.raises(RuntimeError, match=str(MIN_TILE_SIZE)):
        tiled_upscale(model, tensor, tile_size=MIN_TILE_SIZE, overlap=8)


def test_internal_tiling_skips_external_tiles() -> None:
    model = FakeModel(scale=2, tiling=ModelTiling.INTERNAL)
    tensor = torch.rand(1, 3, 1024, 1024)
    output = tiled_upscale(model, tensor, tile_size=512, overlap=32)
    assert output.shape == (1, 3, 2048, 2048)
    assert model.calls == 1


def test_tiled_same_size_preserves_dimensions() -> None:
    calls = {"count": 0}

    def forward(tile: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return tile

    tensor = torch.rand(1, 3, 1024, 1024)
    output = tiled_same_size(forward, tensor, tile_size=512, overlap=32)
    assert output.shape == tensor.shape
    assert calls["count"] > 1


def test_stream_tiled_ndarray_never_builds_full_device_tensor() -> None:
    calls = {"max_elements": 0}

    def forward_tile(tile: torch.Tensor) -> torch.Tensor:
        calls["max_elements"] = max(calls["max_elements"], tile.numel())
        return tile

    image = np.random.rand(1024, 1024, 3).astype(np.float32)
    output = stream_tiled_ndarray(
        image,
        forward_tile,
        torch.device("cpu"),
        tile_size=512,
        overlap=32,
    )
    assert output.shape == image.shape
    assert calls["max_elements"] < image.size
    assert calls["max_elements"] > 0
