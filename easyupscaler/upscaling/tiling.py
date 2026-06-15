import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from spandrel import ImageModelDescriptor

DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 32
MIN_TILE_SIZE = 128

OOM_ERROR_MARKERS = ("out of memory", "mps backend out of memory", "alloc")


def tiled_upscale(
    model: "ImageModelDescriptor",
    tensor: "torch.Tensor",
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> "torch.Tensor":
    """Run tiled super-resolution on a (1, C, H, W) tensor in [0, 1]."""
    from spandrel import ModelTiling

    _, _, height, width = tensor.shape
    tiling_mode = model.tiling

    if tiling_mode == ModelTiling.INTERNAL or (height <= tile_size and width <= tile_size):
        return _forward_with_oom_retry(model, tensor, tile_size, overlap, allow_tiling=False)

    if tiling_mode == ModelTiling.DISCOURAGED:
        try:
            return _forward_with_oom_retry(model, tensor, tile_size, overlap, allow_tiling=False)
        except RuntimeError:
            print(
                "Warning: tiling a model marked DISCOURAGED due to OOM; seams may be visible.",
                file=sys.stderr,
            )
            return _forward_with_oom_retry(model, tensor, tile_size, overlap, allow_tiling=True)

    return _forward_with_oom_retry(model, tensor, tile_size, overlap, allow_tiling=True)


def _forward_with_oom_retry(
    model: "ImageModelDescriptor",
    tensor: "torch.Tensor",
    tile_size: int,
    overlap: int,
    *,
    allow_tiling: bool,
) -> "torch.Tensor":
    current_tile_size = tile_size
    while current_tile_size >= MIN_TILE_SIZE:
        try:
            return _run_inference(
                model,
                tensor,
                current_tile_size,
                overlap,
                allow_tiling=allow_tiling,
            )
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            next_tile_size = current_tile_size // 2
            if next_tile_size < MIN_TILE_SIZE:
                msg = (
                    f"out of memory during upscaling at minimum tile size ({MIN_TILE_SIZE}); "
                    f"try a smaller image or use CPU"
                )
                raise RuntimeError(msg) from exc
            print(
                f"Warning: out of memory at tile size {current_tile_size}; "
                f"retrying with tile size {next_tile_size}.",
                file=sys.stderr,
            )
            current_tile_size = next_tile_size

    msg = f"out of memory during upscaling at minimum tile size ({MIN_TILE_SIZE})"
    raise RuntimeError(msg)


def _run_inference(
    model: "ImageModelDescriptor",
    tensor: "torch.Tensor",
    tile_size: int,
    overlap: int,
    *,
    allow_tiling: bool,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    _, _, height, width = tensor.shape
    if not allow_tiling or (height <= tile_size and width <= tile_size):
        with torch.inference_mode():
            return model(tensor)

    scale = model.scale
    output_channels = tensor.shape[1]
    output = torch.zeros(
        (1, output_channels, height * scale, width * scale),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    weights = torch.zeros_like(output)

    stride = tile_size - overlap
    for top in _tile_starts(height, tile_size, stride):
        for left in _tile_starts(width, tile_size, stride):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = tensor[:, :, top:bottom, left:right]
            with torch.inference_mode():
                upscaled_tile = model(tile)

            tile_h = bottom - top
            tile_w = right - left
            mask = _feather_mask(tile_h, tile_w, overlap, tensor.device, tensor.dtype)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask,
                size=(tile_h * scale, tile_w * scale),
                mode="bilinear",
                align_corners=False,
            )

            out_top = top * scale
            out_left = left * scale
            out_bottom = out_top + tile_h * scale
            out_right = out_left + tile_w * scale

            region = output[:, :, out_top:out_bottom, out_left:out_right]
            weight_region = weights[:, :, out_top:out_bottom, out_left:out_right]
            region.add_(upscaled_tile * mask)
            weight_region.add_(mask)

    weights = weights.clamp_min(1e-8)
    return output / weights


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _feather_mask(
    height: int,
    width: int,
    overlap: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    import torch

    mask = torch.ones((height, width), device=device, dtype=dtype)
    if overlap <= 0:
        return mask

    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    effective_overlap = min(overlap, height // 2, width // 2)
    if effective_overlap == 0:
        return mask

    ramp = ramp[:effective_overlap]
    for index in range(effective_overlap):
        factor = ramp[index]
        mask[index, :] *= factor
        mask[-(index + 1), :] *= factor
        mask[:, index] *= factor
        mask[:, -(index + 1)] *= factor
    return mask


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in OOM_ERROR_MARKERS)
