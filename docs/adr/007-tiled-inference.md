# ADR-007: Tiled inference for large images

**Status:** Accepted  
**Date:** 2025-06-15

## Context

Spandrel runs a single forward pass per call but does not provide tiling, memory estimation, or OOM recovery. Full-frame upscaling of photos (3000×2000 and larger) exhausts unified memory on Apple Silicon and causes OOM on CPU for heavy models.

ComfyUI and AUTOMATIC1111 both wrap Spandrel with tiled inference:

- ComfyUI: `tiled_scale` with tile=512, overlap=32; halve tile on OOM until minimum 128
- A1111: configurable tile size and overlap; grid-based merge

Spandrel exposes `model.tiling` metadata:

| Value | Meaning |
|-------|---------|
| `SUPPORTED` | External tiling is safe |
| `DISCOURAGED` | Tiling may cause seam artifacts |
| `INTERNAL` | Model tiles internally; do not tile externally |

Spandrel also handles `size_requirements` padding inside `ImageModelDescriptor.__call__`.

## Decision

Implement tiled inference in `easyupscaler.upscaling.tiling` (or within `SpandrelBackend`).

### Defaults

| Constant | Value |
|----------|-------|
| `DEFAULT_TILE_SIZE` | 512 |
| `DEFAULT_TILE_OVERLAP` | 32 |
| `MIN_TILE_SIZE` | 128 |

### Algorithm

1. Convert input to tensor `(1, C, H, W)` in `[0, 1]` on model device/dtype
2. If image fits in a single tile (both dimensions ≤ tile size), run one forward pass
3. Otherwise split into overlapping tiles, run `model(tile)` per tile, merge with overlap blending (weighted average in overlap regions — same approach as ComfyUI/A1111)
4. On OOM or memory error: halve tile size and retry until `MIN_TILE_SIZE` or fail with clear message
5. Honor `model.tiling`:
   - `INTERNAL`: run full frame only (no external tiling); rely on model internals
   - `DISCOURAGED`: run full frame if it fits; if OOM, tile with stderr warning about possible artifacts
   - `SUPPORTED`: tile when image exceeds tile size

### Tensor contract

Follow Spandrel `ImageModelDescriptor.__call__`:

- Input: `(1, input_channels, H, W)`, `[0, 1]`, same device/dtype as model
- Output: `(1, output_channels, H*scale, W*scale)`, `[0, 1]`

Convert between NumPy RGB (ImageIO) and torch tensors in the backend layer.

## Consequences

**Positive**

- Practical upscaling of real-world photo sizes on Apple Silicon
- OOM recovery without user configuration
- Aligns with proven patterns from ComfyUI and A1111

**Negative**

- Tiling adds implementation complexity and test surface
- `DISCOURAGED` models may show seams when forced to tile after OOM
- Tiled runs are slower than single-pass for small images (acceptable)

**Follow-up**

- Expose `--tile-size` and `--tile-overlap` flags
- Auto-tune tile size from available memory
