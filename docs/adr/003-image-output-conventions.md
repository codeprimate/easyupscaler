# ADR-003: Always JPEG output with `-upscaled` naming

**Status:** Accepted  
**Date:** 2025-06-15

## Context

Upscaled outputs need a predictable path and format. Options:

| Approach | Example |
|----------|---------|
| Preserve input format | `foo.png` → `foo-upscaled.png` |
| Always JPEG | `foo.png` → `foo-upscaled.jpg` |
| Configurable via flag | Default preserve; `--format jpeg` |

Mission examples consistently show JPEG output (`input-upscaled.jpg`). PNG upscales are often saved as JPEG in practice to reduce file size.

Pillow JPEG defaults use chroma subsampling (4:2:0), which softens fine detail in upscaled output. PNG inputs with alpha must be converted before JPEG write.

## Decision

**Always write JPEG** for upscaled output.

- Output path: same directory as input
- Filename: `{input_stem}-upscaled.jpg`
- Examples: `photo.png` → `photo-upscaled.jpg`; `scan.jpeg` → `scan-upscaled.jpg`
- If `{stem}-upscaled.jpg` already exists, **overwrite without prompt** (MVP)

### JPEG encoding defaults

```python
image.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)
```

- `quality=95` — high quality without bloating files
- `subsampling=0` (4:4:4) — preserve detail from upscaling
- Convert RGBA/grayscale PNG to RGB before save

Input formats supported for reading: PNG and JPEG.

## Consequences

**Positive**

- Matches mission examples and user expectations from docs
- Smaller output files than PNG for photographic content
- Subsampling=0 avoids unnecessary blur on edges and text

**Negative**

- PNG inputs lose alpha channel and lossless encoding
- Re-running upscale silently replaces prior output

**Follow-up**

- Add `--format` or `--output` flag in a future version if users need PNG or custom paths
