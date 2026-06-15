# ADR-011: Indexed output filenames on conflict

**Status:** Accepted  
**Date:** 2025-06-15

## Context

[ADR-003](./003-image-output-conventions.md) chose silent overwrite when `{stem}-upscaled.jpg` already exists. Re-running upscale on the same input destroys the prior result with no recovery path.

Users may want to compare outputs from different models or re-runs without losing earlier files.

## Decision

When `{stem}-upscaled.jpg` already exists, append a **4-digit zero-padded index** before the extension instead of overwriting:

| Existing files | Next output |
|----------------|-------------|
| (none) | `photo-upscaled.jpg` |
| `photo-upscaled.jpg` | `photo-upscaled-0001.jpg` |
| `photo-upscaled.jpg`, `photo-upscaled-0001.jpg` | `photo-upscaled-0002.jpg` |

Resolution picks the **lowest available index** starting at `0001`. If indices `0001`–`9999` are all taken, write fails with an OS error.

Base naming, JPEG encoding, and output directory are unchanged from ADR-003.

## Consequences

**Positive**

- Re-runs and batch overlaps preserve prior outputs
- First upscale keeps the simple `{stem}-upscaled.jpg` name users expect

**Negative**

- Same directory can accumulate many indexed files over repeated runs
- Supersedes the overwrite clause in ADR-003 (ADR-003 remains accepted for format and base naming)

**Follow-up**

- Optional `--force` overwrite flag if users want the old behavior
