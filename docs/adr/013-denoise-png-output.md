# ADR-013: PNG output for denoise command

**Status:** Accepted  
**Date:** 2026-06-20

## Context

[ADR-003](./003-image-output-conventions.md) defines JPEG output with `-upscaled` suffix for the scale (upscale) command. Denoise operates at 1× resolution and benefits from lossless output to preserve cleaned detail without JPEG recompression artifacts.

[ADR-011](./011-output-conflict-indexing.md) defines indexed filenames when the base output path already exists.

## Decision

Denoise output is always **PNG** at full input resolution (1×):

| Case | Output path |
|------|-------------|
| First run | `{stem}-denoised.png` |
| Conflict | `{stem}-denoised-0001.png`, then `-0002`, … |

Resolution uses the same **lowest available 4-digit index** algorithm as ADR-011, with `-denoised` instead of `-upscaled`. **Never overwrite** an existing output file.

The scale command retains ADR-003 JPEG `-upscaled` output unchanged.

Grayscale manga inputs may be written as grayscale PNG (`mode="L"`) when colorspace preservation applies (see denoise spec §2.4).

## Consequences

**Positive**

- Lossless denoise output avoids generation loss at 1×
- Conflict indexing matches upscale UX from ADR-011

**Negative**

- Larger output files than JPEG for photographic content
- Two output conventions in one tool (JPEG upscale, PNG denoise)

**Follow-up**

- None for MVP
