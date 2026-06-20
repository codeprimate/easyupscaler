# ADR-017: scikit-image as hard runtime dependency for document mode

**Status:** Accepted  
**Date:** 2026-06-20

## Context

The `document` denoise mode applies Sauvola adaptive thresholding as a post-processing step after two AI passes. Sauvola requires local window statistics over grayscale pixels — a capability not available in Pillow alone. The algorithm is defined in [specification-document-mode.md](../specification-document-mode.md) §2.6–2.7.

Stakeholder decision: **scikit-image is a required core dependency** — no optional extras, no install-time gating for document mode.

## Decision

1. Add `scikit-image>=0.22` to `[project] dependencies` in `pyproject.toml`.
2. Import `skimage.filters.threshold_sauvola` lazily inside `enhance_document_contrast` (not at module top level), consistent with [ADR-008](./008-lazy-torch-imports.md) spirit for heavy scientific deps.
3. Document mode is the sole caller of scikit-image in the codebase.

No `[project.optional-dependencies]` section for document mode. No runtime error instructing users to install an extra.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Optional `[document]` extra | Install friction; inconsistent with pillow-heif decision in [ADR-015](./015-heic-pillow-heif.md) |
| `doxapy` (C++ document binarization binding) | Niche ecosystem fit; less discoverable and maintained than scikit-image |
| Pure PIL fixed-threshold Otsu | No adaptive local thresholding; poor on camera captures with uneven lighting |

## Consequences

**Positive**

- Document mode works out of the box after `pip install easyupscaler`
- Single install path; no feature-gated extras
- scikit-image is a standard scientific Python package with stable Sauvola API

**Negative**

- Adds `scipy` and `imageio` as transitive dependencies for all installs
- Slightly larger install footprint for users who never use document mode

**Follow-up**

- None for MVP
