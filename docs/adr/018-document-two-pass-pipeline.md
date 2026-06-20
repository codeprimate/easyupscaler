# ADR-018: Two-pass AI pipeline for document denoise mode

**Status:** Superseded by [ADR-019](./019-document-binarize-antialias.md)  
**Date:** 2026-06-20

## Context

Document scans and camera photos of text combine general artifact types (film grain, compression noise) with document-specific types (yellowing, color bleed from adjacent pages, ink fading, mould stains). No single community super-resolution/denoise model covers both reliably.

The `document` mode spec ([specification-document-mode.md](../specification-document-mode.md) §2.3–2.5) requires a two-pass AI pipeline followed by Sauvola-guided contrast enhancement. `--strength` controls only the Sauvola parameters, not the model passes.

## Decision

### Document mode pipeline

| Strength | Passes (in order) |
|----------|-------------------|
| `low` | Archiver Medium → Book Compact → Sauvola (low params) |
| `high` | Archiver Medium → Book Compact → Sauvola (high params) |

Both AI passes always run regardless of strength or input format (including HEIC). HEIC does **not** trigger an additional FBCNN pass — unlike photo mode ([ADR-014](./014-heic-two-pass-denoise.md)).

### Model roles

- **Archiver Medium** (`archivist_medium`): General restoration — removes grain, scan film, and compression artifacts at the global level. Already in the managed catalog.
- **Book Compact** (`book_compact`): Compact/SRVGGNet architecture (~2.3 MB), purpose-trained for book and document scan cleanup — color bleed, stains, fine hairlines. Loaded via Spandrel like other denoise backends.

Inference runs on RGB throughout both passes. Grayscale conversion and Sauvola enhancement happen after the second pass.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Archiver Medium only | Not document-trained; leaves color bleed and document-specific stains |
| Book Compact only | May leave general sensor/compression artifacts |
| DocRes (CVPR 2024 SOTA) | Custom `.pkl` weights in non-Spandrel architecture; not loadable via existing backend as of 2026-06-20 |

## Consequences

**Positive**

- Covers both general and document-specific artifact types
- Book Compact is small and fast on MPS/CPU
- Reuses existing Spandrel backend infrastructure

**Negative**

- Document mode runs are slower than single-pass modes (two AI passes + Sauvola)
- Downloads two model weights on first invocation (~Archiver + ~2.3 MB Book Compact)

**Follow-up**

- None for MVP
