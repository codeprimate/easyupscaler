# ADR-020: Document mode post-processing — morph cleanup, edge-only anti-alias, flat snap

**Status:** Accepted  
**Date:** 2026-06-20  
**Extends:** [ADR-019](./019-document-binarize-antialias.md)

## Context

ADR-019 adopted Sauvola binarization followed by Gaussian anti-aliasing on the full binary mask. Camera-capture testing (`document-IMG_8202.HEIC`) showed clear, readable text but mid-gray mottling in large flat paper regions — speckle not correlated with stroke edges. Applying Gaussian blur to the entire mask propagates isolated binary noise into flat areas as gray patches.

Stakeholder confirmed overall text quality was good; the remaining issue was flat-region speckle only. Aggressive morph/snap tuning (5×5 kernel, tighter snap thresholds) was tried and reverted — it removed speckle but degraded stroke quality.

## Decision

After Sauvola binarization, apply three CPU post-processing steps before writing PNG:

1. **Morphological open/close** (`3×3` kernel) — removes isolated binary speckle from the mask.
2. **Edge-only anti-alias** — Gaussian blur (`σ` per `--strength`) applies only within `DOCUMENT_EDGE_BAND_WIDTH` (2.0 px) of a stroke boundary, measured by distance transform. Flat ink and paper regions keep hard `0.0` / `1.0`.
3. **Flat-region snap** — values ≤ `DOCUMENT_FLAT_INK_SNAP` (0.25) → pure ink; values ≥ `DOCUMENT_FLAT_PAPER_SNAP` (0.75) → pure paper. Preserves gray only in the edge band.

Implementation: `easyupscaler/denoise/document_enhance.py` (`enhance_document_contrast`). Constants: `easyupscaler/denoise/document_constants.py`. Algorithm contract: [specification-document-mode.md](../specification-document-mode.md) §2.6.

`scipy.ndimage` provides morphology, distance transform, and Gaussian filter (transitive dep via scikit-image per ADR-017).

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Full-mask Gaussian (ADR-019 as written) | Mid-gray mottling in flat paper blocks on camera captures |
| Larger morph kernel + tighter snap (5×5, 0.70/0.30) | Removed speckle but stakeholder reported overall quality loss; reverted |
| Sauvola sigmoid stretch (pre-ADR-019) | Did not match 1-bit → anti-alias workflow; Book Compact path had already failed |

## Consequences

**Positive**

- Flat paper and ink regions stay clean; gray values concentrated at stroke edges
- Text legibility preserved on iPhone HEIC test input
- Tunable via named constants without changing AI pass

**Negative**

- Slightly more complex post-processing than ADR-019's full-mask blur
- Edge band width and snap thresholds require per-input tuning if quality regresses

**Follow-up**

- None for MVP
