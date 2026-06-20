# ADR-019: Document mode — single AI pass, Sauvola binarize, anti-aliased grayscale

**Status:** Accepted  
**Date:** 2026-06-20  
**Supersedes:** [ADR-018](./018-document-two-pass-pipeline.md)

## Context

ADR-018 routed document mode through Archiver Medium → Book Compact → Sauvola sigmoid contrast. Testing on iPhone HEIC camera captures showed Book Compact washed ~96% of pixels to white and erased faint text before post-processing. Sauvola sigmoid could not recover detail already removed by the second AI pass.

Stakeholder direction: binarize to 1-bit first, then emit smooth grayscale via anti-aliasing — not a hard binary PNG, but not the prior sigmoid stretch either.

## Decision

### AI pass

Document mode runs **one** AI pass only:

| Strength | AI pass | Post-processing |
|----------|---------|-----------------|
| `low` | Archiver Medium | Sauvola binarize (window 75) → Gaussian anti-alias (σ 1.5) |
| `high` | Archiver Medium | Sauvola binarize (window 25) → Gaussian anti-alias (σ 0.75) |

`--strength` controls Sauvola window size and anti-alias sigma. No Book Compact pass.

### Post-processing algorithm

1. Convert Archiver RGB output to grayscale (`uint8`).
2. Compute per-pixel Sauvola threshold.
3. Binarize: `1.0` = background (pixel ≥ threshold), `0.0` = text (pixel < threshold).
4. Apply `scipy.ndimage.gaussian_filter` to the binary mask.
5. Write as grayscale PNG (float `[0, 1]` → `uint8` via existing `write_png`).

Output preserves smooth strokes at edges (anti-aliased gray) while keeping high ink-to-paper separation in flat regions.

### Book Compact

`book_compact` remains in the managed catalog and backend code for potential future use. It is **not** invoked by document mode.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Keep ADR-018 two-pass pipeline | Book Compact destroys text on camera captures; confirmed by histogram analysis |
| Book Compact on `--strength high` only | Still fails on HEIC test input after Archiver; anti-alias cannot restore lost ink |
| Sauvola sigmoid only (no binarize) | Stakeholder chose explicit 1-bit → anti-alias workflow |
| PIL-only anti-alias | scipy already present via scikit-image; Gaussian filter is standard |

## Consequences

**Positive**

- Camera photos and flatbed scans share one reliable pipeline
- Text preserved through Archiver + adaptive threshold
- Smooth grayscale output suitable for screen and OCR

**Negative**

- Book Compact weight no longer downloaded for document mode
- ADR-018 superseded; docs and tests updated

**Follow-up**

- None for MVP
