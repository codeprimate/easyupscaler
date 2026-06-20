# ADR-014: Two-pass denoise pipeline for HEIC photo inputs

**Status:** Accepted  
**Date:** 2026-06-20

## Context

iPhone HEIC files carry sensor noise (ISO grain) and potential compression artifacts from the HEIC encode. A single denoise model cannot optimally address both in one pass. SCUNet targets real-world sensor noise; FBCNN targets JPEG/codec compression artifacts with an optional quality-factor override.

Art and manga modes treat HEIC as any other format (single pass). Only **photo** mode applies the two-pass pipeline for HEIC inputs.

## Decision

### Photo mode pipeline

| Strength | Input | Passes (in order) |
|----------|-------|-------------------|
| `low` | non-HEIC | SCUNet PSNR |
| `high` | non-HEIC | SCUNet GAN |
| `low` | HEIC | SCUNet PSNR → FBCNN (auto QF) |
| `high` | HEIC | SCUNet GAN → FBCNN (QF override) |

Both HEIC passes always run regardless of strength. Strength selects the SCUNet variant and FBCNN aggressiveness.

### FBCNN quality factor

- **`--strength low`:** auto QF via `descriptor.model(image_tensor)` (blind estimate).
- **`--strength high`:** QF override constant `FBCNN_HIGH_STRENGTH_QF = 20`, passed as `qf_input = torch.tensor([[1.0 - 20 / 100]])`.

FBCNN forward bypasses the Spandrel descriptor wrapper only for the QF parameter; all other models use standard `descriptor(image_tensor)`.

### Art / manga HEIC

Single pass only — same model selection as non-HEIC art/manga. No FBCNN second pass.

## Consequences

**Positive**

- Addresses both noise types in iPhone photos
- Clear separation: photo HEIC is special-cased; art/manga stay simple

**Negative**

- HEIC photo runs are slower and download more weights (~340 MB SCUNet + FBCNN)
- FBCNN requires direct model call, diverging from other backends

**Follow-up**

- None for MVP
