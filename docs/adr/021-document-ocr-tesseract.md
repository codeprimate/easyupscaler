# ADR-021: Default Tesseract OCR for document denoise mode

**Status:** Accepted  
**Date:** 2026-06-20  
**Extends:** [ADR-019](./019-document-binarize-antialias.md), [ADR-020](./020-document-postprocessing-refinements.md)

## Context

Document denoise mode produces high-contrast grayscale PNG optimized for readability and OCR input ([specification-document-mode.md](../specification-document-mode.md)). Users processing scans and photos of printed text often want searchable plain text alongside the cleaned image.

Stakeholder decision: **default-on OCR** with `--no-text` to opt out. Missing or failing OCR must not fail the denoise job when the PNG is written successfully (Option B soft default).

## Decision

1. After `enhance_document_contrast`, run Tesseract OCR on the in-memory grayscale array (no PNG re-read).
2. Write UTF-8 plain text to `{input_stem}.txt` beside the input (or under `--output`), with indexed conflict resolution independent of the PNG path ([ADR-011](./011-output-conflict-indexing.md) pattern: `scan.txt`, then `scan-0001.txt`, …).
3. Add `pytesseract>=0.3` as a hard Python dependency. The system `tesseract` binary must be on PATH at runtime (not pip-installable).
4. OCR settings: English (`eng`); no explicit `--psm` (Tesseract default).
5. `--no-text` on `denoise` skips OCR (no-op outside `document` mode).
6. Failure semantics:
   - Tesseract not found: warn **once per batch** on stderr; PNG succeeds; file counts as success.
   - OCR runtime failure: warn **per file** on stderr; PNG succeeds; no `.txt`; file counts as success.
   - Empty OCR result: write empty `.txt` (predictable for scripts).

Implementation: `easyupscaler/denoise/document_ocr.py`, `ImageIO.write_txt()`, document branch in `DenoiseService._process_path`. `DenoiseResult.text_output` holds the text path when written.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Opt-in `--text` flag | Stakeholder wants default-on with `--no-text` opt-out |
| Strict default (fail without Tesseract) | Breaks image-only workflows; harsh for single-user rollout |
| `{stem}-denoised.txt` paired with PNG name | Stakeholder chose input-stem naming (`scan.txt`) |
| subprocess only (no pytesseract) | Stakeholder chose pytesseract for PIL/numpy integration |
| Optional `[ocr]` extra | Inconsistent with document-mode hard deps (ADR-017, ADR-015) |

## Consequences

**Positive**

- Document mode delivers image + text in one command
- PNG remains the primary artifact; OCR failures are non-fatal
- Injectable OCR for fast tests without system Tesseract

**Negative**

- Requires system `tesseract` install for OCR (documented in README)
- Adds `pytesseract` to all installs; handwriting OCR quality poor (README caveat)
- TXT and PNG paths resolve independently — pairing is by input file, not shared index

**Follow-up**

- Optional `--ocr-lang`, `--psm` flags if users need non-English or layout tuning
