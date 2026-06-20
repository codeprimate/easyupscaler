# ADR-023: `--ocrai` writes additive Markdown (Tesseract unchanged)

**Status:** Accepted  
**Date:** 2026-06-20  
**Supersedes:** [ADR-022](./022-opt-in-vlm-ocr-ocrai.md) decision items 1 and 7 (backend selection)

## Context

[ADR-022](./022-opt-in-vlm-ocr-ocrai.md) made `--ocrai` **replace** Tesseract for `.txt` output. Stakeholder revision: Tesseract remains the default plain-text path; `--ocrai` **additionally** writes structured Markdown via the VLM.

## Decision

1. **Tesseract always runs** for `{stem}.txt` when OCR is enabled (default; skipped only by `--no-text`).
2. **`--ocrai` adds `{stem}.md`** via Qwen2.5-VL; it does not skip or replace Tesseract.
3. **VLM prompt** instructs minimal Markdown structure (headings, standard syntax); output is written UTF-8 to `.md` with the same indexed conflict pattern as `.txt`.
4. **`--no-text`** skips both Tesseract and `--ocrai` (unchanged silent honor when combined with `--ocrai`).
5. **Failure semantics:** Tesseract and `--ocrai` fail independently (soft per file). PNG success is unchanged.

All other ADR-022 decisions (hard dep, in-process llama.cpp, download abort, 2 MP cap, repo fallback) remain in force.

## Consequences

**Positive**

- Scripts keep predictable plain `.txt` from Tesseract
- Markdown suited to notes apps and layout-aware content
- Handwriting/multilingual enrichment without losing fast default OCR

**Negative**

- `--ocrai` runs two OCR backends when both succeed (latency + Tesseract dependency remains)
