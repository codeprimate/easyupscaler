# ADR-022: Opt-in VLM OCR via `--ocrai`

**Status:** Accepted  
**Date:** 2026-06-20  
**Extends:** [ADR-021](./021-document-ocr-tesseract.md), [ADR-012](./012-denoise-model-auto-download.md)

## Context

Document denoise mode writes a cleaned grayscale PNG and, by default, plain text via Tesseract ([ADR-021](./021-document-ocr-tesseract.md)). Tesseract is fast and lightweight but English-centric (`eng`), weak on handwriting, and requires a system binary.

Stakeholder decision: add an **opt-in** `--ocrai` flag that runs **Qwen2.5-VL-3B-Instruct** (Q8_0 GGUF) through **in-process llama.cpp** (`llama-cpp-python`) instead of Tesseract. The VLM path targets **multilingual** extraction and better handwriting at the cost of download size (~4.1 GB), RAM, and per-image latency.

## Decision

1. **`--ocrai` is opt-in.** Default document mode keeps Tesseract OCR unchanged.
2. **`--ocrai` applies only in `document` mode.** Ignored elsewhere (same pattern as `--no-text`).
3. **`--no-text` wins silently.** When both `--ocrai` and `--no-text` are passed, skip all OCR with no stderr warning.
4. **Hard Python dependency:** add `llama-cpp-python` (vision-capable build with `Qwen25VLChatHandler`) to `[project.dependencies]`. Import lazily when `--ocrai` is active ([ADR-008](./008-lazy-torch-imports.md) spirit).
5. **Managed VLM weights:** two GGUF files (text backbone Q8_0 + mmproj Q8_0) live in `$XDG_DATA_HOME/easyupscaler/models/` alongside denoise weights ([ADR-012](./012-denoise-model-auto-download.md)). Not registry entries; not listed by `models list`.
6. **Download on first `--ocrai` use:** try an ordered list of Hugging Face repos; abort the **entire job** if all sources fail (same severity as Archiver Medium download failure).
7. **Inference:** in-process `Llama` + `Qwen25VLChatHandler`; load model **once per batch** (`DenoiseService.run()`); resize denoised grayscale to ≤2 MP before VLM input; convert to RGB for the vision encoder.
8. **Runtime OCR failure** (inference error after model is loaded): warn per file on stderr; PNG succeeds; no `.txt` (soft fail, same as Tesseract runtime failure).
9. **Empty model output:** write empty `{stem}.txt`.
10. **Future batching:** keep a dedicated OCR-AI service object loaded for the batch so multi-image VLM batching can be added without CLI changes.

## Model source

Primary and fallback repos (try in order until both files download):

| Priority | Repo | Rationale |
|----------|------|-----------|
| 1 | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` | Official ggml-org quantizations |
| 2 | `unsloth/Qwen2.5-VL-3B-Instruct-GGUF` | Widely mirrored; same filenames |

Canonical filenames in `MODELS_DIR`:

- `Qwen2.5-VL-3B-Instruct-Q8_0.gguf` (~3.3 GB)
- `Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf` (~805 MB; vision encoder, Q8_0 from ggml-org; unsloth fallback uses F16 mmproj)

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Replace Tesseract as default | Stakeholder wants opt-in only |
| Optional `[ocrai]` extra | Stakeholder chose hard dep |
| Subprocess to `llama-cli` | Stakeholder chose in-process for future batching |
| Hugging Face `from_pretrained` cache | Inconsistent storage with managed denoise weights |
| Soft-fail on VLM download | Stakeholder chose job abort — user explicitly opted in |
| Per-file download failure | Same as ADR-012: partial catalog unusable |
| `--ocrai --no-text` error | Stakeholder chose silent honor of `--no-text` |
| English-only VLM prompt | Stakeholder chose multilingual extraction |

## Consequences

**Positive**

- Handwriting and non-Latin scripts without `--ocr-lang` tuning
- Same `.txt` naming and conflict rules as Tesseract path
- Injectable backend for fast tests; model singleton enables future batch inference

**Negative**

- ~4.1 GB additional disk on first `--ocrai` run
- Hard dep on native `llama-cpp-python` wheels (platform/Python-version sensitive)
- Slower per-image OCR than Tesseract (seconds vs sub-second)
- Two OCR backends to maintain and document

**Follow-up**

- True multi-image VLM batching in one forward pass
- Optional `--ocr-lang` for Tesseract-only tuning (does not affect `--ocrai`)
