# Specification: Denoise Document Mode

This document extends [`specification-denoise.md`](./specification-denoise.md) with two changes:

1. **`document` mode for `easyupscaler denoise`** — a fourth mode alongside `photo`, `art`, and `manga`, targeting scanned documents and camera photos of text. It runs an Archiver Medium AI cleanup pass, then Sauvola adaptive binarization and Gaussian anti-aliasing to produce smooth high-contrast grayscale output optimized for readability.
2. **Opt-in VLM OCR (`--ocrai`)** — an alternative text-extraction backend using Qwen2.5-VL-3B-Instruct via in-process llama.cpp ([ADR-022](./adr/022-opt-in-vlm-ocr-ocrai.md)).

All existing behavior, error messages, exit codes, and architectural constraints from `specification.md`, `specification-denoise.md`, and the accepted ADRs remain in force unless explicitly superseded here.

---

## 1. Updated command syntax

```
easyupscaler denoise <mode> [--strength low|high] [--no-text] [--ocrai] [--output DIR] <image> [<image> ...]
```

`<mode>` now accepts: `photo`, `art`, `manga`, **`document`**.

| Flag | Document mode | Other modes |
|------|---------------|-------------|
| `--no-text` | Skip OCR; PNG only | Ignored |
| `--ocrai` | Use VLM OCR instead of Tesseract | Ignored |

When both `--ocrai` and `--no-text` are passed, `--no-text` wins silently (no OCR, no stderr warning).

All other flags and positional arguments are unchanged. `document` mode behaves identically to other modes in terms of batch processing, exit codes, TTY detection, and `--output` handling.

---

## 2. `document` mode

### 2.1 Purpose

`document` mode is optimized for scans and photos of text: printed pages, handwritten notes, receipts, whiteboards, and book pages. Archiver Medium removes grain and compression artifacts; CPU post-processing (Sauvola binarization, morph cleanup, edge anti-aliasing) produces high-contrast grayscale optimized for readability.

The output is a clean grayscale PNG. It is **not** a hard 1-bit binary image — gray values are preserved to maintain smooth strokes and avoid pixel-level harshness at low-resolution edges. This makes the output legible both on screen and as OCR input.

**Out of scope:** perspective distortion, uneven geometry, and severely underexposed camera captures. Users with heavy lighting gradients should consider `photo` mode for sensor noise removal before passing to `document` mode.

---

### 2.2 Output conventions

PNG output matches all `denoise` modes (§2.1 of `specification-denoise.md`):

- Output is always **PNG** at full input resolution (1×), named `{stem}-denoised.png`.
- Output is always written as **grayscale** (mode `L`), regardless of input colorspace.
- Color and RGBA inputs are converted to **RGB** for inference; output is always written as grayscale PNG after post-processing.
- On conflict: `{stem}-denoised-NNNN.png` (lowest available 4-digit index).

**Text extraction (default):** When OCR is enabled (default), document mode also writes a UTF-8 plain-text file named `{stem}.txt` beside the input (or under `--output`). Conflict resolution is **independent** of the PNG path: `{stem}.txt`, then `{stem}-0001.txt`, `{stem}-0002.txt`, … (lowest available 4-digit index per [ADR-011](./adr/011-output-conflict-indexing.md)). Pass `--no-text` to skip OCR and write PNG only.

**OCR backend selection:**

| Condition | Backend |
|-----------|---------|
| `--no-text` | None |
| `--ocrai` | Qwen2.5-VL-3B-Instruct via llama.cpp ([§6](#6-opt-in-vlm-ocr-ocrai)) |
| (default) | Tesseract ([ADR-021](./adr/021-document-ocr-tesseract.md)) |

Both backends run on the in-memory grayscale array after post-processing (no PNG re-read). Output path and conflict rules are identical.

**Tesseract (default):** English (`eng`); Tesseract default page segmentation (no explicit `--psm`).

**VLM (`--ocrai`):** Multilingual; no language flag. The model preserves original script and language (see §6.4).

---

### 2.3 Model selection matrix

`document` mode applies one AI pass (Archiver Medium), then CPU post-processing. `--strength` controls Sauvola window size and anti-alias sigma (§2.6).

| Mode       | Strength | Pipeline |
|------------|----------|----------|
| `document` | `low`    | Archiver Medium → Sauvola binarize (window 75) → morph cleanup → edge anti-alias (σ 1.5) → flat snap |
| `document` | `high`   | Archiver Medium → Sauvola binarize (window 25) → morph cleanup → edge anti-alias (σ 0.75) → flat snap |

**Rationale for single AI pass:**
- **Archiver Medium** removes general grain, scan film, and compression artifacts without the aggressive binarization that destroys faint text on camera captures.
- Post-processing handles ink-to-paper separation via adaptive thresholding and edge smoothing.

Inference runs on RGB. Grayscale conversion happens after the AI pass (see §2.5).

See [ADR-019](./adr/019-document-binarize-antialias.md) (supersedes ADR-018) and [ADR-020](./adr/020-document-postprocessing-refinements.md) (post-processing detail).

---

### 2.4 Managed model catalog

Document mode uses the existing `archivist_medium` entry. A `book_compact` entry remains in the catalog for potential future use but is **not** invoked by document mode ([ADR-019](./adr/019-document-binarize-antialias.md)).

`document` mode requires `archivist_medium` only. It is downloaded lazily on first invocation. If download fails, the entire job fails before inference begins.

---

### 2.5 Colorspace pipeline

1. Open input image (any supported format).
2. Flatten alpha to white background if RGBA.
3. Convert to **RGB** (required by Archiver Medium, which is trained on RGB data).
4. Run Archiver Medium pass → RGB tensor output.
5. Convert the RGB tensor to **grayscale** (`uint8` via standard luminance weights).
6. Sauvola adaptive binarization → 1-bit mask (§2.6).
7. Morphological open/close on the mask (speckle removal).
8. Edge-only Gaussian anti-alias → smooth grayscale at stroke boundaries.
9. Flat-region snap → force uniform ink/paper areas to pure black/white.
10. Write as grayscale PNG.

Color information carries no useful signal for text readability and is discarded during post-processing. Converting to grayscale after (not before) the AI pass avoids inference on single-channel input that the RGB-trained Archiver model does not expect.

---

### 2.6 Sauvola binarization and anti-aliased grayscale

After the AI pass and grayscale conversion, the pipeline binarizes with Sauvola adaptive thresholding, cleans the mask with morphological open/close, smooths stroke edges with a Gaussian filter, then snaps flat ink and paper regions to pure black/white. Output is **not** a hard 1-bit PNG — anti-aliasing preserves gray values at stroke boundaries only.

**Algorithm:**

```python
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, gaussian_filter
from skimage.filters import threshold_sauvola
import numpy as np

def enhance_document_contrast(
    rgb: np.ndarray,           # float32 (H, W, 3) RGB in [0, 1]
    strength: str,             # "low" | "high"
) -> np.ndarray:               # float32 (H, W) in [0, 1]
    gray = rgb_to_gray_uint8(rgb)
    thresh = threshold_sauvola(gray, window_size=window, k=k)
    binary = (gray.astype(np.float32) >= thresh).astype(np.float32)
    binary = morphological_open_close(binary)
    edge_band = distance_to_nearest_boundary(binary) <= DOCUMENT_EDGE_BAND_WIDTH
    smoothed = gaussian_filter(binary, sigma=sigma)
    result = binary.copy()
    result[edge_band] = smoothed[edge_band]
    return snap_flat_regions(result)
```

Pixels ≥ local Sauvola threshold map to `1.0` (paper/background). Pixels below map to `0.0` (ink/text). Morphological cleanup removes isolated speckle. Gaussian blur applies only within `DOCUMENT_EDGE_BAND_WIDTH` pixels of a stroke boundary — flat paper and ink regions stay pure. Flat-region snap forces remaining values ≤ `DOCUMENT_FLAT_INK_SNAP` to `0.0` and ≥ `DOCUMENT_FLAT_PAPER_SNAP` to `1.0`.

**Named constants** (defined in `easyupscaler/denoise/document_constants.py`):

| Constant | Value | Role |
|---|---|---|
| `DOCUMENT_SAUVOLA_WINDOW_LOW` | `75` | Large local context; conservative; handles mild gradients |
| `DOCUMENT_SAUVOLA_WINDOW_HIGH` | `25` | Tight local context; aggressive local adaptation |
| `DOCUMENT_SAUVOLA_K` | `0.2` | Standard Sauvola sensitivity parameter |
| `DOCUMENT_ANTIALIAS_SIGMA_LOW` | `1.5` | Softer edge smoothing (default) |
| `DOCUMENT_ANTIALIAS_SIGMA_HIGH` | `0.75` | Sharper edges; higher ink-to-paper separation |
| `DOCUMENT_MORPH_STRUCTURE_SIZE` | `3` | Open/close kernel size for binary speckle removal |
| `DOCUMENT_EDGE_BAND_WIDTH` | `2.0` | Pixels from stroke boundary that receive anti-alias blur |
| `DOCUMENT_FLAT_INK_SNAP` | `0.25` | Values at or below snap to pure ink (0) |
| `DOCUMENT_FLAT_PAPER_SNAP` | `0.75` | Values at or above snap to pure paper (1) |

**`--strength` mapping:**

| `--strength` | `window` | `k` | anti-alias `sigma` |
|---|---|---|---|
| `low` | `DOCUMENT_SAUVOLA_WINDOW_LOW` (75) | `DOCUMENT_SAUVOLA_K` (0.2) | `DOCUMENT_ANTIALIAS_SIGMA_LOW` (1.5) |
| `high` | `DOCUMENT_SAUVOLA_WINDOW_HIGH` (25) | `DOCUMENT_SAUVOLA_K` (0.2) | `DOCUMENT_ANTIALIAS_SIGMA_HIGH` (0.75) |

`low` is the default, consistent with all other modes. Prefer `low` for flatbed scans and clean camera shots. Use `high` when ink-to-paper contrast is poor (aged documents, faint print, pencil).

---

### 2.7 scikit-image dependency

`scikit-image` is added as a **hard dependency** in `pyproject.toml`. It is required at runtime for all invocations; the `document` mode post-processing step is the only caller.

```toml
# pyproject.toml [project.dependencies] — add:
"scikit-image>=0.22",
```

`scikit-image` pulls in `scipy` and `imageio` as transitive dependencies. Document mode also calls `scipy.ndimage` directly for morphological cleanup, distance transform, and Gaussian filtering ([ADR-020](./adr/020-document-postprocessing-refinements.md)). Both are standard scientific Python packages acceptable for this project's dependency profile.

No optional extra is introduced. There is no install-time gating for this feature.

---

### 2.8 HEIC handling

HEIC inputs in `document` mode follow the same logic as in `art` and `manga` modes: a single AI pipeline (no FBCNN pass). HEIC is decoded to RGB via `pillow-heif` and treated identically to any other RGB input from that point forward.

---

### 2.9 stdout / stderr contract

Consistent with `specification-denoise.md §2.5`. Per-file stdout lists all artifacts written for that input.

```
# TTY (OCR succeeded):
Denoising 2 images [document, low] [███████████████░░░░] 1/2
  ✓ scan.jpg   → scan-denoised.png, scan.txt
  ✓ notes.heic → notes-denoised.png, notes.txt
Completed: 2 succeeded, 0 failed in 0:12.

# TTY (OCR skipped or failed — PNG only):
  ✓ scan.jpg   → scan-denoised.png

# Non-TTY (both artifacts):
scan.jpg → scan-denoised.png, scan.txt

# stderr (Tesseract missing — once per batch; default OCR only):
Warning: Tesseract not found; skipping text extraction. Install tesseract to enable OCR.

# stderr (OCR runtime failure — per file; either backend):
Warning: OCR failed for scan.jpg: {reason}

# TTY (--ocrai, first run — download before denoise progress):
Downloading Qwen2.5-VL-3B-Instruct-Q8_0.gguf...
Downloading Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf...
Denoising 1 images [document, low] [████████████████████] 1/1
  ✓ scan.jpg   → scan-denoised.png, scan.txt
Completed: 1 succeeded, 0 failed in 0:45.

# stderr (VLM download failure — aborts entire job before inference):
Error: could not download Qwen2.5-VL-3B-Instruct-Q8_0.gguf. Check your network or download manually from https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF and place in {MODELS_DIR}.
```

Document mode uses a single AI pass; no multi-pass note appears in TTY output (unlike HEIC photo mode). OCR skip or failure does not change exit code when the PNG is written successfully.

---

### 2.10 Edge cases

These extend the table in `specification-denoise.md §2.10`. All existing rows remain in force.

| Scenario | Behavior |
|----------|----------|
| Input is color (RGB) | Converted to RGB for inference; written as grayscale PNG after Sauvola step |
| Input is already grayscale | Converted to RGB for inference, back to grayscale for output; identical result |
| Input is HEIC | Decoded via `pillow-heif`; single AI pipeline (no FBCNN); grayscale output |
| `archivist_medium` download fails | Fail entire job before inference: `Error: could not download 1x-Archivist_Medium.pth. …` |
| Model already cached | No download; proceed directly to inference |
| Sauvola window larger than image dimension | `scikit-image` raises `ValueError`; treat as per-file failure: `{path} FAILED: image too small for document mode (minimum dimension: {window}px)` |
| `--no-text` passed | PNG only; no OCR; no stderr warning |
| `--ocrai` passed (without `--no-text`) | VLM OCR; Tesseract not invoked |
| `--ocrai --no-text` passed | PNG only; no OCR; no stderr warning |
| `--ocrai` outside `document` mode | Flag ignored |
| Tesseract not on PATH (default OCR) | Warn once per batch on stderr; PNG succeeds; file counts as success |
| Tesseract present, OCR runtime failure | Warn per file on stderr; PNG succeeds; no `.txt`; file counts as success |
| VLM weights download fails | Fail entire job before inference (same pattern as `archivist_medium`) |
| VLM weights already cached | No download; load model once per batch |
| VLM loaded, inference runtime failure | Warn per file on stderr; PNG succeeds; no `.txt`; file counts as success |
| `llama-cpp-python` missing or lacks `Qwen25VLChatHandler` | Fail entire job before inference with install/platform message |
| OCR returns empty string | Write empty `{stem}.txt`; file counts as success |
| `{stem}.txt` already exists | Write `{stem}-0001.txt` (independent of PNG index) |

The minimum image dimension for `--strength low` is 75 px (the Sauvola window size). For `--strength high` it is 25 px. Images smaller than the window fail per-file; the batch continues.

---

## 3. Architecture decisions

| Topic | ADR |
|-------|-----|
| `scikit-image` as hard runtime dependency | [ADR-017](./adr/017-scikit-image-dependency.md) |
| Single AI pass; Sauvola binarize + anti-alias | [ADR-019](./adr/019-document-binarize-antialias.md) (supersedes ADR-018) |
| Morph cleanup, edge-only anti-alias, flat snap | [ADR-020](./adr/020-document-postprocessing-refinements.md) |
| Default Tesseract OCR + `--no-text` | [ADR-021](./adr/021-document-ocr-tesseract.md) |
| Opt-in VLM OCR + `--ocrai` | [ADR-022](./adr/022-opt-in-vlm-ocr-ocrai.md) |

---

## 4. Affected components

Extensions to the existing denoise module ([specification-denoise.md §2.11](./specification-denoise.md)):

```
easyupscaler/
  denoise/
    catalog.py              # DenoiseMode "document"; book_compact catalog entry (unused)
    document_constants.py   # Sauvola, anti-alias, morph, snap constants
    document_enhance.py     # enhance_document_contrast(): binarize + post-process
    document_ocr.py         # Tesseract OCR on grayscale array
    document_ocrai.py       # VLM OCR: resize, prompt, llama.cpp inference
    ocrai_catalog.py        # VLM filenames, URLs, repo fallback order, constants
    ocrai_downloader.py     # ensure_ocrai_models(): two-file download lifecycle
    ocrai_service.py        # OcraiService: load model once per batch; extract_text()
    pipeline.py             # DenoiseService: document branch after _run_passes
    backends/
      book_compact_backend.py   # catalog/backend only; not invoked by document mode
  cli/
    denoise.py              # VALID_MODES includes "document"
tests/
  test_denoise_catalog.py   # resolve_models("document", ...)
  test_denoise_document.py  # enhance_document_contrast unit tests
  test_document_ocr.py      # Tesseract OCR unit tests
  test_document_ocrai.py    # VLM OCR unit tests (injected fake service)
  test_ocrai_downloader.py  # VLM weight download tests
  test_denoise_pipeline.py  # document mode integration via DenoiseService
  test_cli_denoise.py       # document mode CLI routing
```

Document mode extends the existing `DenoiseService` — no separate service class. When `mode == "document"`, `_process_path` calls `enhance_document_contrast` on Archiver output and writes via `ImageIO.write_png(..., mode="L")`.

No new config keys in `config.toml`.

---

## 5. Success criteria

- `easyupscaler denoise document scan.jpg` downloads Archiver Medium on first run and writes a clean high-contrast grayscale `scan-denoised.png` in under 30 s on M2 (16 GB).
- `easyupscaler denoise document --strength high notes.png` produces visibly higher ink-to-paper contrast than `--strength low` on the same input.
- A color input and a grayscale input of the same document produce identical output (both written as grayscale PNG).
- `easyupscaler denoise photo scan.jpg` is unaffected; mode routing is unchanged.
- `easyupscaler denoise document tiny.png` (< 25 px in any dimension) fails per-file with a clear minimum-dimension error and exits `1`.
- All `document` mode tests pass alongside existing tests; package coverage remains ≥ 80%.
- `models list` and `--help` do not load PyTorch and respond under 500 ms regardless of whether document-mode models have been downloaded.
- `easyupscaler denoise document --ocrai scan.jpg` downloads VLM weights on first run (~4.5 GB), writes `scan-denoised.png` and multilingual `scan.txt`.
- `easyupscaler denoise document --ocrai --no-text scan.jpg` writes PNG only with no download of VLM weights.
- VLM download failure exits `1` before any denoise inference begins.

---

## 6. Opt-in VLM OCR (`--ocrai`)

### 6.1 Purpose

`--ocrai` replaces Tesseract with **Qwen2.5-VL-3B-Instruct** running through **in-process llama.cpp** (`llama-cpp-python`). It is opt-in; default document mode behavior is unchanged.

Use `--ocrai` when:

- Text is handwritten or in non-Latin scripts
- Tesseract quality is insufficient
- Multilingual extraction without per-language configuration is needed

Trade-offs: ~4.5 GB first-run download, higher RAM use (~5–6 GB loaded), and slower per-image OCR than Tesseract.

### 6.2 Dependency

Add to `[project.dependencies]` in `pyproject.toml`:

```toml
"llama-cpp-python>=0.3",
```

This is a **hard dependency** (installed with every `easyupscaler` install). The package must include vision support (`Qwen25VLChatHandler`). Import `llama_cpp` and handlers **lazily** when `--ocrai` is active — not at CLI startup ([ADR-008](./adr/008-lazy-torch-imports.md)).

If the installed wheel lacks vision handlers or fails to import when `--ocrai` is requested, abort the entire job before inference:

```
Error: llama-cpp-python vision support is unavailable. Reinstall easyupscaler or use a build with Qwen25VLChatHandler support.
```

### 6.3 Managed model catalog

Two GGUF files are required. Both are stored in `$XDG_DATA_HOME/easyupscaler/models/` ([ADR-012](./adr/012-denoise-model-auto-download.md)). They are **not** registry entries and do **not** appear in `models list`.

| Constant | Filename | Role | Approx. size |
|----------|----------|------|--------------|
| `OCRAI_BACKBONE_FILENAME` | `Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Text backbone (Q8_0) | ~3.3 GB |
| `OCRAI_MMPROJ_FILENAME` | `Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` | Vision encoder (always F16) | ~1.25 GB |

**Repo fallback order** (try each repo until both files download successfully):

| Priority | Hugging Face repo |
|----------|-------------------|
| 1 | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` |
| 2 | `unsloth/Qwen2.5-VL-3B-Instruct-GGUF` |

URL pattern per file:

```
https://huggingface.co/{repo}/resolve/main/{filename}
```

Download lifecycle matches denoise weights ([ADR-012](./adr/012-denoise-model-auto-download.md)):

1. Before the first `--ocrai` inference in a job, ensure both files exist.
2. Stream each missing file to a temp file in `MODELS_DIR`, then rename atomically.
3. Show download progress (Rich bar in TTY; status line when piped).
4. If **all repos fail** for a required file, abort the **entire job** with a clear error including the last attempted URL and `MODELS_DIR`.
5. If a downloaded file is corrupt (< `MIN_VALID_MODEL_BYTES`), delete it and abort the entire job.

VLM weights download only when `--ocrai` is active and OCR is not skipped (`--no-text` suppresses download).

### 6.4 Image preparation

OCR input is the same in-memory grayscale array used by Tesseract (post-`enhance_document_contrast`, float32 `[0, 1]`).

Before VLM inference:

1. Convert to `uint8` grayscale (`mode L`).
2. **Downscale to ≤2 megapixels** if `width × height > OCRAI_MAX_PIXELS` (`2_000_000`):
   - `scale = sqrt(OCRAI_MAX_PIXELS / (width × height))`
   - New dimensions: `round(width × scale)`, `round(height × scale)` (minimum 1 px each)
   - Resampling: Lanczos (PIL `Image.Resampling.LANCZOS`)
3. Convert grayscale to **RGB** (three identical channels) for the vision encoder.

Images already ≤2 MP are not upscaled.

### 6.5 Inference

**Service:** `OcraiService` in `easyupscaler/denoise/ocrai_service.py`.

- Constructed once per `DenoiseService.run()` when `--ocrai` is active.
- Loads `Llama` with `Qwen25VLChatHandler(clip_model_path=…)` using cached GGUF paths.
- `n_gpu_layers=-1` when GPU offload is available; otherwise CPU.
- Reused for every file in the batch (model loaded once).

**Generation settings** (named constants in `ocrai_catalog.py`):

| Constant | Value | Role |
|----------|-------|------|
| `OCRAI_MAX_PIXELS` | `2_000_000` | Max input area before downscale |
| `OCRAI_PROMPT` | (see below) | Fixed extraction instruction |
| `OCRAI_MAX_TOKENS` | `4096` | Output token cap |
| `OCRAI_TEMPERATURE` | `0.0` | Deterministic extraction |
| `OCRAI_N_CTX` | `8192` | Context window |

**Builtin prompt** (`OCRAI_PROMPT`):

```
Extract all visible text from this document image.
Output plain UTF-8 text only.
Preserve the original language, script, and reading order (top to bottom, left to right where applicable).
Do not translate.
Do not add commentary, markdown, labels, or descriptions.
If there is no text, output nothing.
```

Pass the prepared RGB image to `create_chat_completion` via a `file://` URI or equivalent encoding supported by `Qwen25VLChatHandler`.

**Post-processing:** trim leading/trailing whitespace from model output. Do not strip interior content. Write result via `ImageIO.write_txt()` (same path rules as Tesseract).

### 6.6 Failure semantics

| Scenario | Behavior |
|----------|----------|
| `--ocrai` without `--no-text`, weights missing | Download; abort job on failure |
| `--ocrai --no-text` | No VLM download; no inference |
| Model loaded, inference raises | Warn per file: `Warning: OCR failed for {filename}: {reason}`; PNG succeeds |
| Model returns empty/whitespace-only string | Write empty `{stem}.txt` |
| `--ocrai` in non-`document` mode | Flag ignored |

Runtime inference failures do **not** change exit code when the PNG is written successfully (same soft-fail as Tesseract in [ADR-021](./adr/021-document-ocr-tesseract.md)).

### 6.7 Future batching

The current contract runs **one image per `create_chat_completion` call**. `OcraiService` is structured so a future implementation can batch multiple images in one forward pass without CLI or flag changes. Tests should inject a fake `OcraiService` rather than loading real GGUF weights.

### 6.8 pyproject.toml addition

```toml
# pyproject.toml [project.dependencies] — add:
"llama-cpp-python>=0.3",
```

No optional extra. Platform-specific wheel selection is documented in README (CPU vs Metal/CUDA builds as needed).
