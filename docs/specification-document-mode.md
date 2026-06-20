# Specification: Denoise Document Mode

This document extends [`specification-denoise.md`](./specification-denoise.md) with one change:

1. **`document` mode for `easyupscaler denoise`** — a fourth mode alongside `photo`, `art`, and `manga`, targeting scanned documents and camera photos of text. It runs an Archiver Medium AI cleanup pass, then Sauvola adaptive binarization and Gaussian anti-aliasing to produce smooth high-contrast grayscale output optimized for readability.

All existing behavior, error messages, exit codes, and architectural constraints from `specification.md`, `specification-denoise.md`, and the accepted ADRs remain in force unless explicitly superseded here.

---

## 1. Updated command syntax

```
easyupscaler denoise <mode> [--strength low|high] [--output DIR] <image> [<image> ...]
```

`<mode>` now accepts: `photo`, `art`, `manga`, **`document`**.

All other flags and positional arguments are unchanged. `document` mode behaves identically to other modes in terms of batch processing, exit codes, TTY detection, and `--output` handling.

---

## 2. `document` mode

### 2.1 Purpose

`document` mode is optimized for scans and photos of text: printed pages, handwritten notes, receipts, whiteboards, and book pages. It removes background stains, compression artifacts, paper texture, and color bleed, then applies adaptive contrast enhancement to push the result toward high-contrast grayscale.

The output is a clean grayscale PNG. It is **not** a hard 1-bit binary image — gray values are preserved to maintain smooth strokes and avoid pixel-level harshness at low-resolution edges. This makes the output legible both on screen and as OCR input.

**Out of scope:** perspective distortion, uneven geometry, and severely underexposed camera captures. Users with heavy lighting gradients should consider `photo` mode for sensor noise removal before passing to `document` mode.

---

### 2.2 Output conventions

Identical to all `denoise` modes (§2.1 of `specification-denoise.md`):

- Output is always **PNG** at full input resolution (1×), named `{stem}-denoised.png`.
- Output is always written as **grayscale** (mode `L`), regardless of input colorspace.
- Color and RGBA inputs are converted to grayscale before inference (see §2.5).
- On conflict: `{stem}-denoised-NNNN.png` (lowest available 4-digit index).

---

### 2.3 Model selection matrix

`document` mode applies one AI pass (Archiver Medium), then CPU post-processing. `--strength` controls Sauvola window size and anti-alias sigma (§2.6).

| Mode       | Strength | Pipeline |
|------------|----------|----------|
| `document` | `low`    | Archiver Medium → Sauvola binarize (window 75) → anti-alias (σ 1.5) |
| `document` | `high`   | Archiver Medium → Sauvola binarize (window 25) → anti-alias (σ 0.75) |

**Rationale for single AI pass:**
- **Archiver Medium** removes general grain, scan film, and compression artifacts without the aggressive binarization that destroys faint text on camera captures.
- Post-processing handles ink-to-paper separation via adaptive thresholding and edge smoothing.

Inference runs on RGB. Grayscale conversion happens after the AI pass (see §2.5).

See [ADR-019](./adr/019-document-binarize-antialias.md) (supersedes ADR-018).

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
5. Convert the RGB tensor to **grayscale** (`L` mode via standard luminance weights).
6. Sauvola adaptive binarization → 1-bit mask (§2.6).
7. Gaussian anti-alias the mask → smooth grayscale array.
8. Write as grayscale PNG.

Color information carries no useful signal for text readability and is discarded after inference. Converting to grayscale after (not before) the AI passes avoids inference on single-channel input that the RGB-trained models do not expect.

---

### 2.6 Sauvola binarization and anti-aliased grayscale

After the AI pass and grayscale conversion, the pipeline binarizes with Sauvola adaptive thresholding, cleans the mask with morphological open/close, smooths stroke edges with a Gaussian filter, then snaps flat ink and paper regions to pure black/white. Output is **not** a hard 1-bit PNG — anti-aliasing preserves gray values at stroke boundaries only.

**Algorithm:**

```python
from scipy.ndimage import gaussian_filter
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
    smoothed = antialias_edges_only(binary, sigma=sigma)
    return snap_flat_regions(smoothed)
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

`scikit-image` pulls in `scipy` and `imageio` as transitive dependencies. Both are standard scientific Python packages acceptable for this project's dependency profile.

No optional extra is introduced. There is no install-time gating for this feature.

---

### 2.8 HEIC handling

HEIC inputs in `document` mode follow the same logic as in `art` and `manga` modes: a single AI pipeline (no FBCNN pass). HEIC is decoded to RGB via `pillow-heif` and treated identically to any other RGB input from that point forward.

---

### 2.9 stdout / stderr contract

Consistent with `specification-denoise.md §2.5`. No new output lines.

```
# TTY:
Denoising 2 images [document, low] [███████████████░░░░] 1/2
  ✓ scan.jpg   → scan-denoised.png
  ✓ notes.heic → notes-denoised.png
Completed: 2 succeeded, 0 failed in 0:12.

# Non-TTY:
scan.jpg → scan-denoised.png
notes.heic → notes-denoised.png
```

Two-pass note is not shown for `document` mode in TTY output (unlike HEIC photo mode). The two AI passes are an implementation detail, not a user-visible distinction.

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

The minimum image dimension for `--strength low` is 75 px (the Sauvola window size). For `--strength high` it is 25 px. Images smaller than the window fail per-file; the batch continues.

---

## 3. Decisions deferred to ADR

| Topic | ADR |
|-------|-----|
| `scikit-image` as hard runtime dependency | ADR-017 |
| Document pipeline (Archiver + Sauvola binarize + anti-alias) | ADR-019 (supersedes ADR-018) |

These ADRs must be written and accepted before implementation begins.

---

## 4. Affected components

Extensions to the existing denoise module layout defined in `specification-denoise.md §2.11`:

```
easyupscaler/
  denoise/
    catalog.py          # add book_compact entry (filename, URL, architecture notes)
    pipeline.py         # add DocumentDenoiseService: two AI passes + Sauvola step
    document_constants.py   # DOCUMENT_SAUVOLA_WINDOW_LOW/HIGH, _K, CONTRAST_FACTOR_LOW/HIGH
    backends/
      book_compact_backend.py   # BookCompactBackend: Compact arch via Spandrel
  io/
    images.py           # extend with: rgb_to_grayscale(), grayscale PIL→numpy round-trip
tests/
  test_denoise_document_pipeline.py
  test_cli_denoise_document.py
```

`DocumentDenoiseService` is injected with backend factories for `ArchiverBackend` and `BookCompactBackend`, and a contrast-enhancement callable, following the same dependency injection pattern as `UpscaleService` and the existing denoise services.

No new CLI changes are required beyond registering `document` as a valid mode value in `easyupscaler/cli/denoise.py`. The existing mode enum and routing logic extend naturally.

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
