# Specification: Denoise Command and Scale Command Rename

This document extends [`specification.md`](./specification.md) with two changes:

1. **`scale` subcommand** — the bare `easyupscaler <files>` default command is replaced by `easyupscaler scale [--model NAME] <files>`. Behavior is identical; only the invocation changes.
2. **`denoise` subcommand** — new command for AI-assisted image denoising and compression artifact removal. Selects models automatically based on mode, strength, and input format; no user-managed model imports.

All existing behavior, error messages, exit codes, and architectural constraints from `specification.md` and the ADRs remain in force unless explicitly superseded here.

---

## 1. `scale` Subcommand

### Command syntax

```
easyupscaler scale [--model NAME] <image> [<image> ...]
```

This replaces the bare `easyupscaler <files>` default command. All behavior is identical to the upscale command defined in `specification.md §1`, with the single change that `scale` is now an explicit subcommand.

**Breaking change:** the old bare invocation `easyupscaler file.jpg` no longer works. Users receive a Typer error directing them to `easyupscaler scale` or `easyupscaler denoise`.

**stdout / stderr contract:** identical to the existing upscale contract (progress bar in TTY, plain lines when piped; errors to stderr).

**Exit codes:** identical to existing upscale (`0` all succeeded, `1` any failure).

---

## 2. `denoise` Subcommand

### Command syntax

```
easyupscaler denoise <mode> [--strength low|high] <image> [<image> ...]
```

| Argument / flag | Values | Default | Notes |
|-----------------|--------|---------|-------|
| `<mode>` | `photo`, `art`, `manga` | Required | Controls model selection and output colorspace |
| `--strength` | `low`, `high` | `low` | Controls which model variant is selected |
| `<image> ...` | One or more file paths | Required | Shell expands globs before invocation |

**`--model` is not accepted.** Model selection is automatic, governed by `<mode>`, `--strength`, and input file format. Passing `--model` is an error.

---

### 2.1 Output conventions

- Output is always **PNG** at full input resolution (1×): `{stem}-denoised.png` beside the input file.
- This applies to all input formats including JPEG, HEIC, WebP, and PNG.
- On conflict: `{stem}-denoised-NNNN.png` (lowest available 4-digit index, identical to ADR-011 pattern).
- The `-upscaled` suffix is not used; `-denoised` is the canonical denoise suffix.
- Input HEIC files are read as RGB (via `pillow-heif`), processed, and written as PNG. HEIC is never written as output.
- Input alpha channel (RGBA PNG): flattened to white background before inference; output is RGB PNG.
- Input grayscale: behavior depends on mode — see §2.4 Manga mode.

---

### 2.2 Model selection matrix

Model selection is determined at runtime by mode + strength + input format. The tool never asks the user to specify a model name.

| Mode | Strength | Input format | Models applied (in order) |
|------|----------|--------------|---------------------------|
| `photo` | `low` | non-HEIC | SCUNet PSNR pass |
| `photo` | `high` | non-HEIC | SCUNet GAN pass |
| `photo` | `low` | HEIC | SCUNet PSNR pass → FBCNN pass (auto QF) |
| `photo` | `high` | HEIC | SCUNet GAN pass → FBCNN pass (QF override ≈ 20) |
| `art` | `low` | any | 1xDeJPG_realplksr_otf pass |
| `art` | `high` | any | Archiver Medium pass |
| `manga` | `low` | any | 1xDeJPG_realplksr_otf pass (colorspace-preserved) |
| `manga` | `high` | any | Archiver Medium pass (colorspace-preserved) |

**HEIC two-pass rationale:** iPhone HEIC files carry sensor noise (ISO grain) plus potential compression artifacts. SCUNet removes sensor noise first; FBCNN then cleans any residual compression artifacts introduced by the HEIC encode or downstream processing. Both passes are always applied to HEIC regardless of strength; strength controls which SCUNet variant and FBCNN aggressiveness.

**`art` vs `manga`:** identical model selection; the only runtime difference is colorspace handling (§2.4).

---

### 2.3 Managed model catalog

Denoise weights are downloaded automatically and stored in `$XDG_DATA_HOME/easyupscaler/models/` (same directory as user-imported upscaler weights). They are **not** listed in `models list` (which shows only user-imported upscaler models) and are identified by canonical filename only.

| Catalog key | Filename | Architecture | Used for | Source |
|-------------|----------|--------------|----------|--------|
| `scunet_psnr` | `scunet_color_real_psnr.pth` | SCUNet | photo --low | KAIR releases (cszn) |
| `scunet_gan` | `scunet_color_real_gan.pth` | SCUNet | photo --high | KAIR releases (cszn) |
| `fbcnn_color` | `fbcnn_color.pth` | FBCNN | HEIC second pass | FBCNN releases (jiaxi-jiang) |
| `dejpg_art` | `1xDeJPG_realplksr_otf.safetensors` | RealPLKSR | art/manga --low | Phhofm GitHub releases |
| `archivist_medium` | `1x-Archivist_Medium.pth` | ESRGAN | art/manga --high | Archivist-Project-Denoiser (Loganavter) |

**Lazy download:** the tool downloads only the model(s) required for the current invocation. It does not pre-download the full catalog. For HEIC inputs in photo mode, both `scunet_psnr` (or `scunet_gan`) and `fbcnn_color` are downloaded as needed in the same run.

**Model presence check:** before inference, verify each required model file exists at its expected path. If absent, attempt download. If download fails, fail the entire job (not per-file) with a clear error.

**Download progress:** show a Rich progress bar for each download in TTY mode; single status line in non-TTY mode:

```
# TTY:
Downloading scunet_color_real_psnr.pth (15 MB) [████████████████░░░░] 12/15 MB

# Non-TTY:
Downloading scunet_color_real_psnr.pth...
```

**Source URLs:** exact URLs are hardcoded in a `DENOISE_MODEL_CATALOG` constant in the implementation:

- `scunet_color_real_psnr.pth` — `https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth`
- `scunet_color_real_gan.pth` — `https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth`
- `fbcnn_color.pth` — `https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth`
- `1xDeJPG_realplksr_otf.safetensors` — `https://github.com/Phhofm/models/releases/download/1xDeJPG_realplksr_otf/1xDeJPG_realplksr_otf.safetensors`
- `1x-Archivist_Medium.pth` — `https://github.com/Loganavter/Archivist-Project-Denoiser/releases/download/v1.0/1x-Archivist_Medium.pth`

All URLs verified via HTTP HEAD (200 OK) on 2026-06-20. File sizes:

| Filename | Verified size |
|----------|---------------|
| `scunet_color_real_psnr.pth` | ~68.7 MB |
| `scunet_color_real_gan.pth` | ~68.7 MB |
| `fbcnn_color.pth` | ~274.6 MB |
| `1xDeJPG_realplksr_otf.safetensors` | ~28.1 MB |
| `1x-Archivist_Medium.pth` | ~1.3 MB |

---

### 2.4 Mode behavior details

#### `photo` mode

- Input is treated as a real-world photograph (sensor noise, ISO grain, possible JPEG compression).
- Always processed as RGB; grayscale inputs are converted to RGB for inference and written as RGB PNG.
- HEIC inputs trigger a two-pass pipeline (§2.2).
- Non-HEIC inputs (JPEG, PNG, WebP, etc.) use a single SCUNet pass.

#### `art` mode

- Input is treated as digital illustration, line art, or an image with JPEG compression artifacts.
- Always processed and written as RGB PNG.
- `--strength low` applies `1xDeJPG_realplksr_otf` — balanced compression cleanup, trained on photograph + digital art data, handles down to JPEG quality 40 with OTF augmentation.
- `--strength high` applies `Archiver Medium` — aggressive grain + artifact removal, preserves drawing texture, best for scanned or re-compressed sources at 720p–1080p.

#### `manga` mode

- Input is treated as manga, manhwa, webtoon, or other lineart with compression artifacts.
- Model selection is identical to `art` mode.
- **Colorspace is preserved:** if the input is grayscale, run inference on an RGB version (convert → infer → convert back), then write grayscale PNG. If the input is color (RGB), write color PNG. This matches how readers expect manga output: BW originals stay BW.
- Archiver Medium at `--strength high` targets the scan-restoration use case (film grain, physical artifacts, 720p–1080p source).

---

### 2.5 stdout / stderr contract

```
# TTY (progress bar + per-file lines):
Denoising 3 images [photo, low] [███████████████░░░░] 2/3
  ✓ photo.heic → photo-denoised.png   (2 passes: SCUNet PSNR + FBCNN)
  ✓ scan.jpg   → scan-denoised.png
  ✗ broken.png — cannot read image
Completed: 2 succeeded, 1 failed in 0:08.

# Non-TTY (pipe/redirect):
photo.heic → photo-denoised.png
scan.jpg → scan-denoised.png
broken.png FAILED: cannot read image
```

- Errors and warnings go to **stderr**; progress and file status go to **stdout**.
- Two-pass HEIC output lines note `(2 passes: ...)` in TTY mode only.
- Download progress lines precede the denoising progress bar; they use the same TTY/non-TTY detection.

---

### 2.6 Batch behavior

Identical to the upscale command (ADR-006):

- Continue on per-file failure; do not abort the batch.
- Exit `0` if all files succeeded; exit `1` if any file failed.
- Exit `1` with an error before inference if zero paths are given.
- A download failure before inference starts fails the entire job (not per-file) — exit `1`.

---

### 2.7 PyTorch loading

`denoise` always loads PyTorch and Spandrel (all modes require inference). There is no fast path for any `denoise` invocation. This is expected behavior; no 500 ms constraint applies.

`scale` (the renamed upscale command) retains the same lazy-load behavior as the original upscale command.

---

### 2.8 HEIC support

- HEIC / HEIF files are read via `pillow-heif` as a Pillow plugin (`register_heif_opener()` called before any image read in the CLI layer).
- `pillow-heif` is a new **optional dependency** gated on the `[heic]` extra: `pip install easyupscaler[heic]`.
- If `pillow-heif` is not installed and a `.heic` or `.heif` file is passed to any command, fail with a clear error per-file:
  ```
  photo.heic FAILED: HEIC support requires pillow-heif. Install with: pip install easyupscaler[heic]
  ```
- HEIC inputs are valid in all three modes, but only `photo` mode triggers the two-pass pipeline. In `art` and `manga` modes, HEIC is treated as any other input format (single pass).

---

### 2.9 FBCNN inference detail

FBCNN accepts an optional quality-factor tensor that overrides its blind QF estimate. The standard Spandrel `ImageModelDescriptor.__call__(image_tensor)` does not expose this parameter. The implementation must call the underlying model directly for the FBCNN pass:

```python
# Auto QF (photo --low HEIC):
output_tensor, predicted_qf = descriptor.model(image_tensor)

# QF override (photo --high HEIC, more aggressive cleaning):
qf_input = torch.tensor([[1.0 - 20 / 100]])  # QF=20 → input=0.80
output_tensor, _ = descriptor.model(image_tensor, qf_input.to(device))
```

This bypasses Spandrel's wrapper only for the FBCNN pass. All other model passes use the standard `descriptor(image_tensor)` call.

The FBCNN `qf_input` value for `--strength high` is a named constant `FBCNN_HIGH_STRENGTH_QF = 20`. The implementation must not hardcode the literal `20` elsewhere.

---

### 2.10 Edge cases and error handling

| Scenario | Behavior |
|----------|----------|
| Zero paths given | Error before torch load: `Error: no input images. Pass one or more file paths.` Exit 1. |
| `--model NAME` passed to `denoise` | Error before torch load: `Error: --model is not supported for 'denoise'. Model is selected automatically by mode and strength.` Exit 1. |
| Input path does not exist | Per-file failure: `{path} FAILED: file not found`. Continue batch. |
| Input is a directory | Per-file failure: `{path} FAILED: not a file`. Continue batch. |
| Corrupt / unreadable image | Per-file failure: `{path} FAILED: cannot read image`. Continue batch. |
| HEIC file, pillow-heif not installed | Per-file failure with install instructions. Continue batch. |
| Output directory is read-only | Per-file failure with OS error. Continue batch. |
| `{stem}-denoised.png` already exists | Write `{stem}-denoised-NNNN.png` (lowest available index). Never overwrite. |
| Model download fails (network error) | Fail entire job before inference: `Error: could not download {filename}. Check your network or download manually from {url} and place in {models_dir}.` Exit 1. |
| Model file corrupt after download | Delete the corrupt file, fail entire job: `Error: downloaded {filename} appears corrupt. Delete it and retry.` Exit 1. |
| MPS unavailable | Warn to stderr, fall back to CPU (same policy as upscale, ADR-002). |
| MPS op failure mid-inference | Retry on CPU (same policy as upscale, ADR-002). |
| OOM during inference | Halve tile size and retry (same policy as upscale, ADR-007). |
| Old bare `easyupscaler file.jpg` invocation | Typer error: unrecognised command; help text lists `scale` and `denoise`. |
| `easyupscaler scale` with no files | Error: `Error: no input images. Pass one or more file paths.` Exit 1. |

---

### 2.11 Affected components (implementation sketch)

New modules required:

```
easyupscaler/
  cli/
    denoise.py               # denoise command; mode/strength parsing; progress display
    scale.py                 # renamed from upscale.py; subcommand changed from default to `scale`
  denoise/
    catalog.py               # DENOISE_MODEL_CATALOG: names, filenames, URLs, required for which mode/strength
    downloader.py            # download_model(key): fetch + verify + save; progress callback
    pipeline.py              # DenoiseService: single-pass and two-pass orchestration
    backends/
      scunet_backend.py      # SCUNetBackend: load + forward via Spandrel
      fbcnn_backend.py       # FBCNNBackend: load + forward with optional qf_input override
      dejpg_backend.py       # DeJPGBackend: RealPLKSR model via Spandrel
      archiver_backend.py    # ArchiverBackend: ESRGAN model via Spandrel
  io/
    images.py                # extend with: HEIC detection, grayscale round-trip, PNG write
tests/
  test_denoise_catalog.py
  test_denoise_downloader.py
  test_denoise_pipeline.py
  test_cli_denoise.py
  test_cli_scale.py
```

**Backends share the same Protocol** as `UpscalerBackend` where possible. `FBCNNBackend` extends the protocol to accept an optional `qf_override: float | None` parameter.

**`DenoiseService`** is injected with backend factories (not constructed internally), following the same dependency injection pattern as `UpscaleService`.

**`ImageIO.write_png`** is a new write method alongside the existing `write` (JPEG). It follows the same conflict-indexing logic but writes PNG and appends `-denoised` rather than `-upscaled`.

**No new config keys** in `config.toml`. Denoise has no user-configurable defaults beyond the CLI flags.

---

## 3. Decisions deferred to ADR

The following decisions introduced by this spec require new ADRs before implementation:

| Topic | Proposed ADR number |
|-------|---------------------|
| Auto-download of managed denoise model weights | ADR-012 |
| PNG as denoise output format (separate from JPEG upscale convention in ADR-003) | ADR-013 |
| Two-pass pipeline for HEIC photo inputs | ADR-014 |
| `pillow-heif` as an optional dependency for HEIC support | ADR-015 |

These ADRs must be written and accepted before implementation begins. The spec records the decision outcomes; the ADRs record the rationale and alternatives considered.

---

## 4. Success criteria

- `easyupscaler denoise photo --strength low photo.heic` downloads required models on first run, runs two passes, and writes `photo-denoised.png` in under 60 s on M2 (16 GB).
- `easyupscaler denoise art *.jpg` processes a folder of JPEG illustrations and exits `1` only when a file genuinely fails, not on model-selection uncertainty.
- `easyupscaler denoise manga --strength high page.png` writes grayscale PNG when input is grayscale.
- `easyupscaler scale photo.jpg` upscales identically to the previous bare `easyupscaler photo.jpg`.
- The old bare invocation `easyupscaler photo.jpg` fails with a clear Typer error pointing to `scale`.
- `models list` and `--help` do not load PyTorch and respond under 500 ms regardless of whether any denoise models have been downloaded.
- All `denoise` and `scale` tests pass alongside existing tests; package coverage remains ≥ 80%.
