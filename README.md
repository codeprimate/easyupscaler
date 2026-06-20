# easyupscaler

Upscale images from the terminal with community super-resolution models — no GUI, no workflow editor, no weight conversion.

```bash
easyupscaler models import ~/Downloads/RealESRGAN_x4plus.pth
easyupscaler models default RealESRGAN_x4plus
easyupscaler scale photo.png
# → photo-upscaled.jpg
```

> **Platform:** easyupscaler is tested only on **Apple Silicon Mac** (macOS with MPS). Linux, Windows, and Intel Mac are unsupported here — they may work, but are not verified.

## Why this exists

Tools like ComfyUI, A1111, and Upscayl are built around larger products or desktop apps. If you already have a `.pth` or `.safetensors` upscaler and want to run it against local files from a script, you usually end up wiring PyTorch, Spandrel, tiling, and device selection yourself.

easyupscaler is a single command that handles model import, defaults, batch runs, and tiled inference on Apple Silicon. Import a weight file once, then upscale with one argument.

## What you get

- **One entry point** — `easyupscaler scale` for upscaling, `easyupscaler denoise` for AI denoising, and `easyupscaler models` for model management
- **Community weights, unchanged** — import `.pth` and `.safetensors` files that work in ComfyUI and [OpenModelDB](https://openmodeldb.info/)
- **Batch-friendly** — pass multiple paths or shell globs; exit code `1` if any file fails
- **Large images** — tiled inference with automatic tile-size reduction on out-of-memory
- **Fully offline** — no internet access; all models are stored locally in standard XDG directories

Output is always `{stem}-upscaled.jpg` beside each input by default (JPEG quality 95, 4:4:4 chroma subsampling). Use `--output DIR` or `-o DIR` to write all outputs under one directory instead. If that file already exists, the next run writes `{stem}-upscaled-0001.jpg`, then `-0002`, and so on. PNG inputs (including RGBA and grayscale) are converted to RGB.

## Requirements

| | |
|---|---|
| Python | 3.13+ |
| Primary platform | Apple Silicon macOS (MPS inference) |
| Secondary | Linux x86_64/arm64 (CPU, best-effort) |
| Not tested | Windows, Intel Mac |

First inference loads PyTorch and Spandrel — expect a few seconds of startup. Model management commands stay under 500 ms.

## Install

```bash
git clone https://github.com/codeprimate/easyupscaler
cd easyupscaler
make install
```

`make install` runs `python3 -m pip install .` — it installs into whatever `python3` on your PATH resolves to (system or Homebrew Python), not the uv project venv.

Requires Python 3.13+ on PATH as `python3`.

**Developing the project** uses uv for the locked dev environment:

```bash
uv sync
make test
```

## Quickstart

Download a super-resolution weight file from a source you trust — [Real-ESRGAN x4plus](https://openmodeldb.info/) is a common starting point. Prefer `.safetensors` when available.

```bash
# 1. Import the model (validates through Spandrel, copies into local storage)
easyupscaler models import /path/to/RealESRGAN_x4plus.pth

# 2. Set it as the default
easyupscaler models default RealESRGAN_x4plus

# 3. Upscale
easyupscaler scale ~/Pictures/photo.jpg
```

Output lands next to the input: `~/Pictures/photo-upscaled.jpg`. Dimensions are input size × model scale (a 2048×2048 image with a 4× model becomes 8192×8192; a 1× model keeps the same size for detail enhancement).

Override the default for one run:

```bash
easyupscaler scale --model RealESRGAN_x4plus scan.png print.png
easyupscaler scale photo.jpg --output ./results
```

## Denoise

Denoise removes sensor noise and compression artifacts at 1× resolution. Models are downloaded automatically on first use.

```bash
# Photo mode (JPEG, PNG, HEIC)
easyupscaler denoise photo --strength low ~/Pictures/photo.heic

# Art or manga illustrations
easyupscaler denoise art *.jpg
easyupscaler denoise manga --strength high page.png
```

Output is `{stem}-denoised.png` beside each input by default (PNG, lossless). Use `--output DIR` or `-o DIR` to write under one directory. HEIC photo mode runs a two-pass pipeline (SCUNet + FBCNN). See [docs/specification-denoise.md](docs/specification-denoise.md) for the full model selection matrix.

**Photo strength:** `--strength low` (default) uses SCUNet PSNR and is the right choice for typical phone/camera JPEGs. `--strength high` switches to SCUNet GAN for heavy sensor noise; on already-clean JPEGs it can add visible speckle, color blotches, and synthetic texture in smooth areas — not “better,” just more aggressive.

Batch via shell globs:

```bash
easyupscaler scale *.png
easyupscaler scale photos/*.jpg --model 4xUltrasharp
```

## Commands

### Scale (upscale)

```
easyupscaler scale [--model NAME] [--output DIR] <image> [<image> ...]
```

| Flag / arg | Description |
|---|---|
| `--model NAME` | Use this registry model instead of the configured default |
| `--output`, `-o DIR` | Write all outputs to this directory (created if missing) |
| `<image> ...` | One or more file paths (shell expands globs before the process starts) |

Progress goes to stdout. Errors and warnings go to stderr. In a TTY you get a Rich progress bar and per-file ✓/✗ lines. When piped or redirected, output is one plain line per file.

Exit `0` when every file succeeds. Exit `1` on any failure, including an empty argument list.

### Denoise

```
easyupscaler denoise <mode> [--strength low|high] [--output DIR] <image> [<image> ...]
```

| Flag / arg | Description |
|---|---|
| `<mode>` | `photo`, `art`, or `manga` — selects models automatically |
| `--strength` | `low` (default) or `high` — in photo mode, `high` is SCUNet GAN (noisy sources); `low` is SCUNet PSNR (typical JPEGs) |
| `--output`, `-o DIR` | Write all outputs to this directory (created if missing) |
| `<image> ...` | One or more file paths |

Output is PNG at 1× resolution (`{stem}-denoised.png`). Denoise models download automatically on first use. Always loads PyTorch.

### Model management

```
easyupscaler models list
easyupscaler models import <path> [--force]
easyupscaler models default <name>
easyupscaler models remove <name> [--yes]
```

`list`, `default`, and `remove` do not load PyTorch. `import` loads weights to validate architecture, purpose, and scale.

Import derives the registry name from the filename stem (`RealESRGAN_x4plus.pth` → `RealESRGAN_x4plus`). Accepted purposes are `SR` (upscalers) and `Restoration` (1× detail enhancers). Scale is read from the model. Duplicate names are rejected unless you pass `--force`.

`.pth` imports emit a pickle security warning — only import models from sources you trust.

## Where files live

Uses [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) paths. When `XDG_*` variables are unset, defaults are `~/.config` and `~/.local/share`.

| File | Path |
|---|---|
| Default model preference | `~/.config/easyupscaler/config.toml` |
| Installed model registry | `~/.local/share/easyupscaler/registry.json` |
| Copied weight files | `~/.local/share/easyupscaler/models/` |

## Shell globs

Glob expansion is the shell's job, not easyupscaler's. If no files match and your shell lacks `nullglob`, you may get a literal `*.png` path and a per-file "not found" error.

```bash
shopt -s nullglob   # bash
setopt nullglob     # zsh
```

## How inference works

1. **Spandrel** loads the weight file and detects architecture and scale.
2. **Device** — MPS on Apple Silicon when available, otherwise CPU (with a stderr warning).
3. **Tiling** — images larger than 512 px on a side are processed in overlapping tiles (512 px, 32 px overlap). On OOM, tile size halves until 128 px or failure.
4. **MPS fallback** — if an MPS op fails mid-run, the image retries on CPU.

This mirrors patterns from ComfyUI and A1111, which also wrap Spandrel with their own tiling code.

## Troubleshooting

| Symptom | What to try |
|---|---|
| `no default model set` | Run `easyupscaler models default <name>` or pass `--model` |
| `model 'foo' not found` | Run `easyupscaler models list` to see installed names |
| `architecture not recognised` | Update easyupscaler; confirm the file is an SR model Spandrel supports |
| `purpose '…' is not supported` | The checkpoint is inpainting, face restoration, or another unsupported type |
| Slow on first run | PyTorch cold start is normal; later files in the same batch reuse the loaded model |
| Out of memory | Tiling retries with smaller tiles automatically; very large images may still fail at the 128 px floor |
| MPS unavailable | Inference falls back to CPU automatically |
| Photo denoise looks speckled or blotchy at `--strength high` | Use `--strength low` (PSNR). GAN is for heavy noise; clean JPEGs often look worse |

## Development

Requires [uv](https://docs.astral.sh/uv/) for the locked dev environment:

```bash
make sync       # uv sync — dev dependencies into .venv
make test       # ruff + mypy + pytest with ≥80% coverage (slow tests excluded)
make build      # build wheel
```

`make install` is separate: it installs the CLI into system `python3` via pip, not into `.venv`.

Slow end-to-end tests need real weights:

```bash
export EASYUPSCALER_TEST_WEIGHTS=/path/to/RealESRGAN_x4plus.pth
uv run pytest -m slow
```

Architecture and design decisions live in [`docs/`](docs/).

## License

MIT — see [LICENSE](LICENSE).

`spandrel-extra-arches` (a runtime dependency) includes architectures with separate license restrictions. Review that package if you need unrestricted commercial use of every supported architecture.
