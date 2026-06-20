# easyupscaler

Upscale and clean up images from the terminal. No GUI, no workflow editor — just commands you can put in a script.

**Upscale** makes images larger with AI models you install yourself (typically 2× or 4×). **Denoise** removes sensor noise and compression artifacts at the same size — models download automatically.

Tested on **Apple Silicon Mac** (macOS with MPS). Linux may work on CPU. Windows and Intel Mac are untested.

## Install

You need **Python 3.13+** on your PATH as `python3`.

```bash
pip install git+https://github.com/codeprimate/easyupscaler.git
```

That installs the `easyupscaler` command into whatever environment `pip` belongs to (system Python, Homebrew, or an active venv). Use `python3 -m pip install …` if `pip` is not on your PATH.

First upscale or denoise run loads PyTorch — expect a few seconds of startup. Listing or managing models stays fast.

## Your first upscale

Upscale models are **not bundled**. Download a weight file from a source you trust. [Real-ESRGAN x4plus](https://openmodeldb.info/) on [OpenModelDB](https://openmodeldb.info/) is a solid starting point for photos. Prefer `.safetensors` over `.pth` when both are available.

```bash
# Import once — validates the file and copies it into local storage
easyupscaler models import ~/Downloads/RealESRGAN_x4plus.pth

# Remember it for future runs
easyupscaler models default RealESRGAN_x4plus

# Upscale
easyupscaler scale ~/Pictures/photo.jpg
# → ~/Pictures/photo-upscaled.jpg
```

That is the whole workflow: import, set default, run.

## Understanding upscale models

This is the part that takes a minute to learn. The app itself is simple. Picking the right model for your image is not.

### What you are installing

An upscale model is a single weight file (`.pth` or `.safetensors`) trained for a specific job. easyupscaler accepts models that work in ComfyUI, A1111, and OpenModelDB — no conversion step.

Each model has a **scale factor** baked in:

| Scale | What it does |
|-------|--------------|
| **4×** | Most common. A 1000×800 photo becomes 4000×3200. |
| **2×** | Half the enlargement, often sharper on very large sources. |
| **1×** | Same pixel dimensions, but adds detail and sharpness. Sometimes called "restoration" rather than upscaling. |

Check scale after import (`easyupscaler models list`) or on the model's OpenModelDB page before you run.

### Which model for which image

There is no single best model. Community models differ by training data and architecture:

- **General photos** — Real-ESRGAN x4plus, 4x-UltraSharp, and similar "photo" models on OpenModelDB.
- **Anime and illustration** — Models trained on anime art (search OpenModelDB for "anime"). A photo model on line art often looks soft or waxy.
- **1× detail pass** — Some workflows run a 1× restoration model before or instead of a 4× upscale. Import it like any other model; output size matches input.

When results look wrong, try a different model before blaming the tool. Model pages on OpenModelDB usually say what they were trained for.

### Security note on `.pth` files

`.pth` files use Python pickle and can run arbitrary code when loaded. Only import weights from sources you trust. `.safetensors` avoids that risk.

## Upscaling

```bash
easyupscaler scale photo.jpg
easyupscaler scale scan.png print.png          # multiple files
easyupscaler scale *.jpg                       # shell expands globs
easyupscaler scale photo.jpg --model 4xUltrasharp   # override default for one run
easyupscaler scale *.png --output ./results    # write all outputs to one folder
```

**Output:** `{name}-upscaled.jpg` next to each input (JPEG, quality 95). PNG and HEIC inputs are read fine; output is always JPEG. If the file already exists, the next run writes `-upscaled-0001.jpg`, then `-0002`, and so on.

**Exit code:** `0` if every file succeeded, `1` if any failed. Useful in scripts.

Large images are processed in tiles automatically. If memory runs out, tile size shrinks and the run retries.

## Denoising

Denoise cleans noise and compression artifacts **without enlarging** the image. You do not import models — pick a **mode** and easyupscaler downloads the right weights on first use.

```bash
easyupscaler denoise photo ~/Pictures/IMG_1234.heic
easyupscaler denoise art illustration.png
easyupscaler denoise manga --strength high page.png
easyupscaler denoise document scan.jpg
easyupscaler denoise photo *.jpg --output ./cleaned
```

**Output:** `{name}-denoised.png` (lossless PNG, same resolution as input).

### Picking a mode

| Mode | Use for |
|------|---------|
| `photo` | Camera and phone shots — JPEG, PNG, HEIC |
| `art` | Digital art, game textures, non-manga illustration |
| `manga` | Manga and comic pages (same models as `art`, different color handling) |
| `document` | Scanned or photographed text — receipts, notes, book pages (grayscale output; use `--strength high` for faint ink) |

### Strength (`--strength low` or `high`)

For **photo** mode, strength matters:

- **`low` (default)** — Best for typical phone and camera JPEGs. Removes noise without inventing texture.
- **`high`** — For visibly noisy sources (high ISO, low light). On already-clean JPEGs it can add speckle, color blotches, and fake detail in smooth areas like sky and skin. If `high` looks worse, switch back to `low`.

For **art** and **manga**, `low` is lighter cleanup; `high` is more aggressive on compression artifacts.

**HEIC (iPhone photos):** Photo mode runs two passes automatically — noise removal, then compression cleanup. Strength controls how aggressive the first pass is.

Denoise requires network access **only** the first time each model is needed. After download, runs are offline like upscale.

## Managing models

These commands apply to **upscale models you imported**. Denoise models are managed automatically and do not appear in `models list`.

```bash
easyupscaler models list
easyupscaler models import /path/to/model.safetensors
easyupscaler models import /path/to/model.pth --force   # replace existing same name
easyupscaler models default RealESRGAN_x4plus
easyupscaler models remove RealESRGAN_x4plus              # prompts for confirmation
easyupscaler models remove RealESRGAN_x4plus --yes        # skip prompt
```

Import derives the registry name from the filename: `RealESRGAN_x4plus.pth` → `RealESRGAN_x4plus`.

Supported model types: super-resolution (`SR`) and 1× restoration (`Restoration`). Face fixers, inpainting, and other specialty checkpoints are rejected at import.

## Where files are stored

| What | Where |
|------|-------|
| Default model preference | `~/.config/easyupscaler/config.toml` |
| Installed upscale models (registry + weights) | `~/.local/share/easyupscaler/` |

Paths follow [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) conventions. Override with `XDG_CONFIG_HOME` and `XDG_DATA_HOME` if you use them.

## Shell globs

Glob expansion is the shell's job, not easyupscaler's. If `*.png` matches nothing and your shell does not have `nullglob`, you may get a literal `*.png` path and a "file not found" error.

```bash
shopt -s nullglob   # bash
setopt nullglob     # zsh
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `no default model set` | Run `easyupscaler models default <name>` or pass `--model` |
| `model 'foo' not found` | Run `easyupscaler models list` |
| `architecture not recognised` | Update easyupscaler, or confirm the file is a supported upscale model |
| `purpose '…' is not supported` | That checkpoint is inpainting, face restoration, or another unsupported type |
| Slow first run | PyTorch cold start is normal; later files in the same batch reuse the loaded model |
| Out of memory on huge images | Tiling retries with smaller tiles; very large images may still fail |
| Photo denoise looks speckled at `--strength high` | Use `--strength low` — `high` is for noisy sources only |

Run `easyupscaler scale --help` or `easyupscaler denoise --help` for full flag details.

Developers: see [README-DEV.md](README-DEV.md) for clone setup, `make test`, and architecture pointers.

## License

MIT — see [LICENSE](LICENSE).

The `spandrel-extra-arches` dependency includes some architectures with separate license terms. Review that package if unrestricted commercial use matters for your workflow.
