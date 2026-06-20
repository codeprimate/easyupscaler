# Mission

easyupscaler is a Python CLI for upscaling local images with community super-resolution models. One command imports weights, remembers a default, and runs batch upscales from the terminal — no GUI, no workflow editor, no weight conversion.

## Problem

Running GAN and ESRGAN-style upscalers against your own files means assembling PyTorch, Spandrel, tiling, device selection, and output handling. That work belongs in a tool, not in every script or one-off job.

Existing options embed upscaling inside larger products. ComfyUI and A1111 assume a node graph or web UI. Upscayl is a desktop app. None offer a focused, scriptable CLI that accepts arbitrary community `.pth` and `.safetensors` weights unchanged.

## Who this is for

- Developers and power users who already have upscaler weights from ComfyUI, A1111, or [OpenModelDB](https://openmodeldb.info/)
- Anyone who wants `easyupscaler photo.png` in a shell script, Makefile, or batch folder workflow
- Apple Silicon Mac users who want MPS inference with sensible CPU fallback

Secondary support on Linux (CPU, best-effort) is acceptable. Windows and Intel Mac are not MVP targets.

## What we are building

A single entry point, `easyupscaler`, with two concerns:

1. **Model lifecycle** — import local weight files, list what is installed, set a default, remove models you no longer need
2. **Upscaling** — run the default or a named model against one or more image paths, with predictable outputs beside each input

The shell expands globs. The CLI receives a flat list of paths. Runtime is local only: no network calls, no remote inference, no training.

## MVP scope

Capabilities in scope for the first release:

### Upscaling

- One or many images per invocation; continue on per-file failure
- Override model per run or use a saved default
- Common input formats (PNG, JPEG at minimum); consistent JPEG output beside each source file
- Tiled inference for large images, with automatic tile-size reduction on out-of-memory
- Progress and per-file success/failure reporting suitable for interactive use and piping

### Model management

- Import from a local file path; validate through Spandrel
- List installed models with name, scale, and filename
- Set and clear the default model preference
- Remove a model from local storage

Supported model purposes: super-resolution (`SR`) and 1× restoration/detail enhancement (`Restoration`). Scale comes from model metadata.

### Runtime qualities

- Fast housekeeping commands that never load PyTorch (`models list`, `models default`, `models remove`, `--help`)
- Local persistence under XDG paths for config, registry, and copied weights
- Clear errors for missing default, unknown model, bad input, failed import, and batch partial failure
- Scriptable exit codes: success only when every requested file succeeds

Command syntax, messages, and edge cases live in [specification.md](./specification.md). Install steps and troubleshooting live in [README.md](../README.md).

## Success criteria

The MVP succeeds when:

- A user with Python 3.13+, a weight file, and five commands or fewer can import a model and upscale a photo
- Batch runs are automation-friendly: predictable exit codes, per-file status, and progress on stdout
- `models list` and `--help` respond without importing PyTorch
- Any SR or Restoration model loadable by Spandrel can be imported and used without conversion

## Out of scope (MVP)

- GUI or web UI
- Model training or fine-tuning
- URL or remote model import
- Cloud upload or remote inference API
- ncnn-vulkan or additional inference backends
- In-app glob expansion or recursive directory walks
- Parallel batch workers
- Custom output formats or quality settings (optional `--output` directory is supported; see [ADR-016](./adr/016-optional-output-directory.md))

## Related documents

| Document | Use when |
|----------|----------|
| [specification.md](./specification.md) | CLI contracts, flags, messages, exit codes |
| [architecture.md](./architecture.md) | Layers, components, data flow, testing strategy |
| [adr.md](./adr.md) | Immutable architectural decisions |
| [README.md](../README.md) | Install, quickstart, user troubleshooting |
