Mission:  create a python cli that provides a simple and straightforward way to run GAN and other upscalers to resize images.

Example:

$ easyupscaler models list => installed models
$ easyupscaler models import /path/to/4xUltrasharp.pth => import/save a model
$ easyupscaler models default 4xUltrasharp => saves preference for model as default
$ easyupscaler input.png => input-upscaled.jpg
$ easyupscaler --model remacri input.png => input-upscaled.jpg
$ easyupscaler *.png => upscales each match (foo.png => foo-upscaled.jpg)
$ easyupscaler photos/*.jpg --model 4xUltrasharp => batch with chosen model

Features (MVP):

## Core upscaling
- Upscale one or more images: `easyupscaler input.png` → `input-upscaled.jpg`
- Batch via globbing: `easyupscaler *.png` or `easyupscaler photos/*.jpg` (shell expands patterns into multiple inputs)
- Per-file output naming: `foo.png` → `foo-upscaled.jpg` (same directory as each input)
- Override model per run: `easyupscaler --model remacri input.png`
- Use saved default model when `--model` is omitted
- Support common image formats for input and output (PNG, JPEG at minimum)
- Tiled inference for large images (avoid GPU/memory exhaustion)

## Model management
- List installed models: `easyupscaler models list`
- Import a model from a local file: `easyupscaler models import <path>`
- Set the default upscaler model: `easyupscaler models default <model-name>`

## CLI and runtime
- Single `easyupscaler` entry point (Python CLI)
- Local storage for imported models and default-model preference
- Fast commands that do not load PyTorch (`models list`, `models default`, `--help`)
- Clear errors for missing model, bad input path, failed import, and empty glob (no matches)
- Report progress and per-file success/failure during batch runs

## Out of scope for MVP
- GUI or web UI
- Model training or fine-tuning
- URL or remote model import
- Cloud upload or remote inference API
- ncnn-vulkan or multi-backend inference
