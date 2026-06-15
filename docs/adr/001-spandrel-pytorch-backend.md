# ADR-001: Spandrel + PyTorch as sole upscaling backend

**Status:** Accepted  
**Date:** 2025-06-15

## Context

The mission requires running GAN and other upscaler models from imported local weight files. Several execution strategies exist:

- **PyTorch + Spandrel** — loads community `.pth` / `.safetensors` models (ESRGAN, Real-ESRGAN, SwinIR, etc.) in process; used by ComfyUI, AUTOMATIC1111, InvokeAI, and chaiNNer
- **ncnn-vulkan subprocess** — fast inference for pre-converted `.param`/`.bin` models; used by Upscayl and Real-ESRGAN-ncnn-vulkan; does not accept arbitrary `.pth` imports
- **Hybrid** — Spandrel for PyTorch weights, ncnn for converted models

MVP must stay simple: one import path, one inference path, one set of dependencies.

Spandrel provides **architecture detection and model loading** only. It does not ship a complete inference pipeline (tiling, I/O, device management). ComfyUI and A1111 wrap Spandrel with their own tiling and tensor conversion code. easyupscaler must do the same ([ADR-007](./007-tiled-inference.md)).

## Decision

Use **Spandrel on PyTorch** as the only upscaling backend for MVP.

- Supported import formats: `.pth`, `.safetensors`, and other formats Spandrel loads natively
- Single `SpandrelBackend` implements the `UpscalerBackend` protocol
- Use `ModelLoader(device=...)` and `ImageModelDescriptor.__call__` for forward passes
- At import, require `isinstance(model, ImageModelDescriptor)` and `purpose == "SR"`
- ncnn-vulkan and additional backends are out of scope until a follow-up ADR

Optional: call `spandrel_extra_arches.install()` at startup to support additional architectures with restrictive licenses. Document that this is for private/non-commercial use per Spandrel licensing. Not required for MVP core path.

## Consequences

**Positive**

- Same `.pth` models work as in ComfyUI and OpenModelDB without conversion
- Pure Python API: no subprocess or external binary management
- Spandrel inspects architecture and scale from weights at load time
- `.safetensors` avoids pickle deserialization risks present in `.pth` files

**Negative**

- Heavy dependency tree (`torch`, large install size; multi-second cold import)
- Spandrel does not provide tiling — easyupscaler must implement it
- Not every community checkpoint is supported (`UnsupportedModelError`)
- Slower on CPU than ncnn-vulkan on some hardware
- Users with only ncnn-format models cannot import them in MVP

**Follow-up**

- Add ADR for ncnn or hybrid backend when format coverage or install size becomes a requirement
- Pin Spandrel version; track upstream architecture support
