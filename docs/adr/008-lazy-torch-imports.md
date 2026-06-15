# ADR-008: Lazy imports for PyTorch and Spandrel

**Status:** Accepted  
**Date:** 2025-06-15

## Context

Importing `torch` typically adds **2–6 seconds** to process startup, even when the command only prints help or lists registry entries. A CLI that imports torch at module load time makes `easyupscaler models list` and `easyupscaler --help` feel broken.

PyTorch environment variables such as `PYTORCH_ENABLE_MPS_FALLBACK` must be set before the first `import torch`.

## Decision

### Environment before torch

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the CLI bootstrap (e.g. `cli/main.py` top) before any lazy torch import.

### Lazy import boundaries

| Command / path | Import torch/spandrel? |
|----------------|------------------------|
| `easyupscaler --help` | No |
| `easyupscaler models list` | No |
| `easyupscaler models default` | No |
| `easyupscaler models import` | Yes (validation) |
| `easyupscaler <images...>` | Yes (inference) |

Implement lazy loading by importing heavy modules inside service functions or dedicated loader modules, not at package `__init__.py` or Typer module top level (except the env-var bootstrap in `main.py`).

### Structure

```
cli/main.py          # sets env vars; Typer app; no torch import
models/registry.py   # JSON only; no torch
config/settings.py   # TOML only; no torch
upscaling/...        # imports torch when UpscaleService runs
models/import_model.py  # imports torch/spandrel on import_model()
```

## Consequences

**Positive**

- Fast feedback for common non-inference commands
- Env vars applied consistently before torch loads
- Matches CLI best practices for ML-backed tools

**Negative**

- Import paths are less obvious; deferring imports requires discipline in new code
- Type checkers may need `TYPE_CHECKING` guards for torch types in signatures
