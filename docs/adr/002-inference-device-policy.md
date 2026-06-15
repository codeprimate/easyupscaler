# ADR-002: MPS with CPU fallback for inference

**Status:** Accepted  
**Date:** 2025-06-15

## Context

PyTorch supports multiple compute devices: CPU, CUDA (NVIDIA), and MPS (Apple Silicon). The primary development target is Apple Silicon. MVP should use hardware acceleration where available without blocking users when acceleration is unavailable.

Research findings:

- MPS does not implement every PyTorch op; some models hit `NotImplementedError` mid-forward-pass
- `PYTORCH_ENABLE_MPS_FALLBACK=1` routes unsupported ops to CPU but must be set **before** `import torch`
- Device-level fallback (MPS unavailable at startup) is insufficient on its own
- MPS requires arm64 Python and macOS 14.0+ with MPS-enabled hardware

Options considered:

| Policy | Behavior |
|--------|----------|
| GPU required | Fail if no CUDA/MPS |
| CPU only | Simplest; slow on large images |
| MPS then CPU | Prefer Apple GPU; degrade gracefully |
| MPS / CUDA / CPU cascade | Broad platform support; more complexity |

CUDA support adds detection logic and CI matrix cost. MVP does not require NVIDIA GPUs.

## Decision

### Process startup

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the CLI entry path **before** importing `torch` (document in README for direct module use).

### Device selection (once per upscale job)

1. **MPS** if `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()`
2. **CPU** otherwise

Emit a **warning to stderr** when selecting CPU at initialization.

Pass the chosen device to `ModelLoader(device=...)` and move input tensors to `model.device` with matching dtype (float32 unless half is explicitly enabled later).

### Runtime fallback

If a forward pass raises `NotImplementedError` or an MPS-specific runtime error on the selected device:

1. Log a warning naming the failure
2. Retry the **entire image** (or tile — see ADR-007) on CPU
3. Fail with a clear error if CPU also fails

### Memory hygiene

After each large image in a batch, optionally call `torch.mps.empty_cache()` when running on MPS to reduce unified-memory pressure on Apple Silicon.

CUDA is not used in MVP. Intel Mac, Linux, and Windows run on CPU until a future ADR adds CUDA.

## Consequences

**Positive**

- Fast inference on Apple Silicon for supported ops
- Op-level and device-level fallback improve compatibility vs MPS-only
- CLI still works when MPS is unavailable

**Negative**

- MPS/CPU mixed execution can be slower than CPU-only for op-heavy models
- No NVIDIA acceleration in MVP
- CPU inference on large untiled images may be impractically slow (mitigated by ADR-007)

**Implementation note**

Device is chosen once per upscale job and reused for all images in the batch.
