# ADR-012: Auto-download of managed denoise model weights

**Status:** Accepted  
**Date:** 2026-06-20

## Context

Upscaler models are user-imported via `models import` and tracked in `registry.json`. Denoise models are tool-managed: fixed architectures, hardcoded download URLs, and automatic selection by mode and strength. Users must not import or name denoise weights manually.

The spec requires lazy per-invocation download (only weights needed for the current run), job-level failure on download errors (not per-file), and deletion of corrupt downloads before failing.

## Decision

### Storage

Denoise weights live in the same directory as user-imported upscaler weights:

`$XDG_DATA_HOME/easyupscaler/models/`

Each catalog entry uses a **canonical filename** (e.g. `scunet_color_real_psnr.pth`). Denoise files are **not** registry entries and do **not** appear in `models list`.

### Download lifecycle

1. Before inference, resolve required catalog keys from mode, strength, and input format.
2. For each required key, if the file is absent, download from the hardcoded HTTPS URL.
3. Stream to a temporary file in `MODELS_DIR`, then rename atomically.
4. Show download progress (Rich bar in TTY; single status line when piped).
5. If download fails, abort the entire job with a clear error including URL and `MODELS_DIR`.
6. If a downloaded file is corrupt (validation or load failure), delete it and abort the entire job.

### Catalog

Five managed models with fixed URLs are defined in `DENOISE_MODEL_CATALOG` in code. No user configuration or remote catalog updates in MVP.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Separate `denoise/` subdirectory | Spec chooses shared `MODELS_DIR`; simpler path resolution |
| Bundle weights in the package | ~440 MB worst case; bloats install and updates |
| Per-file download failure | Spec requires job-level abort; partial catalog is unusable |
| Registry entries for denoise models | Would pollute `models list` and invite user confusion |

## Consequences

**Positive**

- Zero manual setup for denoise beyond first-run download
- Upscaler and denoise weights coexist without schema changes

**Negative**

- First denoise run requires network access
- Shared directory mixes user and managed files (distinguished by filename only)

**Follow-up**

- Optional offline mirror or checksum verification beyond size check
