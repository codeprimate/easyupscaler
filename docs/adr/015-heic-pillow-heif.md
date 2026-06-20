# ADR-015: HEIC support via required pillow-heif dependency

**Status:** Accepted  
**Date:** 2026-06-20

## Context

The denoise spec originally proposed `pillow-heif` as an optional `[heic]` extra with a per-file error when HEIC files are passed without the extra installed. Stakeholder decision: **pillow-heif is a required core dependency** — no optional extras, no install-instruction error path.

HEIC/HEIF is common on Apple devices and needed for the photo-mode two-pass pipeline ([ADR-014](./014-heic-two-pass-denoise.md)).

## Decision

1. Add `pillow-heif` to `[project] dependencies` in `pyproject.toml`.
2. Call `register_heif_opener()` once before any image read at the CLI/service boundary (guarded by a module-level flag to avoid duplicate registration).
3. HEIC inputs are read as RGB via Pillow; output is always PNG (never HEIC).

No `[project.optional-dependencies]` section for HEIC. No runtime error instructing users to install an extra.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Optional `[heic]` extra | Stakeholder override — simplifies install and error surface |
| Reject HEIC entirely | Blocks primary iPhone photo use case |
| Native macOS HEIC via system APIs | Not portable; pillow-heif works cross-platform |

## Consequences

**Positive**

- HEIC works out of the box after `pip install easyupscaler`
- Single install path; no feature-gated extras

**Negative**

- Adds native dependency chain (libheif) to all installs
- Slightly larger install footprint for users who never use HEIC

**Follow-up**

- None for MVP
