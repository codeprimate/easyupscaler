# ADR-016: Optional output directory via `--output`

**Status:** Accepted  
**Date:** 2026-06-20

## Context

[ADR-003](./003-image-output-conventions.md) and [ADR-011](./011-output-conflict-indexing.md) define upscaled output beside each input file. [ADR-013](./013-denoise-png-output.md) does the same for denoise. Users running batch jobs often want all outputs in one directory (e.g. `./results/`) without copying inputs.

The MVP spec listed `--output` as a future flag. Mission originally excluded custom destination directories.

## Decision

Add optional **`--output DIR`** (short **`-o`**) to both `scale` and `denoise` commands.

| Flag | Behavior |
|------|----------|
| Omitted | Unchanged: write beside each input ([ADR-003](./003-image-output-conventions.md), [ADR-013](./013-denoise-png-output.md)) |
| `--output DIR` / `-o DIR` | Write all outputs under `DIR` |

**Naming and conflict rules are unchanged** ([ADR-011](./011-output-conflict-indexing.md)): `{stem}-upscaled.jpg` / `{stem}-denoised.png` with indexed suffixes when the base name exists in the **output directory**.

**Directory handling (CLI boundary, before inference):**

- If `DIR` does not exist: create with `mkdir(parents=True, exist_ok=True)`
- If `DIR` exists but is a file: fail with `Error: output path is not a directory: <path>` (exit 1)
- If `DIR` is not writable after create: fail before inference (exit 1)

Read-only failures at write time remain per-file batch failures (existing spec).

## Consequences

**Positive**

- Script-friendly batch output to a single folder
- Default beside-input behavior preserved for quick one-off runs
- Same flag on `scale` and `denoise` for consistent UX

**Negative**

- Same stem from different input directories can collide in one output dir (ADR-011 indexing handles this)
- Two output-location policies in one tool (default vs flag)

**Supersedes (partially)**

- ADR-003 and ADR-011 remain accepted for default behavior, format, and naming; their "same directory as input" clauses apply only when `--output` is omitted

**Follow-up**

- Per-input output paths or `--format` remain out of scope
