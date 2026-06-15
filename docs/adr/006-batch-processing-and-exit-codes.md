# ADR-006: Sequential batch processing and exit codes

**Status:** Accepted  
**Date:** 2025-06-15

## Context

MVP supports batch upscaling via shell globbing:

```bash
easyupscaler *.png
easyupscaler photos/*.jpg --model 4xUltrasharp
```

The shell expands globs; the CLI receives multiple file paths. Open questions:

| Topic | Options |
|-------|---------|
| Concurrency | Sequential vs parallel workers |
| Partial failure | Continue vs fail-fast |
| Exit code | Zero if any success vs non-zero if any failure |
| Empty glob | Shell may pass literal `*.png` or zero args depending on `nullglob` |

Parallel upscaling could saturate GPU memory because each Spandrel forward pass is memory-intensive. Fail-fast wastes work when one bad file appears early in a large batch.

## Decision

### Batch semantics

- Process paths **sequentially** in argument order
- Load the upscaler backend **once** per job; reuse for all files
- **Continue** on per-file failure; record each result
- Print per-file status and a final summary (succeeded / failed counts)

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All files processed successfully |
| `1` | Any failure: missing model, no inputs, import error, or one or more files failed to upscale |

### Empty input

If the CLI receives **zero** image paths, fail immediately with a clear message (do not load the model). This covers `nullglob` producing no arguments.

When the shell passes a literal non-existent glob (e.g. `*.png` with no matches and without `nullglob`), treat as a normal file-not-found per-file failure.

### Globbing

Glob expansion is the **shell's responsibility**. The application does not implement in-app glob or recursive directory walk in MVP.

## Consequences

**Positive**

- Predictable GPU memory use
- One failed corrupt image does not block the rest of the batch
- Non-zero exit code enables scripting (`set -e` / CI) to detect partial failure
- Simple loop implementation; easy to test

**Negative**

- Large batches run slower than parallel execution
- Users on shells without `nullglob` may see confusing literal-path errors

**Follow-up**

- Optional `--jobs N` for parallel CPU-bound preprocessing (not MVP)
- Optional `--fail-fast` flag for scripts that prefer early stop
