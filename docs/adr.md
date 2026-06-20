# Architecture Decision Records

Index of accepted decisions for easyupscaler MVP. Each ADR is immutable once accepted; supersede by adding a new numbered ADR.

| ADR | Title | Status |
|-----|-------|--------|
| [001](./adr/001-spandrel-pytorch-backend.md) | Spandrel + PyTorch as sole upscaling backend | Accepted |
| [002](./adr/002-inference-device-policy.md) | MPS with CPU fallback and op-level MPS fallback | Accepted |
| [003](./adr/003-image-output-conventions.md) | Always JPEG output with `-upscaled` naming | Accepted |
| [004](./adr/004-model-registry-and-import.md) | Model registry, XDG storage, and local import rules | Accepted |
| [005](./adr/005-cli-framework-typer.md) | Typer for CLI framework | Accepted |
| [006](./adr/006-batch-processing-and-exit-codes.md) | Sequential batch processing and exit codes | Accepted |
| [007](./adr/007-tiled-inference.md) | Tiled inference for large images | Accepted |
| [008](./adr/008-lazy-torch-imports.md) | Lazy imports for PyTorch and Spandrel | Accepted |
| [009](./adr/009-development-toolchain.md) | uv, ruff, mypy, and Makefile for dev/build/test | Accepted |
| [010](./adr/010-code-coverage-gate.md) | ≥80% line coverage gate via pytest-cov | Accepted |
| [011](./adr/011-output-conflict-indexing.md) | Indexed output filenames on conflict | Accepted |
| [012](./adr/012-denoise-model-auto-download.md) | Auto-download of managed denoise model weights | Accepted |
| [013](./adr/013-denoise-png-output.md) | PNG output for denoise command | Accepted |
| [014](./adr/014-heic-two-pass-denoise.md) | Two-pass denoise pipeline for HEIC photo inputs | Accepted |
| [015](./adr/015-heic-pillow-heif.md) | HEIC support via required pillow-heif dependency | Accepted |
| [016](./adr/016-optional-output-directory.md) | Optional output directory via `--output` / `-o` | Accepted |
| [017](./adr/017-scikit-image-dependency.md) | scikit-image as hard runtime dependency for document mode | Accepted |
| [018](./adr/018-document-two-pass-pipeline.md) | Two-pass AI pipeline for document denoise mode | Superseded by [019](./adr/019-document-binarize-antialias.md) |
| [019](./adr/019-document-binarize-antialias.md) | Document mode: Archiver + Sauvola binarize + anti-alias | Accepted |
