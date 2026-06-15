# ADR-004: Model registry, XDG storage, and import rules

**Status:** Accepted  
**Date:** 2025-06-15

## Context

Models must persist across CLI invocations. The tool needs:

- A directory for weight files
- Metadata mapping logical names to paths and scale
- User preference for the default model
- Import from a **local file path only**

Naming and scale factor were open questions. Options for naming: filename stem, required `--name`, or optional override. Options for scale: assume 4×, user flag at import, or read from model.

`.pth` files use pickle-based deserialization and pose a security risk when loading untrusted checkpoints. Spandrel restricts pickle but does not eliminate arbitrary code execution risk; `.safetensors` is preferred when available.

## Decision

### Storage (XDG Base Directory)

| Resource | Path |
|----------|------|
| Config | `$XDG_CONFIG_HOME/easyupscaler/config.toml` |
| Registry | `$XDG_DATA_HOME/easyupscaler/registry.json` |
| Weights | `$XDG_DATA_HOME/easyupscaler/models/` |

When `XDG_*` variables are unset, use `~/.config` and `~/.local/share`.

### Registry schema

Each model entry stores:

- `name` — logical name used on CLI
- `filename` — basename on disk
- `path` — absolute path to weights
- `scale` — integer upscale factor from Spandrel at import
- `imported_at` — ISO 8601 timestamp

### Import rules

1. **Name:** derived from filename stem (`4xUltrasharp.pth` → `4xUltrasharp`). No `--name` flag in MVP.
2. **Scale:** read when validating weights through Spandrel at import; stored in registry. Reject if scale is missing or not suitable for upscaling.
3. **Source:** local file path only. File must exist and be readable.
4. **Copy:** copy (or hardlink when same filesystem) into the models directory; preserve original filename.
5. **Duplicate name:** reject import if registry already contains that name.
6. **Validation:** load with Spandrel before registering; require `ImageModelDescriptor` with `purpose == "SR"`. On `UnsupportedModelError`, fail with message suggesting an unsupported architecture or outdated Spandrel version.

### Config

`config.toml` holds `default_model = "<name>"`. Setting default validates that the name exists in the registry.

## Consequences

**Positive**

- Standard paths familiar to Linux/macOS users
- JSON registry is human-readable and easy to test
- Scale stored once at import; upscale path avoids re-inspection
- Filename-based naming matches `models default 4xUltrasharp` mission example
- Local-only import avoids URL redirect, auth, and drive-link complexity

**Negative**

- Importing the same weights under different names requires renaming the file first
- Registry and files can drift if user deletes files manually; MVP does not repair automatically
- Users must obtain model files outside easyupscaler (browser, git, etc.)

**Follow-up**

- Optional `--name` on import
- Registry integrity check command (`models verify`)
- Warn when importing `.pth` from untrusted sources
