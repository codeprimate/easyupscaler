# Specification: easyupscaler MVP

## Problem Statement

Image upscaling with GAN-based models requires assembling PyTorch, model weights, and a working inference pipeline — a research exercise most users should not face. Existing tools (ComfyUI, A1111, Upscayl) either embed upscaling inside a larger product or require a GUI. No focused, scriptable Python CLI exists for running arbitrary community `.pth`/`.safetensors` models against local images.

easyupscaler fills that gap: a single `easyupscaler` command that manages models and upscales images, with no GUI, no network dependencies at runtime, and no assumptions about which models are installed.

**Success criteria**

- A user with Python 3.13, uv, and a model file can upscale a photo in under five commands
- Batch runs are scriptable: predictable exit codes, per-file status, and progress feedback
- `models list` and `--help` respond without loading PyTorch
- Any SR model loadable by Spandrel can be imported and used unchanged

---

## Requirements

### Functional Requirements

#### 1. Scale command (upscale)

`easyupscaler scale [--model <name>] [--output DIR] <image> [<image> ...]`

**Breaking change:** the bare invocation `easyupscaler <image>` no longer works. Use `easyupscaler scale` instead. Typer shows an error directing users to `scale` or `denoise`.

Full denoise command specification: [specification-denoise.md](./specification-denoise.md). Document mode: [specification-document-mode.md](./specification-document-mode.md).

- Accepts one or more image paths as positional arguments
- Shell expands glob patterns before the process starts; the CLI receives a flat list of paths
- If `--model` is omitted, reads `default_model` from config; fails with a clear error if neither is set
- Optional `--output DIR` / `-o DIR` writes all outputs under `DIR` (created if missing); see [ADR-016](./adr/016-optional-output-directory.md)
- Processes images **sequentially** in argument order; continues on per-file failure
- By default, writes `{stem}-upscaled.jpg` beside each input (same directory); with `--output`, writes under `DIR` instead
- If the target `{stem}-upscaled.jpg` already exists in the output location, writes `{stem}-upscaled-NNNN.jpg` with the lowest available 4-digit index (`0001`, `0002`, …)
- Exits `0` if all files succeeded; exits `2` if invoked with no arguments (displays help); exits `1` if any file failed or image paths are given but all fail to process
- Shows a Rich progress bar when stdout is a TTY; falls back to one plain status line per file when piped or redirected

**stdout / stderr contract**

```
# TTY (progress bar + per-file lines):
Upscaling 3 images with RealESRGAN_x4plus [███████████████░░░░] 2/3
  ✓ photo.png → photo-upscaled.jpg
  ✓ scan.jpeg → scan-upscaled.jpg
  ✗ broken.png — unsupported format
Completed: 2 succeeded, 1 failed in 1:23.

# Non-TTY (pipe/redirect):
photo.png → photo-upscaled.jpg
scan.jpeg → scan-upscaled.jpg
broken.png FAILED: unsupported format
```

Errors and warnings go to **stderr**. Progress output goes to **stdout**.

#### 2. Model management

##### `easyupscaler models list`

- Prints a table: **Name**, **Scale**, **Filename**
- If no models are registered, prints: `No models installed. Use 'easyupscaler models import <path>' to add one.`
- Does **not** load PyTorch; reads `registry.json` only

```
Name                Scale  Filename
──────────────────────────────────────────────
RealESRGAN_x4plus   4×     RealESRGAN_x4plus.pth
4xUltraSharp        4×     4xUltraSharp.pth
```

##### `easyupscaler models import <path> [--force]`

- `<path>` must be a local file path (`.pth`, `.safetensors`, or other Spandrel-supported format)
- Copies the file into `$XDG_DATA_HOME/easyupscaler/models/` preserving the original filename
- Derives registry name from the filename stem (`RealESRGAN_x4plus.pth` → `RealESRGAN_x4plus`)
- Validates the file through Spandrel; requires `ImageModelDescriptor` with `purpose == "SR"`
- Reads and stores `scale` from model metadata
- Rejects if a model with the same name already exists, unless `--force` is given; `--force` overwrites the registry entry and replaces the weight file
- Emits a **stderr warning** when importing `.pth` files (pickle-based; risk of untrusted code)
- On `UnsupportedModelError`: fails with message naming the architecture; suggests updating Spandrel
- Prints on success: `Imported RealESRGAN_x4plus (4×) from RealESRGAN_x4plus.pth`
- Loads PyTorch/Spandrel (lazy import)

##### `easyupscaler models default <name>`

- Writes `default_model = "<name>"` to `config.toml`
- Validates that `<name>` exists in the registry first; fails with a listing of installed models if not
- Does **not** load PyTorch
- Prints on success: `Default model set to RealESRGAN_x4plus`

##### `easyupscaler models remove <name>`

- Removes the named entry from `registry.json`
- Deletes the associated weight file from `models/`
- If the deleted model was the current default, clears `default_model` from config and warns: `Warning: RealESRGAN_x4plus was the default model. No default is now set.`
- Does **not** load PyTorch
- Asks for confirmation unless `--yes` / `-y` is passed: `Remove RealESRGAN_x4plus and delete RealESRGAN_x4plus.pth? [y/N]`
- Fails with a clear error if `<name>` is not registered

#### 3. CLI top-level flags

- `--version` — print `easyupscaler <version>` and exit `0`; no torch load
- `--help` — Typer-generated help; no torch load
- `--model NAME` — override default model for the upscale command

---

### Technical Constraints

#### Platform

- **Primary:** Apple Silicon macOS (arm64), macOS 14.0+, arm64 Python 3.13+
- **Secondary (best-effort):** Linux x86_64 or arm64, CPU-only
- Windows and Intel macOS are not tested or supported in MVP

#### Python and dependencies

- Minimum Python: **3.13**
- Package manager: **uv** for development; installable via `pip` for end users
- All versions pinned in `pyproject.toml` and `uv.lock`
- No network access at runtime (no HTTP client)

#### Performance

- `models list`, `models default`, `models remove`, `--help`, `--version`: respond in under **500 ms** (no PyTorch import)
- `models import`: under **30 s** for a 100 MB weight file on M2 (dominated by Spandrel load)
- Upscale: no hard latency target in MVP; tiled inference must not OOM on images up to 4000×3000 on 16 GB unified memory

#### Type checking and linting

- `mypy easyupscaler` must pass with no errors (scoped `ignore_missing_imports` permitted for untyped third-party packages)
- `ruff check` must pass with no errors
- Both are enforced by `make test` before pytest runs

---

### Edge Cases and Error Handling

| Scenario | Behavior |
|----------|----------|
| Zero image paths provided | Error before torch load: `Error: no input images. Pass one or more file paths.` Exit 1. |
| Input path does not exist | Per-file failure: `broken.png FAILED: file not found`. Continue batch. |
| Input file is a directory | Per-file failure: `photos/ FAILED: not a file`. Continue batch. |
| Input is a corrupt/truncated image | Per-file failure: `corrupt.jpg FAILED: cannot read image`. Continue batch. |
| Output directory is read-only | Per-file failure with OS error message. Continue batch. |
| `--output` path is an existing file | Fail before inference: `Error: output path is not a directory: <path>`. Exit 1. |
| `--output` path missing | Create directory with `mkdir -p`, then proceed. |
| `--output` path not writable | Fail before inference: `Error: output directory is not writable: <path>`. Exit 1. |
| `{stem}-upscaled.jpg` already exists | Write `{stem}-upscaled-NNNN.jpg` using the lowest available 4-digit index (`0001`–`9999`). Do not overwrite. |
| Model not in registry at upscale time | Fail before inference: `Error: model 'foo' not found. Installed: ...` Exit 1. |
| No default model and `--model` omitted | Fail before inference: `Error: no default model set. Run 'easyupscaler models default <name>'.` Exit 1. |
| Import: file does not exist | Fail: `Error: path not found: /path/to/file.pth` |
| Import: unsupported purpose (e.g. Inpainting, FaceSR) | Fail: `Error: model purpose '{purpose}' is not supported. Only SR and Restoration models are supported.` |
| Import: scale < 1 or missing | Fail: `Error: model reports invalid scale {n}.` |
| 1× SR model (scale = 1) | Allowed. Output dimensions match input; still writes `{stem}-upscaled.jpg`. |
| Import: duplicate name, no `--force` | Fail: `Error: model 'RealESRGAN_x4plus' already exists. Use --force to replace.` |
| Import: duplicate name, `--force` | Overwrite file and registry entry; print: `Replaced RealESRGAN_x4plus.` |
| Import: corrupt weight file | Fail with Spandrel error text. No registry write. |
| Import: `UnsupportedModelError` | Fail: `Error: architecture not recognised by Spandrel. Try updating easyupscaler, or check that this is a supported SR model.` |
| `models default` with unknown name | Fail: `Error: model 'foo' not found. Installed models: <table>` |
| `models remove` with unknown name | Fail: `Error: model 'foo' not found.` |
| `models remove` default model | Remove; clear config default; emit warning. |
| Registry file missing | Treat as empty registry (first run). Create parent dirs on first write. |
| Config file missing | Treat as no default set. Create on first write. |
| MPS unavailable | Warn to stderr: `Warning: MPS not available, using CPU (inference will be slower).` Continue. |
| MPS op failure mid-inference | Warn: `Warning: MPS error on <model>: <msg>. Retrying on CPU.` Retry. Fail if CPU also fails. |
| OOM during tiled inference | Halve tile size; retry transparently. Warn user if minimum tile reached before success. |
| Grayscale PNG input | Convert to RGB before inference and output. |
| RGBA PNG input | Flatten alpha to white background, convert to RGB. |
| `.pth` import (pickle) | Emit to stderr: `Warning: .pth files use Python pickle, which may execute arbitrary code. Only import models from sources you trust.` |

---

## Technical Approach

### Implementation Strategy

The project is built from scratch as a layered Python package. No existing codebase to integrate with; all components are new. Implementation follows the architecture described in `docs/architecture.md`.

**Key design rules:**
- CLI layer does not import torch at module level (except the env-var bootstrap in `main.py`)
- Services do not know about Typer types or stdout; they return plain Python values
- Domain exceptions in `errors.py` are the only coupling between layers
- All configuration and registry paths derive from a single `paths.py` module; never hardcoded elsewhere

### Affected Components

All files are new. The canonical package layout:

```
easyupscaler/
  __init__.py                  # package version constant
  errors.py                    # EasyUpscalerError, ModelNotFoundError, ImportError, etc.
  cli/
    main.py                    # Typer app, PYTORCH_ENABLE_MPS_FALLBACK, console_scripts entry
    models.py                  # list, import, default, remove commands
    upscale.py                 # upscale command; progress display
  config/
    paths.py                   # XDG_CONFIG_HOME, XDG_DATA_HOME resolution
    settings.py                # ConfigService: load/save config.toml via tomlkit
  models/
    registry.py                # ModelRegistry: CRUD on registry.json
    import_model.py            # import_model(path, force): validate + copy + register
  upscaling/
    service.py                 # UpscaleService: batch orchestration
    tiling.py                  # tiled_upscale(): tensor tiling + OOM retry
    backends/
      base.py                  # UpscalerBackend Protocol
      spandrel_backend.py      # SpandrelBackend: load, device select, forward
  io/
    images.py                  # ImageIO: PIL read/write, numpy conversion
tests/
  conftest.py
  test_registry.py
  test_settings.py
  test_import_model.py
  test_tiling.py
  test_service.py
  test_cli_models.py
  test_cli_upscale.py
pyproject.toml
Makefile
```

### Key Interfaces

#### `UpscalerBackend` protocol (`backends/base.py`)

```python
class UpscalerBackend(Protocol):
    scale: int

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Accept and return float32 RGB ndarray in [0, 1], shape (H, W, 3)."""
        ...
```

#### `ModelRegistry` (`models/registry.py`)

```python
@dataclass
class ModelEntry:
    name: str
    filename: str
    path: Path
    scale: int
    imported_at: datetime

class ModelRegistry:
    def list(self) -> list[ModelEntry]: ...
    def get(self, name: str) -> ModelEntry: ...          # raises ModelNotFoundError
    def add(self, entry: ModelEntry) -> None: ...        # raises DuplicateModelError
    def remove(self, name: str) -> ModelEntry: ...       # raises ModelNotFoundError
    def replace(self, entry: ModelEntry) -> None: ...    # used by --force
```

#### `ConfigService` (`config/settings.py`)

```python
class ConfigService:
    def get_default_model(self) -> str | None: ...
    def set_default_model(self, name: str) -> None: ...
    def clear_default_model(self) -> None: ...
```

#### `UpscaleService` (`upscaling/service.py`)

```python
@dataclass
class UpscaleResult:
    path: Path
    output: Path | None
    error: str | None

class UpscaleService:
    def run(
        self,
        paths: list[Path],
        model_name: str | None,
        on_progress: Callable[[UpscaleResult], None] | None = None,
        *,
        output_dir: Path | None = None,
    ) -> list[UpscaleResult]: ...
```

The `on_progress` callback decouples progress display (CLI) from orchestration (service).

#### `ImageIO` (`io/images.py`)

```python
class ImageIO:
    def read(self, path: Path) -> np.ndarray: ...
    # Returns float32 RGB (H, W, 3) in [0, 1]; handles RGBA and grayscale
    # Raises ImageReadError on corrupt/unsupported files

    def write(self, image: np.ndarray, source_path: Path, *, output_dir: Path | None = None) -> Path: ...
    # Writes {stem}-upscaled.jpg beside source_path or under output_dir; on conflict, {stem}-upscaled-NNNN.jpg
    # quality=95, subsampling=0, convert to RGB first
    # Returns output path
```

#### `tiled_upscale` (`upscaling/tiling.py`)

```python
def tiled_upscale(
    model: ImageModelDescriptor,
    tensor: torch.Tensor,           # (1, C, H, W) on model.device, [0, 1]
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> torch.Tensor: ...
# Returns (1, C, H*scale, W*scale)
# Retries with halved tile on OOM; respects model.tiling metadata
```

### Dependencies and Integration

#### Runtime dependencies

| Package | Version constraint | Role |
|---------|-------------------|------|
| `typer` | `>=0.12` | CLI framework |
| `rich` | `>=13` | Progress bar; TTY detection |
| `pillow` | `>=10` | Image I/O |
| `tomlkit` | `>=0.13` | config.toml read/write |
| `torch` | `>=2.3` | Inference runtime (lazy) |
| `spandrel` | `>=0.4` | Model loader (lazy) |
| `spandrel-extra-arches` | matches spandrel | Additional architectures (always installed) |
| `numpy` | `>=1.26` | Tensor ↔ ndarray conversion |

`spandrel_extra_arches.install()` is called once in `SpandrelBackend.__init__` before the first `ModelLoader` call.

#### Dev dependencies (uv dev group)

| Package | Role |
|---------|------|
| `pytest` | Test runner |
| `pytest-mock` | Mocking in tests |
| `mypy` | Type checking |
| `ruff` | Lint + format |
| `types-Pillow` | Pillow type stubs |

#### `pyproject.toml` structure

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "easyupscaler"
requires-python = ">=3.13"
# ... runtime deps above

[project.scripts]
easyupscaler = "easyupscaler.cli.main:app"

[dependency-groups]
dev = ["pytest", "pytest-mock", "mypy", "ruff", "types-Pillow"]

[tool.mypy]
strict = false
ignore_missing_imports = true   # scoped to torch/spandrel; tighten per module

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

#### Makefile

```makefile
.PHONY: sync lint typecheck test build install

sync:
	uv sync

lint:
	uv run ruff check easyupscaler tests

typecheck:
	uv run mypy easyupscaler

test: lint typecheck
	uv run pytest

build:
	uv build

install:
	uv pip install .
```

---

## Acceptance Criteria

### Setup and install

- [ ] `uv sync` from a clean clone installs all dependencies without errors on arm64 macOS
- [ ] `make test` (lint → typecheck → pytest) passes on a fresh clone with no manual steps
- [ ] `make install` installs the `easyupscaler` binary into the active Python environment
- [ ] `easyupscaler --version` prints the version and exits `0` without loading PyTorch
- [ ] `easyupscaler --help` prints usage and exits `0` without loading PyTorch

### Model management

- [ ] `easyupscaler models list` prints an empty-state message when no models are registered, without loading PyTorch
- [ ] `easyupscaler models import /path/to/RealESRGAN_x4plus.pth` copies the file, registers name=`RealESRGAN_x4plus` scale=4, and prints success
- [ ] `easyupscaler models import` accepts scale-1 SR models (e.g. detail enhancers); registry stores `scale: 1`
- [ ] A `.pth` import emits a pickle security warning to stderr
- [ ] `easyupscaler models list` shows a table with Name, Scale, Filename after import
- [ ] Importing the same name a second time fails with an error message referencing `--force`
- [ ] `easyupscaler models import ... --force` replaces the existing entry and weight file
- [ ] `easyupscaler models default RealESRGAN_x4plus` writes the config and prints confirmation
- [ ] `easyupscaler models default nonexistent` fails with exit `1` and lists installed models
- [ ] `easyupscaler models remove RealESRGAN_x4plus` prompts for confirmation, then removes registry entry and weight file
- [ ] `easyupscaler models remove RealESRGAN_x4plus --yes` removes without prompting
- [ ] Removing the default model clears the default and emits a warning
- [ ] Importing a non-SR model (e.g. denoiser) fails with a clear purpose error

### Upscaling

- [ ] `easyupscaler input.png` with a default model set produces `input-upscaled.jpg` in the same directory
- [ ] `easyupscaler --model RealESRGAN_x4plus input.png` produces correct output ignoring default
- [ ] Output JPEG is `quality=95`, `subsampling=0`; dimensions are `input × model.scale` (1× models preserve input size)
- [ ] `easyupscaler *.png` (shell-expanded) upscales all matched files; exit `0` if all succeed
- [ ] One corrupt file in a batch does not abort the remaining files; exit `1` at end
- [ ] A RGBA PNG input produces a valid RGB JPEG output (no crash, alpha discarded)
- [ ] A grayscale PNG input produces a valid RGB JPEG output
- [ ] An image with both dimensions ≤ 512 is processed in a single forward pass (no tiling)
- [ ] An image larger than 512 in either dimension is processed with tiling (no OOM on 16 GB)
- [ ] Running `scale` with no arguments displays help and exits `2` before loading PyTorch
- [ ] Running with no default model and no `--model` exits `1` with the missing-default error
- [ ] TTY output includes a progress bar; non-TTY output is plain per-file lines

### Code quality

- [ ] `ruff check easyupscaler tests` passes with zero warnings
- [ ] `mypy easyupscaler` passes with zero errors
- [ ] All unit tests pass without a GPU or model weights present
- [ ] `models list`, `models default`, `models remove` tests assert that `torch` was not imported

---

## Implementation Tasks

### Phase 1: Scaffold and toolchain

- [ ] Initialise `pyproject.toml` with `hatchling`, runtime deps, dev group, scripts entry
- [ ] Create `Makefile` with `sync`, `lint`, `typecheck`, `test`, `build`, `install` targets
- [ ] Add `ruff` and `mypy` configuration to `pyproject.toml`
- [ ] Create package skeleton: all empty `__init__.py` files, `errors.py` with base exception class
- [ ] Verify `make test` passes on empty skeleton

### Phase 2: Storage and config (no torch)

- [ ] `config/paths.py` — XDG path resolution with macOS fallbacks; create dirs on first use
- [ ] `config/settings.py` — `ConfigService`: load/save/clear `default_model` via tomlkit
- [ ] `models/registry.py` — `ModelEntry` dataclass; `ModelRegistry` CRUD on `registry.json`
- [ ] Unit tests: `test_settings.py`, `test_registry.py` (assert no torch in `sys.modules`)

### Phase 3: CLI skeleton (no torch)

- [ ] `cli/main.py` — Typer app; set `PYTORCH_ENABLE_MPS_FALLBACK=1`; `--version` callback
- [ ] `cli/models.py` — `list`, `default`, `remove` commands (torch-free; import deferred)
- [ ] Integration tests: `test_cli_models.py` using `CliRunner`; assert torch not imported for list/default/remove

### Phase 4: Image I/O

- [ ] `io/images.py` — `ImageIO.read()`: Pillow load, RGBA/grayscale normalise, float32 ndarray
- [ ] `io/images.py` — `ImageIO.write()`: `{stem}-upscaled.jpg`, `quality=95`, `subsampling=0`
- [ ] Unit tests: `test_images.py` with fixture PNG/JPEG files; test RGBA, grayscale, naming, conflict indexing

### Phase 5: Tiling

- [ ] `upscaling/tiling.py` — `tiled_upscale()`: split, per-tile forward, weighted merge, OOM retry
- [ ] Honour `model.tiling`: `SUPPORTED`, `DISCOURAGED`, `INTERNAL`
- [ ] Unit tests: `test_tiling.py` with a fake `ImageModelDescriptor`; test single-pass, tiled, OOM halving

### Phase 6: Backend and service (torch)

- [ ] `upscaling/backends/base.py` — `UpscalerBackend` Protocol
- [ ] `upscaling/backends/spandrel_backend.py` — device selection, `spandrel_extra_arches.install()`, `ModelLoader`, MPS op-failure retry
- [ ] `upscaling/service.py` — `UpscaleService.run()`: resolve model, load backend once, batch loop, `on_progress` callback, MPS cache flush
- [ ] Unit tests: `test_service.py` with a fake backend injected via Protocol
- [ ] `models/import_model.py` — copy, Spandrel validate, purpose/scale check, registry write, `--force` path
- [ ] Unit tests: `test_import_model.py` with mocked Spandrel

### Phase 7: Upscale CLI command

- [ ] `cli/upscale.py` — parse paths and `--model`; TTY detection; Rich progress bar; plain-line fallback; print summary
- [ ] `cli/models.py` — add `import` command (deferred torch)
- [ ] Integration tests: `test_cli_upscale.py` with mocked `UpscaleService`; test progress output, exit codes, empty paths

### Phase 8: Slow / end-to-end tests

- [ ] Add `@pytest.mark.slow` to `conftest.py`
- [ ] `tests/slow/test_e2e.py` — import Real-ESRGAN x4plus fixture, upscale a 64×64 PNG, assert output is 256×256 JPEG
- [ ] Document that slow tests require a real model file and are excluded from CI by default (`pytest -m "not slow"`)

### Phase 9: Polish

- [ ] `README.md` — install, quickstart, shell glob tips, `nullglob` note, model sources (OpenModelDB), troubleshooting (MPS, OOM, unsupported model)
- [ ] Populate `--help` strings on all commands
- [ ] Verify `make install` on clean arm64 macOS; `easyupscaler --version` works from PATH

---

## Risk Assessment

### Potential issues

| Risk | Likelihood | Impact | Notes |
|------|-----------|--------|-------|
| Spandrel cannot load a popular community model | Medium | High | `UnsupportedModelError` at import time; user sees clear error |
| MPS op gaps cause inference failure on some models | Medium | Medium | Op-level fallback + CPU retry mitigates; may be slow |
| OOM even at minimum tile size (128) on CPU | Low | Medium | Fail with clear message; future `--tile-size` flag |
| `spandrel_extra_arches` license conflicts for commercial users | Low | Low | Documented in README; extra_arches is non-commercial for restricted archs |
| torch 2.x + Python 3.13 wheel not available at release | Low | High | Check available wheels before final version pin |
| Progress bar (Rich) conflicts with some terminal emulators | Low | Low | TTY detection falls back to plain lines |
| Registry/file drift (user deletes weight manually) | Low | Low | Graceful error at upscale time; `models remove` is the right path |

### Mitigation strategies

- **Version pins first:** run `uv sync` on target hardware before any other work; unblock torch 3.13 wheel availability immediately
- **Spandrel coverage:** use Real-ESRGAN x4plus (official) as the primary test fixture; it is the most widely supported architecture
- **MPS:** test `PYTORCH_ENABLE_MPS_FALLBACK=1` + CPU retry path manually with one MPS-incompatible model op before shipping
- **Tiling:** port ComfyUI's `tiled_scale` pattern directly; it is battle-tested on Apple Silicon

### Investigation requirements

- [ ] **Spike: Python 3.13 + torch wheel** — confirm `torch >=2.3` wheel exists for `cp313-macosx_arm64` on PyPI before pinning
- [ ] **Spike: Real-ESRGAN x4plus on MPS** — load the official weights through Spandrel, run a forward pass on MPS, confirm no `NotImplementedError` (or document which ops fall back)
- [ ] **Spike: tiling quality** — upscale one 3000×2000 JPEG at tile=512/overlap=32; inspect seam artefacts; adjust overlap if visible

---

## Future Considerations (explicitly out of MVP)

- `--tile-size`, `--tile-overlap` flags
- `models verify` — registry integrity check (detect missing weight files)
- `--name` on `models import`
- URL / remote model import
- ncnn-vulkan backend, CUDA support
- `--jobs N` parallel batch
- `--format png` output option
- Shell completion (`typer --install-completion`)
