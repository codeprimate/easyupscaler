# Implementation TODO: easyupscaler MVP

## Specification Reference
**Source Document**: [docs/specification.md](../docs/specification.md)
*This specification document must be loaded alongside this plan during execution to provide complete context and requirements.*

**Architecture Reference**: [docs/architecture.md](../docs/architecture.md)
**ADRs**: [docs/adr.md](../docs/adr.md)

## Overview
- **Complexity**: Complex
- **Risk Level**: Medium
- **Key Dependencies**: Python 3.13, uv, PyTorch ≥2.3 (cp313-macosx_arm64 wheel), Spandrel ≥0.4, spandrel-extra-arches, Typer, Rich, Pillow, tomlkit, numpy
- **Estimated Effort**: 5–8 developer days (excluding slow E2E validation on real hardware)
- **Specification Sections**: Problem Statement, Requirements (§1–3), Technical Constraints, Edge Cases, Technical Approach, Acceptance Criteria, Implementation Tasks, Risk Assessment

## Phase Strategy

The project is **greenfield** (docs only; no Python package yet). Phases deliver incrementally testable slices:

1. **Phase 0** — De-risk dependency pins before any feature code
2. **Phase 1** — Toolchain and empty package; `make test` passes on skeleton
3. **Phase 2** — Persistence layer (paths, config, registry) with no torch
4. **Phase 3** — CLI skeleton for torch-free commands
5. **Phase 4** — Image I/O (Pillow → ndarray contract)
6. **Phase 5** — Tiling engine (mock model; no real weights)
7. **Phase 6** — Spandrel backend, import pipeline, UpscaleService
8. **Phase 7** — Full CLI (upscale + models import) with Rich progress
9. **Phase 8** — Slow E2E tests with real weights
10. **Phase 9** — README, help strings, install verification

Each phase ends with validation criteria. Later phases depend on earlier ones but can be tested in isolation via mocks.

## Progress Indicators
- 📋 **Planned** - Not started
- 🔄 **In Progress** - Currently being worked on
- ✅ **Completed** - Successfully finished
- ❌ **Blocked/Failed** - Encountered issues or dependencies
- ⏸️ **Paused** - Temporarily suspended
- 🔍 **Under Review** - Completed but needs validation

---

## Phase 0: Pre-flight Investigation Spikes - 📋 Planned
*Incremental Goal: Confirm dependency availability and inference feasibility on target hardware before pinning versions.*

### Task 0.1: Verify Python 3.13 + PyTorch wheel availability - 📋 Planned
*Spec Reference: Risk Assessment — Investigation requirements; Technical Constraints — Python and dependencies*

- [ ] 0.1.1 **Confirm cp313-macosx_arm64 torch wheel on PyPI** - 📋 Planned
  - *Hint*: Spec Risk Assessment table — "torch 2.x + Python 3.13 wheel not available"
  - *Consider*: Block all Phase 1 pinning until this passes; document fallback if wheel missing
  - *Files*: `pyproject.toml` (version pins), `docs/specification.md` (note if constraint changes)
  - *Risk*: No wheel → entire MVP blocked on Apple Silicon
  - **IMPLEMENTATION PLAN**:
    - **Verification**:
      - On arm64 macOS with Python 3.13: `uv pip install "torch>=2.3" --dry-run` or install in throwaway venv
      - Confirm import succeeds: `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`
      - Record exact pin in `pyproject.toml` after successful install
    - **Testing Strategy**:
      - **Code Quality**: N/A (investigation only)
      - **Unit Testing**: N/A
      - **Integration Testing**: Manual import on target platform
      - **Test Cases**: Wheel resolves; MPS backend reports availability on Apple Silicon
      - **Mocks/Fixtures**: None
      - **Coverage**: Document result in commit message or ADR addendum if pin differs from spec

### Task 0.2: Spike Real-ESRGAN x4plus on MPS via Spandrel - 📋 Planned
*Spec Reference: Risk Assessment — MPS op gaps; Technical Approach — SpandrelBackend*

- [ ] 0.2.1 **Load official Real-ESRGAN x4plus weights and run forward pass on MPS** - 📋 Planned
  - *Hint*: ADR-002 (inference device policy); ADR-001 (Spandrel backend)
  - *Consider*: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before import; note any ops that fall back to CPU
  - *Files*: N/A (throwaway script); informs `spandrel_backend.py` error handling
  - *Risk*: Widespread MPS op failures → CPU-only path must be robust
  - **IMPLEMENTATION PLAN**:
    - **Verification**:
      - Download official `RealESRGAN_x4plus.pth` (user-trusted source)
      - Load via Spandrel `ModelLoader`; run 64×64 RGB tensor forward on MPS
      - Confirm output shape is 256×256; log warnings if MPS fallback triggered
    - **Testing Strategy**:
      - **Integration Testing**: Manual one-shot script
      - **Test Cases**: Forward pass completes; scale=4; purpose=SR
      - **Coverage**: Informs Phase 6 MPS retry logic expectations

### Task 0.3: Spike tiling quality on large image - 📋 Planned
*Spec Reference: Risk Assessment — tiling quality; ADR-007*

- [ ] 0.3.1 **Upscale 3000×2000 JPEG with tile=512/overlap=32; inspect seams** - 📋 Planned
  - *Hint*: ADR-007 — ComfyUI `tiled_scale` pattern
  - *Consider*: Adjust `DEFAULT_TILE_OVERLAP` only if visible seams; spec default is 32
  - *Files*: Informs constants in `upscaling/tiling.py`
  - *Risk*: Visible seams on DISCOURAGED models — acceptable with stderr warning per ADR-007
  - **IMPLEMENTATION PLAN**:
    - **Verification**:
      - Prototype tiled upscale (ComfyUI pattern or early Phase 5 code)
      - Visually inspect overlap regions at 100% zoom
      - Document whether default overlap=32 is sufficient
    - **Testing Strategy**:
      - **Integration Testing**: Manual visual inspection
      - **Test Cases**: No obvious seam at tile boundaries on SUPPORTED model

### Phase 0 Validation - 📋 Planned
- **Acceptance Criteria**: torch cp313 wheel confirmed; Real-ESRGAN loads on Spandrel; MPS forward pass result documented; tiling constants validated or adjusted with rationale
- **Testing Strategy**: Manual spikes only; no pytest
- **Rollback Plan**: If torch wheel unavailable, downgrade Python requirement in spec (requires user decision) or wait for wheel release

---

## Phase 1: Scaffold and Toolchain - 📋 Planned
*Incremental Goal: Installable package skeleton with lint, typecheck, and pytest passing on empty code.*

### Task 1.1: Initialise pyproject.toml and uv lockfile - 📋 Planned
*Spec Reference: Technical Approach — pyproject.toml structure; Technical Constraints — Python and dependencies*

- [ ] 1.1.1 **Create pyproject.toml with hatchling, runtime deps, dev group, console script** - 📋 Planned
  - *Hint*: Spec lines 332–356; ADR-009 (development toolchain)
  - *Consider*: Pin exact versions from Phase 0 spike; include `spandrel-extra-arches`
  - *Files*: `pyproject.toml`, `uv.lock`
  - *Risk*: Unpinned torch may break CI reproducibility
  - **IMPLEMENTATION PLAN**:
    - **Project metadata**:
      - `[build-system]` hatchling
      - `[project]` name=`easyupscaler`, `requires-python = ">=3.13"`
      - Runtime deps: typer≥0.12, rich≥13, pillow≥10, tomlkit≥0.13, torch≥2.3, spandrel≥0.4, spandrel-extra-arches, numpy≥1.26
      - `[project.scripts]` easyupscaler = `easyupscaler.cli.main:app`
      - `[dependency-groups]` dev: pytest, pytest-mock, mypy, ruff, types-Pillow
    - **Tool config**:
      - `[tool.mypy]` strict=false, ignore_missing_imports=true
      - `[tool.ruff.lint]` select = ["E", "F", "I", "UP"]
    - **Lock**:
      - Run `uv lock` then `uv sync`
    - **Testing Strategy**:
      - **Code Quality**: `uv sync` exits 0 on arm64 macOS
      - **Unit Testing**: N/A
      - **Integration Testing**: `uv run python -c "import easyupscaler"` after skeleton created

### Task 1.2: Create Makefile - 📋 Planned
*Spec Reference: Technical Approach — Makefile; Acceptance Criteria — Setup and install*

- [ ] 1.2.1 **Add Makefile targets: sync, lint, typecheck, test, build, install** - 📋 Planned
  - *Hint*: Spec lines 358–380
  - *Consider*: `test` must run lint → typecheck → pytest in order
  - *Files*: `Makefile`
  - *Risk*: Missing `.PHONY` causes stale target issues
  - **IMPLEMENTATION PLAN**:
    - **Targets**:
      - `sync`: `uv sync`
      - `lint`: `uv run ruff check easyupscaler tests`
      - `typecheck`: `uv run mypy easyupscaler`
      - `test`: lint + typecheck + `uv run pytest`
      - `build`: `uv build`
      - `install`: `uv pip install .`
    - **Testing Strategy**:
      - **Code Quality**: `make test` runs all three gates
      - **Test Cases**: Each target exits 0 on empty skeleton

### Task 1.3: Create package skeleton and base exceptions - 📋 Planned
*Spec Reference: Technical Approach — Affected Components; Key Interfaces — errors*

- [ ] 1.3.1 **Scaffold directory tree with __init__.py files and errors.py** - 📋 Planned
  - *Hint*: Spec package layout (lines 182–215); architecture.md package layout
  - *Consider*: `__init__.py` exports `__version__` only; no torch imports anywhere
  - *Files*:
    - `easyupscaler/__init__.py` — `__version__ = "0.1.0"`
    - `easyupscaler/errors.py` — `EasyUpscalerError` base; stubs for domain errors
    - Empty packages: `cli/`, `config/`, `models/`, `upscaling/backends/`, `io/`
    - `tests/conftest.py` — empty or shared fixtures
    - `tests/test_smoke.py` — single passing test
  - *Risk*: Accidental torch import at package init breaks lazy-load contract
  - **IMPLEMENTATION PLAN**:
    - **errors.py**:
      - `EasyUpscalerError(Exception)` base
      - Placeholder subclasses (filled in Phase 2): `ModelNotFoundError`, `DuplicateModelError`, `ImageReadError`, `ImportModelError`
    - **cli/main.py** (minimal):
      - Typer `app` with no subcommands yet
      - Set `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` at module top
      - `--version` callback reading `easyupscaler.__version__`
    - **Testing Strategy**:
      - **Code Quality**: `make test` passes
      - **Unit Testing**: `test_smoke.py` asserts version import works
      - **Test Cases**: `easyupscaler --version` via CliRunner (after Phase 3 wiring)

### Phase 1 Validation - 📋 Planned
- **Acceptance Criteria**: `uv sync` succeeds; `make test` passes; `make build` produces wheel; package importable; no torch in sys.modules after `--version`
- **Testing Strategy**: `make test` on clean clone
- **Rollback Plan**: Delete `easyupscaler/` and revert pyproject.toml

---

## Phase 2: Storage and Config (no torch) - 📋 Planned
*Incremental Goal: XDG paths, config.toml, and registry.json CRUD with full unit test coverage.*

### Task 2.1: Implement XDG path resolution - 📋 Planned
*Spec Reference: Technical Approach — paths.py; Edge Cases — Registry/Config missing*

- [ ] 2.1.1 **Create config/paths.py with XDG directory helpers** - 📋 Planned
  - *Hint*: Architecture — Storage layout; ADR-004
  - *Consider*: macOS fallbacks when `XDG_CONFIG_HOME` / `XDG_DATA_HOME` unset; create parent dirs on first write only
  - *Files*: `easyupscaler/config/paths.py`
  - *Risk*: Hardcoded paths elsewhere break portability
  - **IMPLEMENTATION PLAN**:
    - **Constants and functions**:
      - `CONFIG_DIR` → `$XDG_CONFIG_HOME/easyupscaler` (fallback `~/.config/easyupscaler`)
      - `DATA_DIR` → `$XDG_DATA_HOME/easyupscaler`
      - `CONFIG_FILE` → `CONFIG_DIR / "config.toml"`
      - `REGISTRY_FILE` → `DATA_DIR / "registry.json"`
      - `MODELS_DIR` → `DATA_DIR / "models"`
      - `ensure_config_dir()`, `ensure_data_dir()`, `ensure_models_dir()` — mkdir -p semantics
    - **Testing Strategy**:
      - **Unit Testing**: Use `tmp_path` + monkeypatch env vars
      - **Test Cases**: Unset XDG vars use macOS defaults; dirs created on ensure_* calls
      - **Mocks/Fixtures**: `monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))`

### Task 2.2: Implement ConfigService - 📋 Planned
*Spec Reference: Key Interfaces — ConfigService; Edge Cases — Config file missing*

- [ ] 2.2.1 **Create config/settings.py with load/save/clear default_model** - 📋 Planned
  - *Hint*: Spec §2 models default; tomlkit preserves formatting
  - *Consider*: Missing config → `get_default_model()` returns `None`; create file on first write
  - *Files*: `easyupscaler/config/settings.py`, `tests/test_settings.py`
  - *Risk*: Corrupt TOML should fail fast with clear error
  - **IMPLEMENTATION PLAN**:
    - **ConfigService methods**:
      - `get_default_model() -> str | None`
      - `set_default_model(name: str) -> None` — writes `default_model = "<name>"` via tomlkit
      - `clear_default_model() -> None` — removes key or sets empty
    - **Testing Strategy**:
      - **Code Quality**: `mypy easyupscaler/config/settings.py`
      - **Unit Testing**: `tests/test_settings.py`
      - **Test Cases**: Missing file returns None; set then get round-trips; clear removes default
      - **Coverage**: Assert `"torch" not in sys.modules` in each test

### Task 2.3: Implement ModelRegistry - 📋 Planned
*Spec Reference: Key Interfaces — ModelRegistry; Edge Cases — Registry missing, duplicate name*

- [ ] 2.3.1 **Create models/registry.py with ModelEntry and CRUD** - 📋 Planned
  - *Hint*: Spec registry JSON schema; architecture ModelRegistry table
  - *Consider*: Missing registry.json → empty list; atomic write (write temp + rename) optional but good
  - *Files*: `easyupscaler/models/registry.py`, `tests/test_registry.py`
  - *Risk*: Concurrent writes not in MVP scope; document as known limitation
  - **IMPLEMENTATION PLAN**:
    - **ModelEntry dataclass**:
      - Fields: `name`, `filename`, `path`, `scale`, `imported_at` (datetime, ISO8601 in JSON)
    - **ModelRegistry methods**:
      - `list() -> list[ModelEntry]`
      - `get(name) -> ModelEntry` — raises `ModelNotFoundError`
      - `add(entry) -> None` — raises `DuplicateModelError`
      - `remove(name) -> ModelEntry` — raises `ModelNotFoundError`
      - `replace(entry) -> None` — for `--force` import
    - **Testing Strategy**:
      - **Unit Testing**: `tests/test_registry.py` with tmp registry path
      - **Test Cases**: Empty registry; add/list/get; duplicate add fails; remove returns entry; replace overwrites
      - **Coverage**: Assert no torch import in all registry tests

### Phase 2 Validation - 📋 Planned
- **Acceptance Criteria**: Config and registry round-trip via services; missing files treated as empty; all Phase 2 tests pass; torch never imported
- **Testing Strategy**: `RAILS_ENV=test` equivalent: `uv run pytest tests/test_settings.py tests/test_registry.py`
- **Rollback Plan**: Revert `config/` and `models/registry.py`; tests removable independently

---

## Phase 3: CLI Skeleton (no torch) - 📋 Planned
*Incremental Goal: Typer app with models list, default, remove commands; torch-free integration tests.*

### Task 3.1: Wire Typer app and version/help - 📋 Planned
*Spec Reference: §3 CLI top-level flags; ADR-008*

- [ ] 3.1.1 **Complete cli/main.py with app registration and MPS env bootstrap** - 📋 Planned
  - *Hint*: ADR-008 — env before torch; lazy import boundaries
  - *Consider*: `--help` and `--version` must not trigger torch import
  - *Files*: `easyupscaler/cli/main.py`
  - *Risk*: Subcommand imports at module level can pull torch transitively
  - **IMPLEMENTATION PLAN**:
    - **main.py**:
      - `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` at top
      - Typer `app` with callback for `--version` (Typer `@app.callback`)
      - Register models sub-app: `app.add_typer(models_app, name="models")`
      - Defer upscale command registration to Phase 7
    - **Testing Strategy**:
      - **Integration Testing**: CliRunner invoke `--help`, `--version`
      - **Test Cases**: Exit 0; version string contains package version; `"torch" not in sys.modules`

### Task 3.2: Implement torch-free models commands - 📋 Planned
*Spec Reference: §2 Model management — list, default, remove*

- [ ] 3.2.1 **Create cli/models.py with list command** - 📋 Planned
  - *Hint*: Spec stdout table format; empty-state message exact string
  - *Consider*: Use Rich Table or plain formatting; no torch
  - *Files*: `easyupscaler/cli/models.py`
  - *Risk*: Table alignment on narrow terminals — acceptable for MVP
  - **IMPLEMENTATION PLAN**:
    - **list command**:
      - Instantiate `ModelRegistry()`; call `list()`
      - Empty: print `No models installed. Use 'easyupscaler models import <path>' to add one.`
      - Non-empty: print Name / Scale / Filename table with `4×` scale format
    - **Testing Strategy**:
      - **Integration Testing**: `tests/test_cli_models.py::test_list_empty`, `test_list_with_models`
      - **Test Cases**: Exact empty message; table headers; torch not imported

- [ ] 3.2.2 **Add models default command** - 📋 Planned
  - *Hint*: Spec error message lists installed models as table
  - *Files*: `easyupscaler/cli/models.py`
  - **IMPLEMENTATION PLAN**:
    - **default command**:
      - Validate name exists in registry before `ConfigService.set_default_model`
      - Success: `Default model set to {name}`
      - Failure: `Error: model '{name}' not found. Installed models:` + table; exit 1
    - **Testing Strategy**:
      - **Test Cases**: Success path; unknown name exits 1 with listing

- [ ] 3.2.3 **Add models remove command with confirmation** - 📋 Planned
  - *Hint*: Spec confirmation prompt `[y/N]`; `--yes` / `-y` skips
  - *Consider*: If removed model is default, clear config and warn to stderr
  - *Files*: `easyupscaler/cli/models.py`
  - **IMPLEMENTATION PLAN**:
    - **remove command**:
      - Prompt unless `--yes`: `Remove {name} and delete {filename}? [y/N]`
      - Delete registry entry and weight file from `MODELS_DIR`
      - If was default: `ConfigService.clear_default_model()` + stderr warning
      - Unknown name: `Error: model '{name}' not found.` exit 1
    - **Testing Strategy**:
      - **Test Cases**: Prompt shown; `--yes` skips; default cleared with warning; file deleted

### Phase 3 Validation - 📋 Planned
- **Acceptance Criteria**: `models list`, `models default`, `models remove` work via CLI; all respond without torch; exit codes match spec
- **Testing Strategy**: `uv run pytest tests/test_cli_models.py` with torch-import assertions
- **Rollback Plan**: Remove models subcommands; keep Phase 2 services tested in isolation

---

## Phase 4: Image I/O - 📋 Planned
*Incremental Goal: ImageIO read/write with RGBA, grayscale, JPEG output contract.*

### Task 4.1: Implement ImageIO.read - 📋 Planned
*Spec Reference: Key Interfaces — ImageIO; Edge Cases — corrupt image, RGBA, grayscale*

- [ ] 4.1.1 **Create io/images.py with read() returning float32 RGB ndarray** - 📋 Planned
  - *Hint*: ADR-003; spec tensor contract [0, 1] shape (H, W, 3)
  - *Consider*: RGBA → flatten alpha to white background; grayscale → RGB
  - *Files*: `easyupscaler/io/images.py`, `easyupscaler/errors.py` (`ImageReadError`), `tests/test_images.py`
  - *Risk*: Unsupported formats must raise `ImageReadError` with message usable by CLI
  - **IMPLEMENTATION PLAN**:
    - **read(path: Path) -> np.ndarray**:
      - Pillow open; handle `Image.UnidentifiedImageError` → `ImageReadError("cannot read image")`
      - RGBA: composite on white background, convert to RGB
      - Grayscale (L, LA): convert to RGB
      - Return `np.asarray(img, dtype=np.float32) / 255.0`, shape (H, W, 3)
    - **Testing Strategy**:
      - **Unit Testing**: Fixture PNG/JPEG in `tests/fixtures/`
      - **Test Cases**: RGB JPEG; RGBA PNG → RGB; grayscale PNG → RGB; corrupt file raises ImageReadError
      - **Mocks/Fixtures**: `tests/fixtures/rgba.png`, `grayscale.png`, `corrupt.jpg`

### Task 4.2: Implement ImageIO.write - 📋 Planned
*Spec Reference: §1 Upscale command output naming; ADR-003*

- [ ] 4.2.1 **Add write() producing {stem}-upscaled.jpg beside source** - 📋 Planned
  - *Hint*: quality=95, subsampling=0; silent overwrite
  - *Files*: `easyupscaler/io/images.py`
  - **IMPLEMENTATION PLAN**:
    - **write(image, source_path) -> Path**:
      - Output path: `source_path.parent / f"{source_path.stem}-upscaled.jpg"`
      - Clamp ndarray to [0, 1]; convert to uint8 RGB
      - Save JPEG: `quality=95`, `subsampling=0`
      - Return output path
    - **Testing Strategy**:
      - **Test Cases**: Naming convention; overwrite existing; output is valid JPEG readable by Pillow

### Phase 4 Validation - 📋 Planned
- **Acceptance Criteria**: Read/write round-trip for RGB, RGBA, grayscale; output naming and JPEG params match spec
- **Testing Strategy**: `uv run pytest tests/test_images.py`
- **Rollback Plan**: Remove `io/` module; no upstream dependencies yet

---

## Phase 5: Tiling Engine - 📋 Planned
*Incremental Goal: tiled_upscale() with single-pass, tiled merge, OOM halving; tested with fake model.*

### Task 5.1: Implement tiled_upscale core algorithm - 📋 Planned
*Spec Reference: Key Interfaces — tiled_upscale; ADR-007*

- [ ] 5.1.1 **Create upscaling/tiling.py with constants and tiled_upscale()** - 📋 Planned
  - *Hint*: ADR-007 algorithm steps; ComfyUI weighted overlap merge
  - *Consider*: `DEFAULT_TILE_SIZE=512`, `DEFAULT_TILE_OVERLAP=32`, `MIN_TILE_SIZE=128`
  - *Files*: `easyupscaler/upscaling/tiling.py`, `tests/test_tiling.py`
  - *Risk*: Incorrect merge weights cause seam artifacts — test with deterministic fake model
  - **IMPLEMENTATION PLAN**:
    - **Constants**: `DEFAULT_TILE_SIZE`, `DEFAULT_TILE_OVERLAP`, `MIN_TILE_SIZE`
    - **tiled_upscale(model, tensor, tile_size, overlap) -> Tensor**:
      - If both H,W ≤ tile_size: single `model(tensor)` forward
      - Else: split into overlapping tiles, forward each, weighted merge into output canvas
      - Honor `model.tiling`: INTERNAL → no external tile; DISCOURAGED → full frame first, tile on OOM with warning; SUPPORTED → tile when oversized
    - **OOM handling**:
      - Catch runtime OOM; halve tile_size; retry until MIN_TILE_SIZE or re-raise with clear message
      - Warn stderr when halving or when DISCOURAGED model forced to tile
    - **Testing Strategy**:
      - **Unit Testing**: Fake `ImageModelDescriptor` mock with known scale and tiling mode
      - **Test Cases**: 256×256 input → single pass (no tile split); 1024×1024 → multiple tiles; OOM mock triggers halving
      - **Mocks/Fixtures**: Mock model returning input * scale; mock OOM on first attempt

### Phase 5 Validation - 📋 Planned
- **Acceptance Criteria**: Tiling tests pass without GPU; single-pass for small images; OOM retry halves tile size
- **Testing Strategy**: `uv run pytest tests/test_tiling.py`
- **Rollback Plan**: Module isolated; no production callers until Phase 6

---

## Phase 6: Backend, Import Pipeline, and UpscaleService - 📋 Planned
*Incremental Goal: End-to-end upscale orchestration with Spandrel backend; model import with validation.*

### Task 6.1: Define UpscalerBackend protocol - 📋 Planned
*Spec Reference: Key Interfaces — UpscalerBackend protocol*

- [ ] 6.1.1 **Create upscaling/backends/base.py with Protocol** - 📋 Planned
  - *Hint*: Spec Protocol snippet; architecture service boundary uses ndarray
  - *Files*: `easyupscaler/upscaling/backends/base.py`
  - **IMPLEMENTATION PLAN**:
    - **Protocol**:
      - `scale: int` property
      - `upscale(image: np.ndarray) -> np.ndarray` — float32 RGB [0,1], (H,W,3)
    - **Testing Strategy**:
      - **Code Quality**: mypy accepts Protocol usage in service tests

### Task 6.2: Implement SpandrelBackend - 📋 Planned
*Spec Reference: Technical Approach — SpandrelBackend; Edge Cases — MPS unavailable, MPS op failure*

- [ ] 6.2.1 **Create upscaling/backends/spandrel_backend.py** - 📋 Planned
  - *Hint*: ADR-001, ADR-002; call `spandrel_extra_arches.install()` once in `__init__`
  - *Consider*: Lazy import torch/spandrel inside module functions or `__init__`
  - *Files*: `easyupscaler/upscaling/backends/spandrel_backend.py`
  - *Risk*: MPS op failures mid-inference require full-image CPU retry
  - **IMPLEMENTATION PLAN**:
    - **SpandrelBackend**:
      - `__init__(weights_path)`: install extra arches; select device MPS if available else CPU (warn on CPU)
      - Load via `ModelLoader(device=...).load_from_file(path)`
      - `upscale(ndarray)`: convert to tensor (1,C,H,W); call `tiled_upscale()`; convert back to ndarray
      - MPS op failure: catch, warn `Warning: MPS error on {model}: {msg}. Retrying on CPU.`, reload/rerun on CPU
    - **Testing Strategy**:
      - **Unit Testing**: Mock Spandrel in import_model tests; backend tested via service with fake backend
      - **Integration Testing**: Phase 8 slow tests with real weights

### Task 6.3: Implement import_model pipeline - 📋 Planned
*Spec Reference: §2 models import; Edge Cases — import error table*

- [ ] 6.3.1 **Create models/import_model.py** - 📋 Planned
  - *Hint*: Spec import steps 1–5; ADR-004
  - *Files*: `easyupscaler/models/import_model.py`, `tests/test_import_model.py`
  - **IMPLEMENTATION PLAN**:
    - **import_model(path: Path, force: bool = False) -> ModelEntry**:
      - Fail if path missing: `Error: path not found: {path}`
      - Warn stderr for `.pth` pickle security message (exact spec string)
      - Copy to `MODELS_DIR` preserving filename
      - Lazy-load Spandrel; validate `ImageModelDescriptor` with `purpose == "SR"`
      - Reject non-SR: `Error: model is not a super-resolution model (purpose: {purpose}). Only SR models are supported.`
      - Reject scale 1 or missing: `Error: model reports scale 1 — not an upscaling model.`
      - Duplicate without force: `Error: model '{name}' already exists. Use --force to replace.`
      - With force: `replace()` registry + overwrite file; print `Replaced {name}.`
      - UnsupportedModelError: spec message about updating Spandrel
      - Success: `Imported {name} ({scale}×) from {filename}`
    - **Testing Strategy**:
      - **Unit Testing**: Mock Spandrel loader returning descriptors with varied purpose/scale
      - **Test Cases**: All edge cases from spec error table; no registry write on validation failure

### Task 6.4: Implement UpscaleService - 📋 Planned
*Spec Reference: Key Interfaces — UpscaleService; §1 Upscale command behavior*

- [ ] 6.4.1 **Create upscaling/service.py with batch orchestration** - 📋 Planned
  - *Hint*: ADR-006 sequential batch; architecture UpscaleService flow
  - *Consider*: Load backend once per job; `on_progress` callback for CLI decoupling
  - *Files*: `easyupscaler/upscaling/service.py`, `tests/test_service.py`
  - **IMPLEMENTATION PLAN**:
    - **UpscaleResult dataclass**: `path`, `output: Path | None`, `error: str | None`
    - **UpscaleService.run(paths, model_name, on_progress)**:
      - Resolve model from arg or `ConfigService.get_default_model()`; fail fast with spec error messages
      - Load registry entry; construct `SpandrelBackend` once (injectable for tests)
      - For each path sequentially:
        - Validate exists and is file (not dir)
        - Read via ImageIO; upscale via backend; write output
        - On failure: record error string; continue
        - Optional `torch.mps.empty_cache()` after large images on MPS
      - Invoke `on_progress(result)` after each file
      - Return list of UpscaleResult
    - **Testing Strategy**:
      - **Unit Testing**: Inject fake `UpscalerBackend` via constructor or monkeypatch
      - **Test Cases**: All succeed; one corrupt continues batch; missing model fails before backend load; no default fails with spec message
      - **Coverage**: Batch continue-on-failure; exit code logic tested at CLI layer

### Phase 6 Validation - 📋 Planned
- **Acceptance Criteria**: Service tests pass with fake backend; import_model tests pass with mocked Spandrel; domain exceptions mapped correctly
- **Testing Strategy**: `uv run pytest tests/test_service.py tests/test_import_model.py`
- **Rollback Plan**: Keep protocol and fake-backend tests; defer Spandrel integration to Phase 8

---

## Phase 7: Upscale CLI and Models Import Command - 📋 Planned
*Incremental Goal: Full user-facing CLI — upscale with progress display; models import with lazy torch.*

### Task 7.1: Implement upscale CLI command - 📋 Planned
*Spec Reference: §1 Upscale command; stdout/stderr contract*

- [ ] 7.1.1 **Create cli/upscale.py with path parsing and UpscaleService integration** - 📋 Planned
  - *Hint*: Spec TTY vs non-TTY output examples; Rich progress bar
  - *Consider*: Fail empty paths before torch load; `--model` global or command option per Typer design
  - *Files*: `easyupscaler/cli/upscale.py`, `easyupscaler/cli/main.py`, `tests/test_cli_upscale.py`
  - **IMPLEMENTATION PLAN**:
    - **upscale command**:
      - Register as default callback or command on main app
      - Zero paths: stderr `Error: no input images. Pass one or more file paths.` exit 1 before torch
      - Resolve `--model` override
      - TTY: Rich progress bar + per-file ✓/✗ lines + summary `Completed: N succeeded, M failed.`
      - Non-TTY: plain `path → output` or `path FAILED: reason` lines
      - Exit 0 if all succeed; 1 if any failure
      - Errors/warnings to stderr; progress to stdout
    - **Testing Strategy**:
      - **Integration Testing**: Mock `UpscaleService` in tests
      - **Test Cases**: Empty paths exit 1 no torch; TTY vs non-TTY output; partial batch exit 1; missing default exit 1

### Task 7.2: Add models import CLI command - 📋 Planned
*Spec Reference: §2 models import*

- [ ] 7.2.1 **Add import subcommand to cli/models.py with deferred torch** - 📋 Planned
  - *Hint*: ADR-008 — import inside command handler, not module top
  - *Files*: `easyupscaler/cli/models.py`
  - **IMPLEMENTATION PLAN**:
    - **import command**:
      - `--force` flag
      - Call `import_model(path, force=force)` inside function body
      - Print success message to stdout; warnings to stderr
    - **Testing Strategy**:
      - **Integration Testing**: Mock `import_model` in CLI tests
      - **Test Cases**: Success output; force flag forwarded; pickle warning on .pth

### Task 7.3: Map domain exceptions to user-facing CLI messages - 📋 Planned
*Spec Reference: Edge Cases table; architecture Error handling*

- [ ] 7.3.1 **Centralise error formatting in CLI layer** - 📋 Planned
  - *Hint*: errors.py exceptions caught at CLI boundary; programming errors propagate
  - *Files*: `easyupscaler/cli/main.py`, `easyupscaler/errors.py` (complete exception set)
  - **IMPLEMENTATION PLAN**:
    - **Exception mapping**:
      - Ensure all spec error strings match exactly (grep-driven verification)
      - Typer exit code 1 on domain errors
    - **Testing Strategy**:
      - **Test Cases**: Each edge-case row in spec error table has corresponding CLI test

### Phase 7 Validation - 📋 Planned
- **Acceptance Criteria**: Full CLI surface matches spec commands; progress output matches TTY/non-TTY examples; all CLI tests pass
- **Testing Strategy**: `uv run pytest tests/test_cli_models.py tests/test_cli_upscale.py`
- **Rollback Plan**: Disable upscale command registration; services remain testable

---

## Phase 8: Slow End-to-End Tests - 📋 Planned
*Incremental Goal: Optional real-weight integration test validating full pipeline on developer hardware.*

### Task 8.1: Configure slow test marker - 📋 Planned
*Spec Reference: Implementation Tasks — Phase 8*

- [ ] 8.1.1 **Add @pytest.mark.slow to conftest.py and pytest config** - 📋 Planned
  - *Hint*: CI excludes slow by default: `pytest -m "not slow"`
  - *Files*: `tests/conftest.py`, `pyproject.toml` `[tool.pytest.ini_options]`
  - **IMPLEMENTATION PLAN**:
    - Register marker `slow` with description
    - Default pytest addopts or document in README: `-m "not slow"`

### Task 8.2: Write E2E upscale test - 📋 Planned
*Spec Reference: Acceptance Criteria — Upscaling; Investigation — Real-ESRGAN fixture*

- [ ] 8.2.1 **Create tests/slow/test_e2e.py** - 📋 Planned
  - *Hint*: Import Real-ESRGAN x4plus; upscale 64×64 PNG → 256×256 JPEG
  - *Consider*: Skip if fixture weight path env var unset
  - *Files*: `tests/slow/test_e2e.py`, `tests/fixtures/input_64x64.png`
  - **IMPLEMENTATION PLAN**:
    - **Test flow**:
      - `@pytest.mark.slow`
      - Import model from env `EASYUPSCALER_TEST_WEIGHTS` or skip
      - Run CLI upscale; assert output dimensions 256×256; valid JPEG
    - **Testing Strategy**:
      - **Integration Testing**: Manual/optional CI job on macOS arm64
      - **Test Cases**: Full import → default → upscale pipeline

### Phase 8 Validation - 📋 Planned
- **Acceptance Criteria**: Slow test passes locally with real weights; excluded from default `make test` or passes skip gracefully
- **Testing Strategy**: `uv run pytest -m slow` on arm64 macOS with weights present
- **Rollback Plan**: Mark entire slow directory as optional; no impact on default test gate

---

## Phase 9: Polish and Documentation - 📋 Planned
*Incremental Goal: Production-ready developer and user documentation; help strings; install verification.*

### Task 9.1: Write README.md - 📋 Planned
*Spec Reference: Implementation Tasks — Phase 9; Risk — spandrel_extra_arches license*

- [ ] 9.1.1 **Document install, quickstart, glob tips, troubleshooting** - 📋 Planned
  - *Hint*: Spec Phase 9 checklist; mission.md examples
  - *Files*: `README.md`
  - **IMPLEMENTATION PLAN**:
    - **Sections**: Install (uv/pip), quickstart (≤5 commands), shell glob + `nullglob` note, model sources (OpenModelDB), MPS/OOM/unsupported model troubleshooting, spandrel-extra-arches license note

### Task 9.2: Complete CLI help strings - 📋 Planned
*Spec Reference: Acceptance Criteria — Setup; §3 --help*

- [ ] 9.2.1 **Add docstrings/help to all Typer commands and options** - 📋 Planned
  - *Files*: `easyupscaler/cli/main.py`, `models.py`, `upscale.py`
  - **IMPLEMENTATION PLAN**:
    - Every command and option has `help=` text
    - Verify `easyupscaler --help` and `easyupscaler models --help` readable

### Task 9.3: Verify clean install on arm64 macOS - 📋 Planned
*Spec Reference: Acceptance Criteria — Setup and install*

- [ ] 9.3.1 **Run full acceptance checklist from specification** - 📋 Planned
  - *Files*: N/A (verification task)
  - **IMPLEMENTATION PLAN**:
    - Fresh clone → `uv sync` → `make test` → `make install`
    - `easyupscaler --version` from PATH without torch load
    - Manual spot-check: import, default, upscale one image

### Phase 9 Validation - 📋 Planned
- **Acceptance Criteria**: All specification Acceptance Criteria checkboxes satisfied; README complete; `make test` green on fresh clone
- **Testing Strategy**: Walk through Acceptance Criteria section in spec line by line
- **Rollback Plan**: README-only changes are safe; revert if inaccurate

---

## Critical Considerations
- **Performance**: Non-inference commands must stay under 500 ms (no torch); upscale dominated by inference; tiling prevents OOM on 4000×3000 @ 16 GB
- **Security**: Pickle warning on `.pth` import; no network at runtime; only import trusted model files
- **Scalability**: Sequential batch only in MVP; no parallel workers
- **Monitoring**: stderr warnings for MPS fallback, OOM tile halving, default model cleared on remove
- **Cross-Phase Dependencies**: Phase 0 blocks version pins in Phase 1; Phase 2 required by Phase 3+; Phase 5 required by Phase 6 backend; Phase 6 required by Phase 7 CLI; Phase 8 optional after Phase 7

## Research & Validation Completed
- **Dependencies Verified**: Spec pins documented; Phase 0 spike required before final lock
- **Patterns Identified**: Greenfield — patterns from spec, architecture.md, ADRs 001–009 (Typer CLI, XDG storage, ComfyUI tiling, lazy torch)
- **Assumptions Validated**: No existing Python code; all file paths from spec are intentional new files; shell glob expansion is out of scope for CLI

---

## Sanity Check Summary (auto-correction applied)

| Check | Result | Action taken |
|-------|--------|--------------|
| Phase incremental value | Pass | Each phase has testable deliverable |
| Inter-phase dependencies | Pass | Phase 0 added before pinning; Phase 7 after Phase 6 |
| Task completeness | Pass | All spec acceptance criteria mapped to phases |
| Specification coverage | Pass | All FR §1–3, edge cases, and acceptance criteria addressed |
| File path accuracy | Pass | Matches spec/architecture; greenfield — no conflicting existing code |
| Missing requirements | Fixed | Added Phase 0 spikes from Risk Assessment; added Task 7.3 error mapping; added `models remove` confirmation tests in Phase 3 |
| Testing coverage | Pass | Torch-not-imported assertions in Phases 2–3; mocked backend in Phase 6–7 |
| Phase 3 vs spec | Fixed | Spec Implementation Tasks list `remove` in Phase 3 scope — included in Task 3.2.3 |
