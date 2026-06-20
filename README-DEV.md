# easyupscaler — developer guide

Technical reference for working in this repository. End-user install and usage live in [README.md](README.md).

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.13+ | Enforced in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | Locked dev environment and lockfile |
| Apple Silicon Mac | Primary platform for MPS inference testing |
| Git | Clone and contribute |

Linux (CPU) is best-effort. Windows and Intel Mac are untested.

## Clone and set up

```bash
git clone https://github.com/codeprimate/easyupscaler
cd easyupscaler
make sync          # uv sync — runtime + dev deps into .venv
```

Run commands through uv so they use the project venv:

```bash
uv run easyupscaler models list
uv run pytest tests/test_registry.py -k foo
```

## Makefile targets

| Target | What it runs |
|--------|----------------|
| `make sync` | `uv sync` |
| `make lint` | `uv run ruff check easyupscaler tests` |
| `make typecheck` | `uv run mypy easyupscaler` |
| `make test` | lint → typecheck → pytest (quality gate) |
| `make build` | `uv build` |
| `make install` | `python -m pip uninstall easyupscaler` then `python -m pip install .` |

**`make test` is the merge gate.** All three steps must pass.

`make install` installs the CLI into whatever `python` resolves to on your PATH — not necessarily `.venv`. Activate a venv first if you want the console script there. End users install via `pip install git+https://github.com/codeprimate/easyupscaler.git` instead.

## Quality gate

`make test` runs in order:

1. **ruff** — lint (`E`, `F`, `I`, `UP`; line length 100)
2. **mypy** — static types on `easyupscaler` (`ignore_missing_imports` for untyped third-party)
3. **pytest** — fast suite with **≥80% line coverage** on `easyupscaler`

Configuration: `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.coverage.*]`).

See [ADR-009](docs/adr/009-development-toolchain.md) and [ADR-010](docs/adr/010-code-coverage-gate.md).

## Testing

### Fast suite (default)

```bash
make test                                    # full gate
uv run pytest tests/test_cli_scale.py        # one file
uv run pytest tests/test_service.py -k oom   # filter by name
```

Default pytest excludes slow tests (`addopts = -m "not slow" …`). Coverage is measured over the fast suite only.

### Slow / end-to-end

Real-weight inference lives under `tests/slow/`:

```bash
export EASYUPSCALER_TEST_WEIGHTS=/path/to/RealESRGAN_x4plus.pth
uv run pytest -m slow
```

Requires a local `.pth` or `.safetensors` file and loads PyTorch. Use for MPS smoke tests on Apple Silicon.

### Shared fixtures (`tests/conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `isolated_paths` | Redirects XDG config/data to `tmp_path`. Required for any test touching registry, config, or model files on disk. |
| `without_torch` | Removes `torch` from `sys.modules` for the test duration. Use when asserting fast commands do not import PyTorch. |

### Patterns

- **Fakes over mocks** — small in-test classes implementing backend protocols (e.g. `FakeBackend`)
- **Constructor injection** — pass `registry=`, `backend_factory=`, `config_service=` into services
- **CLI tests** — `typer.testing.CliRunner` against `app`; assert exit code, stdout/stderr, and file artifacts
- **Behavior, not internals** — tests encode [docs/specification.md](docs/specification.md) contracts

Test layout mirrors package concerns: `tests/test_<module>.py` for fast tests; `tests/slow/` for GPU/weight-dependent paths.

## Repository layout

```
easyupscaler/
  cli/              # Typer presentation layer (main, scale, denoise, models)
  config/           # XDG paths, ConfigService (no torch)
  models/           # ModelRegistry, import_model (lazy torch)
  upscaling/        # UpscaleService, tiling, Spandrel backend
  denoise/          # DenoiseService, catalog, downloader, mode backends
  io/               # ImageIO, HEIC registration
  errors.py         # Domain exceptions → CLI messages at boundary
tests/
docs/               # mission, specification, architecture, ADRs
pyproject.toml
uv.lock
Makefile
```

Full component diagram and data flow: [docs/architecture.md](docs/architecture.md).

## Layer boundaries

| Layer | Modules | Rules |
|-------|---------|-------|
| Presentation | `cli/*` | Typer, Rich progress, user-facing messages. Map domain exceptions to stderr. |
| Application | `upscaling/service`, `denoise/pipeline`, `models/import_model`, `config/settings` | Orchestration, batch logic, dependency injection. |
| Domain | `io/`, `tiling`, `registry`, backends | No CLI imports. Torch/Spandrel loaded lazily (ADR-008). |
| Persistence | XDG paths | `config.toml`, `registry.json`, copied weights under `~/.local/share/easyupscaler/` |

Let programming errors crash. Catch only for I/O, user input, and unreliable externals (torch, Spandrel, OS).

## Inference pipeline (upscale)

1. **Spandrel** loads the weight file and reads architecture, purpose, and scale.
2. **Device** — MPS on Apple Silicon when available; CPU otherwise (stderr warning). MPS op failures mid-run retry that image on CPU (ADR-002).
3. **Tiling** — sides longer than 512 px use overlapping 512 px tiles (32 px overlap). OOM halves tile size until 128 px floor or failure (ADR-007).
4. **Output** — JPEG quality 95, 4:4:4 subsampling, `{stem}-upscaled.jpg` beside input (ADR-003).

Denoise uses a separate managed catalog with lazy HTTP download on first use (ADR-012). Spec: [docs/specification-denoise.md](docs/specification-denoise.md).

## Documentation map

| Document | Owns |
|----------|------|
| [docs/mission.md](docs/mission.md) | Scope, non-goals |
| [docs/specification.md](docs/specification.md) | CLI contracts: commands, flags, messages, exit codes |
| [docs/specification-denoise.md](docs/specification-denoise.md) | Denoise command and managed model matrix |
| [docs/architecture.md](docs/architecture.md) | Layers, components, error table, testing strategy |
| [docs/adr/](docs/adr/) | Immutable architectural decisions |
| [docs/adr.md](docs/adr.md) | ADR index |
| [AGENTS.md](AGENTS.md) | Agent/implementer process, doc maintenance rules |

**Precedence:** specification for user-visible behavior; ADRs for structural choices; architecture for system design; mission for scope. Tests and spec are dual contracts — update both when behavior changes.

### When you change code

| Change | Update |
|--------|--------|
| CLI command, flag, message, exit code | Tests + `docs/specification.md` (+ denoise spec if applicable); [README.md](README.md) if users need to know |
| New module or layer boundary | `docs/architecture.md` |
| Architectural choice | New ADR, index row in `docs/adr.md`, propagate to architecture |
| Dev workflow / Makefile / coverage | ADR-009/010 or new ADR; this file; `docs/architecture.md` toolchain section |

Accepted ADRs are immutable — supersede with a new numbered ADR instead of editing in place.

## Contributing checklist

Before opening a PR:

- [ ] `make test` passes (lint, mypy, pytest, coverage ≥80%)
- [ ] Behavior changes have tests and spec updates in the same change set
- [ ] Layer boundaries preserved
- [ ] Relevant docs updated (see table above)
