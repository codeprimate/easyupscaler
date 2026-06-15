# ADR-009: Development toolchain (uv, ruff, mypy, Make)

**Status:** Accepted  
**Date:** 2025-06-15

## Context

The project needs a consistent local development workflow before implementation begins. Requirements:

- **uv** for dependency and environment management during development
- **ruff** for linting (and formatting, if enabled)
- **mypy** for static type checking
- **mypy and ruff must run as part of the test suite** — a failing lint or type check fails CI/local `make test`, same as pytest
- **Makefile** targets for build and installing the CLI into the **active** Python (system or selected interpreter), not only an ephemeral venv workflow

Python minimum version is **3.13+** (platform: Apple Silicon macOS primary; see product decisions).

## Decision

### uv (development)

- Use [uv](https://github.com/astral-sh/uv) to manage the dev environment and lock dependencies (`uv lock`, `uv sync`)
- `pyproject.toml` is the single source of truth for runtime and dev dependency groups
- Dev dependency group includes at minimum: `pytest`, `ruff`, `mypy`, and typing stubs as needed

### ruff

- Configure in `pyproject.toml` under `[tool.ruff]` (and `[tool.ruff.lint]` as needed)
- `make lint` runs `ruff check` (and `ruff format --check` if formatting is enforced)

### mypy

- Configure in `pyproject.toml` under `[tool.mypy]`
- Strict enough to catch real issues; pragmatic excludes only for third-party gaps (e.g. untyped `spandrel` via overrides or `ignore_missing_imports` scoped to those modules)
- `make typecheck` runs `mypy` on the `easyupscaler` package

### Test suite composition

`make test` (and CI) runs **in order**:

1. `ruff check` (and format check if enabled)
2. `mypy easyupscaler`
3. `pytest`

Any failure exits non-zero. Lint and typecheck are not optional side commands.

### Makefile

Provide a root `Makefile` with documented targets:

| Target | Purpose |
|--------|---------|
| `make sync` | `uv sync` — install dev dependencies into project env |
| `make lint` | ruff only |
| `make typecheck` | mypy only |
| `make test` | lint + typecheck + pytest |
| `make build` | `uv build` — produce sdist/wheel |
| `make install` | Install package into **active Python**: `uv pip install .` (or `uv pip install -e .` for editable dev install) |

`make install` uses the interpreter uv/pip resolves for the current environment (system or activated venv). Document in README that users should activate the target Python before `make install` if they want a specific interpreter.

No separate `setup.py`; build backend is standard `pyproject.toml` (e.g. `hatchling` or `uv_build`).

## Consequences

**Positive**

- One command (`make test`) gates quality before merge
- uv gives fast, reproducible dev installs
- Makefile gives discoverable install/build without memorizing uv flags
- Type checking and linting stay aligned with implementation from day one

**Negative**

- Contributors must install uv and use Make (common on macOS/Linux)
- mypy may need per-module relaxations for torch/spandrel stubs early on
- Windows users without Make need documented uv/pytest equivalents (best-effort; primary target is macOS)

**Follow-up**

- CI workflow mirroring `make test`
- Pre-commit hooks (optional; not required for MVP if `make test` is sufficient)
