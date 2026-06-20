# AGENTS.md

Guidance for LLM agents working on **easyupscaler** — a Python CLI for upscaling images with community super-resolution models.

This file describes **how to work in the repo**, not what the product does. Behavior, structure, and decisions live in `docs/` and `README.md`. Do not duplicate those details here — link to them and keep them current when code changes.

---

## Document map

| Document | Owns | Read when |
|----------|------|-----------|
| [docs/mission.md](docs/mission.md) | Product intent, MVP scope, non-goals | Scoping features or rejecting scope creep |
| [docs/specification.md](docs/specification.md) | CLI contracts: commands, flags, messages, edge cases, exit codes | Changing user-visible behavior |
| [docs/architecture.md](docs/architecture.md) | Layers, components, data flow, storage layout | Changing structure, modules, or integration |
| [docs/adr/](docs/adr/) | Immutable decisions and rationale | Choosing or changing architectural approach |
| [docs/adr.md](docs/adr.md) | ADR index | Finding the relevant ADR |
| [README.md](README.md) | Install, quickstart, troubleshooting for end users | User-facing workflow changes |
| [README-DEV.md](README-DEV.md) | Clone setup, Makefile, testing, architecture pointers | Dev workflow or toolchain changes |

**Precedence:** specification for user-facing contracts; ADRs for structural choices; architecture for system design; mission for scope. If docs conflict with code, treat that as a bug — fix the code or update the docs in the same change. If docs conflict with each other, ask before proceeding.

**AGENTS.md** owns process only: discovery, implementation discipline, doc maintenance, and where to look. Never copy spec values, error strings, constants, or command tables into this file.

---

## Keeping documentation current

Documentation is part of the deliverable. A code change that shifts behavior or structure without updating the right doc is incomplete.

### When to update what

| Change type | Update |
|-------------|--------|
| New/changed CLI command, flag, message, exit code, or edge-case behavior | Tests encoding the contract; `docs/specification.md`; `README.md` if users need to know |
| New module, layer boundary, component, or data-flow path | `docs/architecture.md` |
| New/changed architectural choice (backend, storage, lazy imports, toolchain, etc.) | New ADR in `docs/adr/`, index row in `docs/adr.md`, then `docs/architecture.md` |
| Scope change (in or out of MVP) | `docs/mission.md`; spec if behavior follows |
| Dev workflow (Make targets, test gate, lint/type tools, coverage) | ADR-009, ADR-010, or new ADR; `docs/architecture.md` toolchain section; `README-DEV.md` |
| Dependency or Python version policy | `pyproject.toml`, `README.md`; ADR if policy-level |

### ADR workflow

1. **Read** existing ADRs before proposing structural changes.
2. **Accepted ADRs are immutable.** Do not rewrite them to reflect new reality.
3. **Supersede** by adding the next numbered ADR (`docs/adr/NNN-title.md`) that states what it replaces and why.
4. **Index** the new ADR in [docs/adr.md](docs/adr.md).
5. **Propagate** — update `docs/architecture.md` (and spec/mission if user-visible or scope-related) to match the new decision. Remove or qualify superseded statements there; do not leave contradictions.

### Specification workflow

- Spec sections should match what tests assert and what the CLI actually prints. **Tests and spec are dual contracts** — update both when behavior changes.
- When adding behavior, add or update tests and the spec section **before or alongside** implementation — not as an afterthought.
- Error messages, stdout/stderr contracts, and exit codes belong in the spec, not in AGENTS.md.
- If implementation reveals a spec gap, update the spec in the same PR/change set.

### Architecture workflow

- `docs/architecture.md` is the living system diagram: layers, components, file layout, error-handling table, testing strategy.
- After structural refactors, update the package layout section and any affected diagrams or tables.
- Link to ADRs for rationale; architecture describes **what is**, ADRs explain **why it was chosen**.
- Do not restate ADR detail in architecture when a link suffices.

### README workflow

- [README.md](README.md) is for **users** — install, commands, paths, troubleshooting only.
- [README-DEV.md](README-DEV.md) is for **implementers** — clone setup, Makefile, testing, architecture pointers. Keep it aligned with `Makefile` targets (see ADR-009).
- Mirror spec-level user contracts in README; omit layer diagrams and ADR rationale there.

### Anti-patterns

- Duplicating spec/arch content in AGENTS.md, code comments, or README in three places
- Changing behavior without updating tests and spec together
- Removing or weakening tests to make broken code pass
- Editing an accepted ADR in place instead of superseding
- Leaving architecture.md describing removed modules or old data flows
- Documenting `_local_docs/` or other local-only notes in shared docs

---

## Development protocol

### Phase 1: Discovery

1. Read the relevant spec section and ADR(s) for the task.
2. Read existing tests for the area (`tests/test_*.py` mirroring the package module). They show enforced contracts and patterns to follow.
3. Search the codebase for similar patterns; verify APIs, schema fields, and paths in code — do not assume.
4. State what exists, what must change, and the gap — including which tests will be added or changed.
5. Stop and ask the user when requirements are unclear, trade-offs are real, or a breaking change has broad impact.

### Phase 2: Implementation

1. Smallest diff that solves the problem; match surrounding style.
2. Preserve layer boundaries (see [docs/architecture.md](docs/architecture.md)).
3. Let programming errors crash; catch only for I/O, user input, and unreliable externals (torch, Spandrel, OS).
4. Inject dependencies in services for testability.
5. **Write or update tests in the same change set as production code** — not in a follow-up. See [Testing](#testing).
6. Update the documentation rows from the table above when applicable.
7. Run `make test` before marking work complete.

### Phase 3: Review

- Change matches spec and mission scope
- Simplest solution that works
- Layer boundaries intact
- **Every behavior change has corresponding test coverage**; no orphaned or stale tests left behind
- **`easyupscaler` package line coverage stays ≥80%** (see [Testing § Code coverage](#code-coverage))
- Relevant docs updated in the same change set
- `make test` passes

---

## Repository layout

```
easyupscaler/     # application package (cli, config, models, upscaling, io)
tests/            # pytest; mirrors package concerns
docs/             # mission, specification, architecture, ADRs
pyproject.toml    # deps, ruff, mypy, pytest config
Makefile          # sync, lint, typecheck, test, build, install
```

Package and layer responsibilities: [docs/architecture.md](docs/architecture.md).

Test layout mirrors application concerns: `tests/test_<module>.py` for fast tests; `tests/slow/` for GPU/weight-dependent end-to-end tests.

---

## Testing

Tests are a first-class deliverable. They encode the same contracts as [docs/specification.md](docs/specification.md) and guard regressions. A feature or fix without tests is incomplete unless the user explicitly waives coverage.

Strategy overview: [docs/architecture.md § Testing strategy](docs/architecture.md#testing-strategy). Tooling config: `pyproject.toml`, ADR-009, `Makefile`.

### Quality gate

`make test` runs **ruff → mypy → pytest (with ≥80% coverage gate)** in order. All must pass. Do not treat lint, type, or coverage failures as separate from testing.

Run targeted tests while iterating (`uv run pytest tests/test_service.py -k foo`), but always finish with full `make test`.

### Code coverage

Maintain **≥80% line coverage** on the `easyupscaler` package. Coverage is measured over the fast test suite (slow tests excluded by default).

- **New or changed modules** must not merge with coverage below 80% for affected files.
- **Dropping below 80% project-wide** is a blocker — add tests before marking work complete.
- Prioritize covering error paths and branch logic, not hitting lines for its own sake. Untested torch-only paths (`spandrel_backend`, import validation) should use fakes/mocks in fast tests; reserve real-weight paths for `tests/slow/` where needed.

Check coverage when adding or removing production code (also enforced by `make test`):

```bash
uv run pytest --cov=easyupscaler --cov-report=term-missing --cov-fail-under=80
```

Configuration: `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.*]`).

### What to test

| Layer | Approach | Avoid |
|-------|----------|-------|
| **Domain** (`io/`, `tiling`, `registry`, `settings`) | Unit tests with real objects and `tmp_path`; no GPU | Asserting private helpers or internal call order |
| **Services** (`upscaling/service`, `import_model`) | Inject fakes via constructor (`FakeBackend`, mocked registry) | Loading real torch weights in fast tests |
| **CLI** | `typer.testing.CliRunner` against `app`; assert exit code, stdout, stderr | Duplicating service logic in CLI tests — mock at service boundary |
| **Lazy torch (ADR-008)** | `without_torch` fixture; assert `"torch" not in sys.modules` after fast commands | Importing torch in tests that claim to verify fast paths |
| **End-to-end** | `@pytest.mark.slow` in `tests/slow/`; real weights via env var | Putting slow/GPU tests in default suite |

Prioritize **error paths and edge cases** from the spec: missing default model, partial batch failure, duplicate import, corrupt input, empty path list. Happy paths alone are not enough.

### When to add, update, or remove tests

- **New behavior** → new test(s) asserting exit code, messages, and side effects (files written, registry entries).
- **Changed behavior** → update affected tests in the same commit; update spec to match.
- **Bug fix** → regression test that fails on old code, passes on the fix.
- **Refactor with no behavior change** → existing tests should still pass unchanged; add tests only if coverage was missing.
- **Removed feature** → delete tests for that feature; do not leave dead tests or skip markers without reason.

Never delete or weaken assertions to greenwash a regression.

### Shared fixtures (`tests/conftest.py`)

Add reusable fixtures here — not copied across test files.

| Fixture | Use |
|---------|-----|
| `isolated_paths` | Any test touching config, registry, or model files on disk. Required to avoid polluting the developer's real XDG directories. |
| `without_torch` | Verifying commands that must not import PyTorch. Clears torch from `sys.modules` for the test duration. |

Extend `conftest.py` when a new cross-cutting test need appears (e.g. a shared fake backend used by multiple modules).

### Patterns to follow

- **File naming:** `tests/test_<area>.py` aligned with the module under test (e.g. `test_registry.py`, `test_cli_upscale.py`).
- **Fakes over mocks:** Prefer small in-test classes like `FakeBackend` implementing the protocol; use `MagicMock` only for narrow CLI boundary stubs.
- **Constructor injection:** Pass `registry=`, `backend_factory=`, `config_service=` into services rather than patching module globals.
- **CLI tests:** Check `result.exit_code`, relevant substrings in `result.stdout` / `result.stderr`, and file artifacts on disk.
- **Slow tests:** Mark with `@pytest.mark.slow`; place under `tests/slow/`. Default pytest excludes them (`pyproject.toml` `addopts`). Document env requirements in test skip messages, not in AGENTS.md.

### Test anti-patterns

- Testing implementation details (call counts, private methods) instead of observable behavior
- Real inference or weight files in the fast suite
- Tests that depend on the developer's home directory or installed models
- One giant test covering an entire workflow when separate cases would localize failures
- Skipping `make test` because "only tests changed" — run the full gate anyway
- Merging production changes that drop package coverage below 80%

---

## Invariants (pointers only)

These rules are enforced in code and documented in ADRs — read the source docs when implementing, do not rely on this list for values or message text.

| Topic | Where defined |
|-------|----------------|
| Lazy PyTorch imports, fast non-inference commands | ADR-008, `docs/architecture.md` |
| Layer boundaries (CLI vs application vs domain) | `docs/architecture.md` |
| Output format, naming, JPEG settings | ADR-003, `docs/specification.md` |
| Tiling constants and OOM retry | ADR-007, `upscaling/tiling.py` |
| Registry, XDG paths, import rules | ADR-004, `docs/specification.md` |
| Batch processing and exit codes | ADR-006, `docs/specification.md` |
| Device policy (MPS, CPU fallback) | ADR-002 |
| Toolchain and quality gate | ADR-009, ADR-010, `Makefile`, `pyproject.toml` |

---

## Coding standards

- **Style and tools:** `pyproject.toml` (ruff, mypy) and ADR-009.
- **Principles:** KISS, single responsibility, explicit interfaces (protocols + constructor injection), named constants for tunable values, typed domain exceptions in `errors.py` mapped to CLI messages at the presentation layer.
- **Testability:** Design new code so it can be exercised without GPU, network, or real weight files. See [Testing](#testing).

Dev commands: `Makefile` and [README-DEV.md](README-DEV.md).

---

## Git

Do not commit or push unless the user explicitly asks.
