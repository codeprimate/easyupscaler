# ADR-010: Code coverage gate (≥80% on easyupscaler)

**Status:** Accepted  
**Date:** 2025-06-15

## Context

The project requires tests as first-class deliverables (see AGENTS.md, architecture testing strategy). Without a numeric floor, untested modules can land while `make test` still passes — notably torch-heavy paths that are skipped in fast tests.

ADR-009 defines `make test` as ruff, mypy, then pytest. Coverage was advisory only.

## Decision

- Add **`pytest-cov`** to the dev dependency group.
- Configure **`≥80% line coverage`** on the `easyupscaler` package in `pyproject.toml` (`[tool.coverage.report] fail_under = 80`).
- Enforce via pytest `addopts`: `--cov=easyupscaler --cov-report=term-missing --cov-fail-under=80`.
- Measurement uses the **fast test suite** (slow tests remain excluded by default `-m "not slow"`).
- `make test` fails if coverage drops below 80%.

## Consequences

**Positive**

- Coverage regressions block merge locally and in CI
- Single config in `pyproject.toml`; no separate coverage command to remember for the gate

**Negative**

- New code in lightly tested modules (e.g. `spandrel_backend`) needs mocked fast tests to stay above the floor
- Slow/e2e tests do not contribute to the default gate; torch paths may rely on fakes in unit tests

**Follow-up**

- CI workflow should mirror `make test` including the coverage gate
