# ADR-005: Typer for CLI framework

**Status:** Accepted  
**Date:** 2025-06-15

## Context

The CLI needs a top-level command, a `models` subcommand group (`list`, `import`, `default`), global options (`--model`), and variadic image path arguments. Python CLI frameworks under consideration:

| Framework | Notes |
|-----------|-------|
| **argparse** | Stdlib; verbose for nested subcommands |
| **Click** | Mature; nested groups well supported |
| **Typer** | Click-based; type hints; less boilerplate |

Mission examples map naturally to `easyupscaler models list` style nesting.

`import torch` adds 2–6 seconds to every process start. Commands that only read config or registry should not pay that cost ([ADR-008](./008-lazy-torch-imports.md)).

## Decision

Use **Typer** for the CLI.

- Entry point: `easyupscaler.cli.main:app` registered in `pyproject.toml` `[project.scripts]`
- Structure:
  - Root callback accepts optional `--model` and trailing image paths
  - `models` Typer sub-app with `list`, `import`, `default` commands
- Testing via `typer.testing.CliRunner`
- Lazy-import `torch` and `spandrel` only in upscale and `models import` code paths

## Consequences

**Positive**

- Nested subcommands match mission UX with minimal code
- Type-annotated parameters reduce parsing bugs
- Built on Click; stable ecosystem
- Lazy imports keep `models list`, `models default`, and `--help` responsive

**Negative**

- Extra dependency (acceptable for a CLI-first project)
- Typer adds magic compared to raw argparse; team must know Typer patterns

**Implementation note**

Root command behavior: if paths are provided, run upscale; if `models` subcommand is invoked, dispatch to model management. Avoid ambiguous overlap between subcommands and path arguments.
