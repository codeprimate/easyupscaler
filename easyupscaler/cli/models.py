from pathlib import Path
from typing import Annotated

import typer

from easyupscaler.config.settings import ConfigService
from easyupscaler.errors import DuplicateModelError, ImportModelError, ModelNotFoundError
from easyupscaler.models.registry import ModelRegistry

models_app = typer.Typer(help="Manage installed upscaling models.")

EMPTY_MODELS_MESSAGE = (
    "No models installed. Use 'easyupscaler models import <path>' to add one."
)


def _format_models_table(entries: list) -> str:
    if not entries:
        return EMPTY_MODELS_MESSAGE

    name_width = max(len("Name"), *(len(entry.name) for entry in entries))
    scale_width = len("Scale")
    filename_width = max(len("Filename"), *(len(entry.filename) for entry in entries))

    name_col = "Name".ljust(name_width)
    scale_col = "Scale".ljust(scale_width)
    filename_col = "Filename".ljust(filename_width)
    header = f"{name_col}  {scale_col}  {filename_col}"
    divider = "─" * len(header)
    rows = []
    for entry in entries:
        scale_text = f"{entry.scale}×".ljust(scale_width)
        row = (
            f"{entry.name.ljust(name_width)}  {scale_text}  "
            f"{entry.filename.ljust(filename_width)}"
        )
        rows.append(row)
    return "\n".join([header, divider, *rows])


@models_app.command("list")
def list_models() -> None:
    """List installed models."""
    registry = ModelRegistry()
    entries = registry.list()
    typer.echo(_format_models_table(entries))


@models_app.command("default")
def set_default(
    name: Annotated[str, typer.Argument(help="Registry name of the model to set as default.")],
) -> None:
    """Set the default upscaling model."""
    registry = ModelRegistry()
    try:
        registry.get(name)
    except ModelNotFoundError:
        typer.echo(f"Error: model '{name}' not found. Installed models:", err=True)
        typer.echo(_format_models_table(registry.list()), err=True)
        raise typer.Exit(code=1) from None

    ConfigService().set_default_model(name)
    typer.echo(f"Default model set to {name}")


@models_app.command("remove")
def remove_model(
    name: Annotated[str, typer.Argument(help="Registry name of the model to remove.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Remove a model from the registry and delete its weight file."""
    registry = ModelRegistry()
    try:
        entry = registry.get(name)
    except ModelNotFoundError:
        typer.echo(f"Error: model '{name}' not found.", err=True)
        raise typer.Exit(code=1) from None

    if not yes:
        confirmed = typer.confirm(
            f"Remove {name} and delete {entry.filename}?",
            default=False,
        )
        if not confirmed:
            raise typer.Exit()

    registry.remove(name)
    if entry.path.exists():
        entry.path.unlink()

    config = ConfigService()
    if config.get_default_model() == name:
        config.clear_default_model()
        typer.echo(
            f"Warning: {name} was the default model. No default is now set.",
            err=True,
        )


@models_app.command("import")
def import_model_command(
    path: Annotated[Path, typer.Argument(help="Path to a local model weight file.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing model with the same name."),
    ] = False,
) -> None:
    """Import a local super-resolution model weight file."""
    from easyupscaler.models.import_model import import_model

    try:
        import_model(path, force=force)
    except DuplicateModelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except ImportModelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
