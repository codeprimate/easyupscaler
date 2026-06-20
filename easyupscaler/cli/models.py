from pathlib import Path
from typing import Annotated

import typer

from easyupscaler.config.settings import ConfigService
from easyupscaler.errors import DuplicateModelError, ImportModelError, ModelNotFoundError
from easyupscaler.models.registry import ModelRegistry

models_app = typer.Typer(
    help="Import and manage models used by the scale command.",
    add_completion=False,
)


@models_app.callback(invoke_without_command=True)
def models_main(ctx: typer.Context) -> None:
    """Show available model commands when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

EMPTY_MODELS_MESSAGE = (
    "No models installed. Use 'easyupscaler models import <path>' to add one."
)
PATH_COLUMN_HEADER = "Path"


def _format_models_table(entries: list) -> str:
    if not entries:
        return EMPTY_MODELS_MESSAGE

    name_width = max(len("Name"), *(len(entry.name) for entry in entries))
    scale_width = len("Scale")
    path_width = max(
        len(PATH_COLUMN_HEADER),
        *(len(str(entry.path)) for entry in entries),
    )

    name_col = "Name".ljust(name_width)
    scale_col = "Scale".ljust(scale_width)
    path_col = PATH_COLUMN_HEADER.ljust(path_width)
    header = f"{name_col}  {scale_col}  {path_col}"
    divider = "─" * len(header)
    rows = []
    for entry in entries:
        scale_text = f"{entry.scale}×".ljust(scale_width)
        row = (
            f"{entry.name.ljust(name_width)}  {scale_text}  "
            f"{str(entry.path).ljust(path_width)}"
        )
        rows.append(row)
    return "\n".join([header, divider, *rows])


@models_app.command("import")
def import_model_command(
    path: Annotated[
        Path,
        typer.Argument(help="Path to a local model weight file (.pth, .pt, .safetensors)."),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing model with the same name."),
    ] = False,
) -> None:
    """Import a model weight file from disk.

    Supported formats: .pth, .pt, .safetensors. The architecture is detected
    automatically. Find community models at openmodeldb.info.
    """
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


@models_app.command("list")
def list_models() -> None:
    """List installed models and which is set as default."""
    registry = ModelRegistry()
    entries = registry.list()
    typer.echo(_format_models_table(entries))


@models_app.command("default")
def set_default(
    name: Annotated[str, typer.Argument(help="Registry name of the model to set as default.")],
) -> None:
    """Set the model used by scale when --model is not specified."""
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
