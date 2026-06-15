import os
import sys
from typing import Annotated

import typer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import easyupscaler
from easyupscaler.cli.models import models_app

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"easyupscaler {easyupscaler.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name to use for upscaling."),
    ] = None,
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Image file paths to upscale."),
    ] = None,
) -> None:
    """Upscale images with imported GAN-based super-resolution models."""
    if ctx.invoked_subcommand is not None:
        return

    from easyupscaler.cli.upscale import run_upscale

    run_upscale(paths or [], model=model)


def main_entry() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "models":
        return (
            models_app(
                args=argv[1:],
                prog_name="easyupscaler models",
                standalone_mode=True,
            )
            or 0
        )
    return app(args=argv, prog_name="easyupscaler", standalone_mode=True) or 0


if __name__ == "__main__":
    main_entry()
