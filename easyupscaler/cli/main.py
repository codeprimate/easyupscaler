import os
import sys

import typer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import easyupscaler
from easyupscaler.cli.models import models_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

app.add_typer(models_app, name="models")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"easyupscaler {easyupscaler.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Upscale and denoise images with GAN-based models."""


@app.command("scale")
def scale_command(
    paths: list[str] = typer.Argument(default=[]),
    model: str | None = typer.Option(None, "--model", help="Model name to use for upscaling."),
) -> None:
    """Upscale images with imported super-resolution models."""
    from easyupscaler.cli.scale import run_scale

    run_scale(paths, model=model)


@app.command("denoise")
def denoise_command(
    mode: str = typer.Argument(..., help="Denoise mode: photo, art, or manga."),
    paths: list[str] = typer.Argument(default=[]),
    strength: str = typer.Option("low", "--strength", help="Denoise strength: low or high."),
) -> None:
    """Denoise images with automatically selected AI models."""
    from easyupscaler.cli.denoise import run_denoise

    run_denoise(paths, mode=mode, strength=strength)


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
