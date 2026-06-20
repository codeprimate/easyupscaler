import os
import sys
from pathlib import Path

import typer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import easyupscaler
from easyupscaler.cli.help_text import (
    DENOISE_HELP,
    DENOISE_SHORT_HELP,
    SCALE_HELP,
    SCALE_SHORT_HELP,
    build_main_help,
)
from easyupscaler.cli.models import models_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

app.add_typer(models_app, name="models")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"easyupscaler {easyupscaler.__version__}")
        raise typer.Exit()


@app.callback(help=build_main_help(easyupscaler.__version__))
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


@app.command("scale", no_args_is_help=True, help=SCALE_HELP, short_help=SCALE_SHORT_HELP)
def scale_command(
    paths: list[str] = typer.Argument(
        default=[],
        metavar="IMAGE",
        help="Image files to upscale (JPEG, PNG, HEIC, WebP, TIFF).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Imported model name (see 'models list'). Uses the registered default when omitted.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to write output files. Defaults to the same directory as the input.",
    ),
) -> None:
    from easyupscaler.cli.scale import run_scale

    output_dir = Path(output) if output is not None else None
    run_scale(paths, model=model, output_dir=output_dir)


@app.command("denoise", no_args_is_help=True, help=DENOISE_HELP, short_help=DENOISE_SHORT_HELP)
def denoise_command(
    mode: str = typer.Argument(
        ...,
        metavar="MODE",
        help=(
            "Content type: photo (photographs), art (digital illustrations), "
            "or manga (manga and comics)."
        ),
    ),
    paths: list[str] = typer.Argument(
        default=[],
        metavar="IMAGE",
        help="Image files to denoise (JPEG, PNG, HEIC, WebP, TIFF).",
    ),
    strength: str = typer.Option(
        "low",
        "--strength",
        help="Denoising intensity: low (subtle, preserves detail) or high (aggressive).",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to write output files. Defaults to the same directory as the input.",
    ),
) -> None:
    from easyupscaler.cli.denoise import run_denoise

    output_dir = Path(output) if output is not None else None
    run_denoise(paths, mode=mode, strength=strength, output_dir=output_dir)


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
