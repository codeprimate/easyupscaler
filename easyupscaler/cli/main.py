import os
import sys
from pathlib import Path

import typer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import easyupscaler
from easyupscaler.cli.models import models_app

ROOT_EPILOG = (
    "Getting started:\n"
    "  Import a model:   easyupscaler models import ~/Downloads/RealESRGAN_x4.pth\n"
    "  Upscale images:   easyupscaler scale photo.jpg\n"
    "  Denoise (no model needed): easyupscaler denoise photo image.jpg"
)

SCALE_EPILOG = (
    "Examples:\n"
    "  easyupscaler scale photo.jpg\n"
    "  easyupscaler scale *.jpg --model RealESRGAN_x4plus\n"
    "  easyupscaler scale *.jpg --output ./upscaled/"
)

DENOISE_EPILOG = (
    "Examples:\n"
    "  easyupscaler denoise photo image.jpg\n"
    "  easyupscaler denoise art artwork.png --strength high\n"
    "  easyupscaler denoise manga page*.png --output ./cleaned/"
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    epilog=ROOT_EPILOG,
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
    """Upscale and denoise images with AI super-resolution models.

    scale requires an imported model. denoise downloads models automatically on first use.
    """


@app.command("scale", no_args_is_help=True, epilog=SCALE_EPILOG)
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
    """Upscale images 2-4x with an imported super-resolution model.

    Requires at least one imported model. Run 'models import' to add one,
    or 'models list' to see what's available.
    """
    from easyupscaler.cli.scale import run_scale

    output_dir = Path(output) if output is not None else None
    run_scale(paths, model=model, output_dir=output_dir)


@app.command("denoise", no_args_is_help=True, epilog=DENOISE_EPILOG)
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
        help=(
            "Denoising intensity: low (subtle, preserves detail) or high (aggressive). "
            "[default: low]"
        ),
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to write output files. Defaults to the same directory as the input.",
    ),
) -> None:
    """Reduce noise and compression artifacts from photos, art, or manga.

    Models are selected automatically by mode and downloaded on first use.
    """
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
