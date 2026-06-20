import time
from pathlib import Path

import typer

from easyupscaler.cli.job_progress import (
    DenoiseJobConfig,
    collect_denoise_outputs,
    create_denoise_job_display,
    is_tty_stdout,
)
from easyupscaler.denoise.pipeline import DenoiseResult, DenoiseService
from easyupscaler.io.heic import ensure_heif_registered

EMPTY_INPUT_ERROR = "Error: no input images. Pass one or more file paths."
VALID_MODES = {"photo", "art", "manga", "document"}
VALID_STRENGTHS = {"low", "high"}


def run_denoise(
    paths: list[str],
    *,
    mode: str,
    strength: str,
    output_dir: Path | None = None,
    extract_text: bool = True,
    use_ocrai: bool = False,
) -> None:
    if not paths:
        typer.echo(EMPTY_INPUT_ERROR, err=True)
        raise typer.Exit(code=1)

    if mode not in VALID_MODES:
        typer.echo(
            f"Error: invalid mode '{mode}'. Choose photo, art, manga, or document.",
            err=True,
        )
        raise typer.Exit(code=1)

    if strength not in VALID_STRENGTHS:
        typer.echo(f"Error: invalid strength '{strength}'. Choose low or high.", err=True)
        raise typer.Exit(code=1)

    if output_dir is not None:
        from easyupscaler.cli.output_dir import prepare_output_dir

        try:
            output_dir = prepare_output_dir(output_dir)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None

    ensure_heif_registered()
    resolved_paths = [Path(path) for path in paths]
    is_tty = is_tty_stdout()
    service = DenoiseService()
    display = create_denoise_job_display(
        is_tty=is_tty,
        config=DenoiseJobConfig(
            mode=mode,
            strength=strength,
            file_count=len(resolved_paths),
            output_dir=output_dir,
            extract_text=extract_text,
            use_ocrai=use_ocrai,
        ),
    )

    results: list[DenoiseResult] = []
    started_at = time.perf_counter()

    try:
        display.begin_job()

        def on_phase(event):
            display.handle_phase(event)

        def on_progress(result: DenoiseResult) -> None:
            display.complete_file(
                path=result.path,
                error=result.error,
                outputs=collect_denoise_outputs(result),
            )

        def on_warning(message: str) -> None:
            typer.echo(message, err=True)

        results = service.run(
            resolved_paths,
            mode,  # type: ignore[arg-type]
            strength,  # type: ignore[arg-type]
            on_progress=on_progress,
            on_phase=on_phase,
            on_download_progress=display.handle_download,
            output_dir=output_dir,
            extract_text=extract_text,
            use_ocrai=use_ocrai,
            on_warning=on_warning,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    succeeded = sum(1 for result in results if result.error is None)
    failed = len(results) - succeeded
    display.finish_job(
        succeeded=succeeded,
        failed=failed,
        elapsed_seconds=time.perf_counter() - started_at,
    )

    if failed > 0:
        raise typer.Exit(code=1)
