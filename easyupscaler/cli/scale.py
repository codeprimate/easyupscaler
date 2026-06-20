import time
from pathlib import Path

import typer

from easyupscaler.cli.job_progress import (
    ScaleJobConfig,
    create_scale_job_display,
    is_tty_stdout,
)
from easyupscaler.io.heic import ensure_heif_registered
from easyupscaler.upscaling.service import UpscaleResult, UpscaleService

EMPTY_INPUT_ERROR = "Error: no input images. Pass one or more file paths."


def run_scale(
    paths: list[str],
    *,
    model: str | None,
    output_dir: Path | None = None,
) -> None:
    if not paths:
        typer.echo(EMPTY_INPUT_ERROR, err=True)
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
    service = UpscaleService()

    from easyupscaler.config.settings import ConfigService

    display_name = model or ConfigService().get_default_model() or "unknown"
    display = create_scale_job_display(
        is_tty=is_tty,
        config=ScaleJobConfig(
            model_name=display_name,
            file_count=len(resolved_paths),
            output_dir=output_dir,
        ),
    )

    results: list[UpscaleResult] = []
    started_at = time.perf_counter()

    try:
        display.begin_job()

        def on_phase(event):
            display.handle_phase(event)

        def on_progress(result: UpscaleResult) -> None:
            outputs = [result.output] if result.output is not None else []
            display.complete_file(path=result.path, error=result.error, outputs=outputs)

        results = service.run(
            resolved_paths,
            model,
            on_progress=on_progress,
            on_phase=on_phase,
            output_dir=output_dir,
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
