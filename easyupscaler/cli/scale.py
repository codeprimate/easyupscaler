import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from easyupscaler.io.heic import ensure_heif_registered
from easyupscaler.upscaling.service import UpscaleResult, UpscaleService

EMPTY_INPUT_ERROR = "Error: no input images. Pass one or more file paths."
COMPLETED_SUMMARY_TEMPLATE = "Completed: {succeeded} succeeded, {failed} failed in {elapsed}."


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
    is_tty = sys.stdout.isatty()
    service = UpscaleService()

    from easyupscaler.config.settings import ConfigService

    display_name = model or ConfigService().get_default_model() or "unknown"

    results: list[UpscaleResult] = []
    progress = None
    task_id = None

    if is_tty:
        progress = Progress(
            TextColumn("Upscaling {task.total} images with {task.fields[model]}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=Console(),
            transient=False,
        )
        progress.start()
        task_id = progress.add_task("upscale", total=len(resolved_paths), model=display_name)

    def on_progress(result: UpscaleResult) -> None:
        if progress is not None and task_id is not None:
            progress.advance(task_id)
        _print_result_line(result, tty=is_tty)

    try:
        started_at = time.perf_counter()
        results = service.run(
            resolved_paths,
            model,
            on_progress=on_progress,
            output_dir=output_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    finally:
        if progress is not None:
            progress.stop()

    succeeded = sum(1 for result in results if result.error is None)
    failed = len(results) - succeeded

    if is_tty:
        elapsed = time.perf_counter() - started_at
        typer.echo(
            COMPLETED_SUMMARY_TEMPLATE.format(
                succeeded=succeeded,
                failed=failed,
                elapsed=_format_elapsed(elapsed),
            )
        )

    if failed > 0:
        raise typer.Exit(code=1)


def _print_result_line(result: UpscaleResult, *, tty: bool) -> None:
    if result.error is None and result.output is not None:
        if tty:
            typer.echo(f"  ✓ {result.path.name} → {result.output.name}")
        else:
            typer.echo(f"{result.path} → {result.output}")
        return

    if tty:
        typer.echo(f"  ✗ {result.path.name} — {result.error}")
    else:
        typer.echo(f"{result.path} FAILED: {result.error}")


def _format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = int(elapsed_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
