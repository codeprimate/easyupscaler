import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from easyupscaler.denoise.pipeline import DenoiseResult, DenoiseService
from easyupscaler.io.heic import ensure_heif_registered

EMPTY_INPUT_ERROR = "Error: no input images. Pass one or more file paths."
UNSUPPORTED_MODEL_FLAG_ERROR = (
    "Error: --model is not supported for 'denoise'. "
    "Model is selected automatically by mode and strength."
)
COMPLETED_SUMMARY_TEMPLATE = "Completed: {succeeded} succeeded, {failed} failed in {elapsed}."
VALID_MODES = {"photo", "art", "manga"}
VALID_STRENGTHS = {"low", "high"}


def run_denoise(
    paths: list[str],
    *,
    mode: str,
    strength: str,
) -> None:
    if not paths:
        typer.echo(EMPTY_INPUT_ERROR, err=True)
        raise typer.Exit(code=1)

    if mode not in VALID_MODES:
        typer.echo(f"Error: invalid mode '{mode}'. Choose photo, art, or manga.", err=True)
        raise typer.Exit(code=1)

    if strength not in VALID_STRENGTHS:
        typer.echo(f"Error: invalid strength '{strength}'. Choose low or high.", err=True)
        raise typer.Exit(code=1)

    ensure_heif_registered()
    resolved_paths = [Path(path) for path in paths]
    is_tty = sys.stdout.isatty()
    service = DenoiseService()

    results: list[DenoiseResult] = []
    download_progress: Progress | None = None
    denoise_progress: Progress | None = None
    denoise_task_id = None

    try:
        started_at = time.perf_counter()

        if is_tty:
            download_progress = Progress(
                TextColumn("Downloading {task.fields[filename]}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=Console(),
                transient=True,
            )

        def download_callback(filename: str, bytes_done: int, bytes_total: int | None) -> None:
            if is_tty:
                if download_progress is not None:
                    if not download_progress.finished:
                        download_progress.start()
                    total = bytes_total or bytes_done or 1
                    if download_progress.tasks:
                        download_progress.update(
                            download_progress.tasks[0].id,
                            completed=min(bytes_done, total),
                            total=total,
                        )
                    else:
                        download_progress.add_task(
                            "download",
                            total=total,
                            completed=bytes_done,
                            filename=filename,
                        )
            else:
                typer.echo(f"Downloading {filename}...")

        if is_tty:
            denoise_progress = Progress(
                TextColumn(
                    "Denoising {task.total} images [{task.fields[mode]}, {task.fields[strength]}]"
                ),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=Console(),
                transient=False,
            )
            denoise_progress.start()
            denoise_task_id = denoise_progress.add_task(
                "denoise",
                total=len(resolved_paths),
                mode=mode,
                strength=strength,
            )

        def on_progress(result: DenoiseResult) -> None:
            if denoise_progress is not None and denoise_task_id is not None:
                denoise_progress.advance(denoise_task_id)
            _print_result_line(result, tty=is_tty)

        results = service.run(
            resolved_paths,
            mode,  # type: ignore[arg-type]
            strength,  # type: ignore[arg-type]
            on_progress=on_progress,
            on_download_progress=download_callback,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    finally:
        if download_progress is not None:
            download_progress.stop()
        if denoise_progress is not None:
            denoise_progress.stop()

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


def _print_result_line(result: DenoiseResult, *, tty: bool) -> None:
    if result.error is None and result.output is not None:
        pass_note = ""
        if tty and result.pass_description:
            pass_note = f"   (2 passes: {result.pass_description})"
        if tty:
            typer.echo(f"  ✓ {result.path.name} → {result.output.name}{pass_note}")
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
