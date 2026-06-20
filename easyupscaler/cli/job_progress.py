from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import typer
from rich.console import Console, Group, RenderableType
from rich.text import Text

from easyupscaler.progress import PhaseEvent, PhaseKind, PhaseStatus

SUMMARY_TEMPLATE = "Done · {succeeded} succeeded · {failed} failed · {elapsed}"

CHECKLIST_LABEL_WIDTH = 10
CHECKLIST_TIME_WIDTH = 8
CHECKLIST_REFRESH_SECONDS = 0.25
CHECKLIST_MIN_PATH_WIDTH = 20

ICON_PENDING = "○"
ICON_RUNNING = "●"
ICON_DONE = "✓"
ICON_SKIPPED = "–"

COLOR_HEADER = "bold"
COLOR_PENDING = "dim"
COLOR_RUNNING = "yellow"
COLOR_DONE = "green"
COLOR_TIME = "green"
COLOR_PATH = "cyan"
COLOR_SKIPPED = "dim italic"
COLOR_ERROR = "bold red"

PhaseCallback = Callable[[PhaseEvent], None]


@dataclass
class _PhaseState:
    status: PhaseStatus = PhaseStatus.PENDING
    elapsed_seconds: float | None = None
    detail: str | None = None
    running_started_at: float | None = None
    output_path: Path | None = None


class JobProgressDisplay(Protocol):
    def handle_download(
        self,
        filename: str,
        bytes_done: int,
        bytes_total: int | None,
    ) -> None: ...

    def begin_job(self) -> None: ...

    def handle_phase(self, event: PhaseEvent) -> None: ...

    def complete_file(self, *, path: Path, error: str | None, outputs: Sequence[Path]) -> None: ...

    def finish_job(self, *, succeeded: int, failed: int, elapsed_seconds: float) -> None: ...


@dataclass
class DenoiseJobConfig:
    mode: str
    strength: str
    file_count: int
    output_dir: Path | None
    extract_text: bool
    use_ocrai: bool


@dataclass
class ScaleJobConfig:
    model_name: str
    file_count: int
    output_dir: Path | None


def create_denoise_job_display(
    *,
    is_tty: bool,
    config: DenoiseJobConfig,
) -> JobProgressDisplay:
    if is_tty:
        return TtyDenoiseJobDisplay(config=config)
    return PlainDenoiseJobDisplay(config=config)


def create_scale_job_display(
    *,
    is_tty: bool,
    config: ScaleJobConfig,
) -> JobProgressDisplay:
    if is_tty:
        return TtyScaleJobDisplay(config=config)
    return PlainScaleJobDisplay(config=config)


def format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = int(elapsed_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_elapsed_seconds(elapsed_seconds: float) -> str:
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.1f}s"
    return format_elapsed(elapsed_seconds)


def format_running_elapsed(running_started_at: float) -> str:
    elapsed = max(0.0, time.perf_counter() - running_started_at)
    return format_elapsed(elapsed)


def format_output_location(output_dir: Path | None) -> str:
    if output_dir is None:
        return "same directory as input"
    return _display_path(output_dir)


def _display_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _create_tty_console() -> tuple[Console, object | None]:
    if sys.stdout.isatty():
        try:
            tty_file = open("/dev/tty", "w")
            return Console(file=tty_file, force_terminal=True), tty_file
        except OSError:
            pass
    return Console(file=sys.stdout, force_terminal=sys.stdout.isatty()), None


def _format_checklist_path(path: Path, *, console: Console) -> str:
    full = _display_path(path)
    reserved = 2 + 2 + CHECKLIST_LABEL_WIDTH + CHECKLIST_TIME_WIDTH
    budget = max(CHECKLIST_MIN_PATH_WIDTH, console.width - reserved)
    if len(full) <= budget:
        return full
    head = budget // 2 - 1
    tail = budget - head - 1
    return f"{full[:head]}…{full[-tail:]}"


def _image_count_label(count: int) -> str:
    if count == 1:
        return "1 image"
    return f"{count} images"


def _mode_title(mode: str) -> str:
    return mode.capitalize()


def _checklist_label(phase: PhaseKind, *, document_mode: bool) -> str:
    if phase == PhaseKind.IMAGE:
        return "PNG" if document_mode else "Denoise"
    if phase == PhaseKind.TEXT:
        return "Text"
    if phase == PhaseKind.MARKDOWN:
        return "Markdown"
    return "Upscale"


def _phase_label(phase: PhaseKind, *, document_mode: bool) -> str:
    if phase == PhaseKind.MARKDOWN:
        return "Markdown (VLM)"
    return _checklist_label(phase, document_mode=document_mode)


def _denoise_phase_plan(config: DenoiseJobConfig) -> tuple[PhaseKind, ...]:
    if config.mode != "document":
        return (PhaseKind.IMAGE,)
    if not config.extract_text:
        return (PhaseKind.IMAGE,)
    phases: list[PhaseKind] = [PhaseKind.IMAGE, PhaseKind.TEXT]
    if config.use_ocrai:
        phases.append(PhaseKind.MARKDOWN)
    return tuple(phases)


def _render_checklist_line(
    phase: PhaseKind,
    label: str,
    state: _PhaseState,
    *,
    console: Console,
) -> Text:
    line = Text()
    line.append("  ")

    if state.status == PhaseStatus.PENDING:
        line.append(f"{ICON_PENDING} ", style=COLOR_PENDING)
        line.append(f"{label:<{CHECKLIST_LABEL_WIDTH}}", style=COLOR_PENDING)
        line.append("pending", style=COLOR_PENDING)
        return line

    if state.status == PhaseStatus.SKIPPED:
        line.append(f"{ICON_SKIPPED} ", style=COLOR_SKIPPED)
        line.append(f"{label:<{CHECKLIST_LABEL_WIDTH}}", style=COLOR_SKIPPED)
        line.append("skipped", style=COLOR_SKIPPED)
        return line

    if state.status == PhaseStatus.RUNNING:
        line.append(f"{ICON_RUNNING} ", style=COLOR_RUNNING)
        line.append(f"{label:<{CHECKLIST_LABEL_WIDTH}}")
        if state.running_started_at is not None:
            elapsed = format_running_elapsed(state.running_started_at)
            line.append(f"running ({elapsed})", style=COLOR_RUNNING)
        else:
            line.append("running", style=COLOR_RUNNING)
        return line

    line.append(f"{ICON_DONE} ", style=COLOR_DONE)
    line.append(f"{label:<{CHECKLIST_LABEL_WIDTH}}")
    if state.elapsed_seconds is not None:
        time_text = format_elapsed_seconds(state.elapsed_seconds)
        line.append(f"{time_text:<{CHECKLIST_TIME_WIDTH}}", style=COLOR_TIME)
    elif state.detail:
        line.append(f"{'':<{CHECKLIST_TIME_WIDTH}}")
    else:
        line.append(f"{'':<{CHECKLIST_TIME_WIDTH}}")
    if state.output_path is not None:
        line.append(_format_checklist_path(state.output_path, console=console), style=COLOR_PATH)
    if state.detail:
        line.append(f"  ({state.detail})", style=COLOR_PENDING)
    return line


def _render_file_checklist(
    *,
    file_index: int,
    file_count: int,
    path: Path,
    phase_plan: tuple[PhaseKind, ...],
    phases: dict[PhaseKind, _PhaseState],
    document_mode: bool,
    console: Console,
) -> Group:
    header = Text(
        f"[{file_index + 1}/{file_count}] {path.name}",
        style=COLOR_HEADER,
    )
    lines = [
        _render_checklist_line(
            phase,
            _checklist_label(phase, document_mode=document_mode),
            phases[phase],
            console=console,
        )
        for phase in phase_plan
    ]
    return Group(header, *lines)


class _TtyJobDisplayBase:
    def __init__(
        self,
        *,
        job_header: str,
        phase_plan: tuple[PhaseKind, ...],
        document_mode: bool,
    ) -> None:
        self._console, self._tty_file = _create_tty_console()
        self._job_header = job_header
        self._phase_plan = phase_plan
        self._document_mode = document_mode
        self._paint_lock = threading.Lock()
        self._refresh_stop: threading.Event | None = None
        self._refresh_thread: threading.Thread | None = None
        self._current_file_index = 0
        self._current_file_count = 0
        self._current_path: Path | None = None
        self._phases: dict[PhaseKind, _PhaseState] = {}
        self._announced_downloads: set[str] = set()
        self._checklist_on_screen = False
        self._checklist_lines = 0

    def handle_download(
        self,
        filename: str,
        bytes_done: int,
        bytes_total: int | None,
    ) -> None:
        if filename not in self._announced_downloads:
            self._announced_downloads.add(filename)
            if bytes_total:
                size_mb = bytes_total / (1024 * 1024)
                typer.echo(f"Downloading {filename} ({size_mb:.0f} MB)...")
            else:
                typer.echo(f"Downloading {filename}...")

    def begin_job(self) -> None:
        self._console.print(self._job_header, style=COLOR_HEADER)
        self._console.print()

    def handle_phase(self, event: PhaseEvent) -> None:
        if event.status == PhaseStatus.RUNNING and self._current_path != event.path:
            self._begin_file(event.file_index, event.file_count, event.path)
        state = self._phases[event.phase]
        if event.status == PhaseStatus.RUNNING:
            state.status = PhaseStatus.RUNNING
            state.running_started_at = time.perf_counter()
        elif event.status == PhaseStatus.DONE:
            state.status = PhaseStatus.DONE
            state.elapsed_seconds = event.elapsed_seconds
            state.detail = event.detail
            state.output_path = event.output_path
            state.running_started_at = None
        elif event.status == PhaseStatus.SKIPPED:
            state.status = PhaseStatus.SKIPPED
            state.running_started_at = None
        with self._paint_lock:
            self._paint_checklist()
        if self._has_refreshable_running_phase():
            self._start_refresh_loop()
        else:
            self._stop_refresh_loop()

    def complete_file(self, *, path: Path, error: str | None, outputs: Sequence[Path]) -> None:
        self._stop_refresh_loop()
        with self._paint_lock:
            if self._current_path == path and self._phases:
                if self._checklist_on_screen:
                    self._console.print()
                    self._checklist_on_screen = False
                    self._checklist_lines = 0
                if error is not None:
                    self._console.print(f"  ✗ {path.name} — {error}", style=COLOR_ERROR)
                    self._console.print()
            elif error is not None:
                self._console.print(f"  ✗ {path.name} — {error}", style=COLOR_ERROR)
                self._console.print()
        self._current_path = None
        self._phases = {}

    def finish_job(self, *, succeeded: int, failed: int, elapsed_seconds: float) -> None:
        self._stop_refresh_loop()
        self._console.print(
            SUMMARY_TEMPLATE.format(
                succeeded=succeeded,
                failed=failed,
                elapsed=format_elapsed(elapsed_seconds),
            )
        )

    def _begin_file(self, file_index: int, file_count: int, path: Path) -> None:
        self._checklist_on_screen = False
        self._checklist_lines = 0
        self._current_file_index = file_index
        self._current_file_count = file_count
        self._current_path = path
        self._phases = {phase: _PhaseState() for phase in self._phase_plan}

    def _has_refreshable_running_phase(self) -> bool:
        return any(state.status == PhaseStatus.RUNNING for state in self._phases.values())

    def _start_refresh_loop(self) -> None:
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self._refresh_stop = threading.Event()
        stop = self._refresh_stop

        def refresh_loop() -> None:
            while not stop.wait(CHECKLIST_REFRESH_SECONDS):
                with self._paint_lock:
                    if not self._has_refreshable_running_phase():
                        break
                    self._paint_checklist()

        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _stop_refresh_loop(self) -> None:
        if self._refresh_stop is not None:
            self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=1.0)
        self._refresh_stop = None
        self._refresh_thread = None

    def _paint_checklist(self) -> None:
        if self._current_path is None:
            return
        panel = self._render_current_file()
        with self._console.capture() as capture:
            self._console.print(panel)
        rendered = capture.get().rstrip("\n")
        new_lines = rendered.split("\n") if rendered else []
        new_line_count = len(new_lines)
        previous_line_count = self._checklist_lines if self._checklist_on_screen else 0

        if previous_line_count > 0:
            self._console.file.write(f"\033[{previous_line_count}A")

        for line in new_lines:
            self._console.file.write("\033[2K\r")
            self._console.file.write(line)
            self._console.file.write("\n")

        if previous_line_count > new_line_count:
            for _ in range(previous_line_count - new_line_count):
                self._console.file.write("\033[2K\r\n")

        self._checklist_on_screen = True
        self._checklist_lines = new_line_count
        self._console.file.flush()

    def _render_current_file(self) -> RenderableType:
        if self._current_path is None:
            return Text("")
        return _render_file_checklist(
            file_index=self._current_file_index,
            file_count=self._current_file_count,
            path=self._current_path,
            phase_plan=self._phase_plan,
            phases=self._phases,
            document_mode=self._document_mode,
            console=self._console,
        )


class TtyDenoiseJobDisplay(_TtyJobDisplayBase):
    def __init__(self, *, config: DenoiseJobConfig) -> None:
        header = (
            f"{_mode_title(config.mode)} denoise · {config.strength} · "
            f"{_image_count_label(config.file_count)} · "
            f"output {format_output_location(config.output_dir)}"
        )
        super().__init__(
            job_header=header,
            phase_plan=_denoise_phase_plan(config),
            document_mode=config.mode == "document",
        )


class TtyScaleJobDisplay(_TtyJobDisplayBase):
    def __init__(self, *, config: ScaleJobConfig) -> None:
        header = (
            f"Upscale · {config.model_name} · "
            f"{_image_count_label(config.file_count)} · "
            f"output {format_output_location(config.output_dir)}"
        )
        super().__init__(
            job_header=header,
            phase_plan=(PhaseKind.UPSCALE,),
            document_mode=False,
        )


class _PlainJobDisplayBase:
    def __init__(self, *, job_header: str, phase_labels: dict[PhaseKind, str]) -> None:
        self._job_header = job_header
        self._phase_labels = phase_labels
        self._announced_downloads: set[str] = set()
        self._announced_running: set[tuple[str, PhaseKind]] = set()

    def handle_download(
        self,
        filename: str,
        bytes_done: int,
        bytes_total: int | None,
    ) -> None:
        if filename not in self._announced_downloads:
            self._announced_downloads.add(filename)
            typer.echo(f"Downloading {filename}...")

    def begin_job(self) -> None:
        typer.echo(self._job_header)

    def handle_phase(self, event: PhaseEvent) -> None:
        label = self._phase_labels[event.phase]
        key = (event.path.name, event.phase)
        if event.status == PhaseStatus.RUNNING:
            if key not in self._announced_running:
                self._announced_running.add(key)
                typer.echo(f"{event.path.name}: {label}...")
            return
        if event.status == PhaseStatus.SKIPPED:
            typer.echo(f"{event.path.name}: {label} skipped")
            return
        if event.status == PhaseStatus.DONE and event.output_path is not None:
            elapsed = ""
            if event.elapsed_seconds is not None:
                elapsed = f" ({format_elapsed_seconds(event.elapsed_seconds)})"
            typer.echo(
                f"{event.path.name}: {label} → {_display_path(event.output_path)}{elapsed}"
            )

    def complete_file(self, *, path: Path, error: str | None, outputs: Sequence[Path]) -> None:
        if error is not None:
            typer.echo(f"{path} FAILED: {error}")

    def finish_job(self, *, succeeded: int, failed: int, elapsed_seconds: float) -> None:
        typer.echo(
            SUMMARY_TEMPLATE.format(
                succeeded=succeeded,
                failed=failed,
                elapsed=format_elapsed(elapsed_seconds),
            )
        )


class PlainDenoiseJobDisplay(_PlainJobDisplayBase):
    def __init__(self, *, config: DenoiseJobConfig) -> None:
        document_mode = config.mode == "document"
        header = (
            f"{_mode_title(config.mode)} denoise · {config.strength} · "
            f"{_image_count_label(config.file_count)} · "
            f"output {format_output_location(config.output_dir)}"
        )
        labels = {
            PhaseKind.IMAGE: _phase_label(PhaseKind.IMAGE, document_mode=document_mode),
            PhaseKind.TEXT: _phase_label(PhaseKind.TEXT, document_mode=document_mode),
            PhaseKind.MARKDOWN: _phase_label(PhaseKind.MARKDOWN, document_mode=document_mode),
        }
        super().__init__(job_header=header, phase_labels=labels)


class PlainScaleJobDisplay(_PlainJobDisplayBase):
    def __init__(self, *, config: ScaleJobConfig) -> None:
        header = (
            f"Upscale · {config.model_name} · "
            f"{_image_count_label(config.file_count)} · "
            f"output {format_output_location(config.output_dir)}"
        )
        labels = {PhaseKind.UPSCALE: "Upscale"}
        super().__init__(job_header=header, phase_labels=labels)


def collect_denoise_outputs(result: object) -> list[Path]:
    outputs: list[Path] = []
    output = getattr(result, "output", None)
    if output is not None:
        outputs.append(output)
    text_output = getattr(result, "text_output", None)
    if text_output is not None:
        outputs.append(text_output)
    markdown_output = getattr(result, "markdown_output", None)
    if markdown_output is not None:
        outputs.append(markdown_output)
    return outputs


def is_tty_stdout() -> bool:
    return sys.stdout.isatty()
