from io import StringIO
from pathlib import Path

from rich.console import Console

from easyupscaler.cli.job_progress import (
    ICON_DONE,
    ICON_PENDING,
    ICON_RUNNING,
    DenoiseJobConfig,
    PlainDenoiseJobDisplay,
    TtyDenoiseJobDisplay,
    _format_checklist_path,
    _PhaseState,
    _render_checklist_line,
    _render_file_checklist,
    format_elapsed,
    format_elapsed_seconds,
)
from easyupscaler.progress import PhaseEvent, PhaseKind, PhaseStatus

TEST_CONSOLE = Console(width=120, force_terminal=True)


def test_format_checklist_path_truncates_long_paths() -> None:
    console = Console(width=60, force_terminal=True)
    path = Path("/Users/someone/Desktop/very/long/directory/document-IMG_8202-denoised-0004.png")
    rendered = _format_checklist_path(path, console=console)
    full = str(path)
    assert "…" in rendered
    assert len(rendered) < len(full)


def test_format_elapsed_minutes_seconds() -> None:
    assert format_elapsed(36.0) == "0:36"
    assert format_elapsed(3661.0) == "1:01:01"


def test_format_elapsed_seconds_under_one_minute() -> None:
    assert format_elapsed_seconds(2.1) == "2.1s"
    assert format_elapsed_seconds(65.0) == "1:05"


def test_render_checklist_line_pending() -> None:
    line = _render_checklist_line(
        PhaseKind.TEXT,
        "Text",
        _PhaseState(status=PhaseStatus.PENDING),
        console=TEST_CONSOLE,
    )
    assert ICON_PENDING in line.plain
    assert "pending" in line.plain


def test_render_checklist_line_done_with_path() -> None:
    line = _render_checklist_line(
        PhaseKind.IMAGE,
        "PNG",
        _PhaseState(
            status=PhaseStatus.DONE,
            elapsed_seconds=6.2,
            output_path=Path("/tmp/out/scan-denoised.png"),
        ),
        console=TEST_CONSOLE,
    )
    plain = line.plain
    assert ICON_DONE in plain
    assert "6.2s" in plain
    assert "scan-denoised.png" in plain


def test_render_checklist_line_running_markdown_shows_elapsed() -> None:
    import time

    line = _render_checklist_line(
        PhaseKind.MARKDOWN,
        "Markdown",
        _PhaseState(status=PhaseStatus.RUNNING, running_started_at=time.perf_counter()),
        console=TEST_CONSOLE,
    )
    plain = line.plain
    assert ICON_RUNNING in plain
    assert "running (" in plain
    assert "…" not in plain


def test_render_file_checklist_includes_all_phases() -> None:
    phases = {
        PhaseKind.IMAGE: _PhaseState(status=PhaseStatus.DONE, elapsed_seconds=2.0),
        PhaseKind.TEXT: _PhaseState(status=PhaseStatus.PENDING),
        PhaseKind.MARKDOWN: _PhaseState(status=PhaseStatus.PENDING),
    }
    panel = _render_file_checklist(
        file_index=0,
        file_count=1,
        path=Path("scan.heic"),
        phase_plan=(PhaseKind.IMAGE, PhaseKind.TEXT, PhaseKind.MARKDOWN),
        phases=phases,
        document_mode=True,
        console=TEST_CONSOLE,
    )
    buffer = StringIO()
    Console(file=buffer, force_terminal=True, width=120).print(panel)
    output = buffer.getvalue()
    assert "[1/1] scan.heic" in output
    assert "PNG" in output
    assert "Text" in output
    assert "Markdown" in output


def test_plain_denoise_document_phase_done(capsys) -> None:
    display = PlainDenoiseJobDisplay(
        config=DenoiseJobConfig(
            mode="document",
            strength="low",
            file_count=1,
            output_dir=Path("/tmp/out"),
            extract_text=True,
            use_ocrai=True,
        )
    )
    display.begin_job()
    path = Path("scan.heic")
    display.handle_phase(
        PhaseEvent(
            file_index=0,
            file_count=1,
            path=path,
            phase=PhaseKind.IMAGE,
            status=PhaseStatus.RUNNING,
        )
    )
    display.handle_phase(
        PhaseEvent(
            file_index=0,
            file_count=1,
            path=path,
            phase=PhaseKind.IMAGE,
            status=PhaseStatus.DONE,
            elapsed_seconds=6.2,
            output_path=Path("/tmp/out/scan-denoised.png"),
        )
    )
    display.finish_job(succeeded=1, failed=0, elapsed_seconds=36.0)
    captured = capsys.readouterr()
    assert "Document denoise · low · 1 image" in captured.out
    assert "scan.heic: PNG..." in captured.out
    assert "scan.heic: PNG →" in captured.out
    assert "(6.2s)" in captured.out
    assert "Done · 1 succeeded · 0 failed · 0:36" in captured.out


def test_tty_checklist_updates_after_each_phase() -> None:
    display = TtyDenoiseJobDisplay(
        config=DenoiseJobConfig(
            mode="document",
            strength="low",
            file_count=1,
            output_dir=Path("/tmp/out"),
            extract_text=True,
            use_ocrai=True,
        )
    )
    buffer = StringIO()
    display._console = Console(file=buffer, force_terminal=True, width=120)
    path = Path("scan.heic")
    display.handle_phase(PhaseEvent(0, 1, path, PhaseKind.IMAGE, PhaseStatus.RUNNING))
    display.handle_phase(
        PhaseEvent(
            0,
            1,
            path,
            PhaseKind.IMAGE,
            PhaseStatus.DONE,
            elapsed_seconds=6.2,
            output_path=Path("/tmp/out/scan-denoised.png"),
        ),
    )
    display.handle_phase(
        PhaseEvent(
            0,
            1,
            path,
            PhaseKind.TEXT,
            PhaseStatus.DONE,
            elapsed_seconds=1.6,
            output_path=Path("/tmp/out/scan.txt"),
        ),
    )
    after_text = buffer.getvalue()
    assert "scan.txt" in after_text
    assert "1.6s" in after_text

    display.handle_phase(PhaseEvent(0, 1, path, PhaseKind.MARKDOWN, PhaseStatus.RUNNING))
    after_markdown_running = buffer.getvalue()
    assert "running (" in after_markdown_running
    assert "scan.txt" in after_markdown_running


def test_tty_denoise_checklist_shows_done_paths_on_complete() -> None:
    display = TtyDenoiseJobDisplay(
        config=DenoiseJobConfig(
            mode="document",
            strength="low",
            file_count=1,
            output_dir=Path("/tmp/out"),
            extract_text=True,
            use_ocrai=True,
        )
    )
    buffer = StringIO()
    display._console = Console(file=buffer, force_terminal=True, width=120)
    path = Path("scan.heic")
    display.begin_job()
    display.handle_phase(
        PhaseEvent(0, 1, path, PhaseKind.IMAGE, PhaseStatus.RUNNING),
    )
    display.handle_phase(
        PhaseEvent(
            0,
            1,
            path,
            PhaseKind.IMAGE,
            PhaseStatus.DONE,
            elapsed_seconds=6.2,
            output_path=Path("/tmp/out/scan-denoised.png"),
        ),
    )
    display.handle_phase(
        PhaseEvent(
            0,
            1,
            path,
            PhaseKind.TEXT,
            PhaseStatus.DONE,
            elapsed_seconds=1.6,
            output_path=Path("/tmp/out/scan.txt"),
        ),
    )
    display.handle_phase(
        PhaseEvent(0, 1, path, PhaseKind.MARKDOWN, PhaseStatus.RUNNING),
    )
    display.handle_phase(
        PhaseEvent(
            0,
            1,
            path,
            PhaseKind.MARKDOWN,
            PhaseStatus.DONE,
            elapsed_seconds=28.3,
            output_path=Path("/tmp/out/scan.md"),
        ),
    )
    display.complete_file(path=path, error=None, outputs=[])
    display._stop_refresh_loop()
    output = buffer.getvalue()
    assert "6.2s" in output
    assert "1.6s" in output
    assert "28.3s" in output
    assert "scan-denoised.png" in output
    assert "scan.txt" in output
    assert "scan.md" in output
