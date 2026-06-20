from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class PhaseKind(StrEnum):
    IMAGE = "image"
    TEXT = "text"
    MARKDOWN = "markdown"
    UPSCALE = "upscale"


class PhaseStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PhaseEvent:
    file_index: int
    file_count: int
    path: Path
    phase: PhaseKind
    status: PhaseStatus
    elapsed_seconds: float | None = None
    detail: str | None = None
    output_path: Path | None = None
