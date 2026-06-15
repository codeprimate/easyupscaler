from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from easyupscaler.config import paths
from easyupscaler.errors import DuplicateModelError, ModelNotFoundError


@dataclass
class ModelEntry:
    name: str
    filename: str
    path: Path
    scale: int
    imported_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "filename": self.filename,
            "path": str(self.path),
            "scale": self.scale,
            "imported_at": self.imported_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelEntry:
        imported_raw = str(data["imported_at"])
        if imported_raw.endswith("Z"):
            imported_raw = imported_raw[:-1] + "+00:00"
        imported_at = datetime.fromisoformat(imported_raw)
        scale_raw = data["scale"]
        if not isinstance(scale_raw, int):
            scale_raw = int(str(scale_raw))
        return cls(
            name=str(data["name"]),
            filename=str(data["filename"]),
            path=Path(str(data["path"])),
            scale=scale_raw,
            imported_at=imported_at,
        )


class ModelRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        self._registry_path = registry_path

    @property
    def registry_path(self) -> Path:
        return self._registry_path or paths.REGISTRY_FILE

    def list(self) -> builtins.list[ModelEntry]:
        return sorted(self._read_entries(), key=lambda entry: entry.name)

    def get(self, name: str) -> ModelEntry:
        for entry in self._read_entries():
            if entry.name == name:
                return entry
        raise ModelNotFoundError(name)

    def add(self, entry: ModelEntry) -> None:
        entries = self._read_entries()
        if any(existing.name == entry.name for existing in entries):
            raise DuplicateModelError(entry.name)
        entries.append(entry)
        self._write_entries(entries)

    def remove(self, name: str) -> ModelEntry:
        entries = self._read_entries()
        for index, entry in enumerate(entries):
            if entry.name == name:
                entries.pop(index)
                self._write_entries(entries)
                return entry
        raise ModelNotFoundError(name)

    def replace(self, entry: ModelEntry) -> None:
        entries = self._read_entries()
        replaced = False
        for index, existing in enumerate(entries):
            if existing.name == entry.name:
                entries[index] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        self._write_entries(entries)

    def _read_entries(self) -> builtins.list[ModelEntry]:
        if not self.registry_path.exists():
            return []
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            msg = f"invalid registry format: {self.registry_path}"
            raise ValueError(msg)
        return [ModelEntry.from_dict(item) for item in raw]

    def _write_entries(self, entries: builtins.list[ModelEntry]) -> None:
        paths.ensure_data_dir()
        payload = [entry.to_dict() for entry in entries]
        temp_path = self.registry_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.registry_path)
