import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from easyupscaler.config import paths as paths_module
from easyupscaler.errors import DuplicateModelError, ModelNotFoundError
from easyupscaler.models.registry import ModelEntry, ModelRegistry


def _entry(name: str = "RealESRGAN_x4plus", scale: int = 4) -> ModelEntry:
    return ModelEntry(
        name=name,
        filename=f"{name}.pth",
        path=Path(f"/tmp/models/{name}.pth"),
        scale=scale,
        imported_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC),
    )


def test_empty_registry(isolated_paths, without_torch) -> None:
    registry = ModelRegistry()
    assert registry.list() == []
    assert "torch" not in sys.modules


def test_add_list_get(isolated_paths, without_torch) -> None:
    registry = ModelRegistry()
    entry = _entry()
    registry.add(entry)
    assert registry.get("RealESRGAN_x4plus") == entry
    assert len(registry.list()) == 1
    assert paths_module.REGISTRY_FILE.exists()
    assert "torch" not in sys.modules


def test_duplicate_add_fails(isolated_paths, without_torch) -> None:
    registry = ModelRegistry()
    registry.add(_entry())
    with pytest.raises(DuplicateModelError):
        registry.add(_entry())
    assert "torch" not in sys.modules


def test_remove_returns_entry(isolated_paths, without_torch) -> None:
    registry = ModelRegistry()
    entry = _entry()
    registry.add(entry)
    removed = registry.remove("RealESRGAN_x4plus")
    assert removed == entry
    assert registry.list() == []
    assert "torch" not in sys.modules


def test_replace_overwrites(isolated_paths, without_torch) -> None:
    registry = ModelRegistry()
    registry.add(_entry(scale=4))
    replacement = _entry(scale=2)
    registry.replace(replacement)
    assert registry.get("RealESRGAN_x4plus").scale == 2
    assert "torch" not in sys.modules


def test_get_missing_raises(isolated_paths) -> None:
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.get("missing")
