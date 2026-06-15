from pathlib import Path

import pytest

from easyupscaler.errors import DuplicateModelError, ImportModelError, UnsupportedModelError
from easyupscaler.models.import_model import import_model
from easyupscaler.models.registry import ModelRegistry


class FakeDescriptor:
    def __init__(self, *, purpose: str = "SR", scale: int = 4) -> None:
        self.purpose = purpose
        self.scale = scale


def test_import_success(isolated_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "RealESRGAN_x4plus.pth"
    source.write_bytes(b"weights")

    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(),
    )

    entry = import_model(source)
    assert entry.name == "RealESRGAN_x4plus"
    assert entry.scale == 4
    assert ModelRegistry().get("RealESRGAN_x4plus").filename == "RealESRGAN_x4plus.pth"


def test_import_missing_path(isolated_paths, tmp_path: Path) -> None:
    with pytest.raises(ImportModelError, match="path not found"):
        import_model(tmp_path / "missing.pth")


def test_import_non_sr(isolated_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "inpaint.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(purpose="Inpainting"),
    )
    with pytest.raises(ImportModelError, match="purpose 'Inpainting' is not supported"):
        import_model(source)
    assert ModelRegistry().list() == []


def test_import_restoration_scale_one(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "1xSkinDetail.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(purpose="Restoration", scale=1),
    )
    entry = import_model(source)
    assert entry.name == "1xSkinDetail"
    assert entry.scale == 1


def test_import_scale_one(isolated_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "1xSkinDetail.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(scale=1),
    )
    entry = import_model(source)
    assert entry.name == "1xSkinDetail"
    assert entry.scale == 1
    assert ModelRegistry().get("1xSkinDetail").scale == 1


def test_import_invalid_scale(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invalid.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(scale=0),
    )
    with pytest.raises(ImportModelError, match="invalid scale 0"):
        import_model(source)
    assert ModelRegistry().list() == []


def test_import_duplicate(isolated_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "demo.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(),
    )
    import_model(source)
    with pytest.raises(DuplicateModelError, match="already exists"):
        import_model(source)


def test_import_force_replaces(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    source = tmp_path / "demo.pth"
    source.write_bytes(b"weights")
    monkeypatch.setattr(
        "easyupscaler.models.import_model._load_descriptor",
        lambda path: FakeDescriptor(scale=4),
    )
    import_model(source)
    import_model(source, force=True)
    captured = capsys.readouterr()
    assert "Replaced demo." in captured.out


def test_import_unsupported_architecture(
    isolated_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "unknown.pth"
    source.write_bytes(b"weights")

    def raise_unsupported(path: Path):
        raise RuntimeError("unsupported architecture foo")

    monkeypatch.setattr("easyupscaler.models.import_model._load_descriptor", raise_unsupported)
    with pytest.raises(UnsupportedModelError, match="architecture not recognised"):
        import_model(source)
