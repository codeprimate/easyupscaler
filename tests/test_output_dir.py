from pathlib import Path

import pytest

from easyupscaler.cli.output_dir import (
    OUTPUT_NOT_DIRECTORY_ERROR,
    OUTPUT_NOT_WRITABLE_ERROR,
    prepare_output_dir,
)


def test_prepare_output_dir_creates_missing_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "results"
    assert not output_dir.exists()

    prepared = prepare_output_dir(output_dir)

    assert prepared == output_dir
    assert output_dir.is_dir()


def test_prepare_output_dir_accepts_existing_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    prepared = prepare_output_dir(output_dir)

    assert prepared == output_dir


def test_prepare_output_dir_rejects_existing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "results"
    file_path.write_bytes(b"not-a-directory")

    with pytest.raises(ValueError, match=OUTPUT_NOT_DIRECTORY_ERROR.format(path=file_path)):
        prepare_output_dir(file_path)


def test_prepare_output_dir_rejects_non_writable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    monkeypatch.setattr("easyupscaler.cli.output_dir.os.access", lambda path, mode: False)

    with pytest.raises(ValueError, match=OUTPUT_NOT_WRITABLE_ERROR.format(path=output_dir)):
        prepare_output_dir(output_dir)
