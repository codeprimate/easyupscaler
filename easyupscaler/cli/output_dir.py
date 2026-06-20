import os
from pathlib import Path

OUTPUT_NOT_DIRECTORY_ERROR = "Error: output path is not a directory: {path}"
OUTPUT_NOT_WRITABLE_ERROR = "Error: output directory is not writable: {path}"


def prepare_output_dir(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        msg = OUTPUT_NOT_DIRECTORY_ERROR.format(path=path)
        raise ValueError(msg)

    path.mkdir(parents=True, exist_ok=True)

    if not os.access(path, os.W_OK):
        msg = OUTPUT_NOT_WRITABLE_ERROR.format(path=path)
        raise ValueError(msg)

    return path
