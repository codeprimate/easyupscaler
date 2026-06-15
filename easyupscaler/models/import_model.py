import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from easyupscaler.config import paths
from easyupscaler.errors import (
    DuplicateModelError,
    ImportModelError,
    UnsupportedModelError,
)
from easyupscaler.models.registry import ModelEntry, ModelRegistry

PTH_PICKLE_WARNING = (
    "Warning: .pth files use Python pickle, which may execute arbitrary code. "
    "Only import models from sources you trust."
)
UNSUPPORTED_ARCHITECTURE_MESSAGE = (
    "Error: architecture not recognised by Spandrel. Try updating easyupscaler, "
    "or check that this is a supported SR model."
)
MIN_MODEL_SCALE = 1
SUPPORTED_MODEL_PURPOSES = frozenset({"SR", "Restoration"})
UNSUPPORTED_PURPOSE_MESSAGE = (
    "Error: model purpose '{purpose}' is not supported. "
    "Only SR and Restoration models are supported."
)


def import_model(path: Path, *, force: bool = False) -> ModelEntry:
    if not path.exists():
        msg = f"Error: path not found: {path}"
        raise ImportModelError(msg)

    if path.suffix.lower() == ".pth":
        print(PTH_PICKLE_WARNING, file=sys.stderr)

    paths.ensure_models_dir()
    destination = paths.MODELS_DIR / path.name
    name = path.stem

    registry = ModelRegistry()
    existing = _find_entry(registry, name)
    if existing is not None and not force:
        msg = f"Error: model '{name}' already exists. Use --force to replace."
        raise DuplicateModelError(msg)

    shutil.copy2(path, destination)
    try:
        descriptor = _load_descriptor(destination)
    except UnsupportedModelError:
        if destination.exists():
            destination.unlink()
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        message = str(exc)
        if _looks_like_unsupported_architecture(message):
            raise UnsupportedModelError(UNSUPPORTED_ARCHITECTURE_MESSAGE) from exc
        raise ImportModelError(message) from exc

    purpose = str(descriptor.purpose)
    if purpose not in SUPPORTED_MODEL_PURPOSES:
        if destination.exists():
            destination.unlink()
        msg = UNSUPPORTED_PURPOSE_MESSAGE.format(purpose=purpose)
        raise ImportModelError(msg)

    scale = int(descriptor.scale)
    if scale < MIN_MODEL_SCALE:
        if destination.exists():
            destination.unlink()
        msg = f"Error: model reports invalid scale {scale}."
        raise ImportModelError(msg)

    entry = ModelEntry(
        name=name,
        filename=path.name,
        path=destination.resolve(),
        scale=scale,
        imported_at=datetime.now(UTC),
    )

    if existing is not None and force:
        registry.replace(entry)
        print(f"Replaced {name}.")
    else:
        registry.add(entry)

    print(f"Imported {name} ({scale}×) from {path.name}")
    return entry


def _find_entry(registry: ModelRegistry, name: str) -> ModelEntry | None:
    from easyupscaler.errors import ModelNotFoundError

    try:
        return registry.get(name)
    except ModelNotFoundError:
        return None


def _load_descriptor(path: Path):
    import spandrel_extra_arches

    spandrel_extra_arches.install()

    import torch
    from spandrel import ModelLoader

    device = torch.device("cpu")
    loader = ModelLoader(device=device)
    try:
        return loader.load_from_file(path)
    except Exception as exc:
        message = str(exc)
        if _looks_like_unsupported_architecture(message):
            raise UnsupportedModelError(UNSUPPORTED_ARCHITECTURE_MESSAGE) from exc
        raise ImportModelError(message) from exc


def _looks_like_unsupported_architecture(message: str) -> bool:
    lowered = message.lower()
    markers = ("unsupported", "unrecognized", "unrecognised", "unknown architecture")
    return any(marker in lowered for marker in markers)
