from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from easyupscaler.config.settings import ConfigService
from easyupscaler.errors import ImageReadError, ModelNotFoundError
from easyupscaler.io.images import ImageIO
from easyupscaler.models.registry import ModelRegistry
from easyupscaler.upscaling.backends.base import UpscalerBackend


def _default_backend_factory(path: Path) -> UpscalerBackend:
    from easyupscaler.upscaling.backends.spandrel_backend import SpandrelBackend

    return SpandrelBackend(path)


@dataclass
class UpscaleResult:
    path: Path
    output: Path | None
    error: str | None


class UpscaleService:
    def __init__(
        self,
        *,
        config_service: ConfigService | None = None,
        registry: ModelRegistry | None = None,
        image_io: ImageIO | None = None,
        backend_factory: Callable[[Path], UpscalerBackend] | None = None,
    ) -> None:
        self._config = config_service or ConfigService()
        self._registry = registry or ModelRegistry()
        self._image_io = image_io or ImageIO()
        self._backend_factory = backend_factory or _default_backend_factory

    def run(
        self,
        paths: list[Path],
        model_name: str | None,
        on_progress: Callable[[UpscaleResult], None] | None = None,
        *,
        output_dir: Path | None = None,
    ) -> list[UpscaleResult]:
        resolved_name = model_name or self._config.get_default_model()
        if resolved_name is None:
            msg = "Error: no default model set. Run 'easyupscaler models default <name>'."
            raise ValueError(msg)

        try:
            entry = self._registry.get(resolved_name)
        except ModelNotFoundError:
            installed = ", ".join(item.name for item in self._registry.list()) or "(none)"
            msg = f"Error: model '{resolved_name}' not found. Installed: {installed}"
            raise ValueError(msg) from None

        backend = self._backend_factory(entry.path)
        results: list[UpscaleResult] = []

        for path in paths:
            result = self._process_path(path, backend, output_dir=output_dir)
            results.append(result)
            if on_progress is not None:
                on_progress(result)
            self._maybe_empty_mps_cache()

        return results

    def _process_path(
        self,
        path: Path,
        backend: UpscalerBackend,
        *,
        output_dir: Path | None = None,
    ) -> UpscaleResult:
        if not path.exists():
            return UpscaleResult(path=path, output=None, error="file not found")
        if not path.is_file():
            return UpscaleResult(path=path, output=None, error="not a file")

        try:
            image = self._image_io.read(path)
            upscaled = backend.upscale(image)
            output = self._image_io.write(upscaled, path, output_dir=output_dir)
        except ImageReadError as exc:
            return UpscaleResult(path=path, output=None, error=str(exc))
        except OSError as exc:
            return UpscaleResult(path=path, output=None, error=str(exc))
        except RuntimeError as exc:
            return UpscaleResult(path=path, output=None, error=str(exc))
        else:
            return UpscaleResult(path=path, output=output, error=None)

    def _maybe_empty_mps_cache(self) -> None:
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            return
