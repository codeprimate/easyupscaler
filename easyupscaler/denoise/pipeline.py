import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from easyupscaler.denoise.backends.base import DenoiseBackend, FBCNNDenoiseBackend
from easyupscaler.denoise.backends.fbcnn_backend import FBCNN_HIGH_STRENGTH_QF
from easyupscaler.denoise.catalog import (
    CatalogKey,
    DenoiseMode,
    DenoiseStrength,
    pass_display_names,
    path_for,
    resolve_models,
)
from easyupscaler.denoise.downloader import DownloadProgressCallback, ensure_models
from easyupscaler.denoise.memory import MULTIPASS_SPILL_ENABLED
from easyupscaler.errors import DenoiseDownloadError, DenoiseModelCorruptError, ImageReadError
from easyupscaler.io.images import ImageIO, is_heic_path

BackendFactory = Callable[[CatalogKey], DenoiseBackend | FBCNNDenoiseBackend]


@dataclass
class DenoiseResult:
    path: Path
    output: Path | None
    error: str | None
    pass_description: str | None = None


class DenoiseService:
    def __init__(
        self,
        *,
        image_io: ImageIO | None = None,
        backend_factory: BackendFactory | None = None,
        download_models: Callable[..., None] | None = None,
    ) -> None:
        self._image_io = image_io or ImageIO()
        self._backend_factory = backend_factory or _default_backend_factory
        self._download_models = download_models or ensure_models

    def run(
        self,
        paths: list[Path],
        mode: DenoiseMode,
        strength: DenoiseStrength,
        on_progress: Callable[[DenoiseResult], None] | None = None,
        on_download_progress: DownloadProgressCallback | None = None,
        *,
        output_dir: Path | None = None,
    ) -> list[DenoiseResult]:
        batch_is_heic = any(is_heic_path(path) for path in paths)
        required_keys = resolve_models(mode, strength, is_heic=batch_is_heic)
        try:
            self._download_models(required_keys, on_download_progress=on_download_progress)
        except (DenoiseDownloadError, DenoiseModelCorruptError) as exc:
            raise ValueError(str(exc)) from None

        results: list[DenoiseResult] = []

        for path in paths:
            result = self._process_path(path, mode, strength, output_dir=output_dir)
            results.append(result)
            if on_progress is not None:
                on_progress(result)
            self._release_inference_memory()

        return results

    def _process_path(
        self,
        path: Path,
        mode: DenoiseMode,
        strength: DenoiseStrength,
        *,
        output_dir: Path | None = None,
    ) -> DenoiseResult:
        if not path.exists():
            return DenoiseResult(path=path, output=None, error="file not found")
        if not path.is_file():
            return DenoiseResult(path=path, output=None, error="not a file")

        file_is_heic = is_heic_path(path)
        keys = resolve_models(mode, strength, is_heic=file_is_heic)
        pass_description = pass_display_names(keys) if len(keys) > 1 else None
        preserve_grayscale = mode == "manga"

        try:
            image, was_grayscale = self._image_io.read_preserving_grayscale_info(path)
            processed = self._run_passes(image, mode, strength, file_is_heic)
            output = self._image_io.write_denoised(
                processed,
                path,
                preserve_grayscale=preserve_grayscale,
                was_grayscale=was_grayscale,
                output_dir=output_dir,
            )
        except ImageReadError as exc:
            return DenoiseResult(path=path, output=None, error=str(exc))
        except OSError as exc:
            return DenoiseResult(path=path, output=None, error=str(exc))
        except RuntimeError as exc:
            return DenoiseResult(path=path, output=None, error=str(exc))
        else:
            return DenoiseResult(
                path=path,
                output=output,
                error=None,
                pass_description=pass_description,
            )

    def _run_passes(
        self,
        image: np.ndarray,
        mode: DenoiseMode,
        strength: DenoiseStrength,
        file_is_heic: bool,
    ) -> np.ndarray:
        keys = resolve_models(mode, strength, is_heic=file_is_heic)
        use_spill = MULTIPASS_SPILL_ENABLED and len(keys) > 1
        temp_paths: list[Path] = []
        current = image
        del image

        try:
            for index, key in enumerate(keys):
                is_last = index == len(keys) - 1
                backend = self._backend_factory(key)
                try:
                    if key == "fbcnn_color":
                        qf_override = FBCNN_HIGH_STRENGTH_QF if strength == "high" else None
                        current = backend.denoise(current, qf_override=qf_override)  # type: ignore[call-arg]
                    else:
                        current = backend.denoise(current)
                finally:
                    del backend
                    self._release_inference_memory()

                if use_spill and not is_last:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                    temp_paths.append(temp_path)
                    self._image_io.write_rgb_png_at(current, temp_path)
                    del current
                    self._release_inference_memory()
                    current = self._image_io.read_rgb_array(temp_path)
                    self._release_inference_memory()
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        return current

    def _release_inference_memory(self) -> None:
        import gc

        gc.collect()
        self._maybe_empty_mps_cache()

    def _maybe_empty_mps_cache(self) -> None:
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            return


def _default_backend_factory(key: CatalogKey) -> DenoiseBackend | FBCNNDenoiseBackend:
    weights_path = path_for(key)
    if key == "fbcnn_color":
        from easyupscaler.denoise.backends.fbcnn_backend import FBCNNBackend

        return FBCNNBackend(weights_path)
    if key in {"scunet_psnr", "scunet_gan"}:
        from easyupscaler.denoise.backends.scunet_backend import SCUNetBackend

        return SCUNetBackend(weights_path)
    if key == "dejpg_art":
        from easyupscaler.denoise.backends.dejpg_backend import DeJPGBackend

        return DeJPGBackend(weights_path)
    from easyupscaler.denoise.backends.archiver_backend import ArchiverBackend

    return ArchiverBackend(weights_path)
