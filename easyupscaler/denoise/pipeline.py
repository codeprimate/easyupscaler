import tempfile
import time
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
from easyupscaler.denoise.document_ocr import extract_document_text, is_tesseract_available
from easyupscaler.denoise.document_ocrai import extract_document_markdown_ocrai
from easyupscaler.denoise.downloader import DownloadProgressCallback, ensure_models
from easyupscaler.denoise.memory import MULTIPASS_SPILL_ENABLED
from easyupscaler.denoise.ocrai_downloader import ensure_ocrai_models
from easyupscaler.denoise.ocrai_service import OcraiService, assert_ocrai_vision_available
from easyupscaler.errors import DenoiseDownloadError, DenoiseModelCorruptError, ImageReadError
from easyupscaler.io.images import ImageIO, is_heic_path
from easyupscaler.progress import PhaseEvent, PhaseKind, PhaseStatus

BackendFactory = Callable[[CatalogKey], DenoiseBackend | FBCNNDenoiseBackend]
OcrExtractor = Callable[[np.ndarray], str]
OcrAvailabilityCheck = Callable[[], bool]
OcraiServiceFactory = Callable[[], OcraiService]
WarningCallback = Callable[[str], None]
PhaseCallback = Callable[[PhaseEvent], None]

TESSERACT_MISSING_WARNING = (
    "Warning: Tesseract not found; skipping text extraction. Install tesseract to enable OCR."
)
OCR_FAILED_WARNING_TEMPLATE = "Warning: OCR failed for {filename}: {reason}"
OCRAI_FAILED_WARNING_TEMPLATE = "Warning: --ocrai failed for {filename}: {reason}"


@dataclass
class DenoiseResult:
    path: Path
    output: Path | None
    error: str | None
    pass_description: str | None = None
    text_output: Path | None = None
    markdown_output: Path | None = None


class DenoiseService:
    def __init__(
        self,
        *,
        image_io: ImageIO | None = None,
        backend_factory: BackendFactory | None = None,
        download_models: Callable[..., None] | None = None,
        ocr_extractor: OcrExtractor | None = None,
        ocr_available: OcrAvailabilityCheck | None = None,
        ocrai_service_factory: OcraiServiceFactory | None = None,
        download_ocrai_models: Callable[..., None] | None = None,
    ) -> None:
        self._image_io = image_io or ImageIO()
        self._backend_factory = backend_factory or _default_backend_factory
        self._download_models = download_models or ensure_models
        self._ocr_extractor = ocr_extractor or extract_document_text
        self._ocr_available = ocr_available or is_tesseract_available
        self._ocrai_service_factory = ocrai_service_factory or OcraiService
        self._download_ocrai_models = download_ocrai_models or ensure_ocrai_models

    def run(
        self,
        paths: list[Path],
        mode: DenoiseMode,
        strength: DenoiseStrength,
        on_progress: Callable[[DenoiseResult], None] | None = None,
        on_phase: PhaseCallback | None = None,
        on_download_progress: DownloadProgressCallback | None = None,
        *,
        output_dir: Path | None = None,
        extract_text: bool = True,
        use_ocrai: bool = False,
        on_warning: WarningCallback | None = None,
    ) -> list[DenoiseResult]:
        batch_is_heic = any(is_heic_path(path) for path in paths)
        required_keys = resolve_models(mode, strength, is_heic=batch_is_heic)
        try:
            self._download_models(required_keys, on_download_progress=on_download_progress)
        except (DenoiseDownloadError, DenoiseModelCorruptError) as exc:
            raise ValueError(str(exc)) from None

        ocrai_service: OcraiService | None = None
        if use_ocrai and extract_text and mode == "document":
            try:
                assert_ocrai_vision_available()
                self._download_ocrai_models(on_download_progress=on_download_progress)
            except (DenoiseDownloadError, DenoiseModelCorruptError) as exc:
                raise ValueError(str(exc)) from None
            except ValueError as exc:
                raise ValueError(str(exc)) from None
            ocrai_service = self._ocrai_service_factory()

        results: list[DenoiseResult] = []
        batch_state = {"tesseract_warned": False}

        try:
            for file_index, path in enumerate(paths):
                result = self._process_path(
                    path,
                    mode,
                    strength,
                    file_index=file_index,
                    file_count=len(paths),
                    output_dir=output_dir,
                    extract_text=extract_text,
                    use_ocrai=use_ocrai,
                    ocrai_service=ocrai_service,
                    on_warning=on_warning,
                    on_phase=on_phase,
                    batch_state=batch_state,
                )
                results.append(result)
                if on_progress is not None:
                    on_progress(result)
                self._release_inference_memory()
        finally:
            if ocrai_service is not None:
                ocrai_service.close()

        return results

    def _process_path(
        self,
        path: Path,
        mode: DenoiseMode,
        strength: DenoiseStrength,
        *,
        file_index: int,
        file_count: int,
        output_dir: Path | None = None,
        extract_text: bool = True,
        use_ocrai: bool = False,
        ocrai_service: OcraiService | None = None,
        on_warning: WarningCallback | None = None,
        on_phase: PhaseCallback | None = None,
        batch_state: dict[str, bool] | None = None,
    ) -> DenoiseResult:
        if not path.exists():
            return DenoiseResult(path=path, output=None, error="file not found")
        if not path.is_file():
            return DenoiseResult(path=path, output=None, error="not a file")

        file_is_heic = is_heic_path(path)
        keys = resolve_models(mode, strength, is_heic=file_is_heic)
        pass_description = pass_display_names(keys) if len(keys) > 1 else None

        try:
            _emit_phase(
                on_phase,
                file_index=file_index,
                file_count=file_count,
                path=path,
                phase=PhaseKind.IMAGE,
                status=PhaseStatus.RUNNING,
            )
            image_started_at = time.perf_counter()
            image, was_grayscale = self._image_io.read_preserving_grayscale_info(path)
            processed = self._run_passes(image, mode, strength, file_is_heic)
            if mode == "document":
                from easyupscaler.denoise.document_enhance import enhance_document_contrast

                try:
                    processed = enhance_document_contrast(processed, strength)
                except ValueError as exc:
                    return DenoiseResult(path=path, output=None, error=str(exc))
                output = self._image_io.write_png(
                    processed, path, mode="L", output_dir=output_dir
                )
                _emit_phase(
                    on_phase,
                    file_index=file_index,
                    file_count=file_count,
                    path=path,
                    phase=PhaseKind.IMAGE,
                    status=PhaseStatus.DONE,
                    elapsed_seconds=time.perf_counter() - image_started_at,
                    output_path=output,
                )
                text_output, markdown_output = self._extract_document_outputs(
                    processed,
                    path,
                    file_index=file_index,
                    file_count=file_count,
                    output_dir=output_dir,
                    extract_text=extract_text,
                    use_ocrai=use_ocrai,
                    ocrai_service=ocrai_service,
                    on_warning=on_warning,
                    on_phase=on_phase,
                    batch_state=batch_state,
                )
            else:
                preserve_grayscale = mode == "manga"
                output = self._image_io.write_denoised(
                    processed,
                    path,
                    preserve_grayscale=preserve_grayscale,
                    was_grayscale=was_grayscale,
                    output_dir=output_dir,
                )
                _emit_phase(
                    on_phase,
                    file_index=file_index,
                    file_count=file_count,
                    path=path,
                    phase=PhaseKind.IMAGE,
                    status=PhaseStatus.DONE,
                    elapsed_seconds=time.perf_counter() - image_started_at,
                    detail=pass_description,
                    output_path=output,
                )
                text_output = None
                markdown_output = None
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
                text_output=text_output,
                markdown_output=markdown_output,
            )

    def _extract_document_outputs(
        self,
        processed: np.ndarray,
        path: Path,
        *,
        file_index: int,
        file_count: int,
        output_dir: Path | None,
        extract_text: bool,
        use_ocrai: bool,
        ocrai_service: OcraiService | None,
        on_warning: WarningCallback | None,
        on_phase: PhaseCallback | None,
        batch_state: dict[str, bool] | None,
    ) -> tuple[Path | None, Path | None]:
        if not extract_text:
            return None, None

        gray_uint8 = (np.clip(processed, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        text_output = self._maybe_extract_tesseract_text(
            gray_uint8,
            path,
            file_index=file_index,
            file_count=file_count,
            output_dir=output_dir,
            on_warning=on_warning,
            on_phase=on_phase,
            batch_state=batch_state,
        )
        markdown_output = None
        if use_ocrai:
            markdown_output = self._maybe_extract_ocrai_markdown(
                gray_uint8,
                path,
                file_index=file_index,
                file_count=file_count,
                output_dir=output_dir,
                ocrai_service=ocrai_service,
                on_warning=on_warning,
                on_phase=on_phase,
            )
        return text_output, markdown_output

    def _maybe_extract_tesseract_text(
        self,
        gray_uint8: np.ndarray,
        path: Path,
        *,
        file_index: int,
        file_count: int,
        output_dir: Path | None,
        on_warning: WarningCallback | None,
        on_phase: PhaseCallback | None,
        batch_state: dict[str, bool] | None,
    ) -> Path | None:
        _emit_phase(
            on_phase,
            file_index=file_index,
            file_count=file_count,
            path=path,
            phase=PhaseKind.TEXT,
            status=PhaseStatus.RUNNING,
        )
        text_started_at = time.perf_counter()
        if not self._ocr_available():
            if batch_state is not None and not batch_state.get("tesseract_warned", False):
                if on_warning is not None:
                    on_warning(TESSERACT_MISSING_WARNING)
                if batch_state is not None:
                    batch_state["tesseract_warned"] = True
            _emit_phase(
                on_phase,
                file_index=file_index,
                file_count=file_count,
                path=path,
                phase=PhaseKind.TEXT,
                status=PhaseStatus.SKIPPED,
            )
            return None

        try:
            text = self._ocr_extractor(gray_uint8)
        except Exception as exc:
            if on_warning is not None:
                on_warning(
                    OCR_FAILED_WARNING_TEMPLATE.format(
                        filename=path.name,
                        reason=str(exc),
                    )
                )
            _emit_phase(
                on_phase,
                file_index=file_index,
                file_count=file_count,
                path=path,
                phase=PhaseKind.TEXT,
                status=PhaseStatus.SKIPPED,
                elapsed_seconds=time.perf_counter() - text_started_at,
            )
            return None

        text_output = self._image_io.write_txt(text, path, output_dir=output_dir)
        _emit_phase(
            on_phase,
            file_index=file_index,
            file_count=file_count,
            path=path,
            phase=PhaseKind.TEXT,
            status=PhaseStatus.DONE,
            elapsed_seconds=time.perf_counter() - text_started_at,
            output_path=text_output,
        )
        return text_output

    def _maybe_extract_ocrai_markdown(
        self,
        gray_uint8: np.ndarray,
        path: Path,
        *,
        file_index: int,
        file_count: int,
        output_dir: Path | None,
        ocrai_service: OcraiService | None,
        on_warning: WarningCallback | None,
        on_phase: PhaseCallback | None,
    ) -> Path | None:
        if ocrai_service is None:
            return None

        _emit_phase(
            on_phase,
            file_index=file_index,
            file_count=file_count,
            path=path,
            phase=PhaseKind.MARKDOWN,
            status=PhaseStatus.RUNNING,
        )
        markdown_started_at = time.perf_counter()
        try:
            markdown = extract_document_markdown_ocrai(gray_uint8, ocrai_service)
        except Exception as exc:
            if on_warning is not None:
                on_warning(
                    OCRAI_FAILED_WARNING_TEMPLATE.format(
                        filename=path.name,
                        reason=str(exc),
                    )
                )
            _emit_phase(
                on_phase,
                file_index=file_index,
                file_count=file_count,
                path=path,
                phase=PhaseKind.MARKDOWN,
                status=PhaseStatus.SKIPPED,
                elapsed_seconds=time.perf_counter() - markdown_started_at,
            )
            return None

        markdown_output = self._image_io.write_md(markdown, path, output_dir=output_dir)
        _emit_phase(
            on_phase,
            file_index=file_index,
            file_count=file_count,
            path=path,
            phase=PhaseKind.MARKDOWN,
            status=PhaseStatus.DONE,
            elapsed_seconds=time.perf_counter() - markdown_started_at,
            output_path=markdown_output,
        )
        return markdown_output

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
    if key == "book_compact":
        from easyupscaler.denoise.backends.book_compact_backend import BookCompactBackend

        return BookCompactBackend(weights_path)
    from easyupscaler.denoise.backends.archiver_backend import ArchiverBackend

    return ArchiverBackend(weights_path)


def _emit_phase(
    on_phase: PhaseCallback | None,
    *,
    file_index: int,
    file_count: int,
    path: Path,
    phase: PhaseKind,
    status: PhaseStatus,
    elapsed_seconds: float | None = None,
    detail: str | None = None,
    output_path: Path | None = None,
) -> None:
    if on_phase is None:
        return
    on_phase(
        PhaseEvent(
            file_index=file_index,
            file_count=file_count,
            path=path,
            phase=phase,
            status=status,
            elapsed_seconds=elapsed_seconds,
            detail=detail,
            output_path=output_path,
        )
    )
