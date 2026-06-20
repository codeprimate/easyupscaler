from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from easyupscaler.errors import ImageReadError

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
OUTPUT_SUFFIX = "-upscaled.jpg"
DENOISE_OUTPUT_SUFFIX = "-denoised.png"
TEXT_OUTPUT_SUFFIX = ".txt"
TEXT_OUTPUT_EXTENSION = ".txt"
OUTPUT_INDEX_WIDTH = 4
OUTPUT_INDEX_START = 1
OUTPUT_INDEX_MAX = 9999
HEIC_SUFFIXES = {".heic", ".heif"}


def is_heic_path(path: Path) -> bool:
    return path.suffix.lower() in HEIC_SUFFIXES


class ImageIO:
    def read(self, path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                converted = self._normalize_to_rgb(image)
                converted.load()
                array = np.asarray(converted, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise ImageReadError("cannot read image") from exc
        if array.ndim != 3 or array.shape[2] != 3:
            msg = f"unexpected image shape after conversion: {array.shape}"
            raise ImageReadError(msg)
        return array

    def read_preserving_grayscale_info(self, path: Path) -> tuple[np.ndarray, bool]:
        try:
            with Image.open(path) as image:
                was_grayscale = image.mode in {"L", "LA"}
                converted = self._normalize_to_rgb(image)
                converted.load()
                array = np.asarray(converted, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise ImageReadError("cannot read image") from exc
        if array.ndim != 3 or array.shape[2] != 3:
            msg = f"unexpected image shape after conversion: {array.shape}"
            raise ImageReadError(msg)
        return array, was_grayscale

    def read_rgb_array(self, path: Path) -> np.ndarray:
        return self.read(path)

    def write_rgb_png_at(self, image: np.ndarray, path: Path) -> None:
        clamped = np.clip(image, 0.0, 1.0)
        uint8_array = (clamped * 255.0).round().astype(np.uint8)
        pil_image = Image.fromarray(uint8_array, mode="RGB")
        pil_image.save(path, format="PNG")

    def write(
        self,
        image: np.ndarray,
        source_path: Path,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        output_path = self._resolve_output_path(source_path, output_dir=output_dir)
        clamped = np.clip(image, 0.0, 1.0)
        uint8_array = (clamped * 255.0).round().astype(np.uint8)
        pil_image = Image.fromarray(uint8_array, mode="RGB")
        pil_image.save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
        )
        return output_path

    def write_png(
        self,
        image: np.ndarray,
        source_path: Path,
        *,
        mode: str = "RGB",
        output_dir: Path | None = None,
    ) -> Path:
        output_path = self._resolve_denoise_output_path(source_path, output_dir=output_dir)
        clamped = np.clip(image, 0.0, 1.0)
        if mode == "L":
            if image.ndim == 3 and image.shape[2] == 3:
                gray = (
                    0.299 * clamped[:, :, 0]
                    + 0.587 * clamped[:, :, 1]
                    + 0.114 * clamped[:, :, 2]
                )
            else:
                gray = clamped.squeeze()
            uint8_array = (gray * 255.0).round().astype(np.uint8)
            pil_image = Image.fromarray(uint8_array, mode="L")
        else:
            uint8_array = (clamped * 255.0).round().astype(np.uint8)
            pil_image = Image.fromarray(uint8_array, mode="RGB")
        pil_image.save(output_path, format="PNG")
        return output_path

    def write_denoised(
        self,
        image: np.ndarray,
        source_path: Path,
        *,
        preserve_grayscale: bool,
        was_grayscale: bool,
        output_dir: Path | None = None,
    ) -> Path:
        if preserve_grayscale and was_grayscale:
            return self.write_png(
                image,
                source_path,
                mode="L",
                output_dir=output_dir,
            )
        return self.write_png(image, source_path, mode="RGB", output_dir=output_dir)

    def write_txt(
        self,
        text: str,
        source_path: Path,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        output_path = self._resolve_text_output_path(source_path, output_dir=output_dir)
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def _resolve_text_output_path(
        self,
        source_path: Path,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        return self._resolve_indexed_output_path(
            source_path,
            base_suffix=TEXT_OUTPUT_SUFFIX,
            indexed_stem_suffix="",
            extension=TEXT_OUTPUT_EXTENSION,
            output_dir=output_dir,
        )

    def _resolve_denoise_output_path(
        self,
        source_path: Path,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        return self._resolve_indexed_output_path(
            source_path,
            base_suffix=DENOISE_OUTPUT_SUFFIX,
            indexed_stem_suffix="-denoised",
            extension=".png",
            output_dir=output_dir,
        )

    def _resolve_output_path(
        self,
        source_path: Path,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        return self._resolve_indexed_output_path(
            source_path,
            base_suffix=OUTPUT_SUFFIX,
            indexed_stem_suffix="-upscaled",
            extension=".jpg",
            output_dir=output_dir,
        )

    def _resolve_indexed_output_path(
        self,
        source_path: Path,
        *,
        base_suffix: str,
        indexed_stem_suffix: str,
        extension: str,
        output_dir: Path | None = None,
    ) -> Path:
        parent = output_dir if output_dir is not None else source_path.parent
        if output_dir is not None:
            parent.mkdir(parents=True, exist_ok=True)
        stem = source_path.stem
        base_path = parent / f"{stem}{base_suffix}"
        if not base_path.exists():
            return base_path

        for index in range(OUTPUT_INDEX_START, OUTPUT_INDEX_MAX + 1):
            indexed_name = (
                f"{stem}{indexed_stem_suffix}-{index:0{OUTPUT_INDEX_WIDTH}d}{extension}"
            )
            indexed_path = parent / indexed_name
            if not indexed_path.exists():
                return indexed_path

        msg = f"no available output path for {source_path.name}"
        raise OSError(msg)

    def _normalize_to_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        if image.mode in {"L", "LA"}:
            return image.convert("RGB")
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
