from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from easyupscaler.errors import ImageReadError

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
OUTPUT_SUFFIX = "-upscaled.jpg"
OUTPUT_INDEX_WIDTH = 4
OUTPUT_INDEX_START = 1
OUTPUT_INDEX_MAX = 9999


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

    def write(self, image: np.ndarray, source_path: Path) -> Path:
        output_path = self._resolve_output_path(source_path)
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

    def _resolve_output_path(self, source_path: Path) -> Path:
        parent = source_path.parent
        stem = source_path.stem
        base_path = parent / f"{stem}{OUTPUT_SUFFIX}"
        if not base_path.exists():
            return base_path

        for index in range(OUTPUT_INDEX_START, OUTPUT_INDEX_MAX + 1):
            indexed_name = f"{stem}-upscaled-{index:0{OUTPUT_INDEX_WIDTH}d}.jpg"
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
