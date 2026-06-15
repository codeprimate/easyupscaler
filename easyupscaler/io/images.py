from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from easyupscaler.errors import ImageReadError

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
OUTPUT_SUFFIX = "-upscaled.jpg"


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
        output_path = source_path.parent / f"{source_path.stem}{OUTPUT_SUFFIX}"
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
