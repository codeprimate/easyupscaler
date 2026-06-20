import math

import numpy as np
from PIL import Image

from easyupscaler.denoise.ocrai_catalog import OCRAI_MAX_PIXELS
from easyupscaler.denoise.ocrai_service import OcraiService

MIN_OCRAI_DIMENSION = 1


def prepare_ocrai_rgb(gray_uint8: np.ndarray, *, max_pixels: int = OCRAI_MAX_PIXELS) -> np.ndarray:
    if gray_uint8.ndim != 2:
        msg = f"expected 2D grayscale array, got shape {gray_uint8.shape}"
        raise ValueError(msg)

    resized = _downscale_to_max_pixels(gray_uint8, max_pixels)
    rgb = np.stack([resized, resized, resized], axis=-1)
    return rgb


def extract_document_markdown_ocrai(gray_uint8: np.ndarray, service: OcraiService) -> str:
    rgb = prepare_ocrai_rgb(gray_uint8)
    return service.extract_markdown(rgb)


def _downscale_to_max_pixels(gray_uint8: np.ndarray, max_pixels: int) -> np.ndarray:
    height, width = gray_uint8.shape
    pixel_count = height * width
    if pixel_count <= max_pixels:
        return gray_uint8

    scale = math.sqrt(max_pixels / pixel_count)
    new_width = max(MIN_OCRAI_DIMENSION, round(width * scale))
    new_height = max(MIN_OCRAI_DIMENSION, round(height * scale))
    image = Image.fromarray(gray_uint8, mode="L")
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
