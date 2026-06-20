import numpy as np
import pytest

from easyupscaler.denoise.document_ocrai import extract_document_markdown_ocrai, prepare_ocrai_rgb
from easyupscaler.denoise.ocrai_catalog import OCRAI_MAX_PIXELS


class FakeOcraiService:
    def __init__(self) -> None:
        self.last_rgb: np.ndarray | None = None

    def extract_markdown(self, rgb_uint8: np.ndarray) -> str:
        self.last_rgb = rgb_uint8
        return "  # Title\n\nbody  "


def test_prepare_ocrai_rgb_keeps_small_image_unchanged() -> None:
    gray = np.full((100, 100), 128, dtype=np.uint8)
    rgb = prepare_ocrai_rgb(gray)
    assert rgb.shape == (100, 100, 3)
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 0] == 128)


def test_prepare_ocrai_rgb_downscales_above_max_pixels() -> None:
    gray = np.zeros((2000, 2000), dtype=np.uint8)
    rgb = prepare_ocrai_rgb(gray, max_pixels=OCRAI_MAX_PIXELS)
    assert rgb.shape[0] * rgb.shape[1] <= OCRAI_MAX_PIXELS
    assert rgb.shape[2] == 3


def test_prepare_ocrai_rgb_rejects_non_2d_array() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 2D grayscale array"):
        prepare_ocrai_rgb(rgb)


def test_extract_document_markdown_ocrai_uses_service() -> None:
    gray = np.zeros((50, 50), dtype=np.uint8)
    service = FakeOcraiService()
    result = extract_document_markdown_ocrai(gray, service)
    assert result == "  # Title\n\nbody  "
    assert service.last_rgb is not None
    assert service.last_rgb.shape == (50, 50, 3)
