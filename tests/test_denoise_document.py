import numpy as np
import pytest

from easyupscaler.denoise.document_constants import (
    DOCUMENT_ANTIALIAS_SIGMA_HIGH,
    DOCUMENT_ANTIALIAS_SIGMA_LOW,
    DOCUMENT_EDGE_BAND_WIDTH,
    DOCUMENT_FLAT_INK_SNAP,
    DOCUMENT_FLAT_PAPER_SNAP,
    DOCUMENT_MORPH_STRUCTURE_SIZE,
    DOCUMENT_SAUVOLA_K,
    DOCUMENT_SAUVOLA_WINDOW_HIGH,
    DOCUMENT_SAUVOLA_WINDOW_LOW,
)
from easyupscaler.denoise.document_enhance import (
    _antialias_edges_only,
    _snap_flat_regions,
    enhance_document_contrast,
)


def test_document_constants_match_spec() -> None:
    assert DOCUMENT_SAUVOLA_WINDOW_LOW == 75
    assert DOCUMENT_SAUVOLA_WINDOW_HIGH == 25
    assert DOCUMENT_SAUVOLA_K == 0.2
    assert DOCUMENT_ANTIALIAS_SIGMA_LOW == 1.5
    assert DOCUMENT_ANTIALIAS_SIGMA_HIGH == 0.75
    assert DOCUMENT_MORPH_STRUCTURE_SIZE == 3
    assert DOCUMENT_EDGE_BAND_WIDTH == 2.0
    assert DOCUMENT_FLAT_INK_SNAP == 0.25
    assert DOCUMENT_FLAT_PAPER_SNAP == 0.75


def _stripe_rgb(height: int = 100, width: int = 100) -> np.ndarray:
    image = np.ones((height, width, 3), dtype=np.float32)
    image[:, : width // 2] = 0.0
    return image


def test_enhance_document_contrast_output_shape_and_range() -> None:
    image = _stripe_rgb()
    result = enhance_document_contrast(image, "low")
    assert result.shape == (100, 100)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_enhance_document_contrast_text_and_background() -> None:
    image = _stripe_rgb()
    result = enhance_document_contrast(image, "low")
    left_mean = result[:, :50].mean()
    right_mean = result[:, 50:].mean()
    assert left_mean < 0.3
    assert right_mean > 0.7


def test_antialiased_edges_are_not_hard_binary() -> None:
    image = _stripe_rgb()
    result = enhance_document_contrast(image, "low")
    unique_count = len(np.unique((result * 255).round().astype(np.uint8)))
    assert unique_count > 2


def test_edge_antialias_leaves_flat_regions_pure() -> None:
    binary = np.zeros((50, 50), dtype=np.float32)
    binary[:, 25:] = 1.0
    result = _antialias_edges_only(binary, sigma=1.5)
    assert result[5, 40] == 1.0
    assert result[5, 10] == 0.0
    edge_values = result[20:30, 23:27]
    assert edge_values.min() < 1.0
    assert edge_values.max() > 0.0


def _faint_document_rgb(height: int = 100, width: int = 100) -> np.ndarray:
    image = np.ones((height, width, 3), dtype=np.float32) * 0.9
    image[35:65, 25:75] = 0.55
    return image


def test_high_strength_more_extreme_than_low() -> None:
    image = _faint_document_rgb()
    low = enhance_document_contrast(image, "low")
    high = enhance_document_contrast(image, "high")
    low_extremeness = np.abs(low - 0.5).mean()
    high_extremeness = np.abs(high - 0.5).mean()
    assert high_extremeness > low_extremeness


@pytest.mark.parametrize(
    ("strength", "window"),
    [
        ("low", DOCUMENT_SAUVOLA_WINDOW_LOW),
        ("high", DOCUMENT_SAUVOLA_WINDOW_HIGH),
    ],
)
def test_image_too_small_raises_value_error(
    strength: str,
    window: int,
) -> None:
    image = np.ones((window - 1, window - 1, 3), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=f"image too small for document mode \\(minimum dimension: {window}px\\)",
    ):
        enhance_document_contrast(image, strength)  # type: ignore[arg-type]


def test_enhance_uses_strength_specific_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    def fake_threshold_sauvola(gray, *, window_size, k):
        captured["window_size"] = window_size
        return np.full(gray.shape, 128.0, dtype=np.float32)

    monkeypatch.setattr(
        "skimage.filters.threshold_sauvola",
        fake_threshold_sauvola,
    )
    image = _stripe_rgb()
    enhance_document_contrast(image, "low")
    assert captured["window_size"] == DOCUMENT_SAUVOLA_WINDOW_LOW

    enhance_document_contrast(image, "high")
    assert captured["window_size"] == DOCUMENT_SAUVOLA_WINDOW_HIGH


def test_enhance_uses_strength_specific_antialias_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def fake_gaussian_filter(array, *, sigma):
        captured["sigma"] = sigma
        return array

    def fake_distance_transform_edt(mask):
        return np.ones(mask.shape, dtype=np.float32)

    monkeypatch.setattr(
        "scipy.ndimage.gaussian_filter",
        fake_gaussian_filter,
    )
    monkeypatch.setattr(
        "scipy.ndimage.distance_transform_edt",
        fake_distance_transform_edt,
    )
    image = _stripe_rgb()
    enhance_document_contrast(image, "low")
    assert captured["sigma"] == DOCUMENT_ANTIALIAS_SIGMA_LOW

    enhance_document_contrast(image, "high")
    assert captured["sigma"] == DOCUMENT_ANTIALIAS_SIGMA_HIGH


def test_snap_flat_regions_removes_mid_gray_speckle_in_background() -> None:
    smoothed = np.ones((20, 20), dtype=np.float32)
    smoothed[5:10, 5:10] = 0.55
    result = _snap_flat_regions(smoothed)
    assert result[0, 0] == 1.0
    assert result[5, 5] == 0.55


def test_snap_flat_regions_preserves_ink() -> None:
    smoothed = np.zeros((20, 20), dtype=np.float32)
    smoothed[10, 10] = 0.1
    result = _snap_flat_regions(smoothed)
    assert result[10, 10] == 0.0
