from __future__ import annotations

import numpy as np

from easyupscaler.denoise.catalog import DenoiseStrength
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


def _rgb_float_to_gray_uint8(image: np.ndarray) -> np.ndarray:
    clamped = np.clip(image, 0.0, 1.0)
    gray = (
        0.299 * clamped[:, :, 0]
        + 0.587 * clamped[:, :, 1]
        + 0.114 * clamped[:, :, 2]
    )
    return (gray * 255.0).round().astype(np.uint8)


def _clean_binary_mask(binary: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_closing, binary_opening

    size = DOCUMENT_MORPH_STRUCTURE_SIZE
    structure = np.ones((size, size), dtype=bool)
    mask = binary >= 0.5
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    return mask.astype(np.float32)


def _antialias_edges_only(binary: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    mask = binary >= 0.5
    edge_distance = np.minimum(
        distance_transform_edt(mask),
        distance_transform_edt(~mask),
    )
    edge_band = edge_distance <= DOCUMENT_EDGE_BAND_WIDTH
    smoothed = gaussian_filter(binary, sigma=sigma)
    result = binary.copy()
    result[edge_band] = smoothed[edge_band]
    return result


def _snap_flat_regions(smoothed: np.ndarray) -> np.ndarray:
    result = smoothed.copy()
    result[result <= DOCUMENT_FLAT_INK_SNAP] = 0.0
    result[result >= DOCUMENT_FLAT_PAPER_SNAP] = 1.0
    return result


def enhance_document_contrast(
    image: np.ndarray,
    strength: DenoiseStrength,
) -> np.ndarray:
    from skimage.filters import threshold_sauvola

    window = DOCUMENT_SAUVOLA_WINDOW_LOW if strength == "low" else DOCUMENT_SAUVOLA_WINDOW_HIGH
    sigma = DOCUMENT_ANTIALIAS_SIGMA_LOW if strength == "low" else DOCUMENT_ANTIALIAS_SIGMA_HIGH

    min_dim = min(image.shape[0], image.shape[1])
    if min_dim < window:
        raise ValueError(
            f"image too small for document mode (minimum dimension: {window}px)"
        )

    gray_uint8 = _rgb_float_to_gray_uint8(image)
    thresh = threshold_sauvola(gray_uint8, window_size=window, k=DOCUMENT_SAUVOLA_K)
    binary = (gray_uint8.astype(np.float32) >= thresh).astype(np.float32)
    cleaned = _clean_binary_mask(binary)
    edge_smoothed = _antialias_edges_only(cleaned, sigma)
    return np.clip(_snap_flat_regions(edge_smoothed), 0.0, 1.0).astype(np.float32)
