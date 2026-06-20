import pytest

from easyupscaler.denoise.catalog import DENOISE_MODEL_CATALOG, resolve_models


def test_photo_low_non_heic() -> None:
    assert resolve_models("photo", "low", is_heic=False) == ["scunet_psnr"]


def test_photo_high_non_heic() -> None:
    assert resolve_models("photo", "high", is_heic=False) == ["scunet_gan"]


def test_photo_low_heic_two_pass() -> None:
    assert resolve_models("photo", "low", is_heic=True) == ["scunet_psnr", "fbcnn_color"]


def test_photo_high_heic_two_pass() -> None:
    assert resolve_models("photo", "high", is_heic=True) == ["scunet_gan", "fbcnn_color"]


def test_art_low() -> None:
    assert resolve_models("art", "low", is_heic=False) == ["dejpg_art"]


def test_art_high() -> None:
    assert resolve_models("art", "high", is_heic=False) == ["archivist_medium"]


def test_manga_matches_art() -> None:
    assert resolve_models("manga", "low", is_heic=True) == ["dejpg_art"]
    assert resolve_models("manga", "high", is_heic=True) == ["archivist_medium"]


def test_book_compact_catalog_entry() -> None:
    entry = DENOISE_MODEL_CATALOG["book_compact"]
    assert entry.filename == "1xBook-Compact.safetensors"
    assert "1xBook-Compact.safetensors" in entry.url


@pytest.mark.parametrize("strength", ["low", "high"])
@pytest.mark.parametrize("is_heic", [False, True])
def test_document_mode_always_single_ai_pass(strength: str, is_heic: bool) -> None:
    assert resolve_models("document", strength, is_heic=is_heic) == ["archivist_medium"]  # type: ignore[arg-type]
