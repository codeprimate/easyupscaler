from easyupscaler.denoise.catalog import resolve_models


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
