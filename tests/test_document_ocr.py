import numpy as np
import pytest

from easyupscaler.denoise.document_ocr import (
    TESSERACT_LANG,
    extract_document_text,
    is_tesseract_available,
)


def test_extract_document_text_calls_pytesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    gray = np.zeros((10, 10), dtype=np.uint8)
    captured: dict[str, object] = {}

    def fake_fromarray(array, mode=None):
        captured["array"] = array
        captured["mode"] = mode
        return "pil-image"

    def fake_image_to_string(image, lang=None):
        captured["image"] = image
        captured["lang"] = lang
        return "extracted text"

    import pytesseract

    monkeypatch.setattr(
        "easyupscaler.denoise.document_ocr.Image.fromarray",
        fake_fromarray,
    )
    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)

    result = extract_document_text(gray)
    assert result == "extracted text"
    assert captured["mode"] == "L"
    assert captured["lang"] == TESSERACT_LANG


def test_extract_document_text_rejects_non_2d_array() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 2D grayscale array"):
        extract_document_text(rgb)


def test_is_tesseract_available_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert is_tesseract_available() is False


def test_is_tesseract_available_true_when_version_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/tesseract")

    import pytesseract

    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.3.0")
    assert is_tesseract_available() is True


def test_is_tesseract_available_false_when_version_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/tesseract")

    import pytesseract

    def raise_runtime():
        raise RuntimeError("broken")

    monkeypatch.setattr(pytesseract, "get_tesseract_version", raise_runtime)
    assert is_tesseract_available() is False
