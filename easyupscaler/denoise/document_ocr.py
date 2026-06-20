import shutil

import numpy as np
from PIL import Image

TESSERACT_LANG = "eng"
TESSERACT_BINARY = "tesseract"


def is_tesseract_available() -> bool:
    if shutil.which(TESSERACT_BINARY) is None:
        return False

    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except (OSError, RuntimeError):
        return False
    else:
        return True


def extract_document_text(gray_uint8: np.ndarray) -> str:
    import pytesseract

    if gray_uint8.ndim != 2:
        msg = f"expected 2D grayscale array, got shape {gray_uint8.shape}"
        raise ValueError(msg)

    image = Image.fromarray(gray_uint8, mode="L")
    return pytesseract.image_to_string(image, lang=TESSERACT_LANG)
