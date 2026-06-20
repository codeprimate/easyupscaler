from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from easyupscaler.errors import ImageReadError
from easyupscaler.io.images import OUTPUT_SUFFIX, ImageIO


def _write_rgb_jpeg(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    image = Image.new("RGB", size, color=(128, 64, 32))
    image.save(path, format="JPEG")


def _write_rgba_png(path: Path) -> None:
    image = Image.new("RGBA", (8, 8), color=(255, 0, 0, 128))
    image.save(path, format="PNG")


def _write_grayscale_png(path: Path) -> None:
    image = Image.new("L", (8, 8), color=128)
    image.save(path, format="PNG")


def test_read_rgb_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "rgb.jpg"
    _write_rgb_jpeg(source)
    array = ImageIO().read(source)
    assert array.shape == (8, 8, 3)
    assert array.dtype == np.float32
    assert array.max() <= 1.0


def test_read_rgba_png_converts_to_rgb(tmp_path: Path) -> None:
    source = tmp_path / "rgba.png"
    _write_rgba_png(source)
    array = ImageIO().read(source)
    assert array.shape == (8, 8, 3)


def test_read_grayscale_png_converts_to_rgb(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    _write_grayscale_png(source)
    array = ImageIO().read(source)
    assert array.shape == (8, 8, 3)


def test_read_corrupt_raises_image_read_error(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.jpg"
    source.write_bytes(b"not-an-image")
    with pytest.raises(ImageReadError, match="cannot read image"):
        ImageIO().read(source)


def test_write_uses_base_name_when_no_conflict(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    image = ImageIO().read(source)
    output = ImageIO().write(image, source)
    assert output.name == f"photo{OUTPUT_SUFFIX}"
    assert output.exists()

    with Image.open(output) as written:
        assert written.format == "JPEG"
        assert written.mode == "RGB"


def test_write_appends_index_on_conflict(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    image = ImageIO().read(source)
    image_io = ImageIO()

    first = image_io.write(image, source)
    assert first.name == f"photo{OUTPUT_SUFFIX}"

    second = image_io.write(image, source)
    assert second.name == "photo-upscaled-0001.jpg"
    assert second.exists()
    assert first.exists()

    third = image_io.write(image, source)
    assert third.name == "photo-upscaled-0002.jpg"
    assert third.exists()


def test_write_fills_lowest_available_index(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    image = ImageIO().read(source)
    image_io = ImageIO()

    image_io.write(image, source)
    (tmp_path / "photo-upscaled-0002.jpg").write_bytes(b"placeholder")

    output = image_io.write(image, source)
    assert output.name == "photo-upscaled-0001.jpg"


def test_write_uses_custom_output_dir(tmp_path: Path) -> None:
    source = tmp_path / "inputs" / "photo.jpg"
    source.parent.mkdir()
    _write_rgb_jpeg(source)
    output_dir = tmp_path / "results"
    image = ImageIO().read(source)

    output = ImageIO().write(image, source, output_dir=output_dir)

    assert output == output_dir / f"photo{OUTPUT_SUFFIX}"
    assert output.exists()
    assert not (source.parent / f"photo{OUTPUT_SUFFIX}").exists()


def test_write_conflict_indexing_uses_custom_output_dir(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    output_dir = tmp_path / "results"
    image = ImageIO().read(source)
    image_io = ImageIO()

    (output_dir / f"photo{OUTPUT_SUFFIX}").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / f"photo{OUTPUT_SUFFIX}").write_bytes(b"existing")

    output = image_io.write(image, source, output_dir=output_dir)

    assert output.name == "photo-upscaled-0001.jpg"
    assert output.parent == output_dir
    assert not (source.parent / "photo-upscaled-0001.jpg").exists()


def test_write_png_uses_denoised_suffix(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    image = ImageIO().read(source)
    output = ImageIO().write_png(image, source)
    assert output.name == "photo-denoised.png"
    assert output.exists()


def test_write_png_conflict_index(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    _write_rgb_jpeg(source)
    image = ImageIO().read(source)
    image_io = ImageIO()
    first = image_io.write_png(image, source)
    second = image_io.write_png(image, source)
    assert first.name == "photo-denoised.png"
    assert second.name == "photo-denoised-0001.png"


def test_write_denoised_grayscale_png(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    _write_grayscale_png(source)
    array, was_grayscale = ImageIO().read_preserving_grayscale_info(source)
    assert was_grayscale is True
    output = ImageIO().write_denoised(
        array,
        source,
        preserve_grayscale=True,
        was_grayscale=True,
    )
    assert output.name == "page-denoised.png"
    with Image.open(output) as written:
        assert written.mode == "L"


def test_is_heic_path() -> None:
    from easyupscaler.io.images import is_heic_path

    assert is_heic_path(Path("photo.heic")) is True
    assert is_heic_path(Path("photo.HEIF")) is True
    assert is_heic_path(Path("photo.jpg")) is False
