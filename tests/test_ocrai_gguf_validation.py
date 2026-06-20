from easyupscaler.denoise.ocrai_downloader import _is_valid_gguf


def test_is_valid_gguf_rejects_non_gguf(tmp_path) -> None:
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"xxxx" + b"x" * 2048)
    assert _is_valid_gguf(path) is False


def test_is_valid_gguf_accepts_gguf_header(tmp_path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 2048)
    assert _is_valid_gguf(path) is True


def test_is_valid_gguf_rejects_too_small(tmp_path) -> None:
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF")
    assert _is_valid_gguf(path) is False
