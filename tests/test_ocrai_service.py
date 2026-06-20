from pathlib import Path

import numpy as np
import pytest

from easyupscaler.denoise.ocrai_service import OcraiService, assert_ocrai_vision_available


def test_assert_ocrai_vision_available_raises_when_handler_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp.llama_chat_format":
            raise ImportError("no vision")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ValueError, match="vision support is unavailable"):
        assert_ocrai_vision_available()


def test_ocrai_service_extract_markdown_uses_injected_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "  hello world  "}}]}

    def llama_factory(**kwargs):
        captured["llama_kwargs"] = kwargs
        return FakeLlama()

    def chat_handler_factory(mmproj_path: str):
        captured["mmproj_path"] = mmproj_path
        return "chat-handler"

    service = OcraiService(
        backbone_path=tmp_path / "backbone.gguf",
        mmproj_path=tmp_path / "mmproj.gguf",
        llama_factory=llama_factory,
        chat_handler_factory=chat_handler_factory,
    )

    monkeypatch.setattr(
        "easyupscaler.denoise.ocrai_service.assert_ocrai_vision_available",
        lambda: None,
    )

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.extract_markdown(rgb)

    assert result == "hello world"
    assert captured["mmproj_path"] == str(tmp_path / "mmproj.gguf")
    assert captured["llama_kwargs"]["verbose"] is False
    messages = captured["kwargs"]["messages"]
    assert messages[0]["content"][1]["text"]
    assert messages[0]["content"][0]["type"] == "image_url"


def test_ocrai_service_rejects_non_rgb_input() -> None:
    service = OcraiService(
        backbone_path=__file__,
        mmproj_path=__file__,
        llama_factory=lambda **kwargs: object(),
        chat_handler_factory=lambda path: object(),
    )
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected RGB uint8 array"):
        service.extract_markdown(gray)


def test_ocrai_service_close_releases_llm_and_chat_handler() -> None:
    class FakeExitStack:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeHandler:
        def __init__(self) -> None:
            self._exit_stack = FakeExitStack()

    class FakeLlama:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    handler = FakeHandler()
    llm = FakeLlama()
    service = OcraiService(
        backbone_path=Path(__file__),
        mmproj_path=Path(__file__),
        llama_factory=lambda **kwargs: llm,
        chat_handler_factory=lambda path: handler,
    )
    service._chat_handler = handler
    service._llm = llm

    service.close()

    assert llm.closed is True
    assert handler._exit_stack.closed is True
    assert service._llm is None
    assert service._chat_handler is None
