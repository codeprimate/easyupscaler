import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from easyupscaler.denoise.ocrai_catalog import (
    OCRAI_MAX_TOKENS,
    OCRAI_N_CTX,
    OCRAI_N_GPU_LAYERS,
    OCRAI_TEMPERATURE,
    OCRAI_VERBOSE,
    OCRAI_VISION_UNAVAILABLE_ERROR,
    ocrai_backbone_path,
    ocrai_mmproj_path,
)
from easyupscaler.denoise.ocrai_prompt import load_ocrai_prompt


def assert_ocrai_vision_available() -> None:
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # noqa: F401
    except ImportError as exc:
        raise ValueError(OCRAI_VISION_UNAVAILABLE_ERROR) from exc


class OcraiService:
    def __init__(
        self,
        *,
        backbone_path: Path | None = None,
        mmproj_path: Path | None = None,
        llama_factory: Callable[..., Any] | None = None,
        chat_handler_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._backbone_path = backbone_path or ocrai_backbone_path()
        self._mmproj_path = mmproj_path or ocrai_mmproj_path()
        self._llama_factory = llama_factory or _default_llama_factory
        self._chat_handler_factory = chat_handler_factory or _default_chat_handler_factory
        self._llm: Any | None = None
        self._chat_handler: Any | None = None

    def close(self) -> None:
        with _suppress_llama_output():
            if self._chat_handler is not None:
                exit_stack = getattr(self._chat_handler, "_exit_stack", None)
                if exit_stack is not None:
                    exit_stack.close()
                self._chat_handler = None
            if self._llm is not None:
                self._llm.close()
                self._llm = None
        _release_llama_runtime()

    def extract_markdown(self, rgb_uint8: np.ndarray) -> str:
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            msg = f"expected RGB uint8 array, got shape {rgb_uint8.shape}"
            raise ValueError(msg)

        with _suppress_llama_output():
            llm = self._ensure_loaded()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                Image.fromarray(rgb_uint8, mode="RGB").save(temp_path, format="PNG")
                image_uri = temp_path.resolve().as_uri()
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_uri}},
                                {"type": "text", "text": load_ocrai_prompt()},
                            ],
                        }
                    ],
                    max_tokens=OCRAI_MAX_TOKENS,
                    temperature=OCRAI_TEMPERATURE,
                )
            finally:
                temp_path.unlink(missing_ok=True)

        content = response["choices"][0]["message"]["content"]
        if content is None:
            return ""
        return str(content).strip()

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm

        assert_ocrai_vision_available()
        chat_handler = self._chat_handler_factory(str(self._mmproj_path))
        self._chat_handler = chat_handler
        self._llm = self._llama_factory(
            model_path=str(self._backbone_path),
            chat_handler=chat_handler,
            n_gpu_layers=OCRAI_N_GPU_LAYERS,
            n_ctx=OCRAI_N_CTX,
            verbose=OCRAI_VERBOSE,
        )
        return self._llm


def _suppress_llama_output():
    if OCRAI_VERBOSE:
        from contextlib import nullcontext

        return nullcontext()
    from llama_cpp._utils import suppress_stdout_stderr

    return suppress_stdout_stderr(disable=False)


def _default_chat_handler_factory(mmproj_path: str) -> Any:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

    return Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=OCRAI_VERBOSE)


def _default_llama_factory(**kwargs: Any) -> Any:
    from llama_cpp import Llama
    from llama_cpp._logger import set_verbose

    set_verbose(kwargs.get("verbose", OCRAI_VERBOSE))
    return Llama(**kwargs)


def _release_llama_runtime() -> None:
    import gc

    gc.collect()
    try:
        import llama_cpp.llama_cpp as llama_cpp
        from llama_cpp import Llama
    except ImportError:
        return

    backend_initialized_attr = "_Llama__backend_initialized"
    if not getattr(Llama, backend_initialized_attr, False):
        return

    with _suppress_llama_output():
        llama_cpp.llama_backend_free()
    setattr(Llama, backend_initialized_attr, False)
