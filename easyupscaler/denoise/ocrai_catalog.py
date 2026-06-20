from pathlib import Path

from easyupscaler.config.paths import MODELS_DIR

OCRAI_MAX_PIXELS: int = 2_000_000
OCRAI_MAX_TOKENS: int = 4096
OCRAI_TEMPERATURE: float = 0.0
OCRAI_N_CTX: int = 8192
OCRAI_N_GPU_LAYERS: int = -1
OCRAI_VERBOSE: bool = False

OCRAI_BACKBONE_FILENAME = "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
OCRAI_MMPROJ_FILENAME = "Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf"

GGUF_MAGIC = b"GGUF"

HUGGINGFACE_RESOLVE_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"

OCRAI_REPO_FALLBACK_ORDER = (
    "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
    "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
)

OCRAI_VISION_UNAVAILABLE_ERROR = (
    "Error: llama-cpp-python vision support is unavailable. "
    "Reinstall easyupscaler or use a build with Qwen25VLChatHandler support."
)


class OcraiModelSource:
    __slots__ = ("local_filename", "remote_by_repo")

    def __init__(self, local_filename: str, remote_by_repo: dict[str, str]) -> None:
        self.local_filename = local_filename
        self.remote_by_repo = remote_by_repo


OCRAI_MODEL_SOURCES: tuple[OcraiModelSource, ...] = (
    OcraiModelSource(
        OCRAI_BACKBONE_FILENAME,
        {
            "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF": "Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
            "unsloth/Qwen2.5-VL-3B-Instruct-GGUF": "Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        },
    ),
    OcraiModelSource(
        OCRAI_MMPROJ_FILENAME,
        {
            "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
            "unsloth/Qwen2.5-VL-3B-Instruct-GGUF": "mmproj-F16.gguf",
        },
    ),
)


def ocrai_backbone_path() -> Path:
    return MODELS_DIR / OCRAI_BACKBONE_FILENAME


def ocrai_mmproj_path() -> Path:
    return MODELS_DIR / OCRAI_MMPROJ_FILENAME


def build_ocrai_download_url(repo: str, remote_filename: str) -> str:
    return HUGGINGFACE_RESOLVE_BASE.format(repo=repo, filename=remote_filename)
