from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from easyupscaler.config import paths as paths_module

CatalogKey = Literal[
    "scunet_psnr",
    "scunet_gan",
    "fbcnn_color",
    "dejpg_art",
    "archivist_medium",
    "book_compact",
]

DenoiseMode = Literal["photo", "art", "manga", "document"]
DenoiseStrength = Literal["low", "high"]

SCUNET_PSNR_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
)
SCUNET_GAN_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
)
FBCNN_COLOR_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
DEJPG_ART_URL = (
    "https://github.com/Phhofm/models/releases/download/"
    "1xDeJPG_realplksr_otf/1xDeJPG_realplksr_otf.safetensors"
)
ARCHIVIST_MEDIUM_URL = (
    "https://github.com/Loganavter/Archivist-Project-Denoiser/releases/download/"
    "v1.0/1x-Archivist_Medium.pth"
)
BOOK_COMPACT_URL = (
    "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/"
    "1xBook-Compact.safetensors"
)


@dataclass(frozen=True)
class CatalogEntry:
    key: CatalogKey
    filename: str
    url: str
    display_name: str


DENOISE_MODEL_CATALOG: dict[CatalogKey, CatalogEntry] = {
    "scunet_psnr": CatalogEntry(
        key="scunet_psnr",
        filename="scunet_color_real_psnr.pth",
        url=SCUNET_PSNR_URL,
        display_name="SCUNet PSNR",
    ),
    "scunet_gan": CatalogEntry(
        key="scunet_gan",
        filename="scunet_color_real_gan.pth",
        url=SCUNET_GAN_URL,
        display_name="SCUNet GAN",
    ),
    "fbcnn_color": CatalogEntry(
        key="fbcnn_color",
        filename="fbcnn_color.pth",
        url=FBCNN_COLOR_URL,
        display_name="FBCNN",
    ),
    "dejpg_art": CatalogEntry(
        key="dejpg_art",
        filename="1xDeJPG_realplksr_otf.safetensors",
        url=DEJPG_ART_URL,
        display_name="1xDeJPG",
    ),
    "archivist_medium": CatalogEntry(
        key="archivist_medium",
        filename="1x-Archivist_Medium.pth",
        url=ARCHIVIST_MEDIUM_URL,
        display_name="Archiver Medium",
    ),
    "book_compact": CatalogEntry(
        key="book_compact",
        filename="1xBook-Compact.safetensors",
        url=BOOK_COMPACT_URL,
        display_name="Book Compact",
    ),
}


def path_for(key: CatalogKey) -> Path:
    return paths_module.MODELS_DIR / DENOISE_MODEL_CATALOG[key].filename


def resolve_models(
    mode: DenoiseMode,
    strength: DenoiseStrength,
    *,
    is_heic: bool,
) -> list[CatalogKey]:
    if mode == "photo":
        scunet_key: CatalogKey = "scunet_psnr" if strength == "low" else "scunet_gan"
        if is_heic:
            return [scunet_key, "fbcnn_color"]
        return [scunet_key]

    if mode == "document":
        return ["archivist_medium"]

    if strength == "low":
        return ["dejpg_art"]
    return ["archivist_medium"]


def pass_display_names(keys: list[CatalogKey]) -> str:
    return " + ".join(DENOISE_MODEL_CATALOG[key].display_name for key in keys)
