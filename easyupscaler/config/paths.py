import os
from pathlib import Path

_MACOS_CONFIG_FALLBACK = Path.home() / ".config"
_MACOS_DATA_FALLBACK = Path.home() / ".local" / "share"


def _xdg_config_home() -> Path:
    if env := os.environ.get("XDG_CONFIG_HOME"):
        return Path(env)
    return _MACOS_CONFIG_FALLBACK


def _xdg_data_home() -> Path:
    if env := os.environ.get("XDG_DATA_HOME"):
        return Path(env)
    return _MACOS_DATA_FALLBACK


CONFIG_DIR = _xdg_config_home() / "easyupscaler"
DATA_DIR = _xdg_data_home() / "easyupscaler"
CONFIG_FILE = CONFIG_DIR / "config.toml"
REGISTRY_FILE = DATA_DIR / "registry.json"
MODELS_DIR = DATA_DIR / "models"


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
