from pathlib import Path

import pytest

from easyupscaler.config import paths as paths_module


@pytest.fixture
def without_torch() -> None:
    import sys

    torch_keys = [key for key in sys.modules if key == "torch" or key.startswith("torch.")]
    saved = {key: sys.modules.pop(key) for key in torch_keys}
    yield
    sys.modules.update(saved)


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_home = tmp_path / "config"
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    monkeypatch.setattr(paths_module, "CONFIG_DIR", config_home / "easyupscaler")
    monkeypatch.setattr(paths_module, "DATA_DIR", data_home / "easyupscaler")
    monkeypatch.setattr(paths_module, "CONFIG_FILE", config_home / "easyupscaler" / "config.toml")
    monkeypatch.setattr(paths_module, "REGISTRY_FILE", data_home / "easyupscaler" / "registry.json")
    monkeypatch.setattr(paths_module, "MODELS_DIR", data_home / "easyupscaler" / "models")

    return tmp_path
