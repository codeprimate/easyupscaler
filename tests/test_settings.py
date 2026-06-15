import sys

from easyupscaler.config import paths as paths_module
from easyupscaler.config.settings import ConfigService


def test_missing_config_returns_none(isolated_paths, without_torch) -> None:
    service = ConfigService()
    assert service.get_default_model() is None
    assert "torch" not in sys.modules


def test_set_and_get_default_model(isolated_paths, without_torch) -> None:
    service = ConfigService()
    service.set_default_model("RealESRGAN_x4plus")
    assert service.get_default_model() == "RealESRGAN_x4plus"
    assert paths_module.CONFIG_FILE.exists()
    assert "torch" not in sys.modules


def test_clear_default_model(isolated_paths, without_torch) -> None:
    service = ConfigService()
    service.set_default_model("RealESRGAN_x4plus")
    service.clear_default_model()
    assert service.get_default_model() is None
    assert "torch" not in sys.modules
