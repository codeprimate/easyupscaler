from pathlib import Path

import tomlkit
from tomlkit.toml_document import TOMLDocument

from easyupscaler.config import paths

DEFAULT_MODEL_KEY = "default_model"


class ConfigService:
    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path

    @property
    def config_path(self) -> Path:
        return self._config_path or paths.CONFIG_FILE

    def get_default_model(self) -> str | None:
        if not self.config_path.exists():
            return None
        document = self._load_document()
        value = document.get(DEFAULT_MODEL_KEY)
        if value is None or value == "":
            return None
        return str(value)

    def set_default_model(self, name: str) -> None:
        paths.ensure_config_dir()
        document = self._load_or_create_document()
        document[DEFAULT_MODEL_KEY] = name
        self._write_document(document)

    def clear_default_model(self) -> None:
        if not self.config_path.exists():
            return
        document = self._load_document()
        if DEFAULT_MODEL_KEY in document:
            del document[DEFAULT_MODEL_KEY]
            self._write_document(document)

    def _load_or_create_document(self) -> TOMLDocument:
        if self.config_path.exists():
            return self._load_document()
        return tomlkit.document()

    def _load_document(self) -> TOMLDocument:
        content = self.config_path.read_text(encoding="utf-8")
        parsed = tomlkit.parse(content)
        if not isinstance(parsed, TOMLDocument):
            msg = f"invalid config format: {self._config_path}"
            raise ValueError(msg)
        return parsed

    def _write_document(self, document: TOMLDocument) -> None:
        paths.ensure_config_dir()
        self.config_path.write_text(tomlkit.dumps(document), encoding="utf-8")
