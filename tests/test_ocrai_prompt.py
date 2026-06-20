from pathlib import Path

import pytest
import yaml

from easyupscaler.denoise.ocrai_prompt import (
    OCRAI_PROMPT_YAML_KEY,
    load_ocrai_prompt,
)


def test_load_ocrai_prompt_reads_bundled_yaml() -> None:
    prompt = load_ocrai_prompt()
    assert "reading the page naturally" in prompt
    assert "Markdown pipe tables" in prompt
    assert prompt == prompt.strip()


def test_load_ocrai_prompt_reads_custom_yaml(tmp_path: Path) -> None:
    prompt_path = tmp_path / "custom.yaml"
    prompt_path.write_text(
        yaml.dump({OCRAI_PROMPT_YAML_KEY: "Custom prompt text."}),
        encoding="utf-8",
    )
    load_ocrai_prompt.cache_clear()
    try:
        assert load_ocrai_prompt(prompt_path=prompt_path) == "Custom prompt text."
    finally:
        load_ocrai_prompt.cache_clear()


def test_load_ocrai_prompt_missing_file_raises(tmp_path: Path) -> None:
    load_ocrai_prompt.cache_clear()
    try:
        with pytest.raises(ValueError, match="prompt file not found"):
            load_ocrai_prompt(prompt_path=tmp_path / "missing.yaml")
    finally:
        load_ocrai_prompt.cache_clear()


def test_load_ocrai_prompt_empty_prompt_raises(tmp_path: Path) -> None:
    prompt_path = tmp_path / "empty.yaml"
    prompt_path.write_text(yaml.dump({OCRAI_PROMPT_YAML_KEY: "   "}), encoding="utf-8")
    load_ocrai_prompt.cache_clear()
    try:
        with pytest.raises(ValueError, match="must contain a non-empty"):
            load_ocrai_prompt(prompt_path=prompt_path)
    finally:
        load_ocrai_prompt.cache_clear()
