from functools import lru_cache
from pathlib import Path

import yaml

OCRAI_PROMPT_YAML_KEY = "prompt"
OCRAI_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "ocrai.yaml"
OCRAI_PROMPT_MISSING_ERROR = f"Error: OCR-AI prompt file not found: {OCRAI_PROMPT_PATH}"
OCRAI_PROMPT_INVALID_ERROR_TEMPLATE = (
    "Error: OCR-AI prompt file {path} must contain a non-empty '{key}' string."
)


@lru_cache(maxsize=1)
def load_ocrai_prompt(*, prompt_path: Path | None = None) -> str:
    path = prompt_path or OCRAI_PROMPT_PATH
    if not path.is_file():
        raise ValueError(OCRAI_PROMPT_MISSING_ERROR)

    with path.open(encoding="utf-8") as prompt_file:
        data = yaml.safe_load(prompt_file)

    if not isinstance(data, dict):
        raise ValueError(
            OCRAI_PROMPT_INVALID_ERROR_TEMPLATE.format(path=path, key=OCRAI_PROMPT_YAML_KEY)
        )

    prompt = data.get(OCRAI_PROMPT_YAML_KEY)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            OCRAI_PROMPT_INVALID_ERROR_TEMPLATE.format(path=path, key=OCRAI_PROMPT_YAML_KEY)
        )

    return prompt.strip()
