import os
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from easyupscaler.cli.main import app

runner = CliRunner()
WEIGHTS_ENV = "EASYUPSCALER_TEST_WEIGHTS"


@pytest.mark.slow
def test_e2e_import_default_upscale(isolated_paths, tmp_path: Path) -> None:
    weights = os.environ.get(WEIGHTS_ENV)
    if not weights:
        pytest.skip(f"{WEIGHTS_ENV} not set")

    weights_path = Path(weights)
    if not weights_path.exists():
        pytest.skip(f"weights file not found: {weights_path}")

    input_image = tmp_path / "input_64x64.png"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(input_image)

    import_result = runner.invoke(app, ["models", "import", str(weights_path)])
    assert import_result.exit_code == 0, import_result.stdout + import_result.stderr

    model_name = weights_path.stem
    default_result = runner.invoke(app, ["models", "default", model_name])
    assert default_result.exit_code == 0

    upscale_result = runner.invoke(app, [str(input_image)])
    assert upscale_result.exit_code == 0, upscale_result.stdout + upscale_result.stderr

    output_path = tmp_path / "input_64x64-upscaled.jpg"
    assert output_path.exists()
    with Image.open(output_path) as output_image:
        assert output_image.size == (256, 256)
