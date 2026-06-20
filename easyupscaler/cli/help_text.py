"""Shared help text blocks for CLI commands.

Typer's Rich formatter collapses single newlines inside help paragraphs. Multi-line
sections use Markdown list syntax with rich_markup_mode=\"markdown\" on the app.
"""

"""Shared help text blocks for CLI commands.

Typer's Rich formatter collapses single newlines inside help paragraphs. Multi-line
sections use Markdown list syntax with rich_markup_mode=\"markdown\" on the app.
"""


def build_main_help(version: str) -> str:
    return f"""\
easyupscaler {version}

Upscale and denoise images with AI super-resolution models.

scale requires an imported model. denoise downloads models automatically on first use.

**Getting started:**

- Import a model: `easyupscaler models import ~/Downloads/RealESRGAN_x4.pth`
- Upscale images: `easyupscaler scale photo.jpg`
- Denoise (no model needed): `easyupscaler denoise photo image.jpg`
"""

SCALE_HELP = """\
Upscale images 2-4x with an imported super-resolution model.

Requires at least one imported model. Run 'models import' to add one,
or 'models list' to see what's available.

**Examples:**

- `easyupscaler scale photo.jpg`
- `easyupscaler scale *.jpg --model RealESRGAN_x4plus`
- `easyupscaler scale *.jpg --output ./upscaled/`
"""

SCALE_SHORT_HELP = "Upscale images 2-4x with an imported super-resolution model."

DENOISE_HELP = """\
Reduce noise and compression artifacts from photos, art, or manga.

Models are selected automatically by mode and downloaded on first use.

**Examples:**

- `easyupscaler denoise photo image.jpg`
- `easyupscaler denoise art artwork.png --strength high`
- `easyupscaler denoise manga page*.png --output ./cleaned/`
"""

DENOISE_SHORT_HELP = "Reduce noise and compression artifacts from photos, art, or manga."
