class EasyUpscalerError(Exception):
    """Base exception for all easyupscaler domain errors."""


class ModelNotFoundError(EasyUpscalerError):
    """Raised when a model name is not in the registry."""


class DuplicateModelError(EasyUpscalerError):
    """Raised when adding a model that already exists."""


class ImageReadError(EasyUpscalerError):
    """Raised when an image file cannot be read."""


class ImportModelError(EasyUpscalerError):
    """Raised when model import validation fails."""


class UnsupportedModelError(ImportModelError):
    """Raised when Spandrel cannot recognise the model architecture."""


class DenoiseDownloadError(EasyUpscalerError):
    """Raised when a managed denoise model cannot be downloaded."""


class DenoiseModelCorruptError(EasyUpscalerError):
    """Raised when a managed denoise model file is corrupt."""
