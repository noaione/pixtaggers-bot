import pillow_jxl  # ruff: ignore[unused-import]
from PIL import Image
from pillow_heif import register_heif_opener

__all__ = ("Image",)

register_heif_opener()
