import pillow_jxl  # noqa: F401
from PIL import Image
from pillow_heif import register_heif_opener

__all__ = ("Image",)

register_heif_opener()
