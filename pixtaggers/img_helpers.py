from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from PIL import Image

RatingTag: TypeAlias = Literal["safe", "sketchy", "unsafe"]


class TagDetectionResult(TypedDict):
    general: dict[str, float]
    """The value is the score of the tag, not a count."""
    characters: dict[str, float]
    """The value is the score of the tag, not a count."""
    media: dict[str, int | float]
    """The value is the count of tags that belong to the media, not a score."""
    rating: RatingTag | None
    """
    The image rating, either "safe", "sketchy", or "unsafe", or None if no rating tag is detected.
    """


@dataclass
class ModelThreshold:
    general: float
    character: float
    media: float
    rating: float

    # Do attribute lookup
    def __getattr__(self, name: str) -> float:
        if name in ("general", "character", "media", "rating"):
            return self.__dict__[name]
        raise AttributeError(f"ModelThreshold has no attribute '{name}'")


def has_alpha_channel(image: Image.Image) -> bool:
    mode = image.mode

    if mode in ("RGBA", "LA", "PA"):
        return True

    if getattr(image, "palette"):
        try:
            image.palette.getcolor((0, 0, 0, 0))  # type: ignore
            return True  # cannot find a line to trigger this
        except ValueError:
            pass

    return "transparency" in image.info


def load_image(image_data: Image.Image | Path | str | bytes) -> Image.Image:
    if isinstance(image_data, (str, Path)):
        image = Image.open(image_data)
    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        raise ValueError("Unsupported image data type")

    if has_alpha_channel(image):
        try:
            ret_img = Image.new("RGBA", image.size, "white")
            ret_img.paste(image, (0, 0), mask=image)
        except ValueError:
            ret_img = image
    else:
        ret_img = image

    if ret_img.mode != "RGB":
        ret_img = ret_img.convert("RGB")
    return ret_img


def resize_by_longest_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    return image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
