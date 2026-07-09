import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import onnxruntime as ort

from .im_sess import Image
from .img_helpers import ModelThreshold, RatingTag, TagDetectionResult, has_alpha_channel
from .onnx_session import prepare_model_runtime_builders


@dataclass
class TagResult:
    meta: list[str]
    general: list[str]
    media: list[str]
    characters: list[str]
    rating: RatingTag | None

    def count(self) -> int:
        count = len(self.general) + len(self.media) + len(self.characters)
        return count


def splat_tags(data: dict[str, float]) -> list[str]:
    return list(data.keys())


class BaseTaggerSession(ABC):
    def __init__(self, model_path: Path, threshold: ModelThreshold, top_k: int = 64):
        self._model_path = model_path
        self._threshold = threshold
        self._top_k = top_k
        self._session: ort.InferenceSession | None = None

    async def __aenter__(self):
        self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    def load(self):
        self._session = prepare_model_runtime_builders(self._model_path)

    def unload(self):
        self._session = None

    def require_session(self) -> ort.InferenceSession:
        if self._session is None:
            raise RuntimeError("Session not initialized")
        return self._session

    @abstractmethod
    def _detect_tags(self, img: Image.Image) -> TagDetectionResult:
        raise NotImplementedError

    async def detect(self, img: bytes):
        img_data = Image.open(BytesIO(img))
        meta_tags = await asyncio.to_thread(determine_meta_tag_for_images, img_data, img)
        tag_result = await asyncio.to_thread(self._detect_tags, img_data)
        return TagResult(
            meta=meta_tags + splat_tags(tag_result.get("meta", {})),
            general=splat_tags(tag_result["general"]),
            media=splat_tags(tag_result["media"]),
            characters=splat_tags(tag_result["characters"]),
            rating=tag_result["rating"],
        )


def determine_meta_tag_for_images(img_data: Image.Image, raw_bytes: bytes) -> list[str]:
    # check if has alpha channel
    all_meta_tags = []
    if has_alpha_channel(img_data):
        all_meta_tags.append("alpha_transparency")

    # detect tall image
    if img_data.height / img_data.width >= 2:
        all_meta_tags.append("tall_image")
    elif img_data.width / img_data.height >= 2:
        all_meta_tags.append("wide_image")

    # detect for JPEG artifacts
    # naive, just check file size compared to dimensions, if it's very small, it's likely a compressed to hell and back
    img_w, img_h = img_data.size
    pix_count = img_w * img_h
    # Check if JPEG
    jpeg_signature = b"\xff\xd8\xff"
    # this threshold is arbitrary and may need tuning
    if raw_bytes.startswith(jpeg_signature) and len(raw_bytes) / pix_count < 0.12:
        all_meta_tags.append("jpeg_artifacts")

    # check resolution
    # lowres (500x500 or smaller)
    # no resolution tag (larger than 500x500 and smaller than 1600x1200)
    # highres (at least 1600x1200)
    # absurdres (at least 3200x2400)
    # incredibly absurdres (any dimension over 10000)
    lowres_count = 500 * 500
    highres_count = 1600 * 1200
    absurdres_count = 3200 * 2400
    # Go from highest to lowest
    if img_w > 10000 or img_h > 10000:
        all_meta_tags.append("incredibly_absurdres")
    elif pix_count >= absurdres_count:
        all_meta_tags.append("absurdres")
    elif pix_count >= highres_count:
        all_meta_tags.append("highres")
    elif pix_count <= lowres_count:
        all_meta_tags.append("lowres")
    return all_meta_tags
