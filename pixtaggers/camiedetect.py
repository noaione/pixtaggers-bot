import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from pixtaggers.onnx_session import prepare_model_runtime_builders

from .img_helpers import ModelThreshold, RatingTag, TagDetectionResult, has_alpha_channel, load_image

TARGET_SIZE = 512

THIS_DIR = Path(__file__).parent.resolve()
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_PATH = THIS_DIR / "models" / "camie-tagger-v2.onnx"
MODEL_METADATA = json.loads((THIS_DIR / "models" / "camie-tagger-v2-metadata.json").read_text(encoding="utf-8"))

dataset_info = MODEL_METADATA["dataset_info"]
tag_mapping = dataset_info["tag_mapping"]
idx_to_tag = tag_mapping["idx_to_tag"]
tag_to_category = tag_mapping["tag_to_category"]
total_tags = dataset_info["total_tags"]


def preprocess_image(image: Image.Image):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = TARGET_SIZE
        new_height = int(TARGET_SIZE / aspect_ratio)
    else:
        new_height = TARGET_SIZE
        new_width = int(TARGET_SIZE * aspect_ratio)

    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    pad_color = (124, 116, 104)
    new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), pad_color)

    paste_x = (TARGET_SIZE - new_width) // 2
    paste_y = (TARGET_SIZE - new_height) // 2
    new_img.paste(image, (paste_x, paste_y))

    img_tensor = np.array(new_img).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose((2, 0, 1)).astype(np.float32)

    # apply normalization
    mean = NORM_MEAN.reshape(-1, 1, 1)
    std = NORM_STD.reshape(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor


def raw_detect_image_tags(session: ort.InferenceSession, img_tensor: np.ndarray):
    img_numpy = img_tensor.astype(np.float32)
    img_numpy = np.expand_dims(img_numpy, axis=0)  # add batch dimension

    inputs = {session.get_inputs()[0].name: img_numpy}
    outputs = session.run(None, inputs)

    if len(outputs) >= 2:
        return outputs[1]  # logits
    else:
        # Fallback to single output
        return outputs[0]  # logits


def map_rating_tag(tag_dat: str) -> RatingTag | None:
    if tag_dat == "rating_general":
        return "safe"
    elif tag_dat == "rating_sensitive" or tag_dat == "rating_questionable":
        return "sketchy"
    elif tag_dat == "rating_explicit":
        return "unsafe"
    else:
        return None


def detect_image_tags(
    session: ort.InferenceSession, img: Image.Image | Path | str | bytes, thresholds: ModelThreshold, *, top_k: int = 50
) -> TagDetectionResult:
    proc_img = load_image(img)
    proc_img = preprocess_image(proc_img)

    main_logits = raw_detect_image_tags(session, proc_img)

    general_tags: dict[str, float] = {}
    character_tags: dict[str, float] = {}
    media_tags: dict[str, int | float] = {}
    rating_tag: RatingTag | None = None

    # Apply sigmoid to get probabilities
    main_probs = 1.0 / (1.0 + np.exp(-main_logits))  # type: ignore
    # indices = main_probs[0]
    # print(indices)

    # Group by category
    tags_by_category: dict[str, list[tuple[str, float]]] = defaultdict(list)
    # predictions_mask = (main_probs >= 0.2)
    # indices = np.where(predictions_mask[0])[0]
    # indices but without the threshold limit
    indices = np.argsort(main_probs[0])[::-1]  # Get top-k indices

    for idx in indices:
        idx_str = str(idx)
        tag_name = idx_to_tag.get(idx_str, f"unknown-{idx}")
        category = tag_to_category.get(tag_name, "general")
        prob = float(main_probs[0, idx])

        tags_by_category[category].append((tag_name, prob))

    # Filter by thresholds
    if "general" in tags_by_category:
        limit = thresholds.general
        tags_by_category["general"] = [
            (tag, prob)
            for tag, prob in tags_by_category["general"]
            if prob >= limit
        ]
    if "character" in tags_by_category:
        limit = thresholds.character
        tags_by_category["character"] = [
            (tag, prob)
            for tag, prob in tags_by_category["character"]
            if prob >= limit
        ]
    if "media" in tags_by_category:
        limit = thresholds.media
        tags_by_category["media"] = [
            (tag, prob)
            for tag, prob in tags_by_category["media"]
            if prob >= limit
        ]
    if "rating" in tags_by_category:
        limit = thresholds.rating
        tags_by_category["rating"] = [
            (tag, prob)
            for tag, prob in tags_by_category["rating"]
            if prob >= limit
        ]

    # Sort by probability within each category
    for category in tags_by_category:
        tags_by_category[category] = sorted(tags_by_category[category], key=lambda x: x[1], reverse=True)[
            :top_k
        ]  # Limit per category

    # Get for each and remap into dict[str, float]
    general_tags = {tag: prob for tag, prob in tags_by_category.get("general", [])}
    character_tags = {tag: prob for tag, prob in tags_by_category.get("character", [])}
    media_tags = {tag: prob for tag, prob in tags_by_category.get("media", [])}

    # for rating, get the best
    rating_tags = tags_by_category.get("rating", [])
    if len(rating_tags) > 0:
        raw_rating, _ = rating_tags[0]
        rating_tag = map_rating_tag(raw_rating)

    return {
        "general": general_tags,
        "characters": character_tags,
        "media": media_tags,
        "rating": rating_tag,
    }


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


def splat_tags(data: dict[str, float]) -> list[str]:
    return list(data.keys())


def merge_tags(*tag_lists: list[str]) -> list[str]:
    merged = set()
    for tag_list in tag_lists:
        merged.update(tag_list)
    return list(merged)


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


class CamieSession:
    def __init__(self, model_path: Path, threshold: ModelThreshold | None = None, top_k: int = 64):
        self._model_path = model_path
        self._session: ort.InferenceSession | None = None

        self._threshold = threshold or ModelThreshold(
            0.492,
            0.614,
            0.492,
            0.614,
        )
        self._top_k = top_k

    # async with prepare_model_runtime_builders(MODEL_PATH) as session:
    async def __aenter__(self):
        print("Loading CamieTagger model...")
        self._session = prepare_model_runtime_builders(self._model_path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_session"):
            del self._session

    def _detect_tags(self, img: Image.Image) -> TagDetectionResult:
        if self._session is None:
            raise RuntimeError("Session not initialized")
        return detect_image_tags(self._session, img, self._threshold, top_k=self._top_k)

    async def detect(self, img: bytes):
        img_data = Image.open(BytesIO(img))
        meta_tags = await asyncio.to_thread(determine_meta_tag_for_images, img_data, img)

        # Run in thread to avoid blocking event loop, since onnxruntime doesn't support async
        tag_result = await asyncio.to_thread(self._detect_tags, img_data)
        return TagResult(
            meta=meta_tags,
            general=splat_tags(tag_result["general"]),
            media=splat_tags(tag_result["media"]),
            characters=splat_tags(tag_result["characters"]),
            rating=tag_result["rating"],
        )
