import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .commondetect import BaseTaggerSession
from .im_sess import Image
from .img_helpers import ModelThreshold, RatingTag, TagDetectionResult, load_image

TARGET_SIZE = 512

THIS_DIR = Path(__file__).parent.resolve()
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_PATH = THIS_DIR / "models" / "camie-tagger-v2" / "camie-tagger-v2.onnx"
MODEL_METADATA_PATH = THIS_DIR / "models" / "camie-tagger-v2" / "camie-tagger-v2-metadata.json"


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
    session: ort.InferenceSession,
    img: Image.Image | Path | str | bytes,
    thresholds: ModelThreshold,
    *,
    idx_to_tag: dict[str, str],
    tag_to_category: dict[str, str],
    top_k: int = 50,
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


def merge_tags(*tag_lists: list[str]) -> list[str]:
    merged = set()
    for tag_list in tag_lists:
        merged.update(tag_list)
    return list(merged)


class CamieSession(BaseTaggerSession):
    def __init__(
        self,
        model_path: Path,
        threshold: ModelThreshold | None = None,
        top_k: int = 64,
        *,
        metadata_path: Path = MODEL_METADATA_PATH,
    ):
        super().__init__(
            model_path,
            threshold or ModelThreshold(
                0.492,
                0.614,
                0.492,
                0.614,
            ),
            top_k,
        )
        self._metadata_path = metadata_path
        self._idx_to_tag: dict[str, str] = {}
        self._tag_to_category: dict[str, str] = {}

    def load(self):
        print("Loading CamieTagger model...")
        model_metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        tag_mapping = model_metadata["dataset_info"]["tag_mapping"]
        self._idx_to_tag = tag_mapping["idx_to_tag"]
        self._tag_to_category = tag_mapping["tag_to_category"]
        super().load()

    def _detect_tags(self, img: Image.Image) -> TagDetectionResult:
        return detect_image_tags(
            self.require_session(),
            img,
            self._threshold,
            idx_to_tag=self._idx_to_tag,
            tag_to_category=self._tag_to_category,
            top_k=self._top_k,
        )
