import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import onnxruntime as ort

from .commondetect import BaseTaggerSession
from .im_sess import Image
from .img_helpers import ModelThreshold, RatingTag, TagDetectionResult, load_image

TARGET_SIZE = 1008
THIS_DIR = Path(__file__).parent.resolve()

MODEL_DIR = THIS_DIR / "models" / "pixai-tagger-v1"
MODEL_PATH = MODEL_DIR / "model.onnx"
TAGS_METADATA_PATH = MODEL_DIR / "tags.json"


class ResultProbs(TypedDict):
    idx: int
    tag: str
    logit: float
    prob: float


def load_tags_metadata(meta_path: Path) -> dict[str, Any]:
    tag_map = json.loads(meta_path.read_text(encoding="utf-8"))
    if sum(int(category["count"]) for category in tag_map["categories"]) != int(tag_map["num_classes"]):
        raise ValueError(f"Tag map counts in {meta_path} do not match num_classes")
    return tag_map


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = load_image(image)
    width, height = image.size
    if height != TARGET_SIZE or width != TARGET_SIZE:
        scale = min(TARGET_SIZE / height, TARGET_SIZE / width)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        canvas.paste(image, ((TARGET_SIZE - new_size[0]) // 2, (TARGET_SIZE - new_size[1]) // 2))
        image = canvas

    # The source processor normalizes after padding: RGB 0 becomes -1.
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return np.transpose(array, (2, 0, 1))[None, ...]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative = ~positive
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1 + exp_values)
    return result


def _get_threshold(name: str, threshold: ModelThreshold) -> float:
    if name == "rating":
        return threshold.rating
    elif name == "character":
        return threshold.character
    elif name == "copyright":
        return threshold.media
    elif name == "general":
        return threshold.general
    elif name == "style":
        return 0.15  # default
    elif name == "meta":
        return 0.17  # model default
    else:
        raise ValueError(f"Unknown category {name!r}")


def resolve_results(
    logits: np.ndarray,
    tag_map: dict[str, Any],
    thresholds: ModelThreshold,
) -> dict[str, list[ResultProbs]]:
    probabilities = _sigmoid(logits)
    results: dict[str, list[ResultProbs]] = {}
    for category in tag_map["categories"]:
        name = category["name"]
        offset = int(category["offset"])
        count = int(category["count"])
        category_logits = logits[offset : offset + count]
        category_probabilities = probabilities[offset : offset + count]
        tags = category["tags"]
        if len(tags) != count:
            raise ValueError(f"Tag map for {name!r} has an invalid count")
        selected_threshold = _get_threshold(name, thresholds)
        thresh_limit = np.full(count, selected_threshold, dtype=np.float32)
        selected = np.flatnonzero(category_probabilities > thresh_limit)
        results[name] = [
            {
                "idx": offset + int(local_index),
                "tag": tags[int(local_index)],
                "logit": float(category_logits[local_index]),
                "prob": float(category_probabilities[local_index]),
            }
            for local_index in selected
        ]
    return results


def _map_rating_tag(tag: str) -> RatingTag | None:
    normalized = tag.strip().lower()
    if normalized == "rating:g":
        return "safe"
    if normalized in {"rating:s", "rating:q"}:
        return "sketchy"
    if normalized == "rating:e":
        return "unsafe"
    return None


def detect_image_tags(
    session: ort.InferenceSession,
    img: Image.Image | Path | str | bytes,
    thresholds: ModelThreshold,
    *,
    tag_map: dict[str, Any],
    top_k: int = 64,
) -> TagDetectionResult:
    proc_img = preprocess_image(load_image(img))

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: proc_img})

    logits = np.asarray(outputs[0][0], dtype=np.float32)  # pyright: ignore[reportIndexIssue]
    if logits.shape != (int(tag_map["num_classes"]),):
        raise ValueError(f"Expected {tag_map['num_classes']} logits, got {logits.shape}")

    tags_by_category = resolve_results(logits, tag_map, thresholds)

    for category, tags in tags_by_category.items():
        tags_by_category[category] = sorted(tags, key=lambda tag: tag["prob"], reverse=True)[:top_k]

    general_tags = {tag["tag"]: tag["prob"] for tag in tags_by_category.get("general", [])}
    character_tags = {tag["tag"]: tag["prob"] for tag in tags_by_category.get("character", [])}
    copyright_tags = {tag["tag"]: tag["prob"] for tag in tags_by_category.get("copyright", [])}
    # highest rating tag
    rating_tags = tags_by_category.get("rating", [])
    rating_tag = _map_rating_tag(rating_tags[0]["tag"]) if len(rating_tags) > 0 else None

    return {
        "general": general_tags,
        "characters": character_tags,
        "media": copyright_tags,
        "rating": rating_tag,
    }


class PixAiTaggerSession(BaseTaggerSession):
    def __init__(
        self,
        model_path: Path,
        threshold: ModelThreshold | None = None,
        top_k: int = 64,
        *,
        tags_metadata_path: Path = TAGS_METADATA_PATH,
    ):
        super().__init__(model_path, threshold or ModelThreshold(0.17, 0.27, 0.24, 0.41), top_k)
        self._ood_session: ort.InferenceSession | None = None
        self._tags_meta_path = tags_metadata_path
        self._tags_map: dict[str, Any] = {}

    def load(self):
        print("Loading PixAI Tagger v1 model...")
        self._tags_map = load_tags_metadata(self._tags_meta_path)
        super().load()

    def _detect_tags(self, img: Image.Image) -> TagDetectionResult:
        return detect_image_tags(
            self.require_session(),
            img,
            self._threshold,
            tag_map=self._tags_map,
            top_k=self._top_k,
        )
