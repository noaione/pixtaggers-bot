import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .commondetect import BaseTaggerSession
from .im_sess import Image
from .img_helpers import ModelThreshold, RatingTag, TagDetectionResult, load_image
from .onnx_session import prepare_model_runtime_builders

TARGET_SIZE = 384
THIS_DIR = Path(__file__).parent.resolve()

MODEL_DIR = THIS_DIR / "models" / "cl-tagger-v2"
MODEL_PATH = MODEL_DIR / "model.onnx"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_VOCAB_PATH = MODEL_DIR / "model_vocabulary.json"
MODEL_TAG_METRICS_PATH = MODEL_DIR / "model_tag_metrics.npz"
MODEL_OOD_REF_PATH = MODEL_DIR / "model_ood_ref.npz"

NORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
NORM_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

CHAR_THRESHOLD_CATS = {"Character", "Copyright"}
GENERAL_THRESHOLD_CATS = {"General", "Meta"}
PINNED_CATS = {"Quality", "Rating"}
RATING_WORDS = {"general", "sensitive", "questionable", "explicit"}
QUALITY_WORDS = {"best quality", "high quality", "normal quality", "medium quality"}
OOD_EMB_NODE = "/vision_encoder/head/Gather_1_output_0"


def _vocab_get(vocab: dict[str, Any], key: str) -> dict[str, Any]:
    if key in vocab:
        return vocab[key]
    suffix = f"/{key}"
    for vocab_key, value in vocab.items():
        if isinstance(vocab_key, str) and vocab_key.endswith(suffix):
            return value
    return {}


def load_vocabulary(vocab_path: Path) -> tuple[dict[int, str], dict[str, str]]:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    raw_idx = _vocab_get(vocab, "idx_to_tag")
    if not raw_idx:
        raise ValueError(f"'idx_to_tag' not found in {vocab_path}")

    idx_to_tag = {int(idx): str(tag) for idx, tag in raw_idx.items()}
    tag_to_category = {str(tag): str(cat) for tag, cat in _vocab_get(vocab, "tag_to_category").items()}

    for tag in list(tag_to_category):
        normalized = tag.strip().lower().replace("_", " ")
        if normalized in RATING_WORDS:
            tag_to_category[tag] = "Rating"
        elif normalized in QUALITY_WORDS:
            tag_to_category[tag] = "Quality"
    return idx_to_tag, tag_to_category


def load_tag_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    data = np.load(metrics_path, allow_pickle=True)
    metrics = {key: data[key] for key in data.files}
    required = {"best_thr", "pos_hist", "total_hist"}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"Missing CL tag metrics keys: {', '.join(sorted(missing))}")
    return metrics


def load_ood_reference(ood_ref_path: Path) -> dict[str, Any]:
    if not ood_ref_path.is_file():
        raise FileNotFoundError(ood_ref_path)
    data = np.load(ood_ref_path)
    return {
        "mu": data["mu"].astype(np.float64),
        "cov_inv": data["cov_inv"].astype(np.float64),
        "p50": float(data["p50"]),
        "p95": float(data["p95"]),
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = load_image(image)
    image = image.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.BICUBIC)
    img_tensor = np.array(image).astype(np.float32) / 255.0
    img_tensor = (img_tensor - NORM_MEAN.reshape(1, 1, 3)) / NORM_STD.reshape(1, 1, 3)
    return img_tensor.transpose((2, 0, 1)).astype(np.float32)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_jeffreys_calibration_table(metrics: dict[str, Any], eps: float = 0.5) -> np.ndarray:
    pos_h = metrics["pos_hist"].astype(np.float32)
    total_h = metrics["total_hist"].astype(np.float32)
    n_pos_tag = pos_h.sum(axis=1, keepdims=True)
    n_total_tag = total_h.sum(axis=1, keepdims=True)
    pi = np.where(n_total_tag > 0, n_pos_tag / n_total_tag, 0.0)
    calib = (pos_h + eps) / (total_h + 2.0 * eps)
    calib = np.where(total_h > 0, calib, pi)
    return calib.astype(np.float16)


def calibrate_probs(probs: np.ndarray, metrics: dict[str, Any], calib_table: np.ndarray) -> np.ndarray:
    n_bins_raw = metrics.get("n_bins", np.array([calib_table.shape[1]]))
    n_bins = int(n_bins_raw.item() if np.ndim(n_bins_raw) == 0 else n_bins_raw[0])
    bin_idx = np.clip((probs * n_bins).astype(np.int32), 0, n_bins - 1)
    calibrated = calib_table[np.arange(len(probs)), bin_idx].astype(np.float32)
    nan_mask = np.isnan(calibrated)
    if np.any(nan_mask):
        calibrated[nan_mask] = probs[nan_mask]
    return calibrated


def compute_mahalanobis(emb: np.ndarray, ood_ref: dict[str, Any]) -> float:
    diff = emb.astype(np.float64) - ood_ref["mu"]
    return float(np.sqrt(max(0.0, diff @ ood_ref["cov_inv"] @ diff)))


def compute_ood_ramp(ood_distance: float | None, ood_ref: dict[str, Any] | None) -> float:
    if ood_distance is None or ood_ref is None:
        return 0.0
    tail = max(ood_ref["p95"] - ood_ref["p50"], 1e-6)
    return max(0.0, min(1.0, (ood_distance - ood_ref["p95"]) / (2.0 * tail)))


def map_rating_tag(tag_dat: str) -> RatingTag | None:
    normalized = tag_dat.strip().lower().replace("_", " ")
    if normalized == "general":
        return "safe"
    if normalized in {"sensitive", "questionable"}:
        return "sketchy"
    if normalized == "explicit":
        return "unsafe"
    return None


def normalize_tag_name(tag: str) -> str:
    return "_".join(tag.split())


def _best_by_category(items: list[dict[str, Any]], category: str) -> dict[str, Any] | None:
    category_items = [item for item in items if item["category"] == category]
    return max(category_items, key=lambda item: item["raw_prob"]) if category_items else None


def _category_floor(category: str, thresholds: ModelThreshold) -> float:
    if category in CHAR_THRESHOLD_CATS:
        return thresholds.character
    if category in GENERAL_THRESHOLD_CATS:
        return thresholds.general
    return thresholds.general


def detect_image_tags(
    session: ort.InferenceSession,
    img: Image.Image | Path | str | bytes,
    thresholds: ModelThreshold,
    *,
    idx_to_tag: dict[int, str],
    tag_to_category: dict[str, str],
    tag_metrics: dict[str, Any],
    calibration_table: np.ndarray,
    ood_ref: dict[str, Any] | None = None,
    ood_session: ort.InferenceSession | None = None,
    top_k: int = 64,
    min_best_f1: float = 0.05,
) -> TagDetectionResult:
    proc_img = preprocess_image(load_image(img))
    img_numpy = np.expand_dims(proc_img, axis=0)

    ood_distance = None
    if ood_ref is not None and ood_session is not None:
        outputs = ood_session.run(["logits", OOD_EMB_NODE], {"pixel_values": img_numpy})
        logits = outputs[0][0].astype(np.float64)  # type: ignore
        ood_distance = compute_mahalanobis(outputs[1][0], ood_ref)  # type: ignore
    else:
        outputs = session.run(["logits"], {"pixel_values": img_numpy})
        logits = outputs[0][0].astype(np.float64)  # type: ignore

    raw_probs = _sigmoid(logits)
    cal_probs = calibrate_probs(raw_probs, tag_metrics, calibration_table)
    ood_t = compute_ood_ramp(ood_distance, ood_ref)

    best_thr = tag_metrics["best_thr"].astype(np.float32)
    best_f1 = tag_metrics.get("best_f1")
    if best_f1 is not None:
        best_f1 = best_f1.astype(np.float32)

    items: list[dict[str, Any]] = []
    for idx, raw_prob in enumerate(raw_probs):
        tag = idx_to_tag.get(idx)
        if tag is None:
            continue
        category = tag_to_category.get(tag, "Unknown")
        if category == "Unknown":
            continue
        items.append(
            {
                "tag": tag,
                "category": category,
                "raw_prob": float(raw_prob),
                "cal_prob": float(cal_probs[idx]),
                "idx": idx,
            }
        )

    rating_top = _best_by_category(items, "Rating")
    rating_tag = map_rating_tag(rating_top["tag"]) if rating_top is not None else None

    tags_by_category: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for item in items:
        category = item["category"]
        if category in PINNED_CATS:
            continue

        idx = item["idx"]
        if best_f1 is not None:
            f1 = float(best_f1[idx])
            if not np.isnan(f1) and f1 < min_best_f1:
                continue

        threshold = max(float(best_thr[idx]), _category_floor(category, thresholds))
        if ood_t > 0.0 and category in CHAR_THRESHOLD_CATS:
            threshold = threshold + ood_t * (0.85 - threshold)

        if item["raw_prob"] >= threshold:
            tags_by_category[category].append((normalize_tag_name(item["tag"]), item["cal_prob"]))

    for category, tags in tags_by_category.items():
        tags_by_category[category] = sorted(tags, key=lambda tag: tag[1], reverse=True)[:top_k]

    general_tags = {tag: prob for tag, prob in tags_by_category.get("General", [])}
    character_tags = {tag: prob for tag, prob in tags_by_category.get("Character", [])}
    copyright_tags = {tag: prob for tag, prob in tags_by_category.get("Copyright", [])}

    return {
        "general": general_tags,
        "characters": character_tags,
        "media": copyright_tags,
        "rating": rating_tag,
    }


class ClTaggerSession(BaseTaggerSession):
    def __init__(
        self,
        model_path: Path,
        threshold: ModelThreshold | None = None,
        top_k: int = 64,
        *,
        vocab_path: Path = MODEL_VOCAB_PATH,
        metrics_path: Path = MODEL_TAG_METRICS_PATH,
        ood_ref_path: Path = MODEL_OOD_REF_PATH,
    ):
        super().__init__(model_path, threshold or ModelThreshold(0.30, 0.30, 0.30, 0.0), top_k)
        self._vocab_path = vocab_path
        self._metrics_path = metrics_path
        self._ood_ref_path = ood_ref_path
        self._ood_session: ort.InferenceSession | None = None
        self._idx_to_tag: dict[int, str] = {}
        self._tag_to_category: dict[str, str] = {}
        self._tag_metrics: dict[str, Any] = {}
        self._calibration_table: np.ndarray | None = None
        self._ood_ref: dict[str, Any] | None = None

    def load(self):
        print("Loading CL Tagger v2 model...")
        self._idx_to_tag, self._tag_to_category = load_vocabulary(self._vocab_path)
        self._tag_metrics = load_tag_metrics(self._metrics_path)
        self._calibration_table = compute_jeffreys_calibration_table(self._tag_metrics)
        self._ood_ref = load_ood_reference(self._ood_ref_path)
        super().load()
        self._ood_session = self._prepare_ood_session()

    def unload(self):
        super().unload()
        self._ood_session = None

    def _prepare_ood_session(self) -> ort.InferenceSession | None:
        try:
            import onnx

            ood_model_path = self._model_path.with_name(f"{self._model_path.stem}_ood.onnx")
            if not ood_model_path.is_file() or ood_model_path.stat().st_mtime < self._model_path.stat().st_mtime:
                model_proto = onnx.load(self._model_path, load_external_data=False)
                output_names = {output.name for output in model_proto.graph.output}
                if OOD_EMB_NODE not in output_names:
                    emb_type = onnx.helper.make_tensor_value_info(OOD_EMB_NODE, onnx.TensorProto.FLOAT, None)
                    model_proto.graph.output.append(emb_type)
                onnx.save_model(model_proto, ood_model_path)
            return prepare_model_runtime_builders(ood_model_path)
        except Exception as exc:
            print(f"CL Tagger OOD disabled: {exc}")
            return None

    def _detect_tags(self, img: Image.Image) -> TagDetectionResult:
        if self._calibration_table is None:
            raise RuntimeError("Calibration table not initialized")
        return detect_image_tags(
            self.require_session(),
            img,
            self._threshold,
            idx_to_tag=self._idx_to_tag,
            tag_to_category=self._tag_to_category,
            tag_metrics=self._tag_metrics,
            calibration_table=self._calibration_table,
            ood_ref=self._ood_ref,
            ood_session=self._ood_session,
            top_k=self._top_k,
        )
