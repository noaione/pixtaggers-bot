from dataclasses import dataclass, field
from typing import Literal, cast

ModelName = Literal["camie-tagger-v2", "cl-tagger-v2", "pixai-tagger-v1"]
SUPPORTED_MODELS: tuple[ModelName, ...] = ("camie-tagger-v2", "cl-tagger-v2", "pixai-tagger-v1")


@dataclass
class SimpleSnapshot:
    id: str
    operation: Literal["created", "modified", "deleted", "merged"]
    type: Literal["tag", "tag_category", "post", "pool", "pool_category"]


@dataclass
class SzuruConfig:
    host: str
    user: str
    token: str
    tag_cache_path: str = ".cache/szurubooru-tags.json"
    http_impersonate: str | None = None


@dataclass
class ThumbnailVideoConfig:
    enabled: bool
    """:class:`bool` whether to generate thumbnails for videos"""
    extract: int
    """:class:`int` number of frames to be extracted from the video for thumbnail generation"""
    detect: int
    """:class:`int` number of frames to be used for tag detection"""


@dataclass
class ThumbnailsConfig:
    target_size: int
    """:class:`int` the maximum size of the longest side of the thumbnail"""
    alpha_fix: bool
    """:class:`bool` whether to apply the alpha thumbnail fix (white background)"""
    video: ThumbnailVideoConfig
    """:class:`ThumbnailVideoConfig` configuration for video thumbnail generation"""

    @classmethod
    def from_dict(cls, data: dict) -> "ThumbnailsConfig":
        video_config = ThumbnailVideoConfig(**data["video"])
        return cls(
            target_size=data["target_size"],
            alpha_fix=data["alpha_fix"],
            video=video_config,
        )


@dataclass
class TaggingMap:
    general: str
    media: str
    characters: str
    meta: str


@dataclass
class TaggingEnabled:
    general: bool
    media: bool
    characters: bool
    meta: bool
    rating: bool


@dataclass
class TaggingThresholds:
    general: float
    media: float
    characters: float
    rating: float
    top_k: int


@dataclass
class Config:
    szuru: SzuruConfig
    thumbnails: ThumbnailsConfig
    tagging_map: TaggingMap
    tagging_enable: TaggingEnabled
    threshold: TaggingThresholds
    model: ModelName
    trt_prioritize: Literal["rtx", "trt"]
    device_id: int
    onnx_verbose: bool
    key: str
    discord_url: str | None = field(default=None)

    @classmethod
    def from_json(cls, json_data: dict) -> "Config":
        model = json_data.get("model", "cl-tagger-v2")
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model}'. Expected one of: {', '.join(SUPPORTED_MODELS)}")
        trt_prioritize = json_data.get("trt_prioritize", "trt").lower()
        if trt_prioritize not in ("rtx", "trt"):
            raise ValueError(f"Unsupported trt_prioritize '{trt_prioritize}'. Expected one of: 'rtx', 'trt'")
        return cls(
            szuru=SzuruConfig(**json_data["szuru"]),
            thumbnails=ThumbnailsConfig.from_dict(json_data["thumbnails"]),
            tagging_map=TaggingMap(**json_data["tagging_map"]),
            tagging_enable=TaggingEnabled(**json_data["tagging_enable"]),
            threshold=TaggingThresholds(**json_data["threshold"]),
            model=cast(ModelName, model),
            key=json_data["key"],
            discord_url=json_data.get("discord_url"),
            trt_prioritize=trt_prioritize,
            device_id=json_data.get("device_id", 0),
            onnx_verbose=json_data.get("onnx_verbose", False),
        )
