from dataclasses import dataclass, field
from typing import Literal


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
    key: str
    discord_url: str | None = field(default=None)

    @classmethod
    def from_json(cls, json_data: dict) -> "Config":
        return cls(
            szuru=SzuruConfig(**json_data["szuru"]),
            thumbnails=ThumbnailsConfig.from_dict(json_data["thumbnails"]),
            tagging_map=TaggingMap(**json_data["tagging_map"]),
            tagging_enable=TaggingEnabled(**json_data["tagging_enable"]),
            threshold=TaggingThresholds(**json_data["threshold"]),
            key=json_data["key"],
            discord_url=json_data.get("discord_url"),
        )
