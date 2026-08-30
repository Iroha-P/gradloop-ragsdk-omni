from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MediaModality(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


_MEDIA_MODALITIES = {
    "image/png": MediaModality.IMAGE,
    "image/jpeg": MediaModality.IMAGE,
    "image/webp": MediaModality.IMAGE,
    "audio/wav": MediaModality.AUDIO,
    "audio/x-wav": MediaModality.AUDIO,
    "audio/mpeg": MediaModality.AUDIO,
    "audio/mp4": MediaModality.AUDIO,
    "video/mp4": MediaModality.VIDEO,
}


def _has_valid_signature(media_type: str, content: bytes) -> bool:
    if media_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    if media_type == "image/webp":
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    if media_type in {"audio/wav", "audio/x-wav"}:
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WAVE"
    if media_type == "audio/mpeg":
        return content.startswith(b"ID3") or (
            len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
        )
    if media_type in {"audio/mp4", "video/mp4"}:
        return len(content) >= 12 and content[4:8] == b"ftyp"
    return False


@dataclass(frozen=True, slots=True)
class OmniAttachment:
    """An in-memory attachment whose original filename never leaves the API boundary."""

    modality: MediaModality
    media_type: str
    content: bytes
    asset_id: str


def build_attachment(
    *,
    filename: str,
    media_type: str,
    content: bytes,
    max_bytes: int,
) -> OmniAttachment:
    normalized_name = filename.strip()
    if (
        not normalized_name
        or len(normalized_name) > 160
        or PurePath(normalized_name).name != normalized_name
        or "/" in normalized_name
        or "\\" in normalized_name
        or "\x00" in normalized_name
    ):
        raise ValueError("invalid upload filename")
    normalized_type = media_type.strip().lower()
    modality = _MEDIA_MODALITIES.get(normalized_type)
    if modality is None:
        raise ValueError("unsupported media type")
    if not content:
        raise ValueError("empty upload")
    if len(content) > max_bytes:
        raise OverflowError("upload exceeds size limit")
    if not _has_valid_signature(normalized_type, content):
        raise ValueError("media signature does not match declared type")
    digest = hashlib.sha256(content).hexdigest()
    return OmniAttachment(
        modality=modality,
        media_type=normalized_type,
        content=content,
        asset_id=f"asset-{digest[:24]}",
    )


@dataclass(frozen=True, slots=True)
class OmniRequest:
    prompt: str
    attachments: tuple[OmniAttachment, ...]
    max_tokens: int = 512

    def __post_init__(self) -> None:
        prompt = self.prompt.strip()
        if not prompt or len(prompt) > 4096:
            raise ValueError("prompt must contain 1 to 4096 characters")
        if not self.attachments:
            raise ValueError("at least one attachment is required")
        if not 1 <= self.max_tokens <= 2048:
            raise ValueError("max_tokens is outside the supported range")
        object.__setattr__(self, "prompt", prompt)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MultimodalHealth(StrictModel):
    backend: str
    ready: bool
    model: str
    modalities: list[MediaModality] = Field(default_factory=list)
    message: str

    @field_validator("modalities")
    @classmethod
    def unique_modalities(cls, value: list[MediaModality]) -> list[MediaModality]:
        return list(dict.fromkeys(value))


class OmniResponse(StrictModel):
    backend: str
    model: str
    content: str = Field(min_length=1, max_length=20000)
    modalities: list[MediaModality]
    attachment_count: int = Field(ge=1, le=16)
    elapsed_ms: float = Field(ge=0)


@runtime_checkable
class MultimodalModel(Protocol):
    def healthcheck(self) -> MultimodalHealth: ...

    def generate(self, request: OmniRequest) -> OmniResponse: ...
