from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PARSER_VERSION = "extractors-v4"
CHUNKER_VERSION = "structured-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sensitivity(StrEnum):
    PUBLIC = "public"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IngestionStatus(StrEnum):
    INDEXED = "indexed"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED = "rejected"
    FAILED = "failed"
    UNCHANGED = "unchanged"


class SourceSpec(StrictModel):
    source_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    path: str = Field(min_length=1, max_length=4096)
    title: str = Field(min_length=1, max_length=512)
    category: str = Field(default="general", max_length=64)
    authority_level: str = Field(default="unknown", max_length=32)
    published_at: date | None = None
    effective_at: date | None = None
    declared_sensitivity: Sensitivity = Sensitivity.LOW
    privacy_lane: str = Field(default="generic", pattern=r"^(generic|personal)$")
    password_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{2,127}$")


class ExtractionBlock(StrictModel):
    text: str = Field(min_length=1)
    section: str = ""
    page: int | None = Field(default=None, ge=1)
    block_type: str = "paragraph"


class ExtractionResult(StrictModel):
    parser: str
    blocks: list[ExtractionBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PrivacyFindingSummary(StrictModel):
    finding_counts: dict[str, int] = Field(default_factory=dict)
    prompt_injection_detected: bool = False
    path_risk_terms: list[str] = Field(default_factory=list)
    recommended_sensitivity: Sensitivity = Sensitivity.LOW
    warnings: list[str] = Field(default_factory=list)


class ChunkRecord(StrictModel):
    chunk_id: str
    source_id: str
    title: str
    category: str
    authority_level: str
    published_at: date | None = None
    effective_at: date | None = None
    section: str = ""
    page: int | None = None
    text: str
    checksum: str
    sensitivity: Sensitivity
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceManifest(StrictModel):
    source_id: str
    relative_path: str
    title: str
    category: str
    authority_level: str
    published_at: date | None = None
    effective_at: date | None = None
    section: str = ""
    page: int | None = None
    checksum: str
    sensitivity: Sensitivity
    status: IngestionStatus
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    parser: str = ""
    chunk_count: int = Field(default=0, ge=0)
    privacy: PrivacyFindingSummary = Field(default_factory=PrivacyFindingSummary)
    warnings: list[str] = Field(default_factory=list)


class IngestionResult(StrictModel):
    manifest: SourceManifest
    chunks: list[ChunkRecord] = Field(default_factory=list)
    extracted_output: str | None = None
