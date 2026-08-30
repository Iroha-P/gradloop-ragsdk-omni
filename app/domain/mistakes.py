from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.grading import GradeResult


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MasteryStatus(StrEnum):
    OPEN = "open"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class MistakeRecord(StrictModel):
    mistake_id: str
    user_id: str
    question_id: str
    attempt_id: str
    answer_version: int = Field(ge=1)
    score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    rubric_version: str
    error_types: list[str]
    weak_tags: list[str]
    source_ids: list[str]
    recommended_review: str
    next_review_date: date
    redo_count: int = Field(default=0, ge=0)
    latest_score: int | None = Field(default=None, ge=0, le=100)
    consecutive_passes: int = Field(default=0, ge=0)
    last_reviewed_at: datetime | None = None
    mastery_status: MasteryStatus = MasteryStatus.OPEN
    created_at: datetime
    updated_at: datetime


class MistakeRedoResult(StrictModel):
    grade: GradeResult
    mistake: MistakeRecord
    mastery_changed: bool
    next_action: str


class MistakeListRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    topic: str | None = Field(default=None, max_length=64)
    status: MasteryStatus | None = None
    limit: int = Field(default=50, ge=1, le=200)


class DeleteMistakeRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
