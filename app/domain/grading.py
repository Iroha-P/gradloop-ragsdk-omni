from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradeRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    question_id: str = Field(min_length=4, max_length=128)
    answer: str = Field(min_length=1, max_length=12000)
    use_llm_review: bool = True


class DimensionScore(StrictModel):
    criterion_id: str
    label: str
    score: int = Field(ge=0, le=100)
    max_score: int = Field(ge=1, le=100)
    evidence: str


class LlmReviewStatus(StrEnum):
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    INVALID_OUTPUT = "invalid_output"
    COMPLETED = "completed"


class LlmGradeReview(StrictModel):
    status: LlmReviewStatus
    attempts: int = Field(default=0, ge=0, le=2)
    failure_code: str | None = Field(default=None, max_length=64)
    grader_backend: str | None = None
    model: str | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    dimensions: list[DimensionScore] = Field(default_factory=list)
    matched_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    improved_answer: str | None = None
    next_action: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    score_delta: int | None = Field(default=None, ge=-100, le=100)
    elapsed_ms: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class GradeResult(StrictModel):
    attempt_id: str
    user_id: str
    question_id: str
    score: int = Field(ge=0, le=100)
    baseline_score: int = Field(ge=0, le=100)
    dimensions: list[DimensionScore]
    rubric_version: str
    matched_points: list[str]
    missing_points: list[str]
    evidence_source_ids: list[str]
    improved_answer: str
    next_action: str
    confidence: float = Field(ge=0, le=1)
    grader: str
    grading_policy: str
    llm_review: LlmGradeReview
    mistake_id: str | None = None
    revision_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
