from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.base import RetrievalScope


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PracticeRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    topic: str = Field(min_length=2, max_length=256)
    direction: str | None = Field(default=None, max_length=128)
    difficulty: Difficulty = Difficulty.MEDIUM
    count: int = Field(default=1, ge=1, le=5)
    scope: RetrievalScope = RetrievalScope.GENERIC


class RubricCriterion(StrictModel):
    criterion_id: str
    label: str
    description: str
    max_score: int = Field(ge=1, le=100)


class PracticeQuestion(StrictModel):
    question_id: str
    category: str
    tags: list[str]
    difficulty: Difficulty
    question: str
    expected_points: list[str]
    rubric: list[RubricCriterion]
    source_ids: list[str]
    source_chunk_ids: list[str]
    created_by: str
    version: str
    created_at: datetime

    @field_validator("rubric")
    @classmethod
    def rubric_totals_one_hundred(cls, values: list[RubricCriterion]) -> list[RubricCriterion]:
        if sum(item.max_score for item in values) != 100:
            raise ValueError("rubric scores must total 100")
        return values


class PracticeQuestionSet(StrictModel):
    backend: str
    questions: list[PracticeQuestion]
    warnings: list[str] = Field(default_factory=list)
    trace_id: str
