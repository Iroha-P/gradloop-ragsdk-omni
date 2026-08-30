from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StudyIntensity(StrEnum):
    GENTLE = "gentle"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


class PlanStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class StudyPlanRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    goal: str = Field(min_length=4, max_length=512)
    deadline: date
    weekly_hours: float = Field(gt=0, le=80)
    current_level: str = Field(default="beginner", max_length=64)
    weak_tags: list[str] = Field(default_factory=list, max_length=20)
    completed_tasks: list[str] = Field(default_factory=list, max_length=100)
    intensity: StudyIntensity = StudyIntensity.BALANCED
    days: Literal[7, 14, 30] = 7

    @field_validator("weak_tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:64] for value in values if value.strip()]
        return list(dict.fromkeys(cleaned))


class PlanTask(StrictModel):
    task_id: str
    day: int = Field(ge=1, le=30)
    title: str = Field(min_length=2, max_length=256)
    minutes: int = Field(ge=15, le=480)
    deliverable: str = Field(min_length=2, max_length=512)
    task_type: str = Field(max_length=64)
    source_ids: list[str] = Field(default_factory=list)
    due_date: date
    is_review: bool = False
    status: TaskStatus = TaskStatus.PENDING
    completed_at: datetime | None = None


class StudyPlan(StrictModel):
    plan_id: str
    user_id: str
    goal: str
    deadline: date
    weekly_hours: float
    current_level: str
    weak_tags: list[str]
    intensity: StudyIntensity
    days: int
    total_minutes: int
    available_minutes: int
    status: PlanStatus = PlanStatus.ACTIVE
    source_ids: list[str] = Field(default_factory=list)
    tasks: list[PlanTask]
    adjustment_triggers: list[str]
    created_at: datetime
    updated_at: datetime


class CompleteTaskRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")


class PlanRevisionStatus(StrEnum):
    PENDING = "pending_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class PlanRevisionProposal(StrictModel):
    revision_id: str
    plan_id: str
    user_id: str
    reason: str
    weak_tags: list[str]
    proposed_tasks: list[PlanTask]
    status: PlanRevisionStatus = PlanRevisionStatus.PENDING
    created_at: datetime
    confirmed_at: datetime | None = None


class RevisionDecisionRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    confirm: bool = True
