from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.rag.base import RetrievalScope


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    question: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    scope: RetrievalScope = RetrievalScope.GENERIC


class EvaluationRequest(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
