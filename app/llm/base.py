from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmHealth(StrictModel):
    backend: str
    ready: bool
    model: str
    message: str


class LlmResponse(StrictModel):
    backend: str
    model: str
    content: str = Field(min_length=1)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)


@runtime_checkable
class LlmClient(Protocol):
    def healthcheck(self) -> LlmHealth: ...

    def chat(self, *, system: str, user: str) -> LlmResponse: ...
