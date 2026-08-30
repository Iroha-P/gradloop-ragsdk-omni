from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalScope(StrEnum):
    GENERIC = "generic"
    PERSONAL = "personal"
    ALL = "all"


class DocumentInput(StrictModel):
    source_id: str = Field(min_length=1, max_length=256)
    chunk_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1_000_000)
    title: str = Field(default="", max_length=512)
    section: str = Field(default="", max_length=512)
    page: int | None = Field(default=None, ge=1)
    published_at: date | None = None
    authority_level: str = Field(default="unknown", max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResult(StrictModel):
    backend: str
    indexed_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DeleteResult(StrictModel):
    backend: str
    deleted_ids: list[str] = Field(default_factory=list)
    missing_ids: list[str] = Field(default_factory=list)


class RetrievalRequest(StrictModel):
    query: str = Field(min_length=1, max_length=32768)
    top_k: int = Field(default=5, ge=1, le=100)
    scope: RetrievalScope = RetrievalScope.GENERIC
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(StrictModel):
    source_id: str
    chunk_id: str
    title: str = ""
    section: str = ""
    page: int | None = None
    published_at: date | None = None
    text: str
    score: float
    retrieval_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(StrictModel):
    elapsed_ms: float = Field(ge=0)
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    stages: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalResult(StrictModel):
    backend: str
    backend_version: str
    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
    trace_id: str
    retrieval: RetrievalTrace
    warnings: list[str] = Field(default_factory=list)


class AnswerRequest(StrictModel):
    query: str = Field(min_length=1, max_length=32768)
    top_k: int = Field(default=5, ge=1, le=100)
    scope: RetrievalScope = RetrievalScope.GENERIC


class Citation(StrictModel):
    source_id: str
    chunk_id: str
    title: str = ""
    section: str = ""
    page: int | None = None
    published_at: date | None = None


class AnswerResult(StrictModel):
    backend: str
    backend_version: str
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str
    retrieval: RetrievalTrace
    abstained: bool = False
    warnings: list[str] = Field(default_factory=list)
    generation_backend: str = "none"
    generator_model: str | None = None
    llm_called: bool = False


class BackendHealth(StrictModel):
    backend: str
    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    message: str = ""


class BackendInfo(StrictModel):
    backend: str
    version: str
    capabilities: list[str] = Field(default_factory=list)
    runtime_verified: bool = False


@runtime_checkable
class RagBackend(Protocol):
    def index_documents(self, documents: list[DocumentInput]) -> IndexResult: ...

    def delete_documents(self, document_ids: list[str]) -> DeleteResult: ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...

    def answer(self, request: AnswerRequest) -> AnswerResult: ...

    def healthcheck(self) -> BackendHealth: ...

    def backend_info(self) -> BackendInfo: ...
