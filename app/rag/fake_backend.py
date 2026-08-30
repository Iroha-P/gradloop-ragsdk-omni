from __future__ import annotations

import time
import uuid

from .base import (
    AnswerRequest,
    AnswerResult,
    BackendHealth,
    BackendInfo,
    Citation,
    DeleteResult,
    DocumentInput,
    IndexResult,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    RetrievedDocument,
)
from .scoped_corpus import document_matches_scope


class FakeBackend:
    """Deterministic backend for unit tests and CI only."""

    name = "fake"
    version = "test-v1"

    def __init__(self, documents: list[DocumentInput] | None = None):
        self._documents = {document.chunk_id: document for document in documents or []}

    def index_documents(self, documents: list[DocumentInput]) -> IndexResult:
        for document in documents:
            self._documents[document.chunk_id] = document
        return IndexResult(backend=self.name, indexed_ids=[item.chunk_id for item in documents])

    def delete_documents(self, document_ids: list[str]) -> DeleteResult:
        deleted: list[str] = []
        missing: list[str] = []
        for document_id in document_ids:
            if self._documents.pop(document_id, None) is None:
                missing.append(document_id)
            else:
                deleted.append(document_id)
        return DeleteResult(backend=self.name, deleted_ids=deleted, missing_ids=missing)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        started = time.perf_counter()
        eligible = [
            item
            for item in self._documents.values()
            if document_matches_scope(item, request.scope)
        ]
        selected = eligible[: request.top_k]
        documents = [
            RetrievedDocument(
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                title=item.title,
                section=item.section,
                page=item.page,
                published_at=item.published_at,
                text=item.text,
                score=1.0 - index * 0.01,
                retrieval_type="fake",
                metadata=item.metadata,
            )
            for index, item in enumerate(selected)
        ]
        trace = RetrievalTrace(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            candidate_count=len(eligible),
            selected_count=len(documents),
            stages=[{"name": "fake_retrieve", "candidate_count": len(eligible)}],
        )
        return RetrievalResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            documents=documents,
            trace_id=uuid.uuid4().hex,
            retrieval=trace,
        )

    def answer(self, request: AnswerRequest) -> AnswerResult:
        result = self.retrieve(
            RetrievalRequest(
                query=request.query,
                top_k=request.top_k,
                scope=request.scope,
            )
        )
        if not result.documents:
            return AnswerResult(
                backend=self.name,
                backend_version=self.version,
                query=request.query,
                answer="测试后端没有可用证据。",
                trace_id=result.trace_id,
                retrieval=result.retrieval,
                abstained=True,
                warnings=["fake backend has no documents"],
            )
        first = result.documents[0]
        return AnswerResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            answer=f"测试回答：{first.text}",
            citations=[Citation(**first.model_dump(exclude={"text", "score", "retrieval_type", "metadata"}))],
            trace_id=result.trace_id,
            retrieval=result.retrieval,
        )

    def healthcheck(self) -> BackendHealth:
        return BackendHealth(backend=self.name, ready=True, checks={"deterministic": True})

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.name,
            version=self.version,
            capabilities=["index", "delete", "retrieve", "answer"],
            runtime_verified=True,
        )
