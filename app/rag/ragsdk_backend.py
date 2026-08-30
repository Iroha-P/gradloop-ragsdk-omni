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
    RetrievalScope,
    RetrievalTrace,
    RetrievedDocument,
)
from .ragsdk_runtime import RagsdkHit, RagsdkRuntime


class UnsupportedRetrievalScope(ValueError):
    """RAGSDK has not been provisioned with isolated personal collections."""


class RagsdkBackend:
    name = "ragsdk"

    def __init__(self, runtime: RagsdkRuntime):
        self.runtime = runtime
        self.version = runtime.version

    def index_documents(self, documents: list[DocumentInput]) -> IndexResult:
        indexed, failed, warnings = self.runtime.index_documents(documents)
        return IndexResult(
            backend=self.name,
            indexed_ids=indexed,
            failed_ids=failed,
            warnings=warnings,
        )

    def delete_documents(self, document_ids: list[str]) -> DeleteResult:
        deleted, missing, _warnings = self.runtime.delete_documents(document_ids)
        return DeleteResult(backend=self.name, deleted_ids=deleted, missing_ids=missing)

    @staticmethod
    def _document(hit: RagsdkHit) -> RetrievedDocument:
        return RetrievedDocument(
            source_id=hit.source_id,
            chunk_id=hit.chunk_id,
            title=hit.title,
            section=hit.section,
            page=hit.page,
            published_at=hit.published_at,
            text=hit.text,
            score=hit.score,
            retrieval_type=hit.retrieval_type,
            metadata=hit.metadata,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.scope != RetrievalScope.GENERIC:
            raise UnsupportedRetrievalScope(
                "RAGSDK personal/all scope requires separately provisioned isolated collections"
            )
        started = time.perf_counter()
        response = self.runtime.retrieve(request.query, request.top_k, request.filters)
        documents = [self._document(hit) for hit in response.hits]
        trace = RetrievalTrace(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            candidate_count=(
                response.stages[0].candidate_count
                if response.stages
                else len(response.hits)
            ),
            selected_count=len(documents),
            stages=[
                {
                    "name": stage.name,
                    "candidate_count": stage.candidate_count,
                    "selected_count": stage.selected_count,
                    "elapsed_ms": stage.elapsed_ms,
                }
                for stage in response.stages
            ],
        )
        return RetrievalResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            documents=documents,
            trace_id=uuid.uuid4().hex,
            retrieval=trace,
            warnings=[] if documents else ["RAGSDK returned no matching evidence"],
        )

    def answer(self, request: AnswerRequest) -> AnswerResult:
        if request.scope != RetrievalScope.GENERIC:
            raise UnsupportedRetrievalScope(
                "RAGSDK personal/all scope requires separately provisioned isolated collections"
            )
        started = time.perf_counter()
        generated = self.runtime.generate(request.query, request.top_k)
        if generated is None:
            retrieval = self.retrieve(
                RetrievalRequest(
                    query=request.query,
                    top_k=request.top_k,
                    scope=request.scope,
                )
            )
            citations = [
                Citation(
                    source_id=item.source_id,
                    chunk_id=item.chunk_id,
                    title=item.title,
                    section=item.section,
                    page=item.page,
                    published_at=item.published_at,
                )
                for item in retrieval.documents
            ]
            generation_supported = "optional_generate" in self.runtime.capabilities
            if generation_supported:
                answer_text = "RAGSDK 已完成检索，但当前 profile 未配置 LLM，未生成回答。"
                generation_warning = "llm_not_configured"
            else:
                answer_text = (
                    "RAGSDK 已按所选模式完成检索；当前生成链仅支持 dense 模式，"
                    "为避免检索身份混淆，本次未生成回答。"
                )
                generation_warning = "generation_not_supported_for_retrieval_mode"
            return AnswerResult(
                backend=self.name,
                backend_version=self.version,
                query=request.query,
                answer=answer_text,
                citations=citations,
                trace_id=retrieval.trace_id,
                retrieval=retrieval.retrieval,
                abstained=True,
                warnings=[*retrieval.warnings, generation_warning],
            )

        documents = [self._document(hit) for hit in generated.hits]
        trace = RetrievalTrace(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            candidate_count=len(documents),
            selected_count=len(documents),
            stages=[{"name": "ragsdk_chain", "candidate_count": len(documents)}],
        )
        citations = [
            Citation(
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                title=item.title,
                section=item.section,
                page=item.page,
                published_at=item.published_at,
            )
            for item in documents
        ]
        return AnswerResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            answer=generated.text,
            citations=citations,
            trace_id=uuid.uuid4().hex,
            retrieval=trace,
            abstained=not bool(generated.text.strip()),
            warnings=[] if generated.text.strip() else ["RAGSDK chain returned an empty answer"],
        )

    def healthcheck(self) -> BackendHealth:
        checks, missing, message = self.runtime.healthcheck()
        return BackendHealth(
            backend=self.name,
            ready=not missing,
            checks=checks,
            missing=missing,
            message=message,
        )

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.name,
            version=self.version,
            capabilities=list(self.runtime.capabilities),
            runtime_verified=self.runtime.runtime_verified,
        )
