from __future__ import annotations

import math
import re
import time
import uuid
from collections import Counter

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

_LATIN = re.compile(r"[a-zA-Z0-9_+.-]{2,}")
_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    terms = _LATIN.findall(normalized)
    for run in _CHINESE_RUN.findall(normalized):
        terms.extend(run)
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
        if 2 <= len(run) <= 8:
            terms.append(run)
    return terms


class Bm25BaselineBackend:
    """Small, model-free BM25 baseline used for comparison and CPU demos."""

    name = "baseline"
    version = "bm25-v1"

    def __init__(self, documents: list[DocumentInput] | None = None):
        self._documents: dict[str, DocumentInput] = {}
        self._terms: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 1.0
        if documents:
            self.index_documents(documents)

    def _rebuild_statistics(self) -> None:
        self._terms = {
            chunk_id: Counter(tokenize(f"{document.title} {document.section} {document.text}"))
            for chunk_id, document in self._documents.items()
        }
        self._document_frequency = Counter()
        for terms in self._terms.values():
            self._document_frequency.update(terms.keys())
        if self._terms:
            self._average_length = sum(
                sum(terms.values()) for terms in self._terms.values()
            ) / len(self._terms)
        else:
            self._average_length = 1.0

    def index_documents(self, documents: list[DocumentInput]) -> IndexResult:
        for document in documents:
            self._documents[document.chunk_id] = document
        self._rebuild_statistics()
        return IndexResult(backend=self.name, indexed_ids=[item.chunk_id for item in documents])

    def delete_documents(self, document_ids: list[str]) -> DeleteResult:
        deleted: list[str] = []
        missing: list[str] = []
        for document_id in document_ids:
            if self._documents.pop(document_id, None) is None:
                missing.append(document_id)
            else:
                deleted.append(document_id)
        self._rebuild_statistics()
        return DeleteResult(backend=self.name, deleted_ids=deleted, missing_ids=missing)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        started = time.perf_counter()
        query_terms = Counter(tokenize(request.query))
        scored: list[tuple[float, str]] = []
        eligible_ids = [
            chunk_id
            for chunk_id, document in self._documents.items()
            if document_matches_scope(document, request.scope)
        ]
        document_count = len(eligible_ids)
        document_frequency: Counter[str] = Counter()
        for chunk_id in eligible_ids:
            document_frequency.update(self._terms[chunk_id].keys())
        average_length = (
            sum(sum(self._terms[chunk_id].values()) for chunk_id in eligible_ids)
            / document_count
            if document_count
            else 1.0
        )
        k1 = 1.5
        b = 0.75

        for chunk_id in eligible_ids:
            terms = self._terms[chunk_id]
            document_length = sum(terms.values()) or 1
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = terms.get(term, 0)
                if not term_frequency:
                    continue
                term_document_frequency = document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1
                    + (document_count - term_document_frequency + 0.5)
                    / (term_document_frequency + 0.5)
                )
                denominator = term_frequency + k1 * (
                    1 - b + b * document_length / average_length
                )
                score += (
                    inverse_document_frequency
                    * term_frequency
                    * (k1 + 1)
                    / denominator
                    * (1 + math.log1p(query_frequency))
                )
            if score > 0:
                scored.append((score, chunk_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[: request.top_k]
        documents = []
        for score, chunk_id in selected:
            item = self._documents[chunk_id]
            documents.append(
                RetrievedDocument(
                    source_id=item.source_id,
                    chunk_id=item.chunk_id,
                    title=item.title,
                    section=item.section,
                    page=item.page,
                    published_at=item.published_at,
                    text=item.text,
                    score=round(score, 6),
                    retrieval_type="bm25",
                    metadata=item.metadata,
                )
            )

        trace = RetrievalTrace(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            candidate_count=len(scored),
            selected_count=len(documents),
            stages=[
                {
                    "name": "bm25",
                    "corpus_size": document_count,
                    "query_term_count": len(query_terms),
                    "candidate_count": len(scored),
                }
            ],
        )
        warnings = [] if documents else ["no matching evidence in baseline index"]
        return RetrievalResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            documents=documents,
            trace_id=uuid.uuid4().hex,
            retrieval=trace,
            warnings=warnings,
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
                answer="当前公开知识库没有足够证据支持回答。请补充资料；若涉及当年政策，请查阅目标院校官方通知。",
                trace_id=result.trace_id,
                retrieval=result.retrieval,
                abstained=True,
                warnings=result.warnings,
            )

        citations = [
            Citation(
                source_id=document.source_id,
                chunk_id=document.chunk_id,
                title=document.title,
                section=document.section,
                page=document.page,
                published_at=document.published_at,
            )
            for document in result.documents
        ]
        evidence = "\n".join(
            f"- [{index}] {document.text}" for index, document in enumerate(result.documents, start=1)
        )
        answer = (
            "结论：下面仅展示本地 BM25 基线命中的资料片段；基线没有调用大模型，不会补写资料外事实。\n\n"
            f"证据与来源：\n{evidence}\n\n"
            "匹配分析：需要结合个人目标、截止日期和已有证据进一步判断。\n\n"
            "下一步行动：如问题涉及具体院校或当年政策，请用有日期的官方来源再次核实。"
        )
        return AnswerResult(
            backend=self.name,
            backend_version=self.version,
            query=request.query,
            answer=answer,
            citations=citations,
            trace_id=result.trace_id,
            retrieval=result.retrieval,
            warnings=result.warnings,
        )

    def healthcheck(self) -> BackendHealth:
        ready = bool(self._documents)
        return BackendHealth(
            backend=self.name,
            ready=ready,
            checks={"in_memory_index": ready},
            missing=[] if ready else ["corpus"],
            message=f"{len(self._documents)} documents indexed",
        )

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.name,
            version=self.version,
            capabilities=["index", "delete", "retrieve", "extractive_answer"],
            runtime_verified=True,
        )
