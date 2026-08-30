from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.base import DocumentInput, RagBackend, RetrievalRequest


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalChallengeType(StrEnum):
    STANDARD = "standard"
    PARAPHRASE = "paraphrase"
    LEXICAL_DISTRACTOR = "lexical_distractor"
    MULTI_EVIDENCE = "multi_evidence"
    CONFLICT = "conflict"
    TEMPORAL = "temporal"
    UNANSWERABLE = "unanswerable"


class RetrievalBenchmarkCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    question: str = Field(min_length=1, max_length=4096)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    split: Literal["dev", "test"] = "test"
    challenge_type: RetrievalChallengeType = RetrievalChallengeType.STANDARD
    reference_points: list[str] = Field(default_factory=list, max_length=12)


class RetrievalCaseResult(StrictModel):
    case_id: str
    retrieved_chunk_ids: list[str]
    recall: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    no_answer_correct: bool | None = None
    elapsed_ms: float = Field(ge=0.0)
    challenge_type: RetrievalChallengeType = RetrievalChallengeType.STANDARD


class RetrievalSliceMetrics(StrictModel):
    query_count: int = Field(ge=1)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    no_answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalBenchmarkReport(StrictModel):
    benchmark_version: str = "public-synthetic-v1"
    backend: str
    backend_version: str
    top_k: int = Field(ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0)
    query_count: int = Field(ge=1)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    no_answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    average_latency_ms: float = Field(ge=0.0)
    cases: list[RetrievalCaseResult]
    challenge_metrics: dict[str, RetrievalSliceMetrics] = Field(default_factory=dict)


class ThresholdCalibration(StrictModel):
    backend: str
    backend_version: str
    top_k: int = Field(ge=1)
    score_threshold: float = Field(ge=0.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    answerable_recall: float = Field(ge=0.0, le=1.0)
    no_answer_accuracy: float = Field(ge=0.0, le=1.0)
    dev_count: int = Field(ge=2)


def load_benchmark_cases(path: Path) -> list[RetrievalBenchmarkCase]:
    cases: list[RetrievalBenchmarkCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            cases.append(RetrievalBenchmarkCase.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid benchmark case at line {line_number}") from exc
    if not cases:
        raise ValueError("benchmark must contain at least one case")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("benchmark case_id values must be unique")
    for case in cases:
        if case.answerable and not case.relevant_chunk_ids:
            raise ValueError(f"answerable case {case.case_id} needs relevant_chunk_ids")
        if not case.answerable and case.relevant_chunk_ids:
            raise ValueError(f"unanswerable case {case.case_id} cannot have relevant_chunk_ids")
        if not case.answerable and case.reference_points:
            raise ValueError(f"unanswerable case {case.case_id} cannot have reference_points")
    return cases


def validate_benchmark_gold(
    documents: list[DocumentInput], cases: list[RetrievalBenchmarkCase]
) -> None:
    chunk_ids = [document.chunk_id for document in documents]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("corpus chunk_id values must be unique")
    known = set(chunk_ids)
    missing = sorted(
        {
            chunk_id
            for case in cases
            for chunk_id in case.relevant_chunk_ids
            if chunk_id not in known
        }
    )
    if missing:
        raise ValueError("benchmark contains missing gold chunk IDs")


def _ndcg(retrieved: list[str], relevant: set[str], top_k: int) -> float:
    dcg = sum(
        (1.0 if chunk_id in relevant else 0.0) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:top_k], start=1)
    )
    ideal_count = min(len(relevant), top_k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(
    backend: RagBackend,
    cases: list[RetrievalBenchmarkCase],
    *,
    top_k: int = 5,
    score_threshold: float | None = None,
    benchmark_version: str = "public-synthetic-v1",
) -> RetrievalBenchmarkReport:
    if not cases:
        raise ValueError("benchmark must contain at least one case")

    case_results: list[RetrievalCaseResult] = []
    answerable_results: list[RetrievalCaseResult] = []
    unanswerable_results: list[RetrievalCaseResult] = []
    for case in cases:
        retrieval = backend.retrieve(RetrievalRequest(query=case.question, top_k=top_k))
        selected_documents = [
            document
            for document in retrieval.documents
            if score_threshold is None or document.score >= score_threshold
        ]
        retrieved_ids = [document.chunk_id for document in selected_documents]
        relevant = set(case.relevant_chunk_ids)
        if case.answerable:
            matched = relevant.intersection(retrieved_ids[:top_k])
            recall = len(matched) / len(relevant)
            first_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant),
                None,
            )
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
            ndcg = _ndcg(retrieved_ids, relevant, top_k)
            no_answer_correct = None
        else:
            recall = 0.0
            reciprocal_rank = 0.0
            ndcg = 0.0
            no_answer_correct = not retrieved_ids
        result = RetrievalCaseResult(
            case_id=case.case_id,
            retrieved_chunk_ids=retrieved_ids,
            recall=round(recall, 6),
            reciprocal_rank=round(reciprocal_rank, 6),
            ndcg=round(ndcg, 6),
            no_answer_correct=no_answer_correct,
            elapsed_ms=retrieval.retrieval.elapsed_ms,
            challenge_type=case.challenge_type,
        )
        case_results.append(result)
        (answerable_results if case.answerable else unanswerable_results).append(result)

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def slice_metrics(results: list[RetrievalCaseResult]) -> RetrievalSliceMetrics:
        answerable = [item for item in results if item.no_answer_correct is None]
        unanswerable = [item for item in results if item.no_answer_correct is not None]
        return RetrievalSliceMetrics(
            query_count=len(results),
            answerable_count=len(answerable),
            unanswerable_count=len(unanswerable),
            recall_at_k=(average([item.recall for item in answerable]) if answerable else None),
            mrr=(average([item.reciprocal_rank for item in answerable]) if answerable else None),
            ndcg_at_k=(average([item.ndcg for item in answerable]) if answerable else None),
            no_answer_accuracy=(
                average([float(bool(item.no_answer_correct)) for item in unanswerable])
                if unanswerable
                else None
            ),
        )

    challenge_metrics = {
        challenge.value: slice_metrics(
            [item for item in case_results if item.challenge_type == challenge]
        )
        for challenge in RetrievalChallengeType
        if any(item.challenge_type == challenge for item in case_results)
    }

    backend_info = backend.backend_info()
    return RetrievalBenchmarkReport(
        benchmark_version=benchmark_version,
        backend=backend_info.backend,
        backend_version=backend_info.version,
        top_k=top_k,
        score_threshold=score_threshold,
        query_count=len(cases),
        answerable_count=len(answerable_results),
        unanswerable_count=len(unanswerable_results),
        recall_at_k=round(average([item.recall for item in answerable_results]), 6),
        mrr=round(average([item.reciprocal_rank for item in answerable_results]), 6),
        ndcg_at_k=round(average([item.ndcg for item in answerable_results]), 6),
        no_answer_accuracy=(
            round(
                average([float(bool(item.no_answer_correct)) for item in unanswerable_results]),
                6,
            )
            if unanswerable_results
            else None
        ),
        average_latency_ms=round(average([item.elapsed_ms for item in case_results]), 6),
        cases=case_results,
        challenge_metrics=challenge_metrics,
    )


def calibrate_score_threshold(
    backend: RagBackend,
    cases: list[RetrievalBenchmarkCase],
    *,
    top_k: int = 5,
) -> ThresholdCalibration:
    if not cases:
        raise ValueError("calibration needs at least one case")
    if not any(case.answerable for case in cases) or not any(not case.answerable for case in cases):
        raise ValueError("calibration needs answerable and unanswerable cases")

    observations: list[tuple[bool, float]] = []
    for case in cases:
        retrieval = backend.retrieve(RetrievalRequest(query=case.question, top_k=top_k))
        top_score = retrieval.documents[0].score if retrieval.documents else 0.0
        observations.append((case.answerable, top_score))

    candidates = {0.0}
    for _, score in observations:
        candidates.add(score)
        candidates.add(math.nextafter(score, math.inf))

    best: tuple[float, float, float, float] | None = None
    for threshold in sorted(candidates):
        answerable = [score >= threshold for expected, score in observations if expected]
        unanswerable = [score < threshold for expected, score in observations if not expected]
        answerable_recall = sum(answerable) / len(answerable)
        no_answer_accuracy = sum(unanswerable) / len(unanswerable)
        balanced_accuracy = (answerable_recall + no_answer_accuracy) / 2
        candidate = (balanced_accuracy, answerable_recall, -threshold, no_answer_accuracy)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    balanced_accuracy, answerable_recall, negative_threshold, no_answer_accuracy = best
    info = backend.backend_info()
    return ThresholdCalibration(
        backend=info.backend,
        backend_version=info.version,
        top_k=top_k,
        score_threshold=-negative_threshold,
        balanced_accuracy=round(balanced_accuracy, 6),
        answerable_recall=round(answerable_recall, 6),
        no_answer_accuracy=round(no_answer_accuracy, 6),
        dev_count=len(cases),
    )
