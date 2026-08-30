from __future__ import annotations

import math
import re
import time
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.retrieval import RetrievalBenchmarkCase
from app.rag.base import AnswerResult

_NON_TEXT = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
_CITATION_MARKER = re.compile(r"\[S(\d+)\]", flags=re.IGNORECASE)
_ABSTENTION_PHRASES = (
    "资料不足",
    "无法回答",
    "无法确定",
    "不能确定",
    "没有足够资料",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerQualityStatus(StrEnum):
    COMPLETED = "completed"
    NOT_RUN = "not_run"
    FAILED = "failed"


class AnswerQualityMetrics(StrictModel):
    gold_citation_recall: float = Field(ge=0.0, le=1.0)
    gold_citation_precision: float = Field(ge=0.0, le=1.0)
    citation_marker_validity: float = Field(ge=0.0, le=1.0)
    lexical_point_coverage: float = Field(ge=0.0, le=1.0)
    no_answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    answerable_response_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_call_rate: float = Field(ge=0.0, le=1.0)
    gate_abstention_rate: float = Field(ge=0.0, le=1.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)


class AnswerQualityReport(StrictModel):
    report_version: str = "answer-quality-observable-v1"
    benchmark_version: str
    evaluation_split: Literal["dev", "test", "all"]
    status: AnswerQualityStatus
    query_count: int = Field(ge=0)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    generation_backend: str | None = None
    generator_model: str | None = None
    evidence_score_threshold: float | None = Field(default=None, ge=0.0)
    evidence_gate_policy_version: str | None = None
    metrics: AnswerQualityMetrics | None = None
    missing: list[str] = Field(default_factory=list)
    failure_type: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_payload(self) -> AnswerQualityReport:
        if self.status == AnswerQualityStatus.COMPLETED and self.metrics is None:
            raise ValueError("completed answer-quality report requires metrics")
        if self.status != AnswerQualityStatus.COMPLETED and self.metrics is not None:
            raise ValueError("non-completed answer-quality report forbids metrics")
        return self


class AnswerService(Protocol):
    def ask(self, question: str, *, top_k: int) -> AnswerResult: ...


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalized(value: str) -> str:
    return _NON_TEXT.sub("", value.lower())


def _cited_chunk_ids(result: AnswerResult) -> tuple[set[str], float]:
    marker_indexes = [int(value) for value in _CITATION_MARKER.findall(result.answer)]
    valid_indexes = [
        index for index in marker_indexes if 1 <= index <= len(result.citations)
    ]
    cited = {result.citations[index - 1].chunk_id for index in valid_indexes}
    validity = len(valid_indexes) / len(marker_indexes) if marker_indexes else 0.0
    return cited, validity


def _explicitly_abstained(result: AnswerResult) -> bool:
    normalized_answer = _normalized(result.answer)
    return result.abstained or any(
        _normalized(phrase) in normalized_answer for phrase in _ABSTENTION_PHRASES
    )


def evaluate_answer_quality(
    service: AnswerService,
    cases: list[RetrievalBenchmarkCase],
    *,
    benchmark_version: str,
    top_k: int,
    evidence_score_threshold: float | None = None,
    evidence_gate_policy_version: str | None = None,
) -> AnswerQualityReport:
    if not cases:
        raise ValueError("answer-quality evaluation needs at least one case")

    citation_recalls: list[float] = []
    citation_precisions: list[float] = []
    citation_marker_validities: list[float] = []
    point_coverages: list[float] = []
    no_answer_results: list[float] = []
    answerable_responses: list[float] = []
    llm_calls: list[float] = []
    gate_abstentions: list[float] = []
    latencies: list[float] = []
    generation_backends: set[str] = set()
    generator_models: set[str] = set()

    for case in cases:
        started = time.perf_counter()
        result = service.ask(case.question, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        generation_backends.add(result.generation_backend)
        if result.generator_model:
            generator_models.add(result.generator_model)
        gate_triggered = result.generation_backend == "evidence-gate"
        gate_abstentions.append(float(gate_triggered))
        llm_calls.append(float(result.llm_called))

        cited, marker_validity = _cited_chunk_ids(result)
        if case.answerable:
            gold = set(case.relevant_chunk_ids)
            citation_recalls.append(len(gold.intersection(cited)) / len(gold))
            marker_count = len(_CITATION_MARKER.findall(result.answer))
            citation_precisions.append(
                len(gold.intersection(cited)) / marker_count if marker_count else 0.0
            )
            citation_marker_validities.append(marker_validity)
            normalized_answer = _normalized(result.answer)
            point_coverages.append(
                _average(
                    [
                        float(_normalized(point) in normalized_answer)
                        for point in case.reference_points
                    ]
                )
                if case.reference_points
                else 0.0
            )
            answerable_responses.append(float(not _explicitly_abstained(result)))
        else:
            no_answer_results.append(float(_explicitly_abstained(result) and not cited))

    answerable_count = sum(case.answerable for case in cases)
    unanswerable_count = len(cases) - answerable_count
    case_splits = {case.split for case in cases}
    evaluation_split = next(iter(case_splits)) if len(case_splits) == 1 else "all"
    return AnswerQualityReport(
        benchmark_version=benchmark_version,
        evaluation_split=evaluation_split,
        status=AnswerQualityStatus.COMPLETED,
        query_count=len(cases),
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        top_k=top_k,
        generation_backend=(
            next(iter(generation_backends))
            if len(generation_backends) == 1
            else "mixed"
        ),
        generator_model=(next(iter(generator_models)) if len(generator_models) == 1 else "mixed"),
        evidence_score_threshold=evidence_score_threshold,
        evidence_gate_policy_version=evidence_gate_policy_version,
        metrics=AnswerQualityMetrics(
            gold_citation_recall=round(_average(citation_recalls), 6),
            gold_citation_precision=round(_average(citation_precisions), 6),
            citation_marker_validity=round(_average(citation_marker_validities), 6),
            lexical_point_coverage=round(_average(point_coverages), 6),
            no_answer_accuracy=(round(_average(no_answer_results), 6) if no_answer_results else None),
            answerable_response_rate=(
                round(_average(answerable_responses), 6)
                if answerable_responses
                else None
            ),
            llm_call_rate=round(_average(llm_calls), 6),
            gate_abstention_rate=round(_average(gate_abstentions), 6),
            latency_p50_ms=round(_percentile(latencies, 0.5), 6),
            latency_p95_ms=round(_percentile(latencies, 0.95), 6),
        ),
        limitations=[
            "Lexical point coverage is deterministic and does not replace human semantic grading.",
            "Citation metrics use explicit [S#] markers and do not prove full answer faithfulness.",
            "Synthetic public cases are not a real-user quality claim.",
        ],
    )


def build_not_run_answer_quality_report(
    *,
    benchmark_version: str,
    evaluation_split: Literal["dev", "test", "all"],
    top_k: int,
    missing: list[str],
    evidence_score_threshold: float | None = None,
    evidence_gate_policy_version: str | None = None,
) -> AnswerQualityReport:
    return AnswerQualityReport(
        benchmark_version=benchmark_version,
        evaluation_split=evaluation_split,
        status=AnswerQualityStatus.NOT_RUN,
        query_count=0,
        answerable_count=0,
        unanswerable_count=0,
        top_k=top_k,
        evidence_score_threshold=evidence_score_threshold,
        evidence_gate_policy_version=evidence_gate_policy_version,
        missing=sorted(set(missing)),
        limitations=["No model quality metrics are emitted when the local model is unavailable."],
    )
