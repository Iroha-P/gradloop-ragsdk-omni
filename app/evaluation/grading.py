from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradingBenchmarkCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_-]{3,64}$")
    split: Literal["dev", "test"]
    topic: str = Field(min_length=2, max_length=128)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    answer: str = Field(min_length=1, max_length=12000)
    reference_score: int = Field(ge=0, le=100)
    challenge_tag: str = Field(min_length=2, max_length=64)


class GradingObservation(StrictModel):
    case_id: str
    reference_score: int = Field(ge=0, le=100)
    baseline_score: int = Field(ge=0, le=100)
    llm_score: int | None = Field(default=None, ge=0, le=100)
    llm_status: str
    llm_attempts: int = Field(default=0, ge=0, le=2)
    llm_failure_code: str | None = None


class ScoreMetrics(StrictModel):
    count: int = Field(ge=0)
    mean_absolute_error: float | None = Field(default=None, ge=0)
    mean_signed_error: float | None = None
    within_10_accuracy: float | None = Field(default=None, ge=0, le=1)
    band_accuracy: float | None = Field(default=None, ge=0, le=1)


class GradingBenchmarkReport(StrictModel):
    benchmark_version: str
    split: str
    case_count: int
    baseline_grader: str
    llm_backend: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    max_llm_attempts: int = Field(default=2, ge=1, le=5)
    baseline: ScoreMetrics
    llm_review: ScoreMetrics | None
    llm_review_coverage: float = Field(ge=0, le=1)
    llm_retry_case_count: int = Field(ge=0)
    average_llm_attempts: float = Field(ge=0, le=2)
    llm_status_counts: dict[str, int]
    observations: list[GradingObservation]
    warning: str


def load_grading_cases(path: Path) -> list[GradingBenchmarkCase]:
    cases: list[GradingBenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(GradingBenchmarkCase.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid grading benchmark line {line_number}") from exc
    if not cases:
        raise ValueError("grading benchmark is empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("grading benchmark case_id values must be unique")
    return cases


def _band(score: int) -> int:
    if score < 60:
        return 0
    if score < 80:
        return 1
    return 2


def score_metrics(pairs: list[tuple[int, int]]) -> ScoreMetrics:
    if not pairs:
        return ScoreMetrics(count=0)
    errors = [observed - reference for reference, observed in pairs]
    return ScoreMetrics(
        count=len(pairs),
        mean_absolute_error=round(sum(abs(error) for error in errors) / len(errors), 4),
        mean_signed_error=round(sum(errors) / len(errors), 4),
        within_10_accuracy=round(sum(abs(error) <= 10 for error in errors) / len(errors), 4),
        band_accuracy=round(
            sum(_band(reference) == _band(observed) for reference, observed in pairs) / len(pairs),
            4,
        ),
    )


def build_grading_report(
    *,
    benchmark_version: str,
    split: str,
    observations: list[GradingObservation],
    llm_backend: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> GradingBenchmarkReport:
    baseline_pairs = [(item.reference_score, item.baseline_score) for item in observations]
    llm_pairs = [
        (item.reference_score, item.llm_score)
        for item in observations
        if item.llm_status == "completed" and item.llm_score is not None
    ]
    coverage = len(llm_pairs) / len(observations) if observations else math.nan
    if math.isnan(coverage):
        coverage = 0.0
    return GradingBenchmarkReport(
        benchmark_version=benchmark_version,
        split=split,
        case_count=len(observations),
        baseline_grader="deterministic_multidimensional_v1",
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        baseline=score_metrics(baseline_pairs),
        llm_review=score_metrics(llm_pairs) if llm_pairs else None,
        llm_review_coverage=round(coverage, 4),
        llm_retry_case_count=sum(item.llm_attempts > 1 for item in observations),
        average_llm_attempts=round(
            sum(item.llm_attempts for item in observations) / len(observations),
            4,
        )
        if observations
        else 0.0,
        llm_status_counts=dict(Counter(item.llm_status for item in observations)),
        observations=observations,
        warning=(
            "Synthetic reference labels validate the evaluation pipeline only; "
            "they are not human inter-rater reliability or real-user grading accuracy."
        ),
    )
