from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evaluation.answer_quality import (
    AnswerQualityMetrics,
    AnswerQualityReport,
    AnswerQualityStatus,
    build_not_run_answer_quality_report,
)
from app.evaluation.retrieval import ThresholdCalibration

_PUBLIC_GENERATION_BACKENDS = frozenset({"evidence-gate", "llama.cpp", "mixed", "none"})
_PUBLIC_QWEN_MODEL = re.compile(
    r"^(?:qwen/)?qwen[0-9a-z][0-9a-z_.-]{0,95}$",
    flags=re.IGNORECASE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceGateAbDelta(StrictModel):
    no_answer_accuracy: float | None
    answerable_response_rate: float | None
    gold_citation_precision: float
    lexical_point_coverage: float
    llm_call_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


class EvidenceGateAbReport(StrictModel):
    report_version: str = "evidence-gate-ab-v1"
    benchmark_version: str
    status: AnswerQualityStatus
    calibration: ThresholdCalibration | None
    control: AnswerQualityReport
    treatment: AnswerQualityReport
    delta: EvidenceGateAbDelta | None
    balanced_acceptance_passed: bool | None
    missing: list[str] = Field(default_factory=list)
    failure_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    limitations: list[str]

    @field_validator("control", "treatment")
    @classmethod
    def sanitize_child_generation_metadata(
        cls,
        report: AnswerQualityReport,
    ) -> AnswerQualityReport:
        generation_backend = report.generation_backend
        if generation_backend is not None and generation_backend not in _PUBLIC_GENERATION_BACKENDS:
            generation_backend = "local-llm"

        generator_model = report.generator_model
        if generator_model not in {None, "mixed"} and not _PUBLIC_QWEN_MODEL.fullmatch(generator_model):
            generator_model = "local-model"

        return report.model_copy(
            update={
                "generation_backend": generation_backend,
                "generator_model": generator_model,
            }
        )

    @field_validator("missing", mode="before")
    @classmethod
    def deduplicate_missing(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_status_payload(self) -> EvidenceGateAbReport:
        if self.status == AnswerQualityStatus.COMPLETED:
            if self.missing or self.failure_type is not None:
                raise ValueError("completed evidence-gate A/B report forbids status metadata")
            if self.calibration is None:
                raise ValueError("completed evidence-gate A/B report requires calibration")
            if (
                self.control.status != AnswerQualityStatus.COMPLETED
                or self.treatment.status != AnswerQualityStatus.COMPLETED
                or self.control.metrics is None
                or self.treatment.metrics is None
            ):
                raise ValueError("completed evidence-gate A/B report requires completed child metrics")
            if not (
                self.benchmark_version
                == self.control.benchmark_version
                == self.treatment.benchmark_version
            ):
                raise ValueError("completed evidence-gate A/B report requires matching benchmark versions")
            if (
                self.control.evaluation_split != "test"
                or self.treatment.evaluation_split != "test"
            ):
                raise ValueError("completed evidence-gate A/B report requires the test split")
            child_dimensions = (
                "top_k",
                "query_count",
                "answerable_count",
                "unanswerable_count",
            )
            if any(
                getattr(self.control, field) != getattr(self.treatment, field)
                for field in child_dimensions
            ):
                raise ValueError("completed evidence-gate A/B report requires matching Top-K and counts")
            if self.calibration.top_k != self.control.top_k:
                raise ValueError("completed evidence-gate A/B report requires matching calibration Top-K")
            if any(
                child.query_count != child.answerable_count + child.unanswerable_count
                for child in (self.control, self.treatment)
            ):
                raise ValueError("completed evidence-gate A/B report requires valid child query count totals")
            if (
                self.control.evidence_score_threshold is not None
                or self.control.evidence_gate_policy_version is not None
            ):
                raise ValueError("completed control forbids evidence-gate metadata")
            if (
                self.treatment.evidence_score_threshold != self.calibration.score_threshold
                or self.treatment.evidence_gate_policy_version != "bm25-top1-threshold-v1"
            ):
                raise ValueError("completed treatment evidence-gate metadata is inconsistent")
            if self.delta is None:
                raise ValueError("completed evidence-gate A/B report requires delta")
            if self.balanced_acceptance_passed is None:
                raise ValueError("completed evidence-gate A/B report requires acceptance result")
            expected_delta = _build_delta(self.control.metrics, self.treatment.metrics)
            if self.delta != expected_delta:
                raise ValueError("completed evidence-gate A/B report delta is inconsistent")
            expected_acceptance = _balanced_acceptance(self.treatment.metrics)
            if self.balanced_acceptance_passed != expected_acceptance:
                raise ValueError("completed evidence-gate A/B report acceptance is inconsistent")
            return self

        if self.status == AnswerQualityStatus.NOT_RUN:
            if not self.missing:
                raise ValueError("not-run evidence-gate A/B report requires missing metadata")
            if self.failure_type is not None:
                raise ValueError("not-run evidence-gate A/B report forbids failure metadata")
        elif self.status == AnswerQualityStatus.FAILED:
            if self.failure_type is None:
                raise ValueError("failed evidence-gate A/B report requires failure metadata")
            if self.missing:
                raise ValueError("failed evidence-gate A/B report forbids missing metadata")

        if (
            self.calibration is not None
            or self.delta is not None
            or self.balanced_acceptance_passed is not None
        ):
            raise ValueError("non-completed evidence-gate A/B report forbids calibration and results")
        if self.control.metrics is not None or self.treatment.metrics is not None:
            raise ValueError("non-completed evidence-gate A/B report forbids child metrics")
        if self.control.status != self.status or self.treatment.status != self.status:
            raise ValueError("non-completed evidence-gate A/B report requires matching child statuses")
        return self


def _delta(treatment: float | None, control: float | None) -> float | None:
    if treatment is None or control is None:
        return None
    return round(treatment - control, 6)


def _build_delta(
    control: AnswerQualityMetrics,
    treatment: AnswerQualityMetrics,
) -> EvidenceGateAbDelta:
    return EvidenceGateAbDelta(
        no_answer_accuracy=_delta(
            treatment.no_answer_accuracy,
            control.no_answer_accuracy,
        ),
        answerable_response_rate=_delta(
            treatment.answerable_response_rate,
            control.answerable_response_rate,
        ),
        gold_citation_precision=round(
            treatment.gold_citation_precision - control.gold_citation_precision,
            6,
        ),
        lexical_point_coverage=round(
            treatment.lexical_point_coverage - control.lexical_point_coverage,
            6,
        ),
        llm_call_rate=round(treatment.llm_call_rate - control.llm_call_rate, 6),
        latency_p50_ms=round(treatment.latency_p50_ms - control.latency_p50_ms, 6),
        latency_p95_ms=round(treatment.latency_p95_ms - control.latency_p95_ms, 6),
    )


def _balanced_acceptance(treatment: AnswerQualityMetrics) -> bool:
    return (
        treatment.no_answer_accuracy is not None
        and treatment.answerable_response_rate is not None
        and treatment.no_answer_accuracy >= 0.50
        and treatment.answerable_response_rate >= 0.60
    )


def _limitations() -> list[str]:
    return [
        "The dataset is public synthetic and does not represent real-user quality.",
        "Observability metrics do not replace human semantic grading.",
        "The test split is not used for calibration.",
    ]


def build_completed_evidence_gate_ab_report(
    *,
    benchmark_version: str,
    calibration: ThresholdCalibration,
    control: AnswerQualityReport,
    treatment: AnswerQualityReport,
) -> EvidenceGateAbReport:
    if control.metrics is None or treatment.metrics is None:
        raise ValueError("completed evidence-gate A/B report requires child metrics")

    control_metrics = control.metrics
    treatment_metrics = treatment.metrics
    return EvidenceGateAbReport(
        benchmark_version=benchmark_version,
        status=AnswerQualityStatus.COMPLETED,
        calibration=calibration,
        control=control,
        treatment=treatment,
        delta=_build_delta(control_metrics, treatment_metrics),
        balanced_acceptance_passed=_balanced_acceptance(treatment_metrics),
        limitations=_limitations(),
    )


def build_not_run_evidence_gate_ab_report(
    *,
    benchmark_version: str,
    top_k: int,
    missing: list[str],
) -> EvidenceGateAbReport:
    control = build_not_run_answer_quality_report(
        benchmark_version=benchmark_version,
        evaluation_split="test",
        top_k=top_k,
        missing=missing,
    )
    treatment = build_not_run_answer_quality_report(
        benchmark_version=benchmark_version,
        evaluation_split="test",
        top_k=top_k,
        missing=missing,
        evidence_gate_policy_version="bm25-top1-threshold-v1",
    )
    return EvidenceGateAbReport(
        benchmark_version=benchmark_version,
        status=AnswerQualityStatus.NOT_RUN,
        calibration=None,
        control=control,
        treatment=treatment,
        delta=None,
        balanced_acceptance_passed=None,
        missing=missing,
        limitations=_limitations(),
    )


def build_failed_evidence_gate_ab_report(
    *,
    benchmark_version: str,
    top_k: int,
    failure_type: str,
) -> EvidenceGateAbReport:
    child_payload = {
        "benchmark_version": benchmark_version,
        "evaluation_split": "test",
        "status": AnswerQualityStatus.FAILED,
        "query_count": 0,
        "answerable_count": 0,
        "unanswerable_count": 0,
        "top_k": top_k,
        "failure_type": failure_type,
        "limitations": ["Failure reports omit exception messages and raw cases to prevent data leakage."],
    }
    return EvidenceGateAbReport(
        benchmark_version=benchmark_version,
        status=AnswerQualityStatus.FAILED,
        calibration=None,
        control=AnswerQualityReport(**child_payload),
        treatment=AnswerQualityReport(
            **child_payload,
            evidence_gate_policy_version="bm25-top1-threshold-v1",
        ),
        delta=None,
        balanced_acceptance_passed=None,
        failure_type=failure_type,
        limitations=[
            *_limitations(),
            "Failure reports omit exception messages and raw cases to prevent data leakage.",
        ],
    )
