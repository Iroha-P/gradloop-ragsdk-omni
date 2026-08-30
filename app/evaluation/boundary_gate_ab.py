from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.evaluation.answer_quality import (
    AnswerQualityMetrics,
    AnswerQualityReport,
    AnswerQualityStatus,
)
from app.evaluation.boundary_gate import BoundaryGateDevReport
from app.rag.base import StrictModel
from app.services.boundary_gate import CompositeBoundaryGatePolicy

REPORT_VERSION = "boundary-gate-ab-v2"
BENCHMARK_VERSION = "public-synthetic-hard-v1"
DATASET_VERSION = "public-synthetic-boundary-dev-v2"
DATASET_SHA256 = "DE28237389EA7BD0980E966FED0A301B212A020A5345080E282E0EF270BD7F77"
CORPUS_SHA256 = "0B05F6EDE8DAC2C1E3C62BCFF0BDFB403DAE19053E9416C0718991A01F6F918F"
TOKENIZER_VERSION = "bm25-boundary-tokenizer-v2"
FEATURE_VERSION = "bm25-boundary-features-v2"
POLICY_VERSION = "bm25-composite-boundary-gate-v2"

_PUBLIC_GENERATION_BACKENDS = frozenset(
    {"llama.cpp", "boundary-gate", "mixed", "none", "local-llm"}
)
_PUBLIC_QWEN_MODEL = re.compile(
    r"^(?:qwen/)?qwen[0-9a-z][0-9a-z_.-]{0,95}$",
    flags=re.IGNORECASE,
)
_SHA256_PATTERN = r"^[A-F0-9]{64}$"
_FAILURE_TYPE_PATTERN = r"^[A-Za-z][A-Za-z0-9]*(?:Error|Exception)$"
_COMPLETED_CHILD_LIMITATIONS = [
    "Lexical point coverage is deterministic and does not replace human semantic grading.",
    "Citation metrics use explicit [S#] markers and do not prove full answer faithfulness.",
    "Synthetic public cases are not a real-user quality claim.",
]
_NOT_RUN_CHILD_LIMITATIONS = [
    "No model quality metrics are emitted when the local model is unavailable."
]
_FAILED_CHILD_LIMITATIONS = [
    "Failure reports omit exception messages and raw cases to prevent data leakage."
]
_REPORT_LIMITATIONS = [
    "public_synthetic_test_only",
    "observability_does_not_replace_human_grading",
    "test_split_not_used_for_calibration",
]


class BoundaryGateAbDelta(StrictModel):
    no_answer_accuracy: float = Field(allow_inf_nan=False)
    answerable_response_rate: float = Field(allow_inf_nan=False)
    gold_citation_precision: float = Field(allow_inf_nan=False)
    lexical_point_coverage: float = Field(allow_inf_nan=False)
    llm_call_rate: float = Field(allow_inf_nan=False)
    gate_abstention_rate: float = Field(allow_inf_nan=False)
    latency_p50_ms: float = Field(allow_inf_nan=False)
    latency_p95_ms: float = Field(allow_inf_nan=False)


class BoundaryGateAbReport(StrictModel):
    report_version: Literal["boundary-gate-ab-v2"] = REPORT_VERSION
    benchmark_version: Literal["public-synthetic-hard-v1"] = BENCHMARK_VERSION
    evaluation_split: Literal["test"] = "test"
    status: AnswerQualityStatus
    dev_report_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    dev_dataset_version: Literal[
        "public-synthetic-boundary-dev-v2"
    ] = DATASET_VERSION
    dev_dataset_sha256: Literal[
        "DE28237389EA7BD0980E966FED0A301B212A020A5345080E282E0EF270BD7F77"
    ] = DATASET_SHA256
    corpus_sha256: Literal[
        "0B05F6EDE8DAC2C1E3C62BCFF0BDFB403DAE19053E9416C0718991A01F6F918F"
    ] = CORPUS_SHA256
    tokenizer_version: Literal[
        "bm25-boundary-tokenizer-v2"
    ] = TOKENIZER_VERSION
    feature_version: Literal["bm25-boundary-features-v2"] = FEATURE_VERSION
    backend: Literal["baseline"] = "baseline"
    backend_version: Literal["bm25-v1"] = "bm25-v1"
    top_k: Literal[5] = 5
    query_count: Literal[26] = 26
    answerable_count: Literal[20] = 20
    unanswerable_count: Literal[6] = 6
    temperature: Literal[0.0] = 0.0
    parallelism: Literal[1] = 1
    treatment_policy_version: Literal[
        "bm25-composite-boundary-gate-v2"
    ] = POLICY_VERSION
    composite_policy: CompositeBoundaryGatePolicy | None
    composite_policy_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    control: AnswerQualityReport
    treatment: AnswerQualityReport
    delta: BoundaryGateAbDelta | None
    balanced_acceptance_passed: bool | None
    missing: list[Literal["healthy local language model"]] = Field(
        default_factory=list
    )
    failure_type: str | None = Field(
        default=None,
        max_length=128,
        pattern=_FAILURE_TYPE_PATTERN,
    )
    limitations: list[
        Literal[
            "public_synthetic_test_only",
            "observability_does_not_replace_human_grading",
            "test_split_not_used_for_calibration",
        ]
    ] = Field(default_factory=lambda: list(_REPORT_LIMITATIONS))

    @field_validator("control", "treatment", mode="before")
    @classmethod
    def sanitize_child_generation_metadata(
        cls,
        value: AnswerQualityReport | dict[str, object],
    ) -> AnswerQualityReport:
        report = AnswerQualityReport.model_validate(
            value.model_dump(mode="json")
            if isinstance(value, AnswerQualityReport)
            else value
        )
        backend = report.generation_backend
        if backend is not None and backend not in _PUBLIC_GENERATION_BACKENDS:
            backend = "local-llm"
        model = report.generator_model
        if model not in {None, "mixed", "none", "local-model"} and not (
            isinstance(model, str) and _PUBLIC_QWEN_MODEL.fullmatch(model)
        ):
            model = "local-model"
        return report.model_copy(
            update={"generation_backend": backend, "generator_model": model}
        )

    @field_validator("missing", mode="before")
    @classmethod
    def canonical_missing(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_report(self) -> BoundaryGateAbReport:
        if self.limitations != _REPORT_LIMITATIONS:
            raise ValueError("report limitations must use the fixed aggregate set")
        if self.query_count != self.answerable_count + self.unanswerable_count:
            raise ValueError("parent count total is inconsistent")

        if self.composite_policy is None:
            if self.composite_policy_sha256 is not None:
                raise ValueError("policy hash requires composite thresholds")
        else:
            expected_policy_hash = _policy_sha256(self.composite_policy)
            if self.composite_policy_sha256 != expected_policy_hash:
                raise ValueError("composite threshold binding is inconsistent")

        self._validate_child_protocol(self.control, treatment=False)
        self._validate_child_protocol(self.treatment, treatment=True)

        if self.status == AnswerQualityStatus.COMPLETED:
            self._validate_completed()
        elif self.status == AnswerQualityStatus.NOT_RUN:
            self._validate_not_run()
        else:
            self._validate_failed()
        return self

    def _validate_child_protocol(
        self,
        child: AnswerQualityReport,
        *,
        treatment: bool,
    ) -> None:
        if child.benchmark_version != self.benchmark_version:
            raise ValueError("child benchmark mismatch")
        if child.evaluation_split != "test" or child.top_k != self.top_k:
            raise ValueError("child test/Top-K protocol mismatch")
        if child.evidence_score_threshold is not None:
            raise ValueError("v2 children forbid the historical score threshold")
        if treatment:
            if child.evidence_gate_policy_version != self.treatment_policy_version:
                raise ValueError("Treatment policy metadata mismatch")
        elif child.evidence_gate_policy_version is not None:
            raise ValueError("Control forbids gate metadata")
        if not treatment and child.generation_backend == "boundary-gate":
            raise ValueError("Control cannot identify a gate-only generation backend")

    def _validate_completed(self) -> None:
        if (
            self.dev_report_sha256 is None
            or self.composite_policy is None
            or self.composite_policy_sha256 is None
        ):
            raise ValueError("completed report requires frozen dev binding")
        if self.missing or self.failure_type is not None:
            raise ValueError("completed report forbids status metadata")
        for child in (self.control, self.treatment):
            if child.status != AnswerQualityStatus.COMPLETED or child.metrics is None:
                raise ValueError("completed report requires completed child metrics")
            if (
                child.metrics.no_answer_accuracy is None
                or child.metrics.answerable_response_rate is None
            ):
                raise ValueError(
                    "completed report requires both class metrics for both arms"
                )
            if (
                child.query_count,
                child.answerable_count,
                child.unanswerable_count,
            ) != (self.query_count, self.answerable_count, self.unanswerable_count):
                raise ValueError("completed child counts mismatch")
            if child.query_count != child.answerable_count + child.unanswerable_count:
                raise ValueError("completed child count total is inconsistent")
            if child.missing or child.failure_type is not None:
                raise ValueError("completed child forbids status metadata")
            if child.limitations != _COMPLETED_CHILD_LIMITATIONS:
                raise ValueError("completed child limitations are not canonical")
        if self.treatment.generation_backend not in {"boundary-gate", "mixed"}:
            raise ValueError("Treatment must preserve v2 gate generation metadata")
        if self.delta is None or self.balanced_acceptance_passed is None:
            raise ValueError("completed report requires deltas and acceptance")
        assert self.control.metrics is not None
        assert self.treatment.metrics is not None
        expected_delta = _build_delta(self.control.metrics, self.treatment.metrics)
        if self.delta != expected_delta:
            raise ValueError("completed report delta is inconsistent")
        expected_acceptance = _balanced_acceptance(self.treatment.metrics)
        if self.balanced_acceptance_passed is not expected_acceptance:
            raise ValueError("completed report acceptance is inconsistent")

    def _validate_non_completed_children(
        self,
        expected_status: AnswerQualityStatus,
        expected_limitations: list[str],
    ) -> None:
        for child in (self.control, self.treatment):
            if child.status != expected_status:
                raise ValueError("non-completed child status mismatch")
            if child.metrics is not None:
                raise ValueError("non-completed children forbid metrics")
            if (
                child.query_count,
                child.answerable_count,
                child.unanswerable_count,
            ) != (0, 0, 0):
                raise ValueError("non-completed child counts must be zero")
            if child.limitations != expected_limitations:
                raise ValueError("non-completed child limitations are not canonical")
        if self.delta is not None or self.balanced_acceptance_passed is not None:
            raise ValueError("non-completed reports forbid metrics and acceptance")

    def _validate_not_run(self) -> None:
        if (
            self.dev_report_sha256 is None
            or self.composite_policy is None
            or self.composite_policy_sha256 is None
        ):
            raise ValueError("not-run report requires frozen dev binding")
        if not self.missing or self.failure_type is not None:
            raise ValueError("not-run report requires only safe missing metadata")
        self._validate_non_completed_children(
            AnswerQualityStatus.NOT_RUN,
            _NOT_RUN_CHILD_LIMITATIONS,
        )
        for child in (self.control, self.treatment):
            if child.missing != self.missing or child.failure_type is not None:
                raise ValueError("not-run child metadata mismatch")

    def _validate_failed(self) -> None:
        if self.failure_type is None or self.missing:
            raise ValueError("failed report requires only a redacted failure type")
        if self.composite_policy is not None and self.dev_report_sha256 is None:
            raise ValueError("failed report policy requires a dev-report hash")
        self._validate_non_completed_children(
            AnswerQualityStatus.FAILED,
            _FAILED_CHILD_LIMITATIONS,
        )
        for child in (self.control, self.treatment):
            if child.failure_type != self.failure_type or child.missing:
                raise ValueError("failed child metadata mismatch")


def canonical_dev_report_bytes(report: BoundaryGateDevReport) -> bytes:
    validated = BoundaryGateDevReport.model_validate(report.model_dump(mode="json"))
    return (
        json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def validate_passing_dev_report_bytes(
    raw: bytes,
) -> tuple[BoundaryGateDevReport, str]:
    try:
        text = raw.decode("utf-8")
        report = BoundaryGateDevReport.model_validate_json(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("canonical dev report is invalid") from error
    if raw != canonical_dev_report_bytes(report):
        raise ValueError("dev report is not the exact canonical representation")
    digest = hashlib.sha256(raw).hexdigest().upper()
    validate_passing_dev_report_binding(report, digest)
    return report, digest


def validate_passing_dev_report_binding(
    report: BoundaryGateDevReport,
    dev_report_sha256: str,
) -> CompositeBoundaryGatePolicy:
    validated = BoundaryGateDevReport.model_validate(report.model_dump(mode="json"))
    expected_hash = hashlib.sha256(canonical_dev_report_bytes(validated)).hexdigest().upper()
    if not re.fullmatch(_SHA256_PATTERN, dev_report_sha256):
        raise ValueError("dev report hash is not canonical uppercase SHA-256")
    if dev_report_sha256 != expected_hash:
        raise ValueError("dev report hash mismatch")
    if (
        validated.status != "completed"
        or validated.primary_acceptance_passed is not True
    ):
        raise ValueError("dev primary composite acceptance did not pass")
    if (
        validated.dataset_version != DATASET_VERSION
        or validated.dataset_sha256 != DATASET_SHA256
        or validated.corpus_sha256 != CORPUS_SHA256
        or validated.tokenizer_version != TOKENIZER_VERSION
        or validated.feature_version != FEATURE_VERSION
        or validated.backend != "baseline"
        or validated.backend_version != "bm25-v1"
        or validated.top_k != 5
        or validated.primary_policy_version != POLICY_VERSION
    ):
        raise ValueError("dev report fixed protocol identity mismatch")
    if (
        validated.composite is None
        or validated.composite.policy_version != POLICY_VERSION
        or not isinstance(
            validated.composite.selected_policy,
            CompositeBoundaryGatePolicy,
        )
    ):
        raise ValueError("dev report is missing the completed composite policy")
    if validated.composite.selected_policy != validated.composite.calibration.selected_policy:
        raise ValueError("dev composite threshold binding mismatch")
    return validated.composite.selected_policy


def _policy_sha256(policy: CompositeBoundaryGatePolicy) -> str:
    payload = json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _delta(treatment: float, control: float) -> float:
    value = round(treatment - control, 6)
    return 0.0 if value == 0 else value


def _build_delta(
    control: AnswerQualityMetrics,
    treatment: AnswerQualityMetrics,
) -> BoundaryGateAbDelta:
    control_no_answer = _required_class_metric(
        control.no_answer_accuracy,
        "Control no-answer accuracy",
    )
    treatment_no_answer = _required_class_metric(
        treatment.no_answer_accuracy,
        "Treatment no-answer accuracy",
    )
    control_response = _required_class_metric(
        control.answerable_response_rate,
        "Control answerable response rate",
    )
    treatment_response = _required_class_metric(
        treatment.answerable_response_rate,
        "Treatment answerable response rate",
    )
    return BoundaryGateAbDelta(
        no_answer_accuracy=_delta(treatment_no_answer, control_no_answer),
        answerable_response_rate=_delta(treatment_response, control_response),
        gold_citation_precision=_required_delta(
            treatment.gold_citation_precision,
            control.gold_citation_precision,
        ),
        lexical_point_coverage=_required_delta(
            treatment.lexical_point_coverage,
            control.lexical_point_coverage,
        ),
        llm_call_rate=_required_delta(
            treatment.llm_call_rate,
            control.llm_call_rate,
        ),
        gate_abstention_rate=_required_delta(
            treatment.gate_abstention_rate,
            control.gate_abstention_rate,
        ),
        latency_p50_ms=_required_delta(
            treatment.latency_p50_ms,
            control.latency_p50_ms,
        ),
        latency_p95_ms=_required_delta(
            treatment.latency_p95_ms,
            control.latency_p95_ms,
        ),
    )


def _required_delta(treatment: float, control: float) -> float:
    value = _delta(treatment, control)
    assert math.isfinite(value)
    return value


def _required_class_metric(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"completed report requires class metrics: {label}")
    return value


def _balanced_acceptance(treatment: AnswerQualityMetrics) -> bool:
    return (
        treatment.no_answer_accuracy is not None
        and treatment.answerable_response_rate is not None
        and treatment.no_answer_accuracy >= 0.50
        and treatment.answerable_response_rate >= 0.60
    )


def _report_base(
    *,
    dev_report: BoundaryGateDevReport | None,
    dev_report_sha256: str | None,
) -> dict[str, object]:
    policy = (
        validate_passing_dev_report_binding(dev_report, dev_report_sha256)
        if dev_report is not None and dev_report_sha256 is not None
        else None
    )
    return {
        "dev_report_sha256": dev_report_sha256,
        "composite_policy": policy,
        "composite_policy_sha256": _policy_sha256(policy) if policy else None,
    }


def build_completed_boundary_gate_ab_report(
    *,
    dev_report: BoundaryGateDevReport,
    dev_report_sha256: str,
    control: AnswerQualityReport,
    treatment: AnswerQualityReport,
) -> BoundaryGateAbReport:
    if control.metrics is None or treatment.metrics is None:
        raise ValueError("completed report requires child metrics")
    return BoundaryGateAbReport(
        **_report_base(
            dev_report=dev_report,
            dev_report_sha256=dev_report_sha256,
        ),
        status=AnswerQualityStatus.COMPLETED,
        control=control,
        treatment=treatment,
        delta=_build_delta(control.metrics, treatment.metrics),
        balanced_acceptance_passed=_balanced_acceptance(treatment.metrics),
    )


def _empty_child(
    *,
    status: AnswerQualityStatus,
    treatment: bool,
    missing: list[str] | None = None,
    failure_type: str | None = None,
) -> AnswerQualityReport:
    limitations = (
        _NOT_RUN_CHILD_LIMITATIONS
        if status == AnswerQualityStatus.NOT_RUN
        else _FAILED_CHILD_LIMITATIONS
    )
    return AnswerQualityReport(
        benchmark_version=BENCHMARK_VERSION,
        evaluation_split="test",
        status=status,
        query_count=0,
        answerable_count=0,
        unanswerable_count=0,
        top_k=5,
        evidence_gate_policy_version=POLICY_VERSION if treatment else None,
        missing=missing or [],
        failure_type=failure_type,
        limitations=limitations,
    )


def build_not_run_boundary_gate_ab_report(
    *,
    dev_report: BoundaryGateDevReport,
    dev_report_sha256: str,
    missing: list[str],
) -> BoundaryGateAbReport:
    return BoundaryGateAbReport(
        **_report_base(
            dev_report=dev_report,
            dev_report_sha256=dev_report_sha256,
        ),
        status=AnswerQualityStatus.NOT_RUN,
        control=_empty_child(
            status=AnswerQualityStatus.NOT_RUN,
            treatment=False,
            missing=missing,
        ),
        treatment=_empty_child(
            status=AnswerQualityStatus.NOT_RUN,
            treatment=True,
            missing=missing,
        ),
        delta=None,
        balanced_acceptance_passed=None,
        missing=missing,
    )


def build_failed_boundary_gate_ab_report(
    *,
    dev_report: BoundaryGateDevReport | None,
    dev_report_sha256: str | None,
    failure_type: str,
) -> BoundaryGateAbReport:
    safe_failure_type = (
        failure_type
        if re.fullmatch(_FAILURE_TYPE_PATTERN, failure_type)
        else "RuntimeError"
    )
    return BoundaryGateAbReport(
        **_report_base(
            dev_report=dev_report,
            dev_report_sha256=dev_report_sha256,
        ),
        status=AnswerQualityStatus.FAILED,
        control=_empty_child(
            status=AnswerQualityStatus.FAILED,
            treatment=False,
            failure_type=safe_failure_type,
        ),
        treatment=_empty_child(
            status=AnswerQualityStatus.FAILED,
            treatment=True,
            failure_type=safe_failure_type,
        ),
        delta=None,
        balanced_acceptance_passed=None,
        failure_type=safe_failure_type,
    )
