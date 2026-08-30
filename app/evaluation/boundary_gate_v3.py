from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from app.rag.base import StrictModel
from app.services.boundary_gate import BoundaryFeatureVector, BoundaryGateReason

REPORT_VERSION = "boundary-gate-dev-report-v3"
DATASET_VERSION = "public-synthetic-boundary-dev-v3"
DATASET_SHA256 = (
    "68918DAD24B121C20B526698254305D5F38EF823E5A43B401A52B81D1273C5C1"
)
CORPUS_SHA256 = (
    "9027B10ED814C7FBB138AA184D42D6AD4023D1881D7F1D2916565589D9D2ECEE"
)
BLUEPRINT_VERSION = "boundary-blueprints-v3"
TOKENIZER_VERSION = "bm25-boundary-tokenizer-v2"
FEATURE_VERSION = "bm25-boundary-features-v2"
PRIMARY_POLICY_VERSION = "bm25-logistic-boundary-gate-v3"
FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "absolute_margin",
    "normalized_margin",
    "max_document_coverage",
    "union_top5_coverage",
    "anchor_coverage",
)
RANDOM_STATE = 20260724
CANDIDATE_C_VALUES = (0.01, 0.1, 1.0, 10.0)
CALIBRATION_RESPONSE_FLOOR = 0.80
CALIBRATION_RECALL_FLOOR = 0.80
_BACKEND = "baseline"
_BACKEND_VERSION = "bm25-v1"
_TOP_K = 5
_ANSWERABLE_CHALLENGES = frozenset(
    {
        "paraphrase",
        "lexical_distractor",
        "multi_evidence",
        "conflict",
        "temporal",
    }
)
_ALL_CHALLENGES = frozenset(
    {
        *_ANSWERABLE_CHALLENGES,
        "near_domain_missing_attribute",
        "temporal_out_of_scope",
        "entity_or_condition_out_of_scope",
    }
)

try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
except ImportError:
    LogisticRegression = None  # type: ignore[assignment]
    StratifiedKFold = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False
    SKLEARN_VERSION = "unavailable"
else:
    _SKLEARN_AVAILABLE = True
    SKLEARN_VERSION = sklearn.__version__


UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Partition: TypeAlias = Literal["calibration", "validation"]
RuleArm: TypeAlias = Literal["top1", "margin", "coverage"]
PolicyStatus: TypeAlias = Literal["completed", "failed", "not_run"]
ReportStatus: TypeAlias = Literal["completed", "failed", "not_run"]


class DependencyUnavailableError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The optional evaluation dependency is unavailable.")


class NoSurvivingConfigurationError(ValueError):
    def __init__(self) -> None:
        super().__init__("No calibration candidate satisfies the fixed floors.")


class BoundaryV3Observation(StrictModel):
    answerable: bool
    challenge_type: str
    partition: Partition
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    selected_count: int = Field(ge=0)
    retrieved_scores: list[float]
    features: BoundaryFeatureVector | None = None
    forced_rejection_reason: BoundaryGateReason | None = None
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_observation(self) -> BoundaryV3Observation:
        if self.challenge_type not in _ALL_CHALLENGES:
            raise ValueError("unknown v3 challenge type")
        if bool(self.relevant_chunk_ids) is not self.answerable:
            raise ValueError("gold IDs and answerability disagree")
        if (self.features is None) == (self.forced_rejection_reason is None):
            raise ValueError(
                "observation requires exactly features or forced rejection"
            )
        if self.features is None:
            return self
        if len(self.retrieved_chunk_ids) < 2:
            raise ValueError("valid evidence requires at least two documents")
        if self.selected_count != len(self.retrieved_chunk_ids):
            raise ValueError("selected count must match retrieved documents")
        if len(self.retrieved_scores) != len(self.retrieved_chunk_ids):
            raise ValueError("retrieved scores must align with documents")
        if any(not math.isfinite(score) for score in self.retrieved_scores):
            raise ValueError("retrieved scores must be finite")
        if any(score < 0.0 for score in self.retrieved_scores):
            raise ValueError("retrieved scores must be nonnegative")
        if any(
            current > previous
            for previous, current in pairwise(self.retrieved_scores)
        ):
            raise ValueError("retrieved scores must be descending")
        if self.retrieved_scores[0] <= 0.0:
            raise ValueError("top1 score must be positive")
        values = [float(getattr(self.features, name)) for name in FEATURE_NAMES]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("boundary features must be finite")
        top1, top2 = self.retrieved_scores[:2]
        absolute_margin = top1 - top2
        normalized_margin = absolute_margin / top1
        for observed, expected, label in (
            (self.features.top1_score, top1, "top1 score"),
            (self.features.top2_score, top2, "top2 score"),
            (
                self.features.absolute_margin,
                absolute_margin,
                "absolute margin",
            ),
            (
                self.features.normalized_margin,
                normalized_margin,
                "normalized margin",
            ),
        ):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"inconsistent {label}")
        return self


class BoundaryV3Metrics(StrictModel):
    answerable_response_rate: UnitFloat
    no_answer_accuracy: UnitFloat
    balanced_accuracy: UnitFloat
    raw_gold_recall_at_5: UnitFloat
    post_gate_gold_recall_at_5: UnitFloat
    gold_recall_loss: UnitFloat
    llm_call_rate_estimate: UnitFloat
    per_challenge_gold_recall_at_5: dict[str, UnitFloat]
    worst_challenge_gap: UnitFloat

    @model_validator(mode="after")
    def validate_derived_metrics(self) -> BoundaryV3Metrics:
        if set(self.per_challenge_gold_recall_at_5) != _ANSWERABLE_CHALLENGES:
            raise ValueError("all answerable challenge metrics are required")
        response_rate = Fraction(str(self.answerable_response_rate))
        no_answer_accuracy = Fraction(str(self.no_answer_accuracy))
        raw_recall = Fraction(str(self.raw_gold_recall_at_5))
        post_recall = Fraction(str(self.post_gate_gold_recall_at_5))
        expected_balanced = float((response_rate + no_answer_accuracy) / 2)
        expected_loss = raw_recall - post_recall
        expected_gap = float(
            max(
                abs(post_recall - Fraction(str(value)))
                for value in self.per_challenge_gold_recall_at_5.values()
            )
        )
        if expected_loss < 0:
            raise ValueError("post-gate recall cannot exceed raw recall")
        for observed, expected, label in (
            (self.balanced_accuracy, expected_balanced, "balanced accuracy"),
            (self.gold_recall_loss, float(expected_loss), "gold recall loss"),
            (self.worst_challenge_gap, expected_gap, "worst challenge gap"),
        ):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"inconsistent {label}")
        return self


class ChallengeQuota(StrictModel):
    total: Literal[15]
    calibration: Literal[10]
    validation: Literal[5]


class BoundaryV3DependencyVersions(StrictModel):
    scikit_learn: str = Field(
        pattern=r"^(?:unavailable|[0-9]+\.[0-9]+\.[0-9]+)$"
    )


def _aggregate_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _rule_policy_version(arm: RuleArm) -> str:
    return f"bm25-{arm}-boundary-gate-v3"


class RulePolicyResult(StrictModel):
    arm: RuleArm
    policy_version: str
    status: Literal["completed", "failed"]
    failure_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9]*(?:Error|Exception)$",
    )
    selected_on: Literal["calibration"] | None = None
    selected_parameters: dict[str, float] | None = None
    candidate_count: int | None = Field(default=None, ge=1)
    surviving_candidate_count: int | None = Field(default=None, ge=1)
    calibration_metrics: BoundaryV3Metrics | None = None
    validation_metrics: BoundaryV3Metrics | None = None
    validation_observation_count: int | None = Field(default=None, ge=1)
    inference_latency_p50_ms: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    inference_latency_p95_ms: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    calibration_sha256: str | None = Field(
        default=None, pattern=r"^[A-F0-9]{64}$"
    )
    validation_metrics_sha256: str | None = Field(
        default=None, pattern=r"^[A-F0-9]{64}$"
    )

    @model_validator(mode="after")
    def validate_result(self) -> RulePolicyResult:
        if self.policy_version != _rule_policy_version(self.arm):
            raise ValueError("rule arm and policy identity mismatch")
        completed_fields = (
            self.selected_on,
            self.selected_parameters,
            self.candidate_count,
            self.surviving_candidate_count,
            self.calibration_metrics,
            self.validation_metrics,
            self.validation_observation_count,
            self.inference_latency_p50_ms,
            self.inference_latency_p95_ms,
            self.calibration_sha256,
            self.validation_metrics_sha256,
        )
        if self.status == "failed":
            if self.failure_type is None:
                raise ValueError("failed rule result requires failure_type")
            if any(value is not None for value in completed_fields):
                raise ValueError("failed rule result forbids completed fields")
            return self
        if self.failure_type is not None:
            raise ValueError("completed rule result forbids failure_type")
        if any(value is None for value in completed_fields[:9]):
            raise ValueError("completed rule result requires parameters and metrics")
        assert self.candidate_count is not None
        assert self.surviving_candidate_count is not None
        assert self.calibration_metrics is not None
        assert self.validation_metrics is not None
        if self.surviving_candidate_count > self.candidate_count:
            raise ValueError("surviving candidates cannot exceed candidate count")
        expected_calibration_hash = _aggregate_sha256(
            self.calibration_metrics.model_dump(mode="json")
        )
        expected_validation_hash = _aggregate_sha256(
            self.validation_metrics.model_dump(mode="json")
        )
        if self.calibration_sha256 is None:
            object.__setattr__(
                self, "calibration_sha256", expected_calibration_hash
            )
        elif self.calibration_sha256 != expected_calibration_hash:
            raise ValueError("calibration metrics hash mismatch")
        if self.validation_metrics_sha256 is None:
            object.__setattr__(
                self, "validation_metrics_sha256", expected_validation_hash
            )
        elif self.validation_metrics_sha256 != expected_validation_hash:
            raise ValueError("validation metrics hash mismatch")
        return self


class LogisticPolicyResult(StrictModel):
    policy_version: Literal[
        "bm25-logistic-boundary-gate-v3"
    ] = PRIMARY_POLICY_VERSION
    status: PolicyStatus
    failure_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9]*(?:Error|Exception)$",
    )
    selected_on: Literal["calibration"] | None = None
    selected_c: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    probability_threshold: UnitFloat | None = None
    candidate_c_count: int | None = Field(default=None, ge=1)
    surviving_threshold_count: int | None = Field(default=None, ge=1)
    oof_count: int | None = Field(default=None, ge=1)
    oof_unique_count: int | None = Field(default=None, ge=1)
    validation_seen_during_selection: Literal[False] | None = None
    scaler_mean: list[float] | None = Field(default=None, min_length=7, max_length=7)
    scaler_scale: list[float] | None = Field(default=None, min_length=7, max_length=7)
    coefficients: list[float] | None = Field(default=None, min_length=7, max_length=7)
    intercept: float | None = Field(default=None, allow_inf_nan=False)
    calibration_metrics: BoundaryV3Metrics | None = None
    validation_metrics: BoundaryV3Metrics | None = None
    validation_observation_count: int | None = Field(default=None, ge=1)
    inference_latency_p50_ms: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    inference_latency_p95_ms: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    selection_sha256: str | None = Field(
        default=None, pattern=r"^[A-F0-9]{64}$"
    )
    validation_metrics_sha256: str | None = Field(
        default=None, pattern=r"^[A-F0-9]{64}$"
    )

    @model_validator(mode="after")
    def validate_result(self) -> LogisticPolicyResult:
        completed_fields = (
            self.selected_on,
            self.selected_c,
            self.probability_threshold,
            self.candidate_c_count,
            self.surviving_threshold_count,
            self.oof_count,
            self.oof_unique_count,
            self.validation_seen_during_selection,
            self.scaler_mean,
            self.scaler_scale,
            self.coefficients,
            self.intercept,
            self.calibration_metrics,
            self.validation_metrics,
            self.validation_observation_count,
            self.inference_latency_p50_ms,
            self.inference_latency_p95_ms,
            self.selection_sha256,
            self.validation_metrics_sha256,
        )
        if self.status != "completed":
            expected_failure = (
                "DependencyUnavailableError"
                if self.status == "not_run"
                else "NoSurvivingConfigurationError"
            )
            if self.failure_type != expected_failure:
                raise ValueError("non-completed logistic result has wrong failure type")
            if any(value is not None for value in completed_fields):
                raise ValueError(
                    "non-completed logistic result forbids completed fields"
                )
            return self
        if self.failure_type is not None:
            raise ValueError("completed logistic result forbids failure_type")
        if any(value is None for value in completed_fields[:17]):
            raise ValueError(
                "completed logistic result requires selection and metrics"
            )
        assert self.oof_count is not None
        assert self.oof_unique_count is not None
        assert self.scaler_mean is not None
        assert self.scaler_scale is not None
        assert self.coefficients is not None
        assert self.calibration_metrics is not None
        assert self.validation_metrics is not None
        if self.oof_count != self.oof_unique_count:
            raise ValueError("every calibration row needs exactly one OOF prediction")
        numeric_values = [
            *self.scaler_mean,
            *self.scaler_scale,
            *self.coefficients,
        ]
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("logistic aggregate parameters must be finite")
        selection_payload = {
            "selected_c": self.selected_c,
            "probability_threshold": self.probability_threshold,
            "oof_count": self.oof_count,
            "oof_unique_count": self.oof_unique_count,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "calibration_metrics": self.calibration_metrics.model_dump(
                mode="json"
            ),
        }
        expected_selection_hash = _aggregate_sha256(selection_payload)
        expected_validation_hash = _aggregate_sha256(
            self.validation_metrics.model_dump(mode="json")
        )
        if self.selection_sha256 is None:
            object.__setattr__(
                self, "selection_sha256", expected_selection_hash
            )
        elif self.selection_sha256 != expected_selection_hash:
            raise ValueError("logistic selection hash mismatch")
        if self.validation_metrics_sha256 is None:
            object.__setattr__(
                self, "validation_metrics_sha256", expected_validation_hash
            )
        elif self.validation_metrics_sha256 != expected_validation_hash:
            raise ValueError("validation metrics hash mismatch")
        return self


class LogisticSelection(StrictModel):
    selected_c: float = Field(gt=0.0, allow_inf_nan=False)
    probability_threshold: UnitFloat
    candidate_c_count: Literal[4] = 4
    surviving_threshold_count: int = Field(ge=1)
    oof_count: int = Field(ge=1)
    oof_unique_count: int = Field(ge=1)
    validation_seen_during_selection: Literal[False] = False
    scaler_mean: list[float] = Field(min_length=7, max_length=7)
    scaler_scale: list[float] = Field(min_length=7, max_length=7)
    coefficients: list[float] = Field(min_length=7, max_length=7)
    intercept: float = Field(allow_inf_nan=False)
    calibration_metrics: BoundaryV3Metrics
    selection_sha256: str | None = Field(
        default=None, pattern=r"^[A-F0-9]{64}$"
    )

    @model_validator(mode="after")
    def validate_selection(self) -> LogisticSelection:
        if self.selected_c not in CANDIDATE_C_VALUES:
            raise ValueError("selected C is outside the fixed candidate set")
        if self.oof_count != self.oof_unique_count:
            raise ValueError("every calibration row needs one OOF prediction")
        values = [
            *self.scaler_mean,
            *self.scaler_scale,
            *self.coefficients,
            self.intercept,
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("selection aggregates must be finite")
        payload = self.model_dump(
            mode="json", exclude={"selection_sha256"}
        )
        expected = _aggregate_sha256(payload)
        if self.selection_sha256 is None:
            object.__setattr__(self, "selection_sha256", expected)
        elif self.selection_sha256 != expected:
            raise ValueError("selection aggregate hash mismatch")
        return self


def primary_acceptance(metrics: BoundaryV3Metrics) -> bool:
    return (
        metrics.answerable_response_rate >= 0.80
        and metrics.no_answer_accuracy >= 0.80
        and metrics.balanced_accuracy >= 0.80
        and metrics.post_gate_gold_recall_at_5 >= 0.80
        and metrics.gold_recall_loss <= 0.05
        and metrics.worst_challenge_gap <= 0.15
    )


class BoundaryGateV3Report(StrictModel):
    report_version: Literal[
        "boundary-gate-dev-report-v3"
    ] = REPORT_VERSION
    status: ReportStatus
    failure_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9]*(?:Error|Exception)$",
    )
    dataset_version: Literal[
        "public-synthetic-boundary-dev-v3"
    ] = DATASET_VERSION
    dataset_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    corpus_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    protocol_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    builder_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    blueprint_version: Literal["boundary-blueprints-v3"]
    tokenizer_version: Literal[
        "bm25-boundary-tokenizer-v2"
    ] = TOKENIZER_VERSION
    leakage_check_version: Literal["boundary-leakage-v3"]
    feature_version: Literal["bm25-boundary-features-v2"] = FEATURE_VERSION
    backend: Literal["baseline"] = _BACKEND
    backend_version: Literal["bm25-v1"] = _BACKEND_VERSION
    top_k: Literal[5] = _TOP_K
    total_case_count: Literal[120]
    calibration_case_count: Literal[80]
    validation_case_count: Literal[40]
    answerable_case_count: Literal[75]
    unanswerable_case_count: Literal[45]
    challenge_quotas: dict[str, ChallengeQuota]
    dependency_versions: BoundaryV3DependencyVersions
    top1: RulePolicyResult | None = None
    margin: RulePolicyResult | None = None
    coverage: RulePolicyResult | None = None
    logistic: LogisticPolicyResult | None = None
    primary_policy_version: Literal[
        "bm25-logistic-boundary-gate-v3"
    ] = PRIMARY_POLICY_VERSION
    primary_acceptance_passed: bool | None
    limitations: list[
        Literal[
            "synthetic_data_only",
            "bm25_only",
            "not_ragsdk_result",
            "not_real_user_accuracy",
        ]
    ] = Field(
        default_factory=lambda: [
            "synthetic_data_only",
            "bm25_only",
            "not_ragsdk_result",
            "not_real_user_accuracy",
        ],
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_report(self) -> BoundaryGateV3Report:
        if set(self.challenge_quotas) != _ALL_CHALLENGES:
            raise ValueError("all eight challenge quotas are required")
        canonical_limitations = [
            "synthetic_data_only",
            "bm25_only",
            "not_ragsdk_result",
            "not_real_user_accuracy",
        ]
        if self.limitations != canonical_limitations:
            raise ValueError("report limitations must use the canonical set")
        arms = (self.top1, self.margin, self.coverage, self.logistic)
        if (
            self.logistic is not None
            and self.logistic.status == "not_run"
            and self.dependency_versions.scikit_learn != "unavailable"
        ):
            raise ValueError("missing dependency must use unavailable marker")
        if (
            self.logistic is not None
            and self.logistic.status in {"completed", "failed"}
            and self.dependency_versions.scikit_learn == "unavailable"
        ):
            raise ValueError("executed logistic result requires dependency version")
        if self.status == "completed":
            if self.failure_type is not None or any(arm is None for arm in arms):
                raise ValueError("completed report requires all four arms")
            assert self.top1 is not None
            assert self.margin is not None
            assert self.coverage is not None
            assert self.logistic is not None
            if (
                self.top1.arm != "top1"
                or self.margin.arm != "margin"
                or self.coverage.arm != "coverage"
                or self.logistic.status != "completed"
            ):
                raise ValueError("completed report arm identity mismatch")
            assert self.logistic.validation_metrics is not None
            expected_acceptance = primary_acceptance(
                self.logistic.validation_metrics
            )
            if self.primary_acceptance_passed != expected_acceptance:
                raise ValueError("primary acceptance mismatch")
        else:
            if self.failure_type is None:
                raise ValueError("non-completed report requires failure_type")
            if self.primary_acceptance_passed is not None:
                raise ValueError("non-completed report forbids acceptance claim")
        return self


def require_sklearn() -> tuple[type, type, type]:
    if (
        not _SKLEARN_AVAILABLE
        or StandardScaler is None
        or LogisticRegression is None
        or StratifiedKFold is None
    ):
        raise DependencyUnavailableError
    return StandardScaler, LogisticRegression, StratifiedKFold


def build_logistic_not_run() -> LogisticPolicyResult:
    return LogisticPolicyResult(
        status="not_run",
        failure_type="DependencyUnavailableError",
    )


def dependency_versions() -> BoundaryV3DependencyVersions:
    return BoundaryV3DependencyVersions(
        scikit_learn=(
            SKLEARN_VERSION if _SKLEARN_AVAILABLE else "unavailable"
        )
    )


def validate_live_dependency_versions(
    report: BoundaryGateV3Report,
) -> BoundaryGateV3Report:
    expected = SKLEARN_VERSION if _SKLEARN_AVAILABLE else "unavailable"
    if report.dependency_versions.scikit_learn != expected:
        raise ValueError("reported dependency version mismatch")
    return report


def _require_partition(
    observations: Sequence[BoundaryV3Observation],
    expected: Partition,
) -> None:
    if not observations:
        raise ValueError(f"{expected} observations cannot be empty")
    if any(row.partition != expected for row in observations):
        raise ValueError(f"{expected} selection received a different partition")
    for row in observations:
        BoundaryV3Observation.model_validate(row.model_dump())


def _raw_gold_recall(observation: BoundaryV3Observation) -> Fraction:
    if not observation.relevant_chunk_ids:
        raise ValueError("answerable observations require gold IDs")
    gold = set(observation.relevant_chunk_ids)
    return Fraction(
        len(gold & set(observation.retrieved_chunk_ids[:_TOP_K])),
        len(gold),
    )


def aggregate_v3_metrics(
    observations: Sequence[BoundaryV3Observation],
    decisions: Sequence[bool],
) -> BoundaryV3Metrics:
    if len(observations) != len(decisions) or not observations:
        raise ValueError("metrics require one decision per observation")
    answerable = [
        (row, allowed)
        for row, allowed in zip(observations, decisions, strict=True)
        if row.answerable
    ]
    unanswerable = [
        (row, allowed)
        for row, allowed in zip(observations, decisions, strict=True)
        if not row.answerable
    ]
    if not answerable or not unanswerable:
        raise ValueError("metrics require both answerability classes")
    raw_recalls = [_raw_gold_recall(row) for row, _ in answerable]
    post_recalls = [
        recall if allowed else Fraction()
        for (_, allowed), recall in zip(answerable, raw_recalls, strict=True)
    ]
    response_rate = Fraction(
        sum(allowed for _, allowed in answerable), len(answerable)
    )
    no_answer_accuracy = Fraction(
        sum(not allowed for _, allowed in unanswerable), len(unanswerable)
    )
    raw_recall = sum(raw_recalls, start=Fraction()) / len(raw_recalls)
    post_recall = sum(post_recalls, start=Fraction()) / len(post_recalls)
    per_challenge: dict[str, Fraction] = {}
    for challenge in sorted(_ANSWERABLE_CHALLENGES):
        challenge_values = [
            recall
            for (row, _), recall in zip(answerable, post_recalls, strict=True)
            if row.challenge_type == challenge
        ]
        if not challenge_values:
            raise ValueError("metrics require every answerable challenge")
        per_challenge[challenge] = (
            sum(challenge_values, start=Fraction()) / len(challenge_values)
        )
    return BoundaryV3Metrics(
        answerable_response_rate=float(response_rate),
        no_answer_accuracy=float(no_answer_accuracy),
        balanced_accuracy=float((response_rate + no_answer_accuracy) / 2),
        raw_gold_recall_at_5=float(raw_recall),
        post_gate_gold_recall_at_5=float(post_recall),
        gold_recall_loss=float(raw_recall - post_recall),
        llm_call_rate_estimate=float(
            Fraction(sum(decisions), len(decisions))
        ),
        per_challenge_gold_recall_at_5={
            key: float(value) for key, value in per_challenge.items()
        },
        worst_challenge_gap=float(
            max(abs(post_recall - value) for value in per_challenge.values())
        ),
    )


def _scalar_candidate_thresholds(
    calibration: Sequence[BoundaryV3Observation],
    feature_name: str,
) -> list[float]:
    values = sorted(
        {
            float(getattr(row.features, feature_name))
            for row in calibration
            if row.features is not None
        }
    )
    if not values:
        raise NoSurvivingConfigurationError
    midpoints = [(left + right) / 2 for left, right in pairwise(values)]
    return sorted({*values, *midpoints})


def _coverage_candidate_parameters(
    calibration: Sequence[BoundaryV3Observation],
) -> list[dict[str, float]]:
    candidates: set[tuple[float, float, float]] = {(0.0, 0.0, 0.0)}
    for row in calibration:
        if row.features is None:
            continue
        candidates.add(
            (
                row.features.anchor_coverage,
                row.features.union_top5_coverage,
                row.features.max_document_coverage,
            )
        )
    return [
        {
            "min_anchor_coverage": anchor,
            "min_union_top5_coverage": union,
            "min_max_document_coverage": maximum,
        }
        for anchor, union, maximum in sorted(candidates)
    ]


def _rule_candidates(
    arm: RuleArm,
    calibration: Sequence[BoundaryV3Observation],
) -> list[dict[str, float]]:
    if arm == "top1":
        return [
            {"min_top1_score": value}
            for value in _scalar_candidate_thresholds(
                calibration, "top1_score"
            )
        ]
    if arm == "margin":
        return [
            {"min_normalized_margin": value}
            for value in _scalar_candidate_thresholds(
                calibration, "normalized_margin"
            )
        ]
    if arm == "coverage":
        return _coverage_candidate_parameters(calibration)
    raise ValueError("unknown rule arm")


def _rule_allows(
    arm: RuleArm,
    parameters: dict[str, float],
    observation: BoundaryV3Observation,
) -> bool:
    if observation.features is None:
        return False
    features = observation.features
    if arm == "top1":
        return features.top1_score >= parameters["min_top1_score"]
    if arm == "margin":
        return (
            features.normalized_margin
            >= parameters["min_normalized_margin"]
        )
    return (
        features.anchor_coverage >= parameters["min_anchor_coverage"]
        and features.union_top5_coverage
        >= parameters["min_union_top5_coverage"]
        and features.max_document_coverage
        >= parameters["min_max_document_coverage"]
    )


def _rule_decisions(
    arm: RuleArm,
    parameters: dict[str, float],
    observations: Sequence[BoundaryV3Observation],
) -> list[bool]:
    return [_rule_allows(arm, parameters, row) for row in observations]


def _latency_percentiles(samples_ns: Sequence[int]) -> tuple[float, float]:
    if not samples_ns:
        raise ValueError("latency summary requires samples")
    ordered = sorted(max(1, value) for value in samples_ns)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower] / 1_000_000
        weight = position - lower
        interpolated = (
            ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        )
        return interpolated / 1_000_000

    return percentile(0.50), percentile(0.95)


def _timed_rule_decisions(
    arm: RuleArm,
    parameters: dict[str, float],
    observations: Sequence[BoundaryV3Observation],
) -> tuple[list[bool], tuple[float, float]]:
    decisions: list[bool] = []
    samples_ns: list[int] = []
    for row in observations:
        started = time.perf_counter_ns()
        decisions.append(_rule_allows(arm, parameters, row))
        samples_ns.append(time.perf_counter_ns() - started)
    return decisions, _latency_percentiles(samples_ns)


def evaluate_rule_arm(
    name: RuleArm | str,
    calibration: Sequence[BoundaryV3Observation],
    validation: Sequence[BoundaryV3Observation],
) -> RulePolicyResult:
    if name not in {"top1", "margin", "coverage"}:
        raise ValueError("unknown rule arm")
    arm: RuleArm = name  # type: ignore[assignment]
    _require_partition(calibration, "calibration")
    _require_partition(validation, "validation")
    candidates = _rule_candidates(arm, calibration)
    survivors: list[
        tuple[dict[str, float], BoundaryV3Metrics]
    ] = []
    for parameters in candidates:
        metrics = aggregate_v3_metrics(
            calibration,
            _rule_decisions(arm, parameters, calibration),
        )
        if (
            metrics.answerable_response_rate >= CALIBRATION_RESPONSE_FLOOR
            and metrics.post_gate_gold_recall_at_5
            >= CALIBRATION_RECALL_FLOOR
        ):
            survivors.append((parameters, metrics))
    if not survivors:
        return RulePolicyResult(
            arm=arm,
            policy_version=_rule_policy_version(arm),
            status="failed",
            failure_type="NoSurvivingConfigurationError",
        )
    selected_parameters, calibration_metrics = min(
        survivors,
        key=lambda item: (
            -item[1].balanced_accuracy,
            -item[1].no_answer_accuracy,
            item[1].llm_call_rate_estimate,
            json.dumps(item[0], sort_keys=True, separators=(",", ":")),
        ),
    )
    validation_decisions, latency = _timed_rule_decisions(
        arm, selected_parameters, validation
    )
    validation_metrics = aggregate_v3_metrics(
        validation, validation_decisions
    )
    return RulePolicyResult(
        arm=arm,
        policy_version=_rule_policy_version(arm),
        status="completed",
        selected_on="calibration",
        selected_parameters=selected_parameters,
        candidate_count=len(candidates),
        surviving_candidate_count=len(survivors),
        calibration_metrics=calibration_metrics,
        validation_metrics=validation_metrics,
        validation_observation_count=len(validation),
        inference_latency_p50_ms=latency[0],
        inference_latency_p95_ms=latency[1],
    )


def _feature_row(observation: BoundaryV3Observation) -> list[float]:
    if observation.features is None:
        raise ValueError("logistic calibration requires valid feature vectors")
    return [
        float(getattr(observation.features, name)) for name in FEATURE_NAMES
    ]


def _probability_candidate_thresholds(
    probabilities: Sequence[float],
) -> list[float]:
    values = sorted(set(probabilities))
    if not values:
        raise NoSurvivingConfigurationError
    return sorted(
        {
            *values,
            *((left + right) / 2 for left, right in pairwise(values)),
        }
    )


def fit_logistic_oof(
    calibration: Sequence[BoundaryV3Observation],
) -> LogisticSelection:
    scaler_type, model_type, splitter_type = require_sklearn()
    _require_partition(calibration, "calibration")
    if len(calibration) != 80:
        raise ValueError("v3 logistic calibration requires exactly 80 rows")

    import numpy as np

    matrix = np.asarray([_feature_row(row) for row in calibration], dtype=float)
    labels = np.asarray([int(row.answerable) for row in calibration], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("logistic calibration requires both classes")

    candidates: list[
        tuple[float, float, int, BoundaryV3Metrics, list[float]]
    ] = []
    for c_value in CANDIDATE_C_VALUES:
        probabilities: list[float | None] = [None] * len(calibration)
        assignment_counts = [0] * len(calibration)
        splitter = splitter_type(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        for train_idx, holdout_idx in splitter.split(matrix, labels):
            scaler = scaler_type().fit(matrix[train_idx])
            model = model_type(
                C=c_value,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=1000,
            ).fit(scaler.transform(matrix[train_idx]), labels[train_idx])
            fold_probability = model.predict_proba(
                scaler.transform(matrix[holdout_idx])
            )[:, 1]
            for index, probability in zip(
                holdout_idx, fold_probability, strict=True
            ):
                probabilities[int(index)] = float(probability)
                assignment_counts[int(index)] += 1
        if any(count != 1 for count in assignment_counts):
            raise ValueError("OOF assignment must predict each row exactly once")
        if any(probability is None for probability in probabilities):
            raise ValueError("OOF probability collection is incomplete")
        complete_probabilities = [
            float(probability) for probability in probabilities
        ]
        surviving: list[tuple[float, BoundaryV3Metrics]] = []
        for threshold in _probability_candidate_thresholds(
            complete_probabilities
        ):
            metrics = aggregate_v3_metrics(
                calibration,
                [
                    probability >= threshold
                    for probability in complete_probabilities
                ],
            )
            if (
                metrics.answerable_response_rate
                >= CALIBRATION_RESPONSE_FLOOR
                and metrics.post_gate_gold_recall_at_5
                >= CALIBRATION_RECALL_FLOOR
            ):
                surviving.append((threshold, metrics))
        if not surviving:
            continue
        selected_threshold, selected_metrics = min(
            surviving,
            key=lambda item: (
                -item[1].balanced_accuracy,
                -item[1].no_answer_accuracy,
                item[1].llm_call_rate_estimate,
                item[0],
            ),
        )
        candidates.append(
            (
                c_value,
                selected_threshold,
                len(surviving),
                selected_metrics,
                complete_probabilities,
            )
        )
    if not candidates:
        raise NoSurvivingConfigurationError
    selected_c, threshold, survivor_count, metrics, probabilities = min(
        candidates,
        key=lambda item: (
            -item[3].balanced_accuracy,
            item[0],
            -item[3].no_answer_accuracy,
            item[3].llm_call_rate_estimate,
            item[1],
        ),
    )
    scaler = scaler_type().fit(matrix)
    model = model_type(
        C=selected_c,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
        max_iter=1000,
    ).fit(scaler.transform(matrix), labels)
    return LogisticSelection(
        selected_c=selected_c,
        probability_threshold=threshold,
        surviving_threshold_count=survivor_count,
        oof_count=len(probabilities),
        oof_unique_count=len(probabilities),
        validation_seen_during_selection=False,
        scaler_mean=[float(value) for value in scaler.mean_.tolist()],
        scaler_scale=[float(value) for value in scaler.scale_.tolist()],
        coefficients=[
            float(value) for value in model.coef_[0].tolist()
        ],
        intercept=float(model.intercept_[0]),
        calibration_metrics=metrics,
    )


def _logistic_probability(
    selection: LogisticSelection,
    observation: BoundaryV3Observation,
) -> float | None:
    if observation.features is None:
        return None
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            _feature_row(observation),
            selection.scaler_mean,
            selection.scaler_scale,
            strict=True,
        )
    ]
    logit = selection.intercept + sum(
        coefficient * value
        for coefficient, value in zip(
            selection.coefficients, standardized, strict=True
        )
    )
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def evaluate_logistic_arm(
    calibration: Sequence[BoundaryV3Observation],
    validation: Sequence[BoundaryV3Observation],
) -> LogisticPolicyResult:
    _require_partition(calibration, "calibration")
    _require_partition(validation, "validation")
    if not _SKLEARN_AVAILABLE:
        return build_logistic_not_run()
    try:
        selection = fit_logistic_oof(calibration)
    except NoSurvivingConfigurationError:
        return LogisticPolicyResult(
            status="failed",
            failure_type="NoSurvivingConfigurationError",
        )
    decisions = []
    samples_ns: list[int] = []
    for row in validation:
        started = time.perf_counter_ns()
        probability = _logistic_probability(selection, row)
        decisions.append(
            probability is not None
            and probability >= selection.probability_threshold
        )
        samples_ns.append(time.perf_counter_ns() - started)
    latency = _latency_percentiles(samples_ns)
    validation_metrics = aggregate_v3_metrics(validation, decisions)
    return LogisticPolicyResult(
        status="completed",
        selected_on="calibration",
        selected_c=selection.selected_c,
        probability_threshold=selection.probability_threshold,
        candidate_c_count=selection.candidate_c_count,
        surviving_threshold_count=selection.surviving_threshold_count,
        oof_count=selection.oof_count,
        oof_unique_count=selection.oof_unique_count,
        validation_seen_during_selection=False,
        scaler_mean=selection.scaler_mean,
        scaler_scale=selection.scaler_scale,
        coefficients=selection.coefficients,
        intercept=selection.intercept,
        calibration_metrics=selection.calibration_metrics,
        validation_metrics=validation_metrics,
        validation_observation_count=len(validation),
        inference_latency_p50_ms=latency[0],
        inference_latency_p95_ms=latency[1],
    )


def _load_default_manifest() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = (
        project_root / "data/public_eval_boundary_dev_v3/manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v3 manifest must be a JSON object")
    return payload


def evaluate_v3(
    observations: Sequence[BoundaryV3Observation],
) -> BoundaryGateV3Report:
    calibration = [
        row for row in observations if row.partition == "calibration"
    ]
    validation = [
        row for row in observations if row.partition == "validation"
    ]
    _require_partition(calibration, "calibration")
    _require_partition(validation, "validation")
    if len(observations) != 120 or len(calibration) != 80 or len(validation) != 40:
        raise ValueError("v3 evaluation requires the frozen 120/80/40 split")
    answerable_count = sum(row.answerable for row in observations)
    if answerable_count != 75:
        raise ValueError("v3 evaluation requires the frozen 75/45 classes")
    observed_quotas = {
        challenge: {
            "total": sum(
                row.challenge_type == challenge for row in observations
            ),
            "calibration": sum(
                row.challenge_type == challenge
                and row.partition == "calibration"
                for row in observations
            ),
            "validation": sum(
                row.challenge_type == challenge
                and row.partition == "validation"
                for row in observations
            ),
        }
        for challenge in sorted(_ALL_CHALLENGES)
    }
    if any(
        quota != {"total": 15, "calibration": 10, "validation": 5}
        for quota in observed_quotas.values()
    ):
        raise ValueError("v3 challenge quota drift")

    top1 = evaluate_rule_arm("top1", calibration, validation)
    margin = evaluate_rule_arm("margin", calibration, validation)
    coverage = evaluate_rule_arm("coverage", calibration, validation)
    logistic = evaluate_logistic_arm(calibration, validation)
    manifest = _load_default_manifest()
    common = {
        "dataset_sha256": DATASET_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "protocol_sha256": manifest["protocol_sha256"],
        "builder_sha256": manifest["builder_sha256"],
        "blueprint_version": manifest["blueprint_version"],
        "leakage_check_version": manifest["leakage_check_version"],
        "total_case_count": len(observations),
        "calibration_case_count": len(calibration),
        "validation_case_count": len(validation),
        "answerable_case_count": answerable_count,
        "unanswerable_case_count": len(observations) - answerable_count,
        "challenge_quotas": observed_quotas,
        "dependency_versions": dependency_versions(),
        "top1": top1,
        "margin": margin,
        "coverage": coverage,
        "logistic": logistic,
    }
    if logistic.status == "not_run":
        return validate_live_dependency_versions(
            BoundaryGateV3Report(
                status="not_run",
                failure_type="DependencyUnavailableError",
                primary_acceptance_passed=None,
                **common,
            )
        )
    if (
        logistic.status == "failed"
        or top1.status == "failed"
        or margin.status == "failed"
        or coverage.status == "failed"
    ):
        return validate_live_dependency_versions(
            BoundaryGateV3Report(
                status="failed",
                failure_type="NoSurvivingConfigurationError",
                primary_acceptance_passed=None,
                **common,
            )
        )
    assert logistic.validation_metrics is not None
    return validate_live_dependency_versions(
        BoundaryGateV3Report(
            status="completed",
            primary_acceptance_passed=primary_acceptance(
                logistic.validation_metrics
            ),
            **common,
        )
    )
