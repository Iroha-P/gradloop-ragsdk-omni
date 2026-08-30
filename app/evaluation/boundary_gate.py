from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_serializer, model_validator

from app.evaluation.boundary_dataset import (
    AnswerableChallengeQuota,
    AnswerableChallengeQuotas,
    BoundaryDevCase,
    BoundaryDevManifest,
    BoundaryPartition,
    EntityOrConditionOutOfScopeQuota,
    NearDomainMissingAttributeQuota,
    NegativeSubtypeQuotas,
    TemporalOutOfScopeQuota,
)
from app.evaluation.retrieval import RetrievalChallengeType
from app.rag.base import RagBackend, RetrievalRequest, StrictModel
from app.services.boundary_gate import (
    BoundaryFeatureVector,
    BoundaryGateReason,
    BoundaryRuntimeEvidenceError,
    CompositeBoundaryGatePolicy,
    CoverageGatePolicy,
    MarginGatePolicy,
    extract_boundary_features,
)

REPORT_VERSION = "boundary-gate-dev-report-v2"
DATASET_VERSION = "public-synthetic-boundary-dev-v2"
DATASET_SHA256 = "DE28237389EA7BD0980E966FED0A301B212A020A5345080E282E0EF270BD7F77"
CORPUS_SHA256 = "0B05F6EDE8DAC2C1E3C62BCFF0BDFB403DAE19053E9416C0718991A01F6F918F"
TOKENIZER_VERSION = "bm25-boundary-tokenizer-v2"
FEATURE_VERSION = "bm25-boundary-features-v2"
PRIMARY_POLICY_VERSION = "bm25-composite-boundary-gate-v2"
_BACKEND = "baseline"
_BACKEND_VERSION = "bm25-v1"
_TOP_K = 5
_CALIBRATION_RESPONSE_FLOOR = 0.85
_CALIBRATION_RECALL_FLOOR = 0.80
_ANSWERABLE_CHALLENGES = frozenset(
    {
        RetrievalChallengeType.PARAPHRASE.value,
        RetrievalChallengeType.LEXICAL_DISTRACTOR.value,
        RetrievalChallengeType.MULTI_EVIDENCE.value,
        RetrievalChallengeType.CONFLICT.value,
        RetrievalChallengeType.TEMPORAL.value,
    }
)

UnitFloat = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]
Policy: TypeAlias = (
    MarginGatePolicy | CoverageGatePolicy | CompositeBoundaryGatePolicy
)
PolicyType: TypeAlias = (
    type[MarginGatePolicy]
    | type[CoverageGatePolicy]
    | type[CompositeBoundaryGatePolicy]
)
ReportStatus: TypeAlias = Literal["completed", "failed", "not_run"]
PolicyResultStatus: TypeAlias = Literal["completed", "failed"]
_RUNTIME_EVIDENCE_REASONS = frozenset(
    {
        BoundaryGateReason.NO_DOCUMENTS,
        BoundaryGateReason.INSUFFICIENT_DOCUMENTS,
        BoundaryGateReason.SELECTED_COUNT_MISMATCH,
        BoundaryGateReason.NON_FINITE_SCORE,
        BoundaryGateReason.NEGATIVE_SCORE,
        BoundaryGateReason.SCORE_ORDER_INVALID,
        BoundaryGateReason.ZERO_TOP1_SCORE,
    }
)


class NoSurvivingConfigurationError(ValueError):
    def __init__(self) -> None:
        super().__init__("No calibration candidate satisfies the fixed floors.")


class BoundaryObservation(StrictModel):
    """Ephemeral cached inputs used only during deterministic enumeration."""

    answerable: bool
    challenge_type: RetrievalChallengeType
    split: BoundaryPartition = BoundaryPartition.CALIBRATION
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    features: BoundaryFeatureVector | None = None
    forced_rejection_reason: BoundaryGateReason | None = None
    allowed: bool | None = None

    @model_validator(mode="after")
    def validate_evidence_state(self) -> BoundaryObservation:
        has_features = self.features is not None
        has_forced_reason = self.forced_rejection_reason is not None
        if has_features == has_forced_reason:
            raise ValueError(
                "observation requires exactly features or a forced rejection reason"
            )
        if (
            self.forced_rejection_reason is not None
            and self.forced_rejection_reason not in _RUNTIME_EVIDENCE_REASONS
        ):
            raise ValueError("forced rejection reason must identify runtime evidence")
        if self.forced_rejection_reason is not None and self.allowed is True:
            raise ValueError("forced runtime evidence rejection cannot be allowed")
        return self


class BoundaryGateMetrics(StrictModel):
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
    def validate_derived_metrics(self) -> BoundaryGateMetrics:
        if not self.per_challenge_gold_recall_at_5:
            raise ValueError("answerable challenge metrics cannot be empty")
        if not set(self.per_challenge_gold_recall_at_5) <= _ANSWERABLE_CHALLENGES:
            raise ValueError("unknown answerable challenge metric")
        response_rate = Fraction(str(self.answerable_response_rate))
        no_answer_accuracy = Fraction(str(self.no_answer_accuracy))
        raw_recall = Fraction(str(self.raw_gold_recall_at_5))
        post_recall = Fraction(str(self.post_gate_gold_recall_at_5))
        expected_balanced = float((response_rate + no_answer_accuracy) / 2)
        expected_loss = raw_recall - post_recall
        if expected_loss < 0:
            raise ValueError("post-gate recall cannot exceed raw recall")
        expected_gap = float(
            max(
                abs(post_recall - Fraction(str(value)))
                for value in self.per_challenge_gold_recall_at_5.values()
            )
        )
        for observed, expected, label in (
            (self.balanced_accuracy, expected_balanced, "balanced accuracy"),
            (self.gold_recall_loss, float(expected_loss), "gold recall loss"),
            (self.worst_challenge_gap, expected_gap, "worst challenge gap"),
        ):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"inconsistent {label}")
        return self


class BoundaryPolicyCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=256)
    policy: Policy
    metrics: BoundaryGateMetrics


class BoundaryPolicyCalibration(StrictModel):
    policy_version: Literal[
        "bm25-margin-gate-v2",
        "bm25-coverage-gate-v2",
        "bm25-composite-boundary-gate-v2",
    ]
    candidate_count: int = Field(ge=1)
    surviving_candidate_count: int = Field(ge=1)
    selected_policy: Policy
    metrics: BoundaryGateMetrics

    @model_validator(mode="after")
    def validate_calibration(self) -> BoundaryPolicyCalibration:
        if self.selected_policy.policy_version != self.policy_version:
            raise ValueError("calibration policy identity mismatch")
        if self.surviving_candidate_count > self.candidate_count:
            raise ValueError("surviving candidates cannot exceed all candidates")
        if (
            self.metrics.answerable_response_rate
            < _CALIBRATION_RESPONSE_FLOOR
            or self.metrics.post_gate_gold_recall_at_5
            < _CALIBRATION_RECALL_FLOOR
        ):
            raise ValueError("selected calibration metrics do not meet floors")
        return self


class BoundaryPolicyResult(StrictModel):
    policy_version: Literal[
        "bm25-margin-gate-v2",
        "bm25-coverage-gate-v2",
        "bm25-composite-boundary-gate-v2",
    ]
    status: PolicyResultStatus = "completed"
    failure_type: Literal["NoSurvivingConfigurationError"] | None = None
    selected_policy: Policy | None = None
    calibration: BoundaryPolicyCalibration | None = None
    calibration_metrics: BoundaryGateMetrics | None = None
    validation_metrics: BoundaryGateMetrics | None = None
    calibration_sha256: str | None = Field(
        default=None,
        pattern=r"^[A-F0-9]{64}$",
    )
    validation_metrics_sha256: str | None = Field(
        default=None,
        pattern=r"^[A-F0-9]{64}$",
    )

    @model_validator(mode="after")
    def validate_result(self) -> BoundaryPolicyResult:
        completed_values = (
            self.selected_policy,
            self.calibration,
            self.calibration_metrics,
            self.validation_metrics,
            self.calibration_sha256,
            self.validation_metrics_sha256,
        )
        if self.status == "failed":
            if self.failure_type != "NoSurvivingConfigurationError":
                raise ValueError(
                    "failed policy result requires the fixed failure type"
                )
            if any(value is not None for value in completed_values):
                raise ValueError("failed policy result forbids completed fields")
            return self
        if self.failure_type is not None:
            raise ValueError("completed policy result forbids failure type")
        if (
            self.selected_policy is None
            or self.calibration is None
            or self.validation_metrics is None
        ):
            raise ValueError("completed policy result requires thresholds and metrics")
        if (
            self.selected_policy.policy_version != self.policy_version
            or self.calibration.policy_version != self.policy_version
            or self.selected_policy != self.calibration.selected_policy
        ):
            raise ValueError("policy result identity or threshold mismatch")
        if self.calibration_metrics is None:
            object.__setattr__(self, "calibration_metrics", self.calibration.metrics)
        elif self.calibration_metrics != self.calibration.metrics:
            raise ValueError("policy result calibration metric mismatch")
        expected_calibration_sha256 = _aggregate_sha256(self.calibration.model_dump())
        expected_validation_sha256 = _aggregate_sha256(
            self.validation_metrics.model_dump()
        )
        if self.calibration_sha256 is None:
            object.__setattr__(
                self,
                "calibration_sha256",
                expected_calibration_sha256,
            )
        elif self.calibration_sha256 != expected_calibration_sha256:
            raise ValueError("policy calibration aggregate mismatch")
        if self.validation_metrics_sha256 is None:
            object.__setattr__(
                self,
                "validation_metrics_sha256",
                expected_validation_sha256,
            )
        elif self.validation_metrics_sha256 != expected_validation_sha256:
            raise ValueError("policy validation aggregate mismatch")
        return self

    @model_serializer(mode="wrap")
    def serialize_result(self, handler):
        payload = handler(self)
        if self.status == "failed":
            return {
                "policy_version": payload["policy_version"],
                "status": payload["status"],
                "failure_type": payload["failure_type"],
            }
        return payload


class BoundaryGateDevReport(StrictModel):
    report_version: Literal["boundary-gate-dev-report-v2"] = REPORT_VERSION
    status: ReportStatus
    dataset_version: Literal["public-synthetic-boundary-dev-v2"] = DATASET_VERSION
    dataset_sha256: Literal[
        "DE28237389EA7BD0980E966FED0A301B212A020A5345080E282E0EF270BD7F77"
    ] = DATASET_SHA256
    corpus_sha256: Literal[
        "0B05F6EDE8DAC2C1E3C62BCFF0BDFB403DAE19053E9416C0718991A01F6F918F"
    ] = CORPUS_SHA256
    blueprint_version: Literal["boundary-blueprints-v2"]
    tokenizer_version: Literal["bm25-boundary-tokenizer-v2"] = TOKENIZER_VERSION
    leakage_check_version: Literal["boundary-leakage-v2"]
    feature_version: Literal["bm25-boundary-features-v2"] = FEATURE_VERSION
    backend: Literal["baseline"] = _BACKEND
    backend_version: Literal["bm25-v1"] = _BACKEND_VERSION
    top_k: Literal[5] = _TOP_K
    total_case_count: Literal[64]
    calibration_case_count: Literal[48]
    validation_case_count: Literal[16]
    answerable_case_count: Literal[50]
    unanswerable_case_count: Literal[14]
    challenge_quotas: AnswerableChallengeQuotas
    negative_subtype_quotas: NegativeSubtypeQuotas
    margin: BoundaryPolicyResult | None
    coverage: BoundaryPolicyResult | None
    composite: BoundaryPolicyResult | None
    primary_policy_version: Literal[
        "bm25-composite-boundary-gate-v2"
    ] = PRIMARY_POLICY_VERSION
    primary_acceptance_passed: bool | None
    failure_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9]*(?:Error|Exception)$",
        max_length=128,
    )
    limitations: list[
        Literal[
            "synthetic_data_only",
            "bm25_only",
            "llm_call_rate_is_estimate",
        ]
    ] = Field(
        default_factory=lambda: [
            "synthetic_data_only",
            "bm25_only",
            "llm_call_rate_is_estimate",
        ],
        min_length=3,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_report_invariants(self) -> BoundaryGateDevReport:
        if self.limitations != [
            "synthetic_data_only",
            "bm25_only",
            "llm_call_rate_is_estimate",
        ]:
            raise ValueError("report limitations must use the canonical aggregate set")

        results = (self.margin, self.coverage, self.composite)
        if self.status == "completed":
            if any(result is None for result in results):
                raise ValueError("completed reports require all policy results")
            if self.failure_type is not None:
                raise ValueError("completed reports forbid failure_type")
            assert self.margin is not None
            assert self.coverage is not None
            assert self.composite is not None
            if (
                self.margin.policy_version != "bm25-margin-gate-v2"
                or self.coverage.policy_version != "bm25-coverage-gate-v2"
                or self.composite.policy_version
                != "bm25-composite-boundary-gate-v2"
            ):
                raise ValueError("report policy identity mismatch")
            if self.composite.status != "completed":
                raise ValueError("completed reports require a completed composite")
            for result in results:
                assert result is not None
                if result.status == "failed":
                    continue
                assert result.calibration is not None
                assert result.validation_metrics is not None
                for metrics in (
                    result.calibration.metrics,
                    result.validation_metrics,
                ):
                    if (
                        set(metrics.per_challenge_gold_recall_at_5)
                        != _ANSWERABLE_CHALLENGES
                    ):
                        raise ValueError(
                            "canonical report requires every answerable challenge"
                        )
            assert self.composite.validation_metrics is not None
            expected_acceptance = _primary_acceptance(
                self.composite.validation_metrics
            )
            if self.primary_acceptance_passed is not expected_acceptance:
                raise ValueError("primary acceptance does not match composite metrics")
        elif self.status == "failed":
            if any(result is not None for result in results):
                raise ValueError("failed reports cannot contain policy results")
            if self.primary_acceptance_passed is not None:
                raise ValueError("failed reports cannot contain acceptance")
            if self.failure_type is None:
                raise ValueError("failed reports require a redacted failure type")
        else:
            if any(result is not None for result in results):
                raise ValueError("not-run reports cannot contain policy results")
            if self.primary_acceptance_passed is not None:
                raise ValueError("not-run reports cannot contain acceptance")
            if self.failure_type is not None:
                raise ValueError("not-run reports forbid failure_type")
        return self


def _raw_gold_recall(observation: BoundaryObservation) -> Fraction:
    if not observation.relevant_chunk_ids:
        raise ValueError("answerable observations require gold IDs")
    gold = set(observation.relevant_chunk_ids)
    return Fraction(
        len(gold & set(observation.retrieved_chunk_ids[:_TOP_K])),
        len(gold),
    )


def _aggregate_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def aggregate_boundary_metrics(
    observations: Sequence[BoundaryObservation],
) -> BoundaryGateMetrics:
    if not observations:
        raise ValueError("boundary metrics require observations")
    answerable = [item for item in observations if item.answerable]
    unanswerable = [item for item in observations if not item.answerable]
    if not answerable or not unanswerable:
        raise ValueError("boundary metrics require both answerability classes")
    if any(item.allowed is None for item in observations):
        raise ValueError("boundary metrics require a gate decision for every observation")
    if any(not item.relevant_chunk_ids for item in answerable):
        raise ValueError("answerable observations require gold IDs")
    if any(item.relevant_chunk_ids for item in unanswerable):
        raise ValueError("unanswerable observations forbid gold IDs")

    raw_recalls = [_raw_gold_recall(item) for item in answerable]
    post_recalls = [
        recall if item.allowed else Fraction()
        for item, recall in zip(answerable, raw_recalls, strict=True)
    ]
    response_rate = Fraction(
        sum(bool(item.allowed) for item in answerable),
        len(answerable),
    )
    no_answer_accuracy = Fraction(
        sum(not bool(item.allowed) for item in unanswerable),
        len(unanswerable),
    )
    raw_recall = sum(raw_recalls, start=Fraction()) / len(raw_recalls)
    post_recall = sum(post_recalls, start=Fraction()) / len(post_recalls)
    per_challenge: dict[str, Fraction] = {}
    for challenge in sorted(
        {item.challenge_type.value for item in answerable}
    ):
        values = [
            recall
            for item, recall in zip(answerable, post_recalls, strict=True)
            if item.challenge_type.value == challenge
        ]
        per_challenge[challenge] = sum(values, start=Fraction()) / len(values)

    return BoundaryGateMetrics(
        answerable_response_rate=float(response_rate),
        no_answer_accuracy=float(no_answer_accuracy),
        balanced_accuracy=float((response_rate + no_answer_accuracy) / 2),
        raw_gold_recall_at_5=float(raw_recall),
        post_gate_gold_recall_at_5=float(post_recall),
        gold_recall_loss=float(raw_recall - post_recall),
        llm_call_rate_estimate=float(
            Fraction(
                sum(bool(item.allowed) for item in observations),
                len(observations),
            )
        ),
        per_challenge_gold_recall_at_5={
            challenge: float(value) for challenge, value in per_challenge.items()
        },
        worst_challenge_gap=float(
            max(
                abs(post_recall - value) for value in per_challenge.values()
            )
        ),
    )


def candidate_thresholds(
    observations: Sequence[BoundaryObservation],
    feature_name: str = "top1_score",
    *,
    include_coverage_boundaries: bool = False,
) -> list[float]:
    fields = BoundaryFeatureVector.model_fields
    if feature_name not in fields or feature_name == "feature_version":
        raise ValueError("candidate field must be a continuous boundary feature")
    if not observations:
        raise ValueError("candidate generation requires calibration observations")
    _require_partition(
        observations,
        BoundaryPartition.CALIBRATION,
        label="calibration",
    )
    valid_features = [
        item.features for item in observations if item.features is not None
    ]
    if not valid_features:
        raise NoSurvivingConfigurationError
    values = sorted(
        float(getattr(item, feature_name)) for item in valid_features
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("candidate values must be finite")
    candidates = {
        values[round(step / 10 * (len(values) - 1))] for step in range(11)
    }
    if include_coverage_boundaries:
        candidates.update((0.0, 1.0))
    return sorted(candidates)


def _policy_version(policy_type: PolicyType) -> str:
    if policy_type is MarginGatePolicy:
        return "bm25-margin-gate-v2"
    if policy_type is CoverageGatePolicy:
        return "bm25-coverage-gate-v2"
    if policy_type is CompositeBoundaryGatePolicy:
        return "bm25-composite-boundary-gate-v2"
    raise TypeError("unsupported boundary policy type")


def _candidate_fields(
    observations: Sequence[BoundaryObservation],
    policy_type: PolicyType,
) -> tuple[list[str], list[list[float]]]:
    if policy_type is MarginGatePolicy:
        mapping = (
            ("min_top1_score", "top1_score", False),
            ("min_normalized_margin", "normalized_margin", False),
        )
    elif policy_type is CoverageGatePolicy:
        mapping = (
            ("min_anchor_coverage", "anchor_coverage", True),
            ("min_union_coverage", "union_top5_coverage", True),
            (
                "min_max_document_coverage",
                "max_document_coverage",
                True,
            ),
        )
    elif policy_type is CompositeBoundaryGatePolicy:
        mapping = (
            ("min_anchor_coverage", "anchor_coverage", True),
            ("min_union_coverage", "union_top5_coverage", True),
            ("min_top1_score", "top1_score", False),
            ("min_normalized_margin", "normalized_margin", False),
            ("high_coverage_override", "max_document_coverage", True),
        )
    else:
        raise TypeError("unsupported boundary policy type")
    names = [item[0] for item in mapping]
    values = [
        candidate_thresholds(
            observations,
            feature,
            include_coverage_boundaries=coverage,
        )
        for _, feature, coverage in mapping
    ]
    return names, values


def _allows(features: BoundaryFeatureVector, policy: Policy) -> bool:
    if isinstance(policy, MarginGatePolicy):
        return (
            features.top1_score >= policy.min_top1_score
            and features.normalized_margin >= policy.min_normalized_margin
        )
    if isinstance(policy, CoverageGatePolicy):
        return (
            features.anchor_coverage >= policy.min_anchor_coverage
            and features.union_top5_coverage >= policy.min_union_coverage
            and features.max_document_coverage
            >= policy.min_max_document_coverage
        )
    return (
        features.anchor_coverage >= policy.min_anchor_coverage
        and features.union_top5_coverage >= policy.min_union_coverage
        and features.top1_score >= policy.min_top1_score
        and (
            features.normalized_margin >= policy.min_normalized_margin
            or features.max_document_coverage >= policy.high_coverage_override
        )
    )


def _require_partition(
    observations: Sequence[BoundaryObservation],
    expected: BoundaryPartition,
    *,
    label: str,
) -> None:
    if any(item.split is not expected for item in observations):
        raise ValueError(
            f"{label} evaluation requires only the {expected.value} partition"
        )


def _evaluate_policy(
    observations: Sequence[BoundaryObservation],
    policy: Policy,
    *,
    expected_partition: BoundaryPartition,
) -> BoundaryGateMetrics:
    _require_partition(
        observations,
        expected_partition,
        label=expected_partition.value,
    )
    allowed = [
        False
        if item.forced_rejection_reason is not None
        else _allows(item.features, policy)
        for item in observations
    ]
    decided = [
        item.model_copy(update={"allowed": decision})
        for item, decision in zip(observations, allowed, strict=True)
    ]
    return aggregate_boundary_metrics(decided)


def evaluate_policy(
    observations: Sequence[BoundaryObservation],
    policy: Policy,
) -> BoundaryGateMetrics:
    return _evaluate_policy(
        observations,
        policy,
        expected_partition=BoundaryPartition.VALIDATION,
    )


def _threshold_serialization(policy: Policy) -> str:
    payload = policy.model_dump(exclude={"policy_version"})
    return json.dumps(
        tuple(payload.values()),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def select_candidate(
    candidates: Sequence[BoundaryPolicyCandidate],
) -> BoundaryPolicyCandidate:
    surviving = [
        item
        for item in candidates
        if item.metrics.answerable_response_rate
        >= _CALIBRATION_RESPONSE_FLOOR
        and item.metrics.post_gate_gold_recall_at_5
        >= _CALIBRATION_RECALL_FLOOR
    ]
    if not surviving:
        raise NoSurvivingConfigurationError
    return min(surviving, key=_candidate_sort_key)


def _candidate_sort_key(
    item: BoundaryPolicyCandidate,
) -> tuple[float, float, float, float, str]:
    return (
        -item.metrics.balanced_accuracy,
        -item.metrics.no_answer_accuracy,
        -item.metrics.post_gate_gold_recall_at_5,
        item.metrics.llm_call_rate_estimate,
        _threshold_serialization(item.policy),
    )


def calibrate_policy(
    observations: Sequence[BoundaryObservation],
    policy_type: PolicyType,
) -> BoundaryPolicyCalibration:
    _require_partition(
        observations,
        BoundaryPartition.CALIBRATION,
        label="calibration",
    )
    names, value_sets = _candidate_fields(observations, policy_type)
    candidate_count = 0
    surviving_count = 0
    selected: BoundaryPolicyCandidate | None = None
    for index, values in enumerate(product(*value_sets)):
        candidate_count += 1
        policy = policy_type(**dict(zip(names, values, strict=True)))
        metrics = _evaluate_policy(
            observations,
            policy,
            expected_partition=BoundaryPartition.CALIBRATION,
        )
        candidate = BoundaryPolicyCandidate(
            candidate_id=f"{_policy_version(policy_type)}-{index}",
            policy=policy,
            metrics=metrics,
        )
        if (
            metrics.answerable_response_rate >= _CALIBRATION_RESPONSE_FLOOR
            and metrics.post_gate_gold_recall_at_5
            >= _CALIBRATION_RECALL_FLOOR
        ):
            surviving_count += 1
            if selected is None or _candidate_sort_key(
                candidate
            ) < _candidate_sort_key(selected):
                selected = candidate
    if selected is None:
        raise NoSurvivingConfigurationError
    return BoundaryPolicyCalibration(
        policy_version=_policy_version(policy_type),
        candidate_count=candidate_count,
        surviving_candidate_count=surviving_count,
        selected_policy=selected.policy,
        metrics=selected.metrics,
    )


def calibrate_then_validate_policy(
    calibration_observations: Sequence[BoundaryObservation],
    validation_observations: Sequence[BoundaryObservation],
    policy_type: PolicyType,
) -> BoundaryPolicyResult:
    try:
        calibration = calibrate_policy(calibration_observations, policy_type)
    except NoSurvivingConfigurationError:
        return BoundaryPolicyResult(
            policy_version=_policy_version(policy_type),
            status="failed",
            failure_type="NoSurvivingConfigurationError",
        )
    return BoundaryPolicyResult(
        policy_version=calibration.policy_version,
        selected_policy=calibration.selected_policy,
        calibration=calibration,
        validation_metrics=evaluate_policy(
            validation_observations,
            calibration.selected_policy,
        ),
    )


def collect_boundary_observations(
    cases: Sequence[BoundaryDevCase],
    backend: RagBackend,
) -> list[BoundaryObservation]:
    observations: list[BoundaryObservation] = []
    for case in cases:
        retrieval = backend.retrieve(
            RetrievalRequest(query=case.question, top_k=_TOP_K)
        )
        retrieved_chunk_ids = [
            document.chunk_id for document in retrieval.documents[:_TOP_K]
        ]
        try:
            features = extract_boundary_features(
                case.question,
                retrieval,
                top_k=_TOP_K,
            )
            forced_rejection_reason = None
        except BoundaryRuntimeEvidenceError as error:
            features = None
            forced_rejection_reason = error.reason
        observations.append(
            BoundaryObservation(
                answerable=case.answerable,
                challenge_type=case.challenge_type,
                split=case.split,
                relevant_chunk_ids=case.relevant_chunk_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                features=features,
                forced_rejection_reason=forced_rejection_reason,
            )
        )
    return observations


def _primary_acceptance(metrics: BoundaryGateMetrics) -> bool:
    return (
        metrics.post_gate_gold_recall_at_5 >= 0.80
        and metrics.no_answer_accuracy >= 0.80
        and metrics.balanced_accuracy >= 0.80
        and metrics.gold_recall_loss <= 0.05
        and metrics.worst_challenge_gap <= 0.15
    )


def build_boundary_gate_dev_report(
    *,
    manifest: BoundaryDevManifest,
    margin: BoundaryPolicyResult | None = None,
    coverage: BoundaryPolicyResult | None = None,
    composite: BoundaryPolicyResult | None = None,
) -> BoundaryGateDevReport:
    manifest = BoundaryDevManifest.model_validate(manifest.model_dump())
    if manifest.question_sha256 != DATASET_SHA256:
        raise ValueError("canonical boundary dataset hash mismatch")
    if manifest.corpus_sha256 != CORPUS_SHA256:
        raise ValueError("canonical public corpus hash mismatch")
    if composite is None or composite.status != "completed":
        raise ValueError("completed report requires a completed composite policy")
    assert composite.validation_metrics is not None
    acceptance = _primary_acceptance(composite.validation_metrics)
    return BoundaryGateDevReport(
        status="completed",
        blueprint_version=manifest.blueprint_version,
        leakage_check_version=manifest.leakage_check_version,
        total_case_count=manifest.question_count,
        calibration_case_count=manifest.calibration_count,
        validation_case_count=manifest.validation_count,
        answerable_case_count=manifest.answerable_count,
        unanswerable_case_count=manifest.unanswerable_count,
        challenge_quotas=manifest.challenge_quotas,
        negative_subtype_quotas=manifest.negative_subtype_quotas,
        margin=margin,
        coverage=coverage,
        composite=composite,
        primary_acceptance_passed=acceptance,
        failure_type=None,
    )


def _canonical_report_quotas() -> tuple[
    AnswerableChallengeQuotas,
    NegativeSubtypeQuotas,
]:
    challenge = AnswerableChallengeQuota(total=10, calibration=8, validation=2)
    return (
        AnswerableChallengeQuotas(
            paraphrase=challenge,
            lexical_distractor=challenge,
            multi_evidence=challenge,
            conflict=challenge,
            temporal=challenge,
        ),
        NegativeSubtypeQuotas(
            near_domain_missing_attribute=NearDomainMissingAttributeQuota(
                total=8,
                calibration=5,
                validation=3,
            ),
            temporal_out_of_scope=TemporalOutOfScopeQuota(
                total=3,
                calibration=1,
                validation=2,
            ),
            entity_or_condition_out_of_scope=EntityOrConditionOutOfScopeQuota(
                total=3,
                calibration=2,
                validation=1,
            ),
        ),
    )


def _build_non_completed_boundary_gate_dev_report(
    *,
    status: Literal["failed", "not_run"],
    failure_type: str | None,
) -> BoundaryGateDevReport:
    challenge_quotas, negative_subtype_quotas = _canonical_report_quotas()
    return BoundaryGateDevReport(
        status=status,
        blueprint_version="boundary-blueprints-v2",
        leakage_check_version="boundary-leakage-v2",
        total_case_count=64,
        calibration_case_count=48,
        validation_case_count=16,
        answerable_case_count=50,
        unanswerable_case_count=14,
        challenge_quotas=challenge_quotas,
        negative_subtype_quotas=negative_subtype_quotas,
        margin=None,
        coverage=None,
        composite=None,
        primary_acceptance_passed=None,
        failure_type=failure_type,
    )


def build_not_run_boundary_gate_dev_report() -> BoundaryGateDevReport:
    return _build_non_completed_boundary_gate_dev_report(
        status="not_run",
        failure_type=None,
    )


def build_failed_boundary_gate_dev_report(
    failure: BaseException,
) -> BoundaryGateDevReport:
    if not isinstance(failure, BaseException):
        raise TypeError("failure must be an exception instance")
    failure_type = type(failure).__name__
    if not (
        failure_type
        and failure_type[0].isalpha()
        and failure_type.isascii()
        and failure_type.isalnum()
        and failure_type.endswith(("Error", "Exception"))
    ):
        failure_type = "RuntimeError"
    return _build_non_completed_boundary_gate_dev_report(
        status="failed",
        failure_type=failure_type,
    )
