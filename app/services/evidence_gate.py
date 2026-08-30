from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import Field

from app.rag.base import RetrievalResult, StrictModel
from app.services.boundary_gate import (
    BoundaryGateDecision,
    CompositeBoundaryGatePolicy,
    decide_boundary_gate,
)


class EvidenceGateReason(StrEnum):
    DISABLED = "disabled"
    NO_DOCUMENTS = "no_documents"
    BELOW_THRESHOLD = "below_threshold"
    PASSED = "passed"


class EvidenceGatePolicy(StrictModel):
    policy_version: Literal["bm25-top1-threshold-v1"] = "bm25-top1-threshold-v1"
    enabled: bool = False
    score_threshold: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class EvidenceGateDecision(StrictModel):
    allowed: bool
    top_score: float | None = None
    threshold: float = Field(ge=0.0)
    reason: EvidenceGateReason


EvidenceGatePolicyLike: TypeAlias = (
    EvidenceGatePolicy | CompositeBoundaryGatePolicy
)
EvidenceGateDecisionLike: TypeAlias = EvidenceGateDecision | BoundaryGateDecision


def decide_evidence_gate(
    retrieval: RetrievalResult,
    policy: EvidenceGatePolicy,
) -> EvidenceGateDecision:
    if not policy.enabled:
        return EvidenceGateDecision(
            allowed=True,
            threshold=policy.score_threshold,
            reason=EvidenceGateReason.DISABLED,
        )
    if not retrieval.documents:
        return EvidenceGateDecision(
            allowed=False,
            threshold=policy.score_threshold,
            reason=EvidenceGateReason.NO_DOCUMENTS,
        )
    top_score = retrieval.documents[0].score
    allowed = top_score >= policy.score_threshold
    return EvidenceGateDecision(
        allowed=allowed,
        top_score=top_score,
        threshold=policy.score_threshold,
        reason=EvidenceGateReason.PASSED if allowed else EvidenceGateReason.BELOW_THRESHOLD,
    )


def is_composite_policy(policy: EvidenceGatePolicyLike) -> bool:
    return isinstance(policy, CompositeBoundaryGatePolicy)


def is_gate_active(policy: EvidenceGatePolicyLike) -> bool:
    return is_composite_policy(policy) or policy.enabled


def decide_gate(
    query: str,
    retrieval: RetrievalResult,
    policy: EvidenceGatePolicyLike,
    *,
    top_k: int,
) -> EvidenceGateDecisionLike:
    if isinstance(policy, CompositeBoundaryGatePolicy):
        return decide_boundary_gate(
            query,
            retrieval,
            policy,
            top_k=top_k,
        )
    return decide_evidence_gate(retrieval, policy)
