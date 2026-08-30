from __future__ import annotations

import math
import re
from enum import StrEnum
from itertools import pairwise
from typing import Literal

from pydantic import Field

from app.rag.base import RetrievalResult, StrictModel

FeatureVersion = Literal["bm25-boundary-features-v2"]
MarginPolicyVersion = Literal["bm25-margin-gate-v2"]
CoveragePolicyVersion = Literal["bm25-coverage-gate-v2"]
CompositePolicyVersion = Literal["bm25-composite-boundary-gate-v2"]

_FEATURE_VERSION = "bm25-boundary-features-v2"
_REQUIRED_BACKEND = "baseline"
_REQUIRED_BACKEND_VERSION = "bm25-v1"
_REQUIRED_TOP_K = 5
_LEXICAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?%?|[A-Za-z0-9]+)(?![A-Za-z0-9])"
)
_CHINESE_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


class BoundaryFeatureVector(StrictModel):
    feature_version: FeatureVersion = _FEATURE_VERSION
    top1_score: float = Field(allow_inf_nan=False)
    top2_score: float = Field(allow_inf_nan=False)
    absolute_margin: float = Field(allow_inf_nan=False)
    normalized_margin: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    max_document_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    union_top5_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    anchor_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class BoundaryGateReason(StrEnum):
    PASSED = "passed"
    NO_DOCUMENTS = "no_documents"
    INSUFFICIENT_DOCUMENTS = "insufficient_documents"
    SELECTED_COUNT_MISMATCH = "selected_count_mismatch"
    NON_FINITE_SCORE = "non_finite_score"
    NEGATIVE_SCORE = "negative_score"
    SCORE_ORDER_INVALID = "score_order_invalid"
    ZERO_TOP1_SCORE = "zero_top1_score"
    THRESHOLD_NOT_MET = "threshold_not_met"


class BoundaryGateDecision(StrictModel):
    allowed: bool
    reason: BoundaryGateReason


class MarginGatePolicy(StrictModel):
    policy_version: MarginPolicyVersion = "bm25-margin-gate-v2"
    min_top1_score: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    min_normalized_margin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class CoverageGatePolicy(StrictModel):
    policy_version: CoveragePolicyVersion = "bm25-coverage-gate-v2"
    min_anchor_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    min_union_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    min_max_document_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class CompositeBoundaryGatePolicy(StrictModel):
    policy_version: CompositePolicyVersion = "bm25-composite-boundary-gate-v2"
    min_anchor_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    min_union_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    min_top1_score: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    min_normalized_margin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    high_coverage_override: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class BoundaryRuntimeEvidenceError(ValueError):
    def __init__(self, reason: BoundaryGateReason) -> None:
        super().__init__("Retrieval evidence is invalid for boundary-gate evaluation.")
        self.reason = reason


def _validate_protocol(retrieval: RetrievalResult, *, top_k: int) -> None:
    if (
        retrieval.backend != _REQUIRED_BACKEND
        or retrieval.backend_version != _REQUIRED_BACKEND_VERSION
        or top_k != _REQUIRED_TOP_K
    ):
        raise ValueError(
            "Boundary gate requires BM25 Top-5 with backend='baseline', "
            "backend_version='bm25-v1', and top_k=5."
        )


def _validated_scores(retrieval: RetrievalResult) -> list[float]:
    documents = retrieval.documents
    if not documents:
        raise BoundaryRuntimeEvidenceError(BoundaryGateReason.NO_DOCUMENTS)
    if len(documents) < 2:
        raise BoundaryRuntimeEvidenceError(
            BoundaryGateReason.INSUFFICIENT_DOCUMENTS
        )
    if (
        retrieval.retrieval.selected_count != len(documents)
        or len(documents) > _REQUIRED_TOP_K
    ):
        raise BoundaryRuntimeEvidenceError(
            BoundaryGateReason.SELECTED_COUNT_MISMATCH
        )

    scores = [document.score for document in documents]
    if any(not math.isfinite(score) for score in scores):
        raise BoundaryRuntimeEvidenceError(BoundaryGateReason.NON_FINITE_SCORE)
    if any(score < 0.0 for score in scores):
        raise BoundaryRuntimeEvidenceError(BoundaryGateReason.NEGATIVE_SCORE)
    if any(current > previous for previous, current in pairwise(scores)):
        raise BoundaryRuntimeEvidenceError(
            BoundaryGateReason.SCORE_ORDER_INVALID
        )
    if scores[0] == 0.0:
        raise BoundaryRuntimeEvidenceError(BoundaryGateReason.ZERO_TOP1_SCORE)
    return scores


def _lexical_tokens(text: str) -> set[str]:
    normalized = text.lower().replace("％", "%")
    tokens: set[str] = set()
    for match in _LEXICAL_TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if token[-1:] == "%" or token[0].isdigit() or len(token) >= 2:
            tokens.add(token)
    return tokens


def _informative_tokens(text: str) -> set[str]:
    tokens = _lexical_tokens(text)
    for match in _CHINESE_RUN_RE.finditer(text):
        run = match.group(0)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _coverage(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def extract_boundary_features(
    query: str,
    retrieval: RetrievalResult,
    *,
    top_k: int,
) -> BoundaryFeatureVector:
    _validate_protocol(retrieval, top_k=top_k)
    scores = _validated_scores(retrieval)

    query_tokens = _informative_tokens(query)
    anchor_tokens = _lexical_tokens(query)
    document_token_sets = [
        _informative_tokens(
            "\n".join((document.title, document.section, document.text))
        )
        for document in retrieval.documents
    ]
    union_tokens: set[str] = set().union(*document_token_sets)

    top1_score, top2_score = scores[:2]
    absolute_margin = top1_score - top2_score
    return BoundaryFeatureVector(
        top1_score=top1_score,
        top2_score=top2_score,
        absolute_margin=absolute_margin,
        normalized_margin=absolute_margin / top1_score,
        max_document_coverage=max(
            _coverage(query_tokens, document_tokens)
            for document_tokens in document_token_sets
        ),
        union_top5_coverage=_coverage(query_tokens, union_tokens),
        anchor_coverage=(
            _coverage(anchor_tokens, union_tokens) if anchor_tokens else 1.0
        ),
    )


def _policy_allows(
    features: BoundaryFeatureVector,
    policy: MarginGatePolicy | CoverageGatePolicy | CompositeBoundaryGatePolicy,
) -> bool:
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


def decide_boundary_gate(
    query: str,
    retrieval: RetrievalResult,
    policy: MarginGatePolicy | CoverageGatePolicy | CompositeBoundaryGatePolicy,
    *,
    top_k: int,
) -> BoundaryGateDecision:
    _validate_protocol(retrieval, top_k=top_k)
    try:
        features = extract_boundary_features(query, retrieval, top_k=top_k)
    except BoundaryRuntimeEvidenceError as exc:
        return BoundaryGateDecision(allowed=False, reason=exc.reason)

    allowed = _policy_allows(features, policy)
    return BoundaryGateDecision(
        allowed=allowed,
        reason=(
            BoundaryGateReason.PASSED
            if allowed
            else BoundaryGateReason.THRESHOLD_NOT_MET
        ),
    )
