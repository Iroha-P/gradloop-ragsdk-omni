from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import BackendUnavailableError
from app.llm.base import LlmClient

from .models import QuestionCandidate, QuestionReviewStatus, RubricPoint


class QuestionEnrichment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str
    direction: str
    difficulty: str = Field(pattern=r"^(easy|medium|hard)$")
    question_type: str
    answer_points: list[str]
    rubric: list[RubricPoint]
    evidence_quotes: list[str]
    review_status: QuestionReviewStatus = QuestionReviewStatus.ACCEPTED


class SimilarityReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool
    confidence: float = Field(ge=0, le=1)
    evidence: str


class QuestionEnricher:
    def __init__(self, llm: LlmClient):
        self.llm = llm

    def enrich(self, candidate: QuestionCandidate) -> QuestionEnrichment:
        if not candidate.answer_evidence:
            return self._fallback(candidate, QuestionReviewStatus.NEEDS_ANSWER)
        evidence = "\n".join(candidate.answer_evidence)
        for _attempt in range(2):
            try:
                response = self.llm.chat(
                    system=(
                        "只根据证据输出一个 JSON 对象，不要 Markdown、注释或额外字段。"
                        "difficulty must be exactly easy, medium, or hard. "
                        "rubric 必须是 JSON 数组；不能可靠评分时返回空数组。"
                        "answer_points 与 evidence_quotes 中的每一项都必须逐字复制自 evidence，"
                        "不得改写或添加事实。严格按以下形状输出："
                        '{"topic":"主题","direction":"方向","difficulty":"medium",'
                        '"question_type":"interview","answer_points":[],"rubric":[],'
                        '"evidence_quotes":[]}'
                    ),
                    user=json.dumps(
                        {
                            "question": candidate.canonical_text,
                            "evidence": candidate.answer_evidence,
                        },
                        ensure_ascii=False,
                    ),
                )
            except BackendUnavailableError:
                return self._fallback(candidate, QuestionReviewStatus.NEEDS_REVIEW)
            try:
                result = QuestionEnrichment.model_validate_json(response.content)
            except ValidationError:
                continue
            grounded_quotes = result.evidence_quotes and all(
                quote in evidence for quote in result.evidence_quotes
            )
            grounded_points = all(point in evidence for point in result.answer_points)
            if grounded_quotes and grounded_points:
                return result
        return self._fallback(candidate, QuestionReviewStatus.NEEDS_REVIEW)

    def review_similarity(self, left: str, right: str) -> SimilarityReview:
        payload = json.dumps({"left": left, "right": right}, ensure_ascii=False)
        for _attempt in range(2):
            try:
                response = self.llm.chat(
                    system="只判断两个题干是否语义等价，输出严格 JSON。",
                    user=payload,
                )
            except BackendUnavailableError:
                break
            try:
                return SimilarityReview.model_validate_json(response.content)
            except ValidationError:
                continue
        return SimilarityReview(
            equivalent=False,
            confidence=0,
            evidence="invalid_model_output",
        )

    @staticmethod
    def _fallback(
        candidate: QuestionCandidate,
        status: QuestionReviewStatus,
    ) -> QuestionEnrichment:
        return QuestionEnrichment(
            topic="general",
            direction="general",
            difficulty="medium",
            question_type="interview",
            answer_points=list(candidate.answer_evidence),
            rubric=[],
            evidence_quotes=list(candidate.answer_evidence),
            review_status=status,
        )
