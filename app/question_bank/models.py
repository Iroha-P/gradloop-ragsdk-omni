from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ingestion.feishu_snapshot import PrivacyLane


class QuestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuestionReviewStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_ANSWER = "needs_answer"
    NEEDS_REVIEW = "needs_review"
    CONFLICT = "conflict"


class QuestionOrigin(StrEnum):
    EXTRACTED = "extracted"
    SYNTHETIC = "synthetic"


class QuestionSource(QuestionModel):
    source_id: str = Field(min_length=1, max_length=256)
    title: str = Field(default="", max_length=512)
    section: str = Field(default="", max_length=512)
    page: int | None = Field(default=None, ge=1)
    revision_id: int | None = Field(default=None, ge=0)
    text: str
    privacy_lane: PrivacyLane


class SourceReference(QuestionModel):
    source_id: str = Field(min_length=1, max_length=256)
    section: str = Field(default="", max_length=512)
    page: int | None = Field(default=None, ge=1)
    revision_id: int | None = Field(default=None, ge=0)


class RubricPoint(QuestionModel):
    label: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1024)
    max_score: int = Field(ge=1, le=100)


class QuestionCandidate(QuestionModel):
    candidate_id: str
    canonical_text: str = Field(min_length=2, max_length=4096)
    normalized_text: str = Field(min_length=2, max_length=4096)
    answer_evidence: list[str] = Field(default_factory=list)
    source_ref: SourceReference
    privacy_lane: PrivacyLane
    review_status: QuestionReviewStatus
    content_fingerprint: str


class QuestionRecord(QuestionModel):
    question_id: str
    canonical_text: str = Field(min_length=2, max_length=4096)
    normalized_text: str = Field(min_length=2, max_length=4096)
    question_type: str = "interview"
    topic: str = "general"
    direction: str = "general"
    difficulty: str = Field(default="medium", pattern=r"^(easy|medium|hard)$")
    answer_points: list[str] = Field(default_factory=list)
    rubric: list[RubricPoint] = Field(default_factory=list)
    source_refs: list[SourceReference]
    privacy_lane: PrivacyLane
    origin: QuestionOrigin = QuestionOrigin.EXTRACTED
    content_fingerprint: str
    duplicate_group_id: str
    review_status: QuestionReviewStatus
    builder_version: str = "question-bank-v1"

    @field_validator("rubric")
    @classmethod
    def rubric_total_is_valid(cls, values: list[RubricPoint]) -> list[RubricPoint]:
        if values and sum(item.max_score for item in values) != 100:
            raise ValueError("rubric must total 100")
        return values


class DuplicateGroup(QuestionModel):
    duplicate_group_id: str
    member_candidate_ids: list[str]
    privacy_lane: PrivacyLane
    match_type: str


class AnswerVersion(QuestionModel):
    source_ref: SourceReference
    answer_points: list[str]


class AnswerConflict(QuestionModel):
    conflict_id: str
    duplicate_group_id: str
    conflict_type: str = "answer_disagreement"
    versions: list[AnswerVersion]


class SimilarityCandidate(QuestionModel):
    left_candidate_id: str
    right_candidate_id: str
    character_ngram_score: float = Field(ge=0, le=1)
    token_overlap_score: float = Field(ge=0, le=1)
    review_status: str = "needs_review"


class CanonicalizationResult(QuestionModel):
    questions: list[QuestionRecord]
    duplicate_groups: list[DuplicateGroup]
    conflicts: list[AnswerConflict]
    similarity_candidates: list[SimilarityCandidate] = Field(default_factory=list)


class DatasetSplit(QuestionModel):
    dev: list[QuestionRecord]
    test: list[QuestionRecord]
