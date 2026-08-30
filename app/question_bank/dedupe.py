from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Protocol

from .llm_enrichment import QuestionEnrichment, SimilarityReview
from .models import (
    AnswerConflict,
    AnswerVersion,
    CanonicalizationResult,
    DatasetSplit,
    DuplicateGroup,
    QuestionCandidate,
    QuestionOrigin,
    QuestionRecord,
    QuestionReviewStatus,
    SimilarityCandidate,
)


class SimilarityReviewer(Protocol):
    def review_similarity(self, left: str, right: str) -> SimilarityReview: ...


def _stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _ngrams(value: str, size: int) -> set[str]:
    if len(value) <= size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _similarity(left: QuestionCandidate, right: QuestionCandidate) -> tuple[float, float]:
    character_score = _jaccard(
        _ngrams(left.normalized_text, 3),
        _ngrams(right.normalized_text, 3),
    )
    token_score = _jaccard(
        _ngrams(left.normalized_text, 2),
        _ngrams(right.normalized_text, 2),
    )
    return character_score, token_score


def _status_for(members: list[QuestionCandidate], conflict: bool) -> QuestionReviewStatus:
    if conflict:
        return QuestionReviewStatus.CONFLICT
    rank = {
        QuestionReviewStatus.ACCEPTED: 0,
        QuestionReviewStatus.NEEDS_ANSWER: 1,
        QuestionReviewStatus.NEEDS_REVIEW: 2,
        QuestionReviewStatus.CONFLICT: 3,
    }
    return max((item.review_status for item in members), key=rank.__getitem__)


def canonicalize_questions(
    candidates: list[QuestionCandidate],
    enrichments: dict[str, QuestionEnrichment] | None = None,
    similarity_reviewer: SimilarityReviewer | None = None,
) -> CanonicalizationResult:
    enrichments = enrichments or {}
    exact: dict[tuple[str, str], list[QuestionCandidate]] = defaultdict(list)
    for candidate in candidates:
        exact[(candidate.privacy_lane.value, candidate.normalized_text)].append(candidate)
    base_groups = [members for _key, members in sorted(exact.items())]
    parent = list(range(len(base_groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    similarity_candidates: list[SimilarityCandidate] = []
    pending_review: set[int] = set()
    reviewed_semantic_pairs: set[tuple[int, int]] = set()
    for left_index, left_group in enumerate(base_groups):
        for right_index in range(left_index + 1, len(base_groups)):
            right_group = base_groups[right_index]
            left, right = left_group[0], right_group[0]
            if left.privacy_lane != right.privacy_lane:
                continue
            character_score, token_score = _similarity(left, right)
            if character_score < 0.82 or token_score < 0.75:
                continue
            pair = SimilarityCandidate(
                left_candidate_id=left.candidate_id,
                right_candidate_id=right.candidate_id,
                character_ngram_score=round(character_score, 6),
                token_overlap_score=round(token_score, 6),
            )
            if similarity_reviewer:
                review = similarity_reviewer.review_similarity(
                    left.normalized_text,
                    right.normalized_text,
                )
                if review.equivalent and review.confidence >= 0.85:
                    union(left_index, right_index)
                    reviewed_semantic_pairs.add((left_index, right_index))
                    pair = pair.model_copy(update={"review_status": "equivalent"})
                else:
                    pending_review.update((left_index, right_index))
            else:
                pending_review.update((left_index, right_index))
            similarity_candidates.append(pair)

    merged: dict[int, list[QuestionCandidate]] = defaultdict(list)
    base_indices: dict[int, set[int]] = defaultdict(set)
    for index, members in enumerate(base_groups):
        root = find(index)
        merged[root].extend(members)
        base_indices[root].add(index)

    questions: list[QuestionRecord] = []
    duplicate_groups: list[DuplicateGroup] = []
    conflicts: list[AnswerConflict] = []
    for root, members in sorted(merged.items()):
        first = members[0]
        group_value = f"{first.privacy_lane.value}|" + "|".join(
            sorted({item.normalized_text for item in members})
        )
        group_id = _stable_id("dup_", group_value)
        answer_versions = {
            tuple(item.answer_evidence)
            for item in members
            if item.answer_evidence
        }
        has_conflict = len(answer_versions) > 1
        status = _status_for(members, has_conflict)
        if not has_conflict and base_indices[root] & pending_review:
            status = QuestionReviewStatus.NEEDS_REVIEW
        enrichment = enrichments.get(first.candidate_id)
        references = {
            json.dumps(item.source_ref.model_dump(mode="json"), sort_keys=True): item.source_ref
            for item in members
        }
        answer_points = list(
            dict.fromkeys(point for item in members for point in item.answer_evidence)
        )
        if enrichment:
            if enrichment.answer_points:
                answer_points = enrichment.answer_points
            if status == QuestionReviewStatus.ACCEPTED:
                status = enrichment.review_status
        record = QuestionRecord(
            question_id=_stable_id("q_", group_value),
            canonical_text=first.canonical_text,
            normalized_text=first.normalized_text,
            question_type=enrichment.question_type if enrichment else "interview",
            topic=enrichment.topic if enrichment else "general",
            direction=enrichment.direction if enrichment else "general",
            difficulty=enrichment.difficulty if enrichment else "medium",
            answer_points=answer_points,
            rubric=enrichment.rubric if enrichment else [],
            source_refs=list(references.values()),
            privacy_lane=first.privacy_lane,
            origin=QuestionOrigin.EXTRACTED,
            content_fingerprint=first.content_fingerprint,
            duplicate_group_id=group_id,
            review_status=status,
        )
        questions.append(record)
        match_type = "semantic" if any(
            left in base_indices[root] and right in base_indices[root]
            for left, right in reviewed_semantic_pairs
        ) else "exact"
        if len(members) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    duplicate_group_id=group_id,
                    member_candidate_ids=sorted(item.candidate_id for item in members),
                    privacy_lane=first.privacy_lane,
                    match_type=match_type,
                )
            )
        if has_conflict:
            conflicts.append(
                AnswerConflict(
                    conflict_id=_stable_id("conf_", group_id),
                    duplicate_group_id=group_id,
                    versions=[
                        AnswerVersion(
                            source_ref=item.source_ref,
                            answer_points=item.answer_evidence,
                        )
                        for item in members
                        if item.answer_evidence
                    ],
                )
            )
    return CanonicalizationResult(
        questions=sorted(questions, key=lambda item: item.question_id),
        duplicate_groups=sorted(
            duplicate_groups,
            key=lambda item: item.duplicate_group_id,
        ),
        conflicts=sorted(conflicts, key=lambda item: item.conflict_id),
        similarity_candidates=similarity_candidates,
    )


def split_by_duplicate_group(
    questions: list[QuestionRecord],
    *,
    test_ratio: float,
    seed: str,
) -> DatasetSplit:
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")
    group_ids = sorted({item.duplicate_group_id for item in questions})
    random.Random(seed).shuffle(group_ids)
    test_size = round(len(group_ids) * test_ratio)
    test_groups = set(group_ids[:test_size])
    return DatasetSplit(
        dev=[item for item in questions if item.duplicate_group_id not in test_groups],
        test=[item for item in questions if item.duplicate_group_id in test_groups],
    )
