from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.evaluation.retrieval import RetrievalBenchmarkCase, RetrievalChallengeType, StrictModel
from app.rag.base import DocumentInput

DATASET_VERSION = "public-synthetic-boundary-dev-v2"
BLUEPRINT_VERSION = "boundary-blueprints-v2"
TOKENIZER_VERSION = "bm25-boundary-tokenizer-v2"
LEAKAGE_CHECK_VERSION = "boundary-leakage-v2"
NEAR_DUPLICATE_THRESHOLD = 0.70


class BoundaryPartition(StrEnum):
    CALIBRATION = "calibration"
    VALIDATION = "validation"


class NegativeSubtype(StrEnum):
    NEAR_DOMAIN_MISSING_ATTRIBUTE = "near_domain_missing_attribute"
    TEMPORAL_OUT_OF_SCOPE = "temporal_out_of_scope"
    ENTITY_OR_CONDITION_OUT_OF_SCOPE = "entity_or_condition_out_of_scope"


class BoundaryDevCase(StrictModel):
    case_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4096)
    relevant_chunk_ids: list[str]
    answerable: bool
    split: BoundaryPartition
    challenge_type: RetrievalChallengeType
    reference_points: list[str] = Field(max_length=12)
    group_id: str = Field(min_length=1, max_length=128)
    pair_id: str | None = Field(default=None, min_length=1, max_length=128)
    intent_id: str = Field(min_length=1, max_length=128)
    negative_subtype: NegativeSubtype | None = None

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> BoundaryDevCase:
        if self.answerable:
            if not self.relevant_chunk_ids:
                raise ValueError("answerable cases require gold chunk IDs")
            if not self.reference_points:
                raise ValueError("answerable cases require reference points")
            if self.challenge_type not in _ANSWERABLE_CHALLENGES:
                raise ValueError("answerable cases require an answerable challenge type")
            if self.negative_subtype is not None:
                raise ValueError("answerable cases forbid negative_subtype")
        else:
            if self.relevant_chunk_ids:
                raise ValueError("unanswerable cases forbid gold chunk IDs")
            if self.reference_points:
                raise ValueError("unanswerable cases forbid reference points")
            if self.challenge_type != RetrievalChallengeType.UNANSWERABLE:
                raise ValueError("unanswerable cases require challenge_type=unanswerable")
            if self.pair_id is None:
                raise ValueError("unanswerable cases require pair_id")
            if self.negative_subtype is None:
                raise ValueError("unanswerable cases require negative_subtype")
        return self


class AnswerableChallengeQuota(StrictModel):
    total: Literal[10]
    calibration: Literal[8]
    validation: Literal[2]


class AnswerableChallengeQuotas(StrictModel):
    paraphrase: AnswerableChallengeQuota
    lexical_distractor: AnswerableChallengeQuota
    multi_evidence: AnswerableChallengeQuota
    conflict: AnswerableChallengeQuota
    temporal: AnswerableChallengeQuota


class NearDomainMissingAttributeQuota(StrictModel):
    total: Literal[8]
    calibration: Literal[5]
    validation: Literal[3]


class TemporalOutOfScopeQuota(StrictModel):
    total: Literal[3]
    calibration: Literal[1]
    validation: Literal[2]


class EntityOrConditionOutOfScopeQuota(StrictModel):
    total: Literal[3]
    calibration: Literal[2]
    validation: Literal[1]


class NegativeSubtypeQuotas(StrictModel):
    near_domain_missing_attribute: NearDomainMissingAttributeQuota
    temporal_out_of_scope: TemporalOutOfScopeQuota
    entity_or_condition_out_of_scope: EntityOrConditionOutOfScopeQuota


class BoundaryDevManifest(StrictModel):
    dataset_version: Literal["public-synthetic-boundary-dev-v2"]
    question_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    corpus_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    question_count: Literal[64]
    calibration_count: Literal[48]
    validation_count: Literal[16]
    answerable_count: Literal[50]
    unanswerable_count: Literal[14]
    challenge_quotas: AnswerableChallengeQuotas
    negative_subtype_quotas: NegativeSubtypeQuotas
    blueprint_version: Literal["boundary-blueprints-v2"]
    tokenizer_version: Literal["bm25-boundary-tokenizer-v2"]
    leakage_check_version: Literal["boundary-leakage-v2"]
    frozen_test_used_for_authoring: Literal[False]


class BoundaryDatasetIntegrity(StrictModel):
    question_count: int = Field(ge=0)
    calibration_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    within_dev_near_duplicate_count: int = Field(ge=0)
    frozen_case_count: int = Field(ge=0)
    exact_duplicate_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    maximum_normalized_3gram_jaccard: float = Field(ge=0.0, le=1.0)
    passed: bool


_ANSWERABLE_CHALLENGES = frozenset(
    {
        RetrievalChallengeType.PARAPHRASE,
        RetrievalChallengeType.LEXICAL_DISTRACTOR,
        RetrievalChallengeType.MULTI_EVIDENCE,
        RetrievalChallengeType.CONFLICT,
        RetrievalChallengeType.TEMPORAL,
    }
)
_NEGATIVE_QUOTAS = {
    NegativeSubtype.NEAR_DOMAIN_MISSING_ATTRIBUTE: (5, 3),
    NegativeSubtype.TEMPORAL_OUT_OF_SCOPE: (1, 2),
    NegativeSubtype.ENTITY_OR_CONDITION_OUT_OF_SCOPE: (2, 1),
}


def _normalized_question(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def normalized_char_ngram_jaccard(left: str, right: str, *, n: int = 3) -> float:
    if n <= 0:
        raise ValueError("n must be positive")

    def ngrams(value: str) -> set[str]:
        normalized = _normalized_question(value)
        if not normalized:
            return set()
        if len(normalized) < n:
            return {normalized}
        return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}

    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    if not left_ngrams and not right_ngrams:
        return 1.0
    if not left_ngrams or not right_ngrams:
        return 0.0
    return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)


def _validate_case_models(cases: Iterable[BoundaryDevCase]) -> None:
    for case in cases:
        BoundaryDevCase.model_validate(case.model_dump())


def _validate_exact_quotas(cases: Sequence[BoundaryDevCase]) -> None:
    total = len(cases)
    partitions = Counter(case.split for case in cases)
    answerable_count = sum(case.answerable for case in cases)
    unanswerable_count = total - answerable_count
    if (total, partitions[BoundaryPartition.CALIBRATION], partitions[BoundaryPartition.VALIDATION]) != (
        64,
        48,
        16,
    ):
        raise ValueError("boundary dataset must contain exactly 64 cases split 48/16")
    if (answerable_count, unanswerable_count) != (50, 14):
        raise ValueError("boundary dataset must contain exactly 50/14 answerable quotas")

    for challenge in _ANSWERABLE_CHALLENGES:
        challenge_cases = [case for case in cases if case.challenge_type == challenge]
        calibration_count = sum(
            case.split == BoundaryPartition.CALIBRATION for case in challenge_cases
        )
        validation_count = sum(
            case.split == BoundaryPartition.VALIDATION for case in challenge_cases
        )
        if (len(challenge_cases), calibration_count, validation_count) != (10, 8, 2):
            raise ValueError(f"challenge quota drift for {challenge.value}")

    negatives = [case for case in cases if not case.answerable]
    for subtype, expected in _NEGATIVE_QUOTAS.items():
        subtype_cases = [case for case in negatives if case.negative_subtype == subtype]
        observed = (
            sum(case.split == BoundaryPartition.CALIBRATION for case in subtype_cases),
            sum(case.split == BoundaryPartition.VALIDATION for case in subtype_cases),
        )
        if observed != expected:
            raise ValueError(f"negative subtype quota drift for {subtype.value}")


def _validate_identity_and_partition_integrity(cases: Sequence[BoundaryDevCase]) -> None:
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("boundary dataset contains duplicate case IDs")

    questions = [_normalized_question(case.question) for case in cases]
    if len(set(questions)) != len(questions):
        raise ValueError("boundary dataset contains duplicate normalized questions")

    for label, values in (
        ("group", ((case.group_id, case.split) for case in cases)),
        ("pair", ((case.pair_id, case.split) for case in cases if case.pair_id is not None)),
    ):
        partitions: dict[str, set[BoundaryPartition]] = defaultdict(set)
        for value, split in values:
            assert value is not None
            partitions[value].add(split)
        if any(len(splits) != 1 for splits in partitions.values()):
            raise ValueError(f"{label} crosses a boundary partition")

    pairs: dict[str, list[BoundaryDevCase]] = defaultdict(list)
    for case in cases:
        if case.pair_id is not None:
            pairs[case.pair_id].append(case)
    for paired_cases in pairs.values():
        if len(paired_cases) != 2:
            raise ValueError(
                "every non-null pair_id must identify exactly two paired cases"
            )
        if sum(item.answerable for item in paired_cases) != 1:
            raise ValueError("every pair requires one answerable and one unanswerable case")
        if len({item.group_id for item in paired_cases}) != 1:
            raise ValueError("every pair must share the same group")

    intent_gold_pairs: set[tuple[str, tuple[str, ...]]] = set()
    for case in cases:
        key = (case.intent_id, tuple(sorted(case.relevant_chunk_ids)))
        # Rejecting every repeated intent/gold pair is deliberately stricter than
        # template matching, so entity and numeral substitutions in any language
        # cannot evade the leakage check.
        if key in intent_gold_pairs:
            raise ValueError("duplicate same-intent gold pairing is not allowed")
        intent_gold_pairs.add(key)


def _validate_gold_and_evidence_families(
    cases: Sequence[BoundaryDevCase], documents: Sequence[DocumentInput]
) -> None:
    document_by_id = {document.chunk_id: document for document in documents}
    if len(document_by_id) != len(documents):
        raise ValueError("corpus chunk IDs must be unique")
    missing_gold = {
        chunk_id
        for case in cases
        for chunk_id in case.relevant_chunk_ids
        if chunk_id not in document_by_id
    }
    if missing_gold:
        raise ValueError("boundary dataset contains missing gold chunk IDs")

    for challenge in _ANSWERABLE_CHALLENGES:
        families = {
            document_by_id[chunk_id].source_id
            for case in cases
            if case.challenge_type == challenge
            for chunk_id in case.relevant_chunk_ids
        }
        if len(families) < 6:
            raise ValueError(f"{challenge.value} needs at least six evidence families")

    multi_family_cases = sum(
        len({document_by_id[chunk_id].source_id for chunk_id in case.relevant_chunk_ids}) >= 2
        for case in cases
        if case.challenge_type == RetrievalChallengeType.MULTI_EVIDENCE
    )
    if multi_family_cases < 5:
        raise ValueError("multi_evidence needs at least five cross-family cases")


def _validate_dev_similarity(cases: Sequence[BoundaryDevCase]) -> float:
    maximum_similarity = 0.0
    for left_index, left in enumerate(cases):
        for right in cases[left_index + 1 :]:
            if left.pair_id is not None and left.pair_id == right.pair_id:
                continue
            similarity = normalized_char_ngram_jaccard(left.question, right.question)
            maximum_similarity = max(maximum_similarity, similarity)
            if similarity > NEAR_DUPLICATE_THRESHOLD:
                raise ValueError("non-paired questions exceed the near-duplicate threshold")
    return maximum_similarity


def validate_boundary_dataset(
    cases: Sequence[BoundaryDevCase],
    documents: Sequence[DocumentInput],
    *,
    frozen_cases: Sequence[BoundaryDevCase | RetrievalBenchmarkCase] | None = None,
) -> BoundaryDatasetIntegrity:
    """Validate dev cases and return privacy-safe aggregate frozen leakage counts.

    Frozen cases are already-loaded benchmark records; only their question text is
    needed for comparison, so their schema is intentionally not revalidated here.
    """
    _validate_case_models(cases)
    _validate_exact_quotas(cases)
    _validate_identity_and_partition_integrity(cases)
    _validate_gold_and_evidence_families(cases, documents)
    maximum_similarity = _validate_dev_similarity(cases)

    frozen = list(frozen_cases or [])
    exact_duplicate_count = 0
    near_duplicate_count = 0
    dev_questions = {_normalized_question(case.question) for case in cases}
    for frozen_case in frozen:
        frozen_question = _normalized_question(frozen_case.question)
        if frozen_question in dev_questions:
            exact_duplicate_count += 1
            maximum_similarity = 1.0
            continue
        for case in cases:
            similarity = normalized_char_ngram_jaccard(case.question, frozen_case.question)
            maximum_similarity = max(maximum_similarity, similarity)
            if similarity > NEAR_DUPLICATE_THRESHOLD:
                near_duplicate_count += 1

    return BoundaryDatasetIntegrity(
        question_count=len(cases),
        calibration_count=sum(case.split == BoundaryPartition.CALIBRATION for case in cases),
        validation_count=sum(case.split == BoundaryPartition.VALIDATION for case in cases),
        answerable_count=sum(case.answerable for case in cases),
        unanswerable_count=sum(not case.answerable for case in cases),
        within_dev_near_duplicate_count=0,
        frozen_case_count=len(frozen),
        exact_duplicate_count=exact_duplicate_count,
        near_duplicate_count=near_duplicate_count,
        maximum_normalized_3gram_jaccard=maximum_similarity,
        passed=exact_duplicate_count == 0 and near_duplicate_count == 0,
    )


def _load_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}") from error
    return rows


def load_boundary_cases(path: Path) -> list[BoundaryDevCase]:
    return [BoundaryDevCase.model_validate(row) for row in _load_jsonl(path)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _partition_quota(cases: Sequence[BoundaryDevCase]) -> dict[str, int]:
    return {
        "total": len(cases),
        "calibration": sum(
            case.split == BoundaryPartition.CALIBRATION for case in cases
        ),
        "validation": sum(case.split == BoundaryPartition.VALIDATION for case in cases),
    }


def build_boundary_manifest(question_path: Path, corpus_path: Path) -> BoundaryDevManifest:
    cases = load_boundary_cases(question_path)
    documents = [DocumentInput.model_validate(row) for row in _load_jsonl(corpus_path)]
    integrity = validate_boundary_dataset(cases, documents)
    challenge_quotas = {
        challenge.value: _partition_quota(
            [case for case in cases if case.challenge_type == challenge]
        )
        for challenge in _ANSWERABLE_CHALLENGES
    }
    negative_subtype_quotas = {
        subtype.value: _partition_quota(
            [case for case in cases if case.negative_subtype == subtype]
        )
        for subtype in NegativeSubtype
    }
    return BoundaryDevManifest(
        dataset_version=DATASET_VERSION,
        question_sha256=sha256_file(question_path),
        corpus_sha256=sha256_file(corpus_path),
        question_count=integrity.question_count,
        calibration_count=integrity.calibration_count,
        validation_count=integrity.validation_count,
        answerable_count=integrity.answerable_count,
        unanswerable_count=integrity.unanswerable_count,
        challenge_quotas=AnswerableChallengeQuotas.model_validate(challenge_quotas),
        negative_subtype_quotas=NegativeSubtypeQuotas.model_validate(
            negative_subtype_quotas
        ),
        blueprint_version=BLUEPRINT_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        leakage_check_version=LEAKAGE_CHECK_VERSION,
        frozen_test_used_for_authoring=False,
    )
