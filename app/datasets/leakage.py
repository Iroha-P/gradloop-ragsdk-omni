from __future__ import annotations

import hashlib
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Literal

from .models import QA_TASKS

MAX_TOTAL_TEXT_CHARS = 200_000
MAX_REFERENCE_COUNT = 500
MAX_POSTING_SIZE = 128
MAX_CANDIDATE_PAIRS = 10_000
MAX_SCORE_OPERATIONS = 200_000

_POLICY_VALUES = (12, 0.35, 0.86)
_REQUIRED_FIELDS = frozenset({"sample_id", "task", "question", "answer", "rubric", "rights_class"})
_OPTIONAL_FIELDS = frozenset({"group_id", "split", "reference_texts"})
_RIGHTS_CLASSES = frozenset({"user_owned_derived", "open_licensed", "synthetic"})
_SPLITS = frozenset({"train", "dev", "test"})
_SEMANTIC_ALIASES = {
    "authoritative": "official",
    "criteria": "requirement",
    "criterion": "requirement",
    "requirements": "requirement",
    "review": "check",
    "reviewed": "check",
    "reviewing": "check",
    "prior": "before",
    "sending": "submit",
    "submitted": "submit",
    "submitting": "submit",
    "send": "submit",
    "applications": "application",
    "every": "each",
    "inside": "within",
    "later": "subsequent",
}
_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "with",
        "的",
        "了",
        "和",
        "在",
        "是",
    }
)
_TOKEN_SHINGLE_SIZE = 12
_CHAR_NGRAM_SIZE = 5


@dataclass(frozen=True)
class LeakagePolicy:
    max_contiguous_words: int = 12
    max_char_ngram_jaccard: float = 0.35
    max_semantic_similarity: float = 0.86
    version: str = field(default="v1", init=False)

    def __post_init__(self) -> None:
        current = (
            self.max_contiguous_words,
            self.max_char_ngram_jaccard,
            self.max_semantic_similarity,
        )
        if current != _POLICY_VALUES:
            raise ValueError("unsupported_leakage_policy")


ReportStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class ValidationReport:
    status: ReportStatus
    input_count: int
    valid_count: int
    invalid_count: int
    schema_validity: float
    finding_count: int
    finding_categories: dict[str, int]
    affected_sample_hashes: dict[str, list[str]]
    split_counts: dict[str, int]
    schema_version: str = field(
        default="candidate-qa-validation-report.v1",
        init=False,
    )
    threshold_version: str = field(default="v1", init=False)
    classification: str = field(default="private_only", init=False)
    uploadable: bool = field(default=False, init=False)

    def model_dump(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "threshold_version": self.threshold_version,
            "classification": self.classification,
            "uploadable": self.uploadable,
            "status": self.status,
            "input_count": self.input_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "schema_validity": self.schema_validity,
            "finding_count": self.finding_count,
            "finding_categories": self.finding_categories,
            "affected_sample_hashes": self.affected_sample_hashes,
            "split_counts": self.split_counts,
        }


@dataclass(frozen=True)
class _CoreRecord:
    row_index: int
    sample_id: str
    sample_hash: str
    question: str
    answer: str
    references: tuple[str, ...]
    group_id: str | None
    split: str | None


@dataclass(frozen=True)
class _TextFeatures:
    normalized: str
    tokens: tuple[str, ...]
    token_shingles: frozenset[tuple[str, ...]]
    char_ngrams: frozenset[str]
    semantic: Counter[str]
    supported: bool

    @property
    def scorable(self) -> bool:
        return bool(self.supported and self.tokens and self.char_ngrams and self.semantic)


@dataclass(frozen=True)
class _PreparedRecord:
    core: _CoreRecord
    question: _TextFeatures
    answer: _TextFeatures
    combined: _TextFeatures
    template: str
    references: tuple[_TextFeatures, ...]

    @property
    def scorable(self) -> bool:
        return bool(
            self.question.scorable
            and self.answer.scorable
            and all(reference.scorable for reference in self.references)
        )


@dataclass(frozen=True)
class _FieldEntry:
    record_index: int
    features: _TextFeatures


class _ComparisonBudgetExceeded(RuntimeError):
    pass


def category_failure_report(
    category: str,
    *,
    input_count: int = 0,
) -> ValidationReport:
    return ValidationReport(
        status="failed",
        input_count=input_count,
        valid_count=0,
        invalid_count=input_count,
        schema_validity=0.0,
        finding_count=1,
        finding_categories={category: 1},
        affected_sample_hashes={},
        split_counts={},
    )


def _sample_hash(sample_id: str, row_index: int) -> str:
    seed = sample_id if sample_id else f"invalid-row-{row_index}"
    return f"sha256:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x323AF
    )


def _is_latin_letter(character: str) -> bool:
    return character.isalpha() and unicodedata.name(
        character,
        "",
    ).startswith("LATIN ")


def _flush_latin_numeric_run(
    run: list[str],
    tokens: list[str],
) -> None:
    if not run:
        return
    value = "".join(run).strip("_-")
    run.clear()
    if not value:
        return
    has_digit = any(character.isdigit() for character in value)
    has_latin = any(_is_latin_letter(character) for character in value)
    if has_digit and has_latin:
        tokens.append("<id>")
    elif has_digit:
        tokens.append("<num>")
    elif has_latin:
        tokens.append(value)


def _tokenize_normalized(text: str) -> tuple[tuple[str, ...], bool]:
    tokens: list[str] = []
    run: list[str] = []
    supported = True
    for character in text:
        if _is_han(character):
            _flush_latin_numeric_run(run, tokens)
            tokens.append(character)
        elif _is_latin_letter(character) or character.isdigit() or (character in {"_", "-"} and bool(run)):
            run.append(character)
        else:
            _flush_latin_numeric_run(run, tokens)
            if character.isalpha():
                supported = False
    _flush_latin_numeric_run(run, tokens)
    return tuple(tokens), supported


def _word_tokens(text: str) -> tuple[str, ...]:
    tokens, _ = _tokenize_normalized(_normalized_text(text))
    return tokens


def _semantic_token(lexical_unit: str) -> str:
    normalized_unit = _SEMANTIC_ALIASES.get(lexical_unit, lexical_unit)
    if normalized_unit.startswith("<"):
        return normalized_unit
    if len(normalized_unit) > 5 and normalized_unit.endswith("ing"):
        normalized_unit = normalized_unit[:-3]
    elif len(normalized_unit) > 4 and normalized_unit.endswith("ed"):
        normalized_unit = normalized_unit[:-2]
    elif len(normalized_unit) > 4 and normalized_unit.endswith("s"):
        normalized_unit = normalized_unit[:-1]
    return _SEMANTIC_ALIASES.get(normalized_unit, normalized_unit)


def _semantic_counter(tokens: Iterable[str]) -> Counter[str]:
    normalized = (_semantic_token(lexical_unit) for lexical_unit in tokens)
    return Counter(
        lexical_unit
        for lexical_unit in normalized
        if not lexical_unit.startswith("<") and lexical_unit not in _SEMANTIC_STOPWORDS
    )


def _semantic_tokens(text: str) -> Counter[str]:
    return _semantic_counter(_word_tokens(text))


def _token_shingles(
    tokens: tuple[str, ...],
    *,
    size: int = _TOKEN_SHINGLE_SIZE,
) -> frozenset[tuple[str, ...]]:
    if len(tokens) < size:
        return frozenset()
    return frozenset(tokens[index : index + size] for index in range(len(tokens) - size + 1))


def _char_ngrams_from_tokens(
    tokens: Iterable[str],
    *,
    size: int = _CHAR_NGRAM_SIZE,
) -> frozenset[str]:
    compact = "".join(lexical_unit.replace("<", "").replace(">", "") for lexical_unit in tokens)
    if len(compact) < size:
        return frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def _char_ngrams(
    text: str,
    *,
    size: int = _CHAR_NGRAM_SIZE,
) -> frozenset[str]:
    return _char_ngrams_from_tokens(_word_tokens(text), size=size)


def _analyze_normalized(text: str) -> _TextFeatures:
    tokens, supported = _tokenize_normalized(text)
    return _TextFeatures(
        normalized=text,
        tokens=tokens,
        token_shingles=_token_shingles(tokens),
        char_ngrams=_char_ngrams_from_tokens(tokens),
        semantic=_semantic_counter(tokens),
        supported=supported,
    )


def _combine_features(
    question: _TextFeatures,
    answer: _TextFeatures,
) -> _TextFeatures:
    tokens = question.tokens + answer.tokens
    return _TextFeatures(
        normalized=f"{question.normalized}\n{answer.normalized}",
        tokens=tokens,
        token_shingles=_token_shingles(tokens),
        char_ngrams=question.char_ngrams | answer.char_ngrams,
        semantic=question.semantic + answer.semantic,
        supported=question.supported and answer.supported,
    )


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    numerator = sum(left[semantic_unit] * right[semantic_unit] for semantic_unit in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _threshold_fraction(threshold: float) -> Fraction:
    return Fraction(str(threshold))


def _cosine_numerator_at_or_above(
    numerator: int,
    left_norm_squared: int,
    right_norm_squared: int,
    threshold: Fraction,
) -> bool:
    if numerator <= 0 or left_norm_squared <= 0 or right_norm_squared <= 0:
        return False
    scaled_numerator = threshold.denominator * numerator
    return (
        scaled_numerator * scaled_numerator
        >= threshold.numerator * threshold.numerator * left_norm_squared * right_norm_squared
    )


def _cosine_at_or_above(
    left: Counter[str],
    right: Counter[str],
    threshold: float,
) -> bool:
    if not left or not right:
        return False
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    numerator = sum(count * larger.get(semantic_unit, 0) for semantic_unit, count in smaller.items())
    return _cosine_numerator_at_or_above(
        numerator,
        sum(count * count for count in left.values()),
        sum(count * count for count in right.values()),
        _threshold_fraction(threshold),
    )


def _jaccard_at_or_above(
    left: frozenset[str],
    right: frozenset[str],
    threshold: float,
) -> bool:
    if not left or not right:
        return False
    intersection_size = len(left & right)
    return _jaccard_sizes_at_or_above(
        intersection_size,
        len(left),
        len(right),
        _threshold_fraction(threshold),
    )


def _jaccard_sizes_at_or_above(
    intersection_size: int,
    left_size: int,
    right_size: int,
    threshold: Fraction,
) -> bool:
    if intersection_size <= 0:
        return False
    union_size = left_size + right_size - intersection_size
    return threshold.denominator * intersection_size >= threshold.numerator * union_size


def _longest_common_contiguous_words(
    left: frozenset[tuple[str, ...]],
    right: frozenset[tuple[str, ...]],
) -> int:
    return _TOKEN_SHINGLE_SIZE if left & right else 0


def _template_text(text: str) -> str:
    normalized = _normalized_text(text)
    output: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        value = "".join(run).strip("_-")
        run.clear()
        if not value:
            return
        has_digit = any(character.isdigit() for character in value)
        has_latin = any(_is_latin_letter(character) for character in value)
        if has_digit and has_latin:
            output.append("<id>")
        elif has_digit:
            output.append("<num>")
        else:
            output.append(value)

    for character in normalized:
        if _is_latin_letter(character) or character.isdigit() or (character in {"_", "-"} and bool(run)):
            run.append(character)
        else:
            flush()
            output.append(character)
    flush()
    return "".join(output)


def _is_scorable(text: str) -> bool:
    normalized = _normalized_text(text)
    return len(normalized) >= 4 and any(character.isalpha() for character in normalized)


def _is_canonical_group_component(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value)
    canonical = " ".join(normalized.split())
    return bool(canonical) and value == canonical


def _record_is_valid(record: object) -> bool:
    if not isinstance(record, Mapping):
        return False
    keys = set(record)
    if not _REQUIRED_FIELDS.issubset(keys) or not keys.issubset(_REQUIRED_FIELDS | _OPTIONAL_FIELDS):
        return False

    for field_name in (
        "sample_id",
        "task",
        "question",
        "answer",
        "rights_class",
    ):
        value = record[field_name]
        if not isinstance(value, str) or not value.strip():
            return False
    if record["task"] not in QA_TASKS:
        return False
    if record["rights_class"] not in _RIGHTS_CLASSES:
        return False

    rubric = record["rubric"]
    if (
        not isinstance(rubric, list)
        or not rubric
        or any(not isinstance(item, str) or not item.strip() or not _is_scorable(item) for item in rubric)
    ):
        return False

    has_group = "group_id" in record
    has_split = "split" in record
    if has_group != has_split:
        return False
    if has_group:
        group_value = record["group_id"]
        if not isinstance(group_value, str):
            return False
        if not _is_canonical_group_component(group_value):
            return False
        if not isinstance(record["split"], str):
            return False
        if record["split"] not in _SPLITS:
            return False

    if "reference_texts" in record:
        references = record["reference_texts"]
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(reference, str) or not reference.strip() for reference in references)
        ):
            return False
    return True


def _extract_core(
    record: object,
    row_index: int,
    sample_hash: str,
) -> _CoreRecord | None:
    if not isinstance(record, Mapping):
        return None
    sample_id = record.get("sample_id")
    question = record.get("question")
    answer = record.get("answer")
    if not isinstance(sample_id, str) or not sample_id.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str) or not answer.strip():
        return None

    references_value = record.get("reference_texts", ())
    if "reference_texts" in record:
        if (
            not isinstance(references_value, list)
            or not references_value
            or any(not isinstance(reference, str) or not reference.strip() for reference in references_value)
        ):
            return None
    references = tuple(str(reference) for reference in references_value)

    has_group = "group_id" in record
    has_split = "split" in record
    if has_group != has_split:
        return None
    group_id: str | None = None
    split: str | None = None
    if has_group:
        group_value = record["group_id"]
        split_value = record["split"]
        if not isinstance(group_value, str):
            return None
        if not _is_canonical_group_component(group_value):
            return None
        if not isinstance(split_value, str) or split_value not in _SPLITS:
            return None
        group_id = group_value
        split = split_value

    return _CoreRecord(
        row_index=row_index,
        sample_id=sample_id,
        sample_hash=sample_hash,
        question=question,
        answer=answer,
        references=references,
        group_id=group_id,
        split=split,
    )


def _prepare_records(
    cores: Sequence[_CoreRecord],
) -> tuple[list[_PreparedRecord], set[str]]:
    reference_count = sum(len(core.references) for core in cores)
    if reference_count > MAX_REFERENCE_COUNT:
        raise _ComparisonBudgetExceeded

    total_chars = 0
    prepared: list[_PreparedRecord] = []
    unscorable: set[str] = set()
    for core in cores:
        normalized_question = _normalized_text(core.question)
        normalized_answer = _normalized_text(core.answer)
        normalized_references = tuple(_normalized_text(reference) for reference in core.references)
        total_chars += len(normalized_question) + len(normalized_answer)
        total_chars += sum(len(reference) for reference in normalized_references)
        if total_chars > MAX_TOTAL_TEXT_CHARS:
            raise _ComparisonBudgetExceeded

        question = _analyze_normalized(normalized_question)
        answer = _analyze_normalized(normalized_answer)
        references = tuple(_analyze_normalized(reference) for reference in normalized_references)
        combined = _combine_features(question, answer)
        item = _PreparedRecord(
            core=core,
            question=question,
            answer=answer,
            combined=combined,
            template=_template_text(combined.normalized),
            references=references,
        )
        prepared.append(item)
        if not item.scorable:
            unscorable.add(core.sample_hash)
    return prepared, unscorable


def _add_posting(
    postings: defaultdict[object, list[int]],
    feature: object,
    field_index: int,
) -> None:
    posting = postings[feature]
    posting.append(field_index)
    if len(posting) > MAX_POSTING_SIZE:
        raise _ComparisonBudgetExceeded


def _is_low_information_semantic_feature(feature: object) -> bool:
    return isinstance(feature, str) and len(feature) == 1 and _is_han(feature)


def _add_pairs_from_posting(
    posting: Sequence[int],
    fields: Sequence[_FieldEntry],
    pairs: set[tuple[int, int]],
    posting_pair_operations: int,
) -> int:
    if len(posting) < 2:
        return posting_pair_operations
    for left_field_index, right_field_index in combinations(posting, 2):
        posting_pair_operations += 1
        if posting_pair_operations > MAX_SCORE_OPERATIONS:
            raise _ComparisonBudgetExceeded
        left_record_index = fields[left_field_index].record_index
        right_record_index = fields[right_field_index].record_index
        if left_record_index == right_record_index:
            continue
        pair = tuple(sorted((left_record_index, right_record_index)))
        pairs.add(pair)
        if len(pairs) > MAX_CANDIDATE_PAIRS:
            raise _ComparisonBudgetExceeded
    return posting_pair_operations


def _char_jaccard_similarity_join(
    fields: Sequence[_FieldEntry],
    char_postings: Mapping[str, Sequence[int]],
    pairs: set[tuple[int, int]],
    posting_pair_operations: int,
    char_threshold: float,
) -> int:
    """Add record pairs with a field-level Jaccard at the fixed threshold."""
    feature_order = {
        ngram: order
        for order, ngram in enumerate(
            sorted(
                char_postings,
                key=lambda ngram: (
                    len(char_postings[ngram]),
                    ngram,
                ),
            )
        )
    }
    ordered_features = [
        tuple(
            sorted(
                field_entry.features.char_ngrams,
                key=feature_order.__getitem__,
            )
        )
        for field_entry in fields
    ]
    ordered_ranks = [tuple(feature_order[ngram] for ngram in features) for features in ordered_features]
    inverted_index: defaultdict[str, list[int]] = defaultdict(list)
    threshold = _threshold_fraction(char_threshold)

    for field_index, field_entry in enumerate(fields):
        features = field_entry.features.char_ngrams
        feature_count = len(features)
        required_overlap = (
            threshold.numerator * feature_count + threshold.denominator - 1
        ) // threshold.denominator
        prefix_length = feature_count - required_overlap + 1
        prefix = ordered_features[field_index][:prefix_length]
        cutoff_rank = feature_order[prefix[-1]]
        prefix_intersections: defaultdict[int, int] = defaultdict(int)

        for ngram in prefix:
            for previous_field_index in inverted_index[ngram]:
                posting_pair_operations += 1
                if posting_pair_operations > MAX_SCORE_OPERATIONS:
                    raise _ComparisonBudgetExceeded
                if fields[previous_field_index].record_index == field_entry.record_index:
                    continue
                previous_count = len(fields[previous_field_index].features.char_ngrams)
                if threshold.numerator * max(feature_count, previous_count) > threshold.denominator * min(
                    feature_count, previous_count
                ):
                    continue
                prefix_intersections[previous_field_index] += 1

        tail_size = feature_count - prefix_length
        for previous_field_index, prefix_intersection in sorted(prefix_intersections.items()):
            previous_count = len(fields[previous_field_index].features.char_ngrams)
            previous_suffix_size = previous_count - bisect_right(
                ordered_ranks[previous_field_index],
                cutoff_rank,
            )
            maximum_intersection = prefix_intersection + min(tail_size, previous_suffix_size)
            if not _jaccard_sizes_at_or_above(
                maximum_intersection,
                feature_count,
                previous_count,
                threshold,
            ):
                continue
            posting_pair_operations += min(
                feature_count,
                previous_count,
            )
            if posting_pair_operations > MAX_SCORE_OPERATIONS:
                raise _ComparisonBudgetExceeded
            exact_intersection = len(features & fields[previous_field_index].features.char_ngrams)
            if not _jaccard_sizes_at_or_above(
                exact_intersection,
                feature_count,
                previous_count,
                threshold,
            ):
                continue
            pair = tuple(
                sorted(
                    (
                        field_entry.record_index,
                        fields[previous_field_index].record_index,
                    )
                )
            )
            pairs.add(pair)
            if len(pairs) > MAX_CANDIDATE_PAIRS:
                raise _ComparisonBudgetExceeded

        for ngram in ordered_features[field_index]:
            inverted_index[ngram].append(field_index)
    return posting_pair_operations


def _semantic_similarity_join(
    fields: Sequence[_FieldEntry],
    semantic_postings: Mapping[str, Sequence[int]],
    pairs: set[tuple[int, int]],
    posting_pair_operations: int,
    semantic_threshold: float,
) -> int:
    """Add every record pair whose field cosine can reach the fixed threshold.

    A globally ordered L2 prefix guarantees recall: without a shared prefix
    feature, the dot product is bounded by the query tail norm, which is
    strictly below the threshold. Posting hits are accumulated per prior field,
    then each surviving sparse pair receives one exact dot-product evaluation.
    """
    feature_order = {
        semantic_unit: order
        for order, semantic_unit in enumerate(
            sorted(
                semantic_postings,
                key=lambda semantic_unit: (
                    len(semantic_postings[semantic_unit]),
                    semantic_unit,
                ),
            )
        )
    }
    squared_norms: list[int] = []
    maximum_counts: list[int] = []
    for field_entry in fields:
        semantic = field_entry.features.semantic
        squared_norms.append(sum(count * count for count in semantic.values()))
        maximum_counts.append(max(semantic.values(), default=0))

    inverted_index: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    threshold = _threshold_fraction(semantic_threshold)
    for field_index, field_entry in enumerate(fields):
        semantic = field_entry.features.semantic
        squared_norm = squared_norms[field_index]
        ordered_features = sorted(
            semantic.items(),
            key=lambda item: feature_order[item[0]],
        )
        remaining_squared_norm = squared_norm
        prefix_length = 0
        for _semantic_unit, count in ordered_features:
            prefix_length += 1
            remaining_squared_norm -= count * count
            if (
                threshold.denominator * threshold.denominator * remaining_squared_norm
                < threshold.numerator * threshold.numerator * squared_norm
            ):
                break

        prefix_contributions: defaultdict[int, int] = defaultdict(int)
        for semantic_unit, count in ordered_features[:prefix_length]:
            for previous_field_index, previous_count in inverted_index[semantic_unit]:
                posting_pair_operations += 1
                if posting_pair_operations > MAX_SCORE_OPERATIONS:
                    raise _ComparisonBudgetExceeded
                if fields[previous_field_index].record_index == field_entry.record_index:
                    continue
                prefix_contributions[previous_field_index] += count * previous_count

        tail_l1_count = sum(count for _semantic_unit, count in ordered_features[prefix_length:])
        for previous_field_index, prefix_dot in sorted(prefix_contributions.items()):
            upper_numerator = prefix_dot + tail_l1_count * maximum_counts[previous_field_index]
            if not _cosine_numerator_at_or_above(
                upper_numerator,
                squared_norm,
                squared_norms[previous_field_index],
                threshold,
            ):
                continue
            posting_pair_operations += min(
                len(semantic),
                len(fields[previous_field_index].features.semantic),
            )
            if posting_pair_operations > MAX_SCORE_OPERATIONS:
                raise _ComparisonBudgetExceeded
            previous_semantic = fields[previous_field_index].features.semantic
            smaller, larger = (
                (semantic, previous_semantic)
                if len(semantic) <= len(previous_semantic)
                else (previous_semantic, semantic)
            )
            numerator = sum(count * larger.get(semantic_unit, 0) for semantic_unit, count in smaller.items())
            if not _cosine_numerator_at_or_above(
                numerator,
                squared_norm,
                squared_norms[previous_field_index],
                threshold,
            ):
                continue
            pair = tuple(
                sorted(
                    (
                        field_entry.record_index,
                        fields[previous_field_index].record_index,
                    )
                )
            )
            pairs.add(pair)
            if len(pairs) > MAX_CANDIDATE_PAIRS:
                raise _ComparisonBudgetExceeded

        for semantic_unit, count in ordered_features:
            inverted_index[semantic_unit].append((field_index, count))
    return posting_pair_operations


def _candidate_pairs(
    prepared: Sequence[_PreparedRecord],
    unscorable: set[str],
    policy: LeakagePolicy,
) -> tuple[set[tuple[int, int]], int]:
    shingle_postings: defaultdict[object, list[int]] = defaultdict(list)
    char_postings: defaultdict[str, list[int]] = defaultdict(list)
    semantic_postings: defaultdict[str, list[int]] = defaultdict(list)
    fields: list[_FieldEntry] = []

    for record_index, record in enumerate(prepared):
        if record.core.sample_hash in unscorable:
            continue
        for features in (record.question, record.answer):
            field_index = len(fields)
            fields.append(
                _FieldEntry(
                    record_index=record_index,
                    features=features,
                )
            )
            for shingle in features.token_shingles:
                _add_posting(shingle_postings, shingle, field_index)
            for ngram in features.char_ngrams:
                char_postings[ngram].append(field_index)
            for semantic_unit in features.semantic:
                posting = semantic_postings[semantic_unit]
                posting.append(field_index)
                if len(posting) > MAX_POSTING_SIZE and not _is_low_information_semantic_feature(
                    semantic_unit
                ):
                    raise _ComparisonBudgetExceeded

    pairs: set[tuple[int, int]] = set()
    posting_pair_operations = 0
    for posting in shingle_postings.values():
        posting_pair_operations = _add_pairs_from_posting(
            posting,
            fields,
            pairs,
            posting_pair_operations,
        )

    posting_pair_operations = _char_jaccard_similarity_join(
        fields,
        char_postings,
        pairs,
        posting_pair_operations,
        policy.max_char_ngram_jaccard,
    )
    posting_pair_operations = _semantic_similarity_join(
        fields,
        semantic_postings,
        pairs,
        posting_pair_operations,
        policy.max_semantic_similarity,
    )
    return pairs, posting_pair_operations


def _comparison_cost(left: _TextFeatures, right: _TextFeatures) -> int:
    return (
        3
        + min(len(left.token_shingles), len(right.token_shingles))
        + len(left.char_ngrams)
        + len(right.char_ngrams)
        + min(len(left.char_ngrams), len(right.char_ngrams))
        + len(left.semantic)
        + len(right.semantic)
        + min(len(left.semantic), len(right.semantic))
    )


def _enforce_score_budget(
    prepared: Sequence[_PreparedRecord],
    pairs: Iterable[tuple[int, int]],
    posting_pair_operations: int,
) -> None:
    operations = posting_pair_operations
    for left_index, right_index in pairs:
        for left_field in (
            prepared[left_index].question,
            prepared[left_index].answer,
        ):
            for right_field in (
                prepared[right_index].question,
                prepared[right_index].answer,
            ):
                operations += _comparison_cost(
                    left_field,
                    right_field,
                )
                if operations > MAX_SCORE_OPERATIONS:
                    raise _ComparisonBudgetExceeded
    for record in prepared:
        if not record.scorable:
            continue
        for reference in record.references:
            operations += _comparison_cost(record.question, reference)
            operations += _comparison_cost(record.answer, reference)
            if operations > MAX_SCORE_OPERATIONS:
                raise _ComparisonBudgetExceeded


def validate_candidate_records(
    records: Sequence[object],
    *,
    policy: LeakagePolicy | None = None,
) -> ValidationReport:
    selected_policy = LeakagePolicy() if policy is None else policy
    if not isinstance(selected_policy, LeakagePolicy):
        raise ValueError("unsupported_leakage_policy")

    input_count = len(records)
    categories: Counter[str] = Counter()
    affected: defaultdict[str, set[str]] = defaultdict(set)
    safety_finding_count = 0
    valid_count = 0
    cores: list[_CoreRecord] = []

    def add_finding(
        category: str,
        *sample_hashes: str,
        safety: bool = True,
    ) -> None:
        nonlocal safety_finding_count
        categories[category] += 1
        affected[category].update(sample_hashes)
        if safety:
            safety_finding_count += 1

    for row_index, record in enumerate(records):
        raw_sample_id = ""
        if isinstance(record, Mapping):
            candidate_id = record.get("sample_id")
            if isinstance(candidate_id, str):
                raw_sample_id = candidate_id
        hashed_id = _sample_hash(raw_sample_id, row_index)

        is_valid = _record_is_valid(record)
        if is_valid:
            valid_count += 1
        else:
            add_finding("schema_invalid", hashed_id, safety=False)

        if isinstance(record, Mapping):
            has_group = "group_id" in record
            has_split = "split" in record
            if has_group != has_split:
                add_finding("ambiguous_split_metadata", hashed_id)
            if has_group:
                group_value = record["group_id"]
                if not isinstance(group_value, str) or not _is_canonical_group_component(group_value):
                    add_finding("ambiguous_group_value", hashed_id)

        core = _extract_core(record, row_index, hashed_id)
        if core is None:
            add_finding("schema_unscannable", hashed_id)
        else:
            cores.append(core)

    if input_count == 0:
        add_finding("empty_input", safety=False)

    split_counts: Counter[str] = Counter()

    def finalize() -> ValidationReport:
        invalid_count = input_count - valid_count
        schema_validity = valid_count / input_count if input_count else 0.0
        if schema_validity < 0.99:
            add_finding("schema_validity_below_threshold", safety=False)
        finding_categories = dict(sorted(categories.items()))
        finding_count = sum(finding_categories.values())
        status: ReportStatus = "passed" if schema_validity >= 0.99 and safety_finding_count == 0 else "failed"
        return ValidationReport(
            status=status,
            input_count=input_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            schema_validity=schema_validity,
            finding_count=finding_count,
            finding_categories=finding_categories,
            affected_sample_hashes={
                category: sorted(hashes) for category, hashes in sorted(affected.items()) if hashes
            },
            split_counts=dict(sorted(split_counts.items())),
        )

    try:
        prepared, unscorable = _prepare_records(cores)
    except _ComparisonBudgetExceeded:
        add_finding("comparison_budget_exceeded")
        return finalize()

    for sample_hash in sorted(unscorable):
        add_finding("unscorable_similarity", sample_hash)

    metadata_presence = [record.core.group_id is not None for record in prepared]
    if metadata_presence and any(metadata_presence) and not all(metadata_presence):
        add_finding(
            "ambiguous_split_metadata",
            *(record.core.sample_hash for record in prepared),
        )

    ids: defaultdict[str, list[str]] = defaultdict(list)
    contents: defaultdict[str, list[str]] = defaultdict(list)
    templates: defaultdict[str, list[str]] = defaultdict(list)
    group_splits: defaultdict[
        str,
        defaultdict[str, list[str]],
    ] = defaultdict(lambda: defaultdict(list))

    for record in prepared:
        core = record.core
        ids[core.sample_id].append(core.sample_hash)
        contents[record.combined.normalized].append(core.sample_hash)
        templates[record.template].append(core.sample_hash)
        if core.group_id is not None and core.split is not None:
            group_splits[core.group_id][core.split].append(core.sample_hash)
            split_counts[core.split] += 1

    for hashes in ids.values():
        if len(hashes) > 1:
            add_finding("duplicate_sample_id", *hashes)
    for hashes in contents.values():
        if len(hashes) > 1:
            add_finding("duplicate_content", *hashes)
    for hashes in templates.values():
        if len(hashes) > 1:
            add_finding("template_near_duplicate", *hashes)
    for split_members in group_splits.values():
        if len(split_members) > 1:
            add_finding(
                "cross_split_group_contamination",
                *(sample_hash for hashes in split_members.values() for sample_hash in hashes),
            )

    try:
        pairs, posting_operations = _candidate_pairs(
            prepared,
            unscorable,
            selected_policy,
        )
        _enforce_score_budget(prepared, pairs, posting_operations)
    except _ComparisonBudgetExceeded:
        add_finding("comparison_budget_exceeded")
        return finalize()

    for left_index, right_index in sorted(pairs):
        left = prepared[left_index]
        right = prepared[right_index]
        left_hash = left.core.sample_hash
        right_hash = right.core.sample_hash
        if left.combined.normalized == right.combined.normalized:
            continue
        if left.template == right.template:
            continue
        max_contiguous_words = 0
        char_ngram_hit = False
        semantic_hit = False
        for left_field in (left.question, left.answer):
            for right_field in (right.question, right.answer):
                max_contiguous_words = max(
                    max_contiguous_words,
                    _longest_common_contiguous_words(
                        left_field.token_shingles,
                        right_field.token_shingles,
                    ),
                )
                if _jaccard_at_or_above(
                    left_field.char_ngrams,
                    right_field.char_ngrams,
                    selected_policy.max_char_ngram_jaccard,
                ):
                    char_ngram_hit = True
                if _cosine_at_or_above(
                    left_field.semantic,
                    right_field.semantic,
                    selected_policy.max_semantic_similarity,
                ):
                    semantic_hit = True
        if max_contiguous_words >= selected_policy.max_contiguous_words:
            add_finding("contiguous_word_reuse", left_hash, right_hash)
        if char_ngram_hit:
            add_finding("char_ngram_near_copy", left_hash, right_hash)
        if semantic_hit:
            add_finding("semantic_near_paraphrase", left_hash, right_hash)

    for record in prepared:
        if not record.scorable:
            continue
        for reference in record.references:
            for candidate in (record.question, record.answer):
                if (
                    _longest_common_contiguous_words(
                        candidate.token_shingles,
                        reference.token_shingles,
                    )
                    >= selected_policy.max_contiguous_words
                ):
                    add_finding(
                        "contiguous_word_reuse",
                        record.core.sample_hash,
                    )
                if _jaccard_at_or_above(
                    candidate.char_ngrams,
                    reference.char_ngrams,
                    selected_policy.max_char_ngram_jaccard,
                ):
                    add_finding(
                        "char_ngram_near_copy",
                        record.core.sample_hash,
                    )
                if _cosine_at_or_above(
                    candidate.semantic,
                    reference.semantic,
                    selected_policy.max_semantic_similarity,
                ):
                    add_finding(
                        "semantic_near_paraphrase",
                        record.core.sample_hash,
                    )

    return finalize()
