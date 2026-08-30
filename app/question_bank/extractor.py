from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import QuestionCandidate, QuestionReviewStatus, QuestionSource, SourceReference

_NUMBER_PREFIX = re.compile(
    r"^(?:(?:Q|Question|问题)\s*)?[0-9０-９一二三四五六七八九十]+"
    r"[.、:：)）-]?\s*",
    re.IGNORECASE,
)
_BARE_QUESTION_LABEL = re.compile(
    r"^(?:Q|Question|问题)\s*[0-9０-９一二三四五六七八九十]+[.、:：)）-]?$",
    re.IGNORECASE,
)
_QUESTION_WORDS = ("什么", "如何", "为什么", "是否", "请", "介绍", "哪些", "怎样")
_QUESTION_CONTEXT = ("问题", "高频", "qa", "问答", "题库", "面试题")


def normalize_question(text: str) -> str:
    value = _NUMBER_PREFIX.sub("", text.strip())
    value = re.sub(r"[\s\u3000]+", "", value)
    value = re.sub(r"[,，.。;；:：!！?？'\"“”‘’()（）\[\]【】]", "", value)
    return value.casefold()


def _clean_question(text: str) -> str:
    return _NUMBER_PREFIX.sub("", text.strip()).strip()


def _make_candidate(
    source: QuestionSource,
    question: str,
    answers: list[str],
    ordinal: int,
) -> QuestionCandidate:
    canonical = _clean_question(question)
    normalized = normalize_question(canonical)
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    identity = (
        f"{source.source_id}|{source.section}|{ordinal}|{canonical}|{fingerprint}"
    )
    evidence = [item.strip() for item in answers if item.strip()]
    return QuestionCandidate(
        candidate_id="cand_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        canonical_text=canonical,
        normalized_text=normalized,
        answer_evidence=evidence,
        source_ref=SourceReference(
            source_id=source.source_id,
            section=source.section,
            page=source.page,
            revision_id=source.revision_id,
        ),
        privacy_lane=source.privacy_lane,
        review_status=(
            QuestionReviewStatus.ACCEPTED
            if evidence
            else QuestionReviewStatus.NEEDS_ANSWER
        ),
        content_fingerprint=fingerprint,
    )


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _extract_markdown_tables(source: QuestionSource) -> list[QuestionCandidate]:
    lines = source.text.splitlines()
    results: list[QuestionCandidate] = []
    ordinal = 0
    index = 0
    while index + 1 < len(lines):
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        headers = [value.casefold() for value in _split_table_row(lines[index])]
        separator = _split_table_row(lines[index + 1])
        if not separator or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
            index += 1
            continue
        question_column = next(
            (i for i, value in enumerate(headers) if value in {"题目", "问题", "question"}),
            None,
        )
        answer_column = next(
            (i for i, value in enumerate(headers) if value in {"答案", "要点", "answer"}),
            None,
        )
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            cells = _split_table_row(lines[index])
            if question_column is not None and question_column < len(cells):
                question = cells[question_column]
                answer = (
                    cells[answer_column]
                    if answer_column is not None and answer_column < len(cells)
                    else ""
                )
                if question:
                    ordinal += 1
                    results.append(
                        _make_candidate(source, question, [answer] if answer else [], ordinal)
                    )
            index += 1
    return results


def _has_question_context(source: QuestionSource, heading: str = "") -> bool:
    context = f"{source.title} {source.section} {heading}".casefold()
    return any(marker in context for marker in _QUESTION_CONTEXT)


def _is_question_line(text: str, *, in_question_context: bool) -> bool:
    value = text.strip()
    if not value or _BARE_QUESTION_LABEL.fullmatch(value):
        return False
    cleaned = _clean_question(value)
    has_number = bool(_NUMBER_PREFIX.match(value))
    if value.endswith(("?", "？")):
        return True
    if any(word in cleaned for word in _QUESTION_WORDS) and (has_number or in_question_context):
        return True
    return has_number and in_question_context and len(cleaned) >= 4


def extract_questions(sources: Iterable[QuestionSource]) -> list[QuestionCandidate]:
    results: list[QuestionCandidate] = []
    for source in sources:
        source_results = _extract_markdown_tables(source)
        current_question: str | None = None
        answers: list[str] = []
        heading = ""
        ordinal = len(source_results)
        for raw in source.text.splitlines():
            line = raw.strip()
            if not line or line.startswith("|"):
                continue
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                if _BARE_QUESTION_LABEL.fullmatch(heading):
                    continue
                candidate_text = heading
            else:
                candidate_text = line.lstrip("*- ").strip()
            if _is_question_line(
                candidate_text,
                in_question_context=_has_question_context(source, heading),
            ):
                if current_question is not None:
                    ordinal += 1
                    source_results.append(
                        _make_candidate(source, current_question, answers, ordinal)
                    )
                current_question = candidate_text
                answers = []
            elif current_question is not None and not line.startswith("#"):
                answers.append(candidate_text)
        if current_question is not None:
            ordinal += 1
            source_results.append(_make_candidate(source, current_question, answers, ordinal))
        results.extend(source_results)
    return sorted(results, key=lambda item: (item.source_ref.source_id, item.candidate_id))
