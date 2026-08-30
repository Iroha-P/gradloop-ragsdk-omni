from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .models import QuestionRecord, QuestionReviewStatus


def write_question_bank(path: Path, questions: Iterable[QuestionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(questions, key=lambda item: item.question_id)
    payload = "".join(item.model_dump_json() + "\n" for item in rows)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


class QuestionBankRepository:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[QuestionRecord]:
        if not self.path.exists():
            return []
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("question bank path must be a regular file")
        records = [
            QuestionRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [item.question_id for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError("question bank contains duplicate question IDs")
        return records

    def select(
        self,
        *,
        topic: str | None,
        direction: str | None,
        difficulty: str | None,
        scope: str = "generic",
        count: int = 1,
    ) -> list[QuestionRecord]:
        if scope not in {"generic", "personal", "all"}:
            raise ValueError("invalid question bank scope")
        if count < 1:
            raise ValueError("count must be positive")
        allowed = (
            {"generic"}
            if scope == "generic"
            else {"personal"}
            if scope == "personal"
            else {"generic", "personal"}
        )
        rows = [
            item
            for item in self.load()
            if item.privacy_lane.value in allowed
            and item.review_status == QuestionReviewStatus.ACCEPTED
        ]
        if topic:
            needle = topic.casefold()
            rows = [
                item
                for item in rows
                if needle in f"{item.topic} {item.canonical_text}".casefold()
            ]
        if direction:
            rows = [item for item in rows if item.direction == direction]
        if difficulty:
            rows = [item for item in rows if item.difficulty == difficulty]
        return sorted(rows, key=lambda item: item.question_id)[:count]
