from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.domain.grading import GradeResult
from app.domain.mistakes import MasteryStatus, MistakeRecord
from app.domain.plans import PlanRevisionProposal, StudyPlan
from app.domain.practice import PracticeQuestion

SCHEMA_VERSION = 1


class RecordNotFoundError(LookupError):
    pass


class UserScopeError(PermissionError):
    pass


class SQLiteLearningStore:
    """Small transactional learning-state store with explicit user scoping."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    current_level TEXT NOT NULL,
                    weak_tags_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_plans_user ON plans(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS questions (
                    question_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    answer_redacted TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES questions(question_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS mistakes (
                    mistake_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    next_review_date TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mistakes_user ON mistakes(user_id, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS plan_revisions (
                    revision_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(plan_id) REFERENCES plans(plan_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_revisions_user ON plan_revisions(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS tool_runs (
                    trace_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_summary_json TEXT NOT NULL,
                    output_summary_json TEXT NOT NULL,
                    error_type TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    PRIMARY KEY(trace_id, sequence)
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _payload(model) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)

    def upsert_profile(self, user_id: str, current_level: str, weak_tags: list[str]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles(user_id, current_level, weak_tags_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    current_level=excluded.current_level,
                    weak_tags_json=excluded.weak_tags_json,
                    updated_at=excluded.updated_at
                """,
                (user_id, current_level, json.dumps(weak_tags, ensure_ascii=False), now),
            )

    def get_profile(self, user_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT current_level, weak_tags_json, updated_at FROM user_profiles WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            return {"user_id": user_id, "current_level": "beginner", "weak_tags": []}
        return {
            "user_id": user_id,
            "current_level": row["current_level"],
            "weak_tags": json.loads(row["weak_tags_json"]),
            "updated_at": row["updated_at"],
        }

    def add_weak_tags(self, user_id: str, tags: list[str]) -> list[str]:
        profile = self.get_profile(user_id)
        combined = list(dict.fromkeys([*profile["weak_tags"], *tags]))
        self.upsert_profile(user_id, profile["current_level"], combined)
        return combined

    def save_plan(self, plan: StudyPlan) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO plans(plan_id, user_id, status, payload_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    plan.plan_id,
                    plan.user_id,
                    plan.status.value,
                    self._payload(plan),
                    plan.created_at.isoformat(),
                    plan.updated_at.isoformat(),
                ),
            )

    def get_plan(self, plan_id: str, user_id: str) -> StudyPlan:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, payload_json FROM plans WHERE plan_id=?",
                (plan_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("plan not found")
        if row["user_id"] != user_id:
            raise UserScopeError("plan does not belong to this user")
        return StudyPlan.model_validate_json(row["payload_json"])

    def latest_active_plan(self, user_id: str) -> StudyPlan | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM plans
                WHERE user_id=? AND status='active'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return StudyPlan.model_validate_json(row["payload_json"]) if row else None

    def save_question(self, user_id: str, question: PracticeQuestion) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO questions(question_id, user_id, payload_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (question.question_id, user_id, self._payload(question), question.created_at.isoformat()),
            )

    def get_question(self, question_id: str, user_id: str) -> PracticeQuestion:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, payload_json FROM questions WHERE question_id=?",
                (question_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("question not found")
        if row["user_id"] != user_id:
            raise UserScopeError("question does not belong to this user")
        return PracticeQuestion.model_validate_json(row["payload_json"])

    def save_attempt(self, answer_redacted: str, result: GradeResult) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO attempts(
                    attempt_id, user_id, question_id, answer_redacted, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    result.attempt_id,
                    result.user_id,
                    result.question_id,
                    answer_redacted,
                    self._payload(result),
                    result.created_at.isoformat(),
                ),
            )

    def save_mistake(self, mistake: MistakeRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO mistakes(
                    mistake_id, user_id, question_id, status, next_review_date,
                    payload_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mistake_id) DO UPDATE SET
                    status=excluded.status,
                    next_review_date=excluded.next_review_date,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    mistake.mistake_id,
                    mistake.user_id,
                    mistake.question_id,
                    mistake.mastery_status.value,
                    mistake.next_review_date.isoformat(),
                    self._payload(mistake),
                    mistake.created_at.isoformat(),
                    mistake.updated_at.isoformat(),
                ),
            )

    def list_mistakes(
        self,
        user_id: str,
        *,
        status: MasteryStatus | None = None,
        topic: str | None = None,
        limit: int = 50,
    ) -> list[MistakeRecord]:
        query = "SELECT payload_json FROM mistakes WHERE user_id=?"
        params: list[object] = [user_id]
        if status:
            query += " AND status=?"
            params.append(status.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        items = [MistakeRecord.model_validate_json(row["payload_json"]) for row in rows]
        if topic:
            normalized = topic.lower()
            items = [item for item in items if any(normalized in tag.lower() for tag in item.weak_tags)]
        return items

    def get_mistake(self, mistake_id: str, user_id: str) -> MistakeRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, payload_json FROM mistakes WHERE mistake_id=?",
                (mistake_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("mistake not found")
        if row["user_id"] != user_id:
            raise UserScopeError("mistake does not belong to this user")
        return MistakeRecord.model_validate_json(row["payload_json"])

    def delete_mistake(self, mistake_id: str, user_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM mistakes WHERE mistake_id=? AND user_id=?",
                (mistake_id, user_id),
            )
        return cursor.rowcount > 0

    def save_revision(self, revision: PlanRevisionProposal) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO plan_revisions(
                    revision_id, plan_id, user_id, status, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(revision_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json
                """,
                (
                    revision.revision_id,
                    revision.plan_id,
                    revision.user_id,
                    revision.status.value,
                    self._payload(revision),
                    revision.created_at.isoformat(),
                ),
            )

    def get_revision(self, revision_id: str, user_id: str) -> PlanRevisionProposal:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, payload_json FROM plan_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("revision not found")
        if row["user_id"] != user_id:
            raise UserScopeError("revision does not belong to this user")
        return PlanRevisionProposal.model_validate_json(row["payload_json"])

    def record_tool_run(
        self,
        *,
        trace_id: str,
        sequence: int,
        user_id: str,
        tool_name: str,
        status: str,
        input_summary: dict,
        output_summary: dict,
        error_type: str | None,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO tool_runs(
                    trace_id, sequence, user_id, tool_name, status,
                    input_summary_json, output_summary_json, error_type, started_at, ended_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    sequence,
                    user_id,
                    tool_name,
                    status,
                    json.dumps(input_summary, ensure_ascii=False, sort_keys=True),
                    json.dumps(output_summary, ensure_ascii=False, sort_keys=True),
                    error_type,
                    started_at.isoformat(),
                    ended_at.isoformat(),
                ),
            )

    def export_user_data(self, user_id: str) -> dict:
        profile = self.get_profile(user_id)
        with self._connection() as connection:
            plans = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM plans WHERE user_id=? ORDER BY created_at", (user_id,)
                )
            ]
            questions = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM questions WHERE user_id=? ORDER BY created_at", (user_id,)
                )
            ]
            attempts = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM attempts WHERE user_id=? ORDER BY created_at", (user_id,)
                )
            ]
            mistakes = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM mistakes WHERE user_id=? ORDER BY created_at", (user_id,)
                )
            ]
            revisions = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM plan_revisions WHERE user_id=? ORDER BY created_at", (user_id,)
                )
            ]
            tool_runs = [
                {
                    "trace_id": row["trace_id"],
                    "sequence": row["sequence"],
                    "tool_name": row["tool_name"],
                    "status": row["status"],
                    "input_summary": json.loads(row["input_summary_json"]),
                    "output_summary": json.loads(row["output_summary_json"]),
                    "error_type": row["error_type"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                }
                for row in connection.execute(
                    """
                    SELECT trace_id, sequence, tool_name, status, input_summary_json,
                           output_summary_json, error_type, started_at, ended_at
                    FROM tool_runs WHERE user_id=? ORDER BY started_at, sequence
                    """,
                    (user_id,),
                )
            ]
        return {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "plans": plans,
            "questions": questions,
            "attempts": attempts,
            "mistakes": mistakes,
            "plan_revisions": revisions,
            "tool_runs": tool_runs,
        }

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._connection() as connection:
            for table in (
                "tool_runs",
                "plan_revisions",
                "mistakes",
                "attempts",
                "questions",
                "plans",
                "user_profiles",
            ):
                cursor = connection.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                counts[table] = cursor.rowcount
        return counts
