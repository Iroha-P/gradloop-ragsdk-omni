from __future__ import annotations

from datetime import date

from app.domain.grading import GradeRequest
from app.domain.mistakes import MistakeListRequest
from app.domain.plans import RevisionDecisionRequest, StudyPlanRequest
from app.domain.practice import PracticeRequest
from app.rag.base import RetrievalScope
from app.services.learning import LearningService


class LearningTools:
    """Typed tool boundary. Tools receive no filesystem or command execution capability."""

    def __init__(self, service: LearningService):
        self.service = service

    def search_knowledge(
        self,
        *,
        question: str,
        top_k: int = 5,
        scope: str = "generic",
    ) -> dict:
        return self.service.ask(
            question,
            top_k=top_k,
            scope=RetrievalScope(scope),
        ).model_dump(mode="json")

    def create_study_plan(self, *, user_id: str, payload: dict) -> dict:
        data = {**payload, "user_id": user_id}
        if isinstance(data.get("deadline"), str):
            data["deadline"] = date.fromisoformat(data["deadline"])
        return self.service.create_plan(StudyPlanRequest.model_validate(data)).model_dump(mode="json")

    def get_practice_questions(self, *, user_id: str, payload: dict) -> dict:
        return self.service.get_questions(
            PracticeRequest.model_validate({**payload, "user_id": user_id})
        ).model_dump(mode="json")

    def grade_answer(self, *, user_id: str, payload: dict) -> dict:
        return self.service.grade(GradeRequest.model_validate({**payload, "user_id": user_id})).model_dump(
            mode="json"
        )

    def list_mistakes(self, *, user_id: str, payload: dict) -> dict:
        mistakes = self.service.list_mistakes(
            MistakeListRequest.model_validate({**payload, "user_id": user_id})
        )
        return {"mistakes": [item.model_dump(mode="json") for item in mistakes]}

    def redo_mistake(self, *, user_id: str, payload: dict) -> dict:
        data = {**payload, "user_id": user_id}
        mistake_id = str(data.pop("mistake_id", ""))
        return self.service.redo_mistake(
            mistake_id,
            GradeRequest.model_validate(data),
        ).model_dump(mode="json")

    def decide_plan_revision(self, *, user_id: str, payload: dict) -> dict:
        revision_id = str(payload.get("revision_id", ""))
        request = RevisionDecisionRequest(
            user_id=user_id,
            confirm=bool(payload.get("confirm", True)),
        )
        revision, plan = self.service.decide_revision(revision_id, request)
        return {
            "revision": revision.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json") if plan else None,
        }
