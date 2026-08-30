from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.langgraph_agent import LangGraphLearningAgent
from app.agent.state import AgentIntent, AgentResumeRequest, AgentRunRequest
from app.storage.sqlite import SQLiteLearningStore


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentEvalCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_]{6,64}$")
    category: Literal[
        "routing", "retrieval_retry", "human_confirmation", "recovery", "privacy"
    ]
    intent: AgentIntent
    scope: Literal["generic", "personal", "all"] = "generic"
    ask_mode: Literal["sufficient", "insufficient_then_sufficient", "insufficient"]
    grade_revision: bool
    confirm: bool | None
    fail_node: str | None
    expected_tools: list[str]
    expected_status: Literal["completed", "waiting_for_confirmation"]


class AgentEvalObservation(StrictModel):
    case_id: str
    intent_correct: bool
    tool_sequence_correct: bool
    bounded: bool
    interrupt_correct: bool | None
    recovery_success: bool | None
    duplicate_side_effect: bool
    privacy_leakage_count: int = Field(ge=0)
    final_status: str


class AgentEvalReport(StrictModel):
    case_count: int
    intent_routing_accuracy: float
    tool_sequence_accuracy: float
    bounded_completion_rate: float
    interrupt_correctness: float
    recovery_success_rate: float
    duplicate_side_effect_rate: float
    privacy_leakage_count: int
    claim_scope: Literal["workflow_only_public_synthetic"]
    observations: list[AgentEvalObservation]


class SyntheticAgentTools:
    """Deterministic public fixture used only to evaluate orchestration behavior."""

    def __init__(self, case: AgentEvalCase):
        self.case = case
        self.calls: Counter[str] = Counter()

    def search_knowledge(self, **_kwargs):
        self.calls["search_knowledge"] += 1
        hit = self.case.ask_mode == "sufficient" or (
            self.case.ask_mode == "insufficient_then_sufficient"
            and self.calls["search_knowledge"] >= 2
        )
        if hit:
            return {
                "answer": "公开合成回答",
                "citations": [{"chunk_id": "public_synthetic_chunk"}],
                "warnings": [],
            }
        return {
            "answer": "公开合成资料不足",
            "citations": [],
            "warnings": ["insufficient_evidence"],
        }

    def create_study_plan(self, **_kwargs):
        self.calls["create_study_plan"] += 1
        return {"plan_id": "public_synthetic_plan", "tasks": []}

    def get_practice_questions(self, **_kwargs):
        self.calls["get_practice_questions"] += 1
        return {"questions": [{"question_id": "public_synthetic_question"}]}

    def grade_answer(self, **_kwargs):
        self.calls["grade_answer"] += 1
        return {
            "attempt_id": "public_synthetic_attempt",
            "score": 60 if self.case.grade_revision else 90,
            "mistake_id": "public_synthetic_mistake" if self.case.grade_revision else None,
            "revision_id": "public_synthetic_revision" if self.case.grade_revision else None,
            "warnings": [],
        }

    def list_mistakes(self, **_kwargs):
        self.calls["list_mistakes"] += 1
        return {"mistakes": []}

    def decide_plan_revision(self, *, payload, **_kwargs):
        self.calls["decide_plan_revision"] += 1
        return {
            "revision": {
                "revision_id": payload["revision_id"],
                "status": "confirmed" if payload.get("confirm") else "rejected",
            },
            "plan": None,
        }


def load_agent_cases(path: str | Path) -> list[AgentEvalCase]:
    cases: list[AgentEvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(AgentEvalCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid agent eval case at line {line_number}") from exc
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("agent eval case IDs must be unique")
    return cases


def _payload(case: AgentEvalCase, canary: str) -> dict:
    if case.intent == "ask":
        return {"question": f"公开合成问题 {canary}", "scope": case.scope}
    if case.intent == "plan":
        return {"goal": "公开合成目标"}
    if case.intent == "practice":
        return {"topic": "公开合成主题", "scope": case.scope}
    if case.intent == "grade":
        return {"question_id": "public_synthetic_question", "answer": canary}
    if case.intent == "mistakes":
        return {"limit": 5}
    return {"revision_id": "public_synthetic_revision", "confirm": bool(case.confirm)}


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 1.0


def evaluate_agent_cases(
    cases: list[AgentEvalCase], *, workdir: str | Path
) -> AgentEvalReport:
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    observations: list[AgentEvalObservation] = []
    for index, case in enumerate(cases):
        case_root = root / case.case_id
        tools = SyntheticAgentTools(case)
        failed_once = False

        def failure_injector(node_name: str, fail_node: str | None = case.fail_node) -> None:
            nonlocal failed_once
            if fail_node == node_name and not failed_once:
                failed_once = True
                raise RuntimeError("public synthetic fault")

        checkpoint_path = case_root / "checkpoints.sqlite"
        store = SQLiteLearningStore(case_root / "learning.db")
        agent = LangGraphLearningAgent(
            tools,
            store,
            backend_name="synthetic",
            checkpoint_path=checkpoint_path,
            failure_injector=failure_injector,
        )
        user_id = f"eval_user_{index:02d}"
        canary = f"PUBLIC_EVAL_CANARY_{case.case_id}"
        request = AgentRunRequest(
            user_id=user_id,
            intent=case.intent,
            message=canary,
            payload=_payload(case, canary),
        )
        trace_names: list[str] = []
        try:
            first = agent.run(request)
            trace_names.extend(item.tool_name for item in first.tool_trace)
            final = first
            interrupt_correct: bool | None = None
            recovery_success: bool | None = None
            if case.category == "human_confirmation":
                interrupt_correct = first.status == "waiting_for_confirmation"
            if first.status == "waiting_for_confirmation" and case.confirm is not None:
                final = agent.resume(
                    AgentResumeRequest(
                        trace_id=first.trace_id,
                        user_id=user_id,
                        confirm=case.confirm,
                    )
                )
                trace_names.extend(item.tool_name for item in final.tool_trace)
            elif first.status == "failed_recoverable" and case.fail_node:
                final = agent.resume(
                    AgentResumeRequest(
                        trace_id=first.trace_id,
                        user_id=user_id,
                        request=request,
                    )
                )
                trace_names.extend(item.tool_name for item in final.tool_trace)
            if case.category == "recovery":
                recovery_success = (
                    first.status == "failed_recoverable"
                    and final.status == "completed"
                    and final.recovery.resumed
                )
            expected_counts = Counter(case.expected_tools)
            duplicate_side_effect = any(
                tools.calls[name] > count for name, count in expected_counts.items()
            )
            leakage_count = 0
            checkpoint_blob = checkpoint_path.read_bytes()
            for forbidden in (canary, user_id):
                leakage_count += int(forbidden.encode("utf-8") in checkpoint_blob)
            observations.append(
                AgentEvalObservation(
                    case_id=case.case_id,
                    intent_correct=final.intent == case.intent,
                    tool_sequence_correct=trace_names == case.expected_tools,
                    bounded=len(trace_names) <= agent.MAX_STEPS,
                    interrupt_correct=interrupt_correct,
                    recovery_success=recovery_success,
                    duplicate_side_effect=duplicate_side_effect,
                    privacy_leakage_count=leakage_count,
                    final_status=final.status,
                )
            )
        finally:
            agent.close()

    interrupt_values = [
        item.interrupt_correct for item in observations if item.interrupt_correct is not None
    ]
    recovery_values = [
        item.recovery_success for item in observations if item.recovery_success is not None
    ]
    return AgentEvalReport(
        case_count=len(observations),
        intent_routing_accuracy=_rate([item.intent_correct for item in observations]),
        tool_sequence_accuracy=_rate([item.tool_sequence_correct for item in observations]),
        bounded_completion_rate=_rate(
            [
                item.bounded
                and item.final_status
                == next(case.expected_status for case in cases if case.case_id == item.case_id)
                for item in observations
            ]
        ),
        interrupt_correctness=_rate([bool(value) for value in interrupt_values]),
        recovery_success_rate=_rate([bool(value) for value in recovery_values]),
        duplicate_side_effect_rate=_rate(
            [item.duplicate_side_effect for item in observations]
        ),
        privacy_leakage_count=sum(item.privacy_leakage_count for item in observations),
        claim_scope="workflow_only_public_synthetic",
        observations=observations,
    ).model_copy(
        update={
            "duplicate_side_effect_rate": (
                sum(item.duplicate_side_effect for item in observations) / len(observations)
                if observations
                else 0.0
            )
        }
    )


def write_agent_report(path: str | Path, report: AgentEvalReport) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
