from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from datetime import UTC, datetime
from typing import Any

from app.agent.state import AgentIntent, AgentRunRequest, AgentRunResult, ToolTrace
from app.agent.tools import LearningTools
from app.storage.sqlite import SQLiteLearningStore


def build_innovation_coaching(
    *,
    prompt: str,
    modalities: list[str],
    evidence: list[dict[str, Any]],
    backend_mode: str,
    agent_engine: str,
) -> dict[str, Any]:
    """Build the bounded coaching part of the public omni demo.

    The function is intentionally deterministic for fixture replay. It accepts
    only sanitized model observations and retrieval evidence, never a user ID,
    source path, or original upload filename.
    """
    evidence_count = len(evidence)
    if evidence_count:
        coach_feedback = (
            "回答已关联本地证据。请先复述证据结论，再补充一个可验证的步骤；"
            "若无法由证据支持，应明确说明不确定性。"
        )
    else:
        coach_feedback = "当前没有足够证据。请缩小问题范围，并补充可核对的材料。"
    return {
        "agent_engine": agent_engine,
        "workflow": [
            "media_validation",
            "model_understanding",
            "evidence_retrieval",
            "bounded_coaching",
            "next_action",
        ],
        "backend_mode": backend_mode,
        "modalities": modalities,
        "evidence_count": evidence_count,
        "coach_feedback": coach_feedback,
        "next_action": "用 STAR 结构重答，并在结论后附一条证据引用。",
        "retention": "request_scoped",
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }


class LearningAgent:
    """Observable bounded workflow equivalent to a small explicit StateGraph."""

    MAX_STEPS = 6
    engine_name = "baseline"
    checkpointer_name = "none"
    recoverable = False

    def __init__(self, tools: LearningTools, store: SQLiteLearningStore, *, backend_name: str):
        self.tools = tools
        self.store = store
        self.backend_name = backend_name

    @staticmethod
    def _classify(request: AgentRunRequest) -> AgentIntent:
        if request.intent:
            return request.intent
        text = request.message.lower()
        if any(token in text for token in ("计划", "plan")):
            return "plan"
        if any(token in text for token in ("练习", "题目", "practice")):
            return "practice"
        if any(token in text for token in ("批改", "评分", "grade")):
            return "grade"
        if any(token in text for token in ("重做", "重答", "redo")):
            return "redo"
        if any(token in text for token in ("错题", "mistake")):
            return "mistakes"
        if any(token in text for token in ("确认调整", "revision")):
            return "revision"
        return "ask"

    @staticmethod
    def _input_summary(payload: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        text_lengths = {key: len(value) for key, value in payload.items() if isinstance(value, str)}
        return {
            "keys": sorted(payload),
            "text_lengths": text_lengths,
            "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        }

    @staticmethod
    def _output_summary(result: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {"keys": sorted(result)}
        for key in ("plan_id", "question_id", "attempt_id", "mistake_id", "revision_id", "score"):
            if key in result and result[key] is not None:
                summary[key] = result[key]
        for key in ("questions", "mistakes", "tasks", "citations"):
            if isinstance(result.get(key), list):
                summary[f"{key}_count"] = len(result[key])
        return summary

    def _call(
        self,
        *,
        trace_id: str,
        sequence: int,
        user_id: str,
        tool_name: str,
        payload: dict[str, Any],
        function: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], ToolTrace]:
        started = datetime.now(UTC)
        clock = time.perf_counter()
        status = "success"
        error_type = None
        result: dict[str, Any] = {}
        try:
            result = function()
            return result, self._finish_trace(
                trace_id,
                sequence,
                user_id,
                tool_name,
                payload,
                result,
                started,
                clock,
                status,
                error_type,
            )
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            self._finish_trace(
                trace_id,
                sequence,
                user_id,
                tool_name,
                payload,
                result,
                started,
                clock,
                status,
                error_type,
            )
            raise

    def _finish_trace(
        self,
        trace_id: str,
        sequence: int,
        user_id: str,
        tool_name: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        started: datetime,
        clock: float,
        status: str,
        error_type: str | None,
    ) -> ToolTrace:
        ended = datetime.now(UTC)
        trace = ToolTrace(
            sequence=sequence,
            tool_name=tool_name,
            status=status,
            input_summary=self._input_summary(payload),
            output_summary=self._output_summary(result),
            error_type=error_type,
            started_at=started,
            ended_at=ended,
            elapsed_ms=(time.perf_counter() - clock) * 1000,
            personal_scope_used=str(payload.get("scope", "generic")) in {"personal", "all"},
        )
        self.store.record_tool_run(
            trace_id=trace_id,
            sequence=sequence,
            user_id=user_id,
            tool_name=tool_name,
            status=status,
            input_summary=trace.input_summary,
            output_summary=trace.output_summary,
            error_type=error_type,
            started_at=started,
            ended_at=ended,
        )
        return trace

    @staticmethod
    def _progress_event(trace_id: str, trace: ToolTrace) -> dict[str, object]:
        return {
            "event": "tool",
            "trace_id": trace_id,
            "sequence": trace.sequence,
            "tool_name": trace.tool_name,
            "status": trace.status,
        }

    def _execute(
        self,
        request: AgentRunRequest,
        *,
        trace_id: str,
    ) -> Generator[dict[str, object], None, AgentRunResult]:
        intent = self._classify(request)
        payload = dict(request.payload)
        traces: list[ToolTrace] = []
        if intent == "ask":
            question = str(payload.get("question") or request.message).strip()
            scope = str(payload.get("scope", "generic"))
            result, trace = self._call(
                trace_id=trace_id,
                sequence=1,
                user_id=request.user_id,
                tool_name="search_knowledge",
                payload={
                    "question": question,
                    "top_k": payload.get("top_k", 5),
                    "scope": scope,
                },
                function=lambda: self.tools.search_knowledge(
                    question=question,
                    top_k=int(payload.get("top_k", 5)),
                    scope=scope,
                ),
            )
        else:
            routes: dict[AgentIntent, tuple[str, Callable[[], dict[str, Any]]]] = {
                "plan": (
                    "create_study_plan",
                    lambda: self.tools.create_study_plan(user_id=request.user_id, payload=payload),
                ),
                "practice": (
                    "get_practice_questions",
                    lambda: self.tools.get_practice_questions(user_id=request.user_id, payload=payload),
                ),
                "grade": (
                    "grade_answer",
                    lambda: self.tools.grade_answer(user_id=request.user_id, payload=payload),
                ),
                "mistakes": (
                    "list_mistakes",
                    lambda: self.tools.list_mistakes(user_id=request.user_id, payload=payload),
                ),
                "redo": (
                    "redo_mistake",
                    lambda: self.tools.redo_mistake(user_id=request.user_id, payload=payload),
                ),
                "revision": (
                    "decide_plan_revision",
                    lambda: self.tools.decide_plan_revision(user_id=request.user_id, payload=payload),
                ),
                "ask": ("search_knowledge", lambda: {}),
            }
            tool_name, function = routes[intent]
            result, trace = self._call(
                trace_id=trace_id,
                sequence=1,
                user_id=request.user_id,
                tool_name=tool_name,
                payload=payload,
                function=function,
            )
        traces.append(trace)
        yield self._progress_event(trace_id, trace)

        if intent == "grade" and result.get("mistake_id"):
            for tool_name, key in (
                ("save_mistake", "mistake_id"),
                ("propose_plan_revision", "revision_id"),
            ):
                if result.get(key):
                    sequence = len(traces) + 1
                    virtual_result, virtual_trace = self._call(
                        trace_id=trace_id,
                        sequence=sequence,
                        user_id=request.user_id,
                        tool_name=tool_name,
                        payload={"attempt_id": result.get("attempt_id")},
                        function=lambda key=key: {key: result.get(key)},
                    )
                    del virtual_result
                    traces.append(virtual_trace)
                    yield self._progress_event(trace_id, virtual_trace)

        if len(traces) > self.MAX_STEPS:
            raise RuntimeError("agent exceeded maximum step count")
        warnings = list(result.get("warnings", [])) if isinstance(result.get("warnings"), list) else []
        return AgentRunResult(
            trace_id=trace_id,
            user_id=request.user_id,
            intent=intent,
            backend=self.backend_name,
            result=result,
            tool_trace=traces,
            steps=len(traces),
            warnings=warnings,
        )

    def run(self, request: AgentRunRequest, *, trace_id: str | None = None) -> AgentRunResult:
        execution = self._execute(request, trace_id=trace_id or uuid.uuid4().hex)
        while True:
            try:
                next(execution)
            except StopIteration as completed:
                return completed.value

    def stream(self, request: AgentRunRequest, *, trace_id: str | None = None) -> Iterator[dict[str, object]]:
        stream_trace_id = trace_id or uuid.uuid4().hex
        yield {"event": "start", "trace_id": stream_trace_id, "engine": self.engine_name}
        execution = self._execute(request, trace_id=stream_trace_id)
        while True:
            try:
                yield next(execution)
            except StopIteration as completed:
                result = completed.value
                yield {
                    "event": "done",
                    "trace_id": stream_trace_id,
                    "status": result.status,
                    "steps": result.steps,
                }
                return
            except Exception:
                yield {
                    "event": "error",
                    "trace_id": stream_trace_id,
                    "code": "agent_run_failed",
                    "retryable": True,
                }
                return
