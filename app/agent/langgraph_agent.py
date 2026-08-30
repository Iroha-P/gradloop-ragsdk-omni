from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from app.agent.checkpoint import SQLiteAgentCheckpointer
from app.agent.graph import LearningAgent
from app.agent.state import (
    AgentGraphState,
    AgentResumeRequest,
    AgentRunRequest,
    AgentRunResult,
    PendingAction,
    RecoveryInfo,
    ToolTrace,
    build_checkpoint_seed,
)
from app.agent.tools import LearningTools
from app.storage.sqlite import RecordNotFoundError, SQLiteLearningStore, UserScopeError


@dataclass
class AgentRunContext:
    request: AgentRunRequest | None
    user_id: str
    trace_id: str
    result_sink: dict[str, Any] = field(default_factory=dict)
    traces: list[ToolTrace] = field(default_factory=list)
    resumed: bool = False


class RecoverableAgentNodeError(RuntimeError):
    """Sanitized error persisted by LangGraph without the original exception message."""


class LangGraphLearningAgent:
    MAX_STEPS = 8
    engine_name = "langgraph"
    checkpointer_name = "sqlite"
    recoverable = True

    def __init__(
        self,
        tools: LearningTools,
        store: SQLiteLearningStore,
        *,
        backend_name: str,
        checkpoint_path: str | Path,
        failure_injector: Callable[[str], None] | None = None,
    ):
        self.tools = tools
        self.store = store
        self.backend_name = backend_name
        self.failure_injector = failure_injector
        self.checkpointer = SQLiteAgentCheckpointer(checkpoint_path)
        self._trace_helper = LearningAgent(tools, store, backend_name=backend_name)
        self.graph = self._build_graph()

    @staticmethod
    def _user_hash(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]

    def _build_graph(self):
        builder = StateGraph(AgentGraphState, context_schema=AgentRunContext)
        builder.add_node("route_intent", self._guard("route_intent", self._route_intent))
        builder.add_node("search_knowledge", self._guard("search_knowledge", self._search_knowledge))
        builder.add_node("retry_search", self._guard("retry_search", self._retry_search))
        builder.add_node("create_study_plan", self._guard("create_study_plan", self._create_plan))
        builder.add_node(
            "get_practice_questions",
            self._guard("get_practice_questions", self._get_practice),
        )
        builder.add_node("grade_answer", self._guard("grade_answer", self._grade_answer))
        builder.add_node("list_mistakes", self._guard("list_mistakes", self._list_mistakes))
        builder.add_node("redo_mistake", self._guard("redo_mistake", self._redo_mistake))
        builder.add_node(
            "decide_plan_revision",
            self._guard("decide_plan_revision", self._decide_revision),
        )
        builder.add_node("await_revision_decision", self._await_revision_decision)
        builder.add_node("finalize", self._guard("finalize", self._finalize))

        builder.add_edge(START, "route_intent")
        builder.add_conditional_edges(
            "route_intent",
            lambda state: state["intent"],
            {
                "ask": "search_knowledge",
                "plan": "create_study_plan",
                "practice": "get_practice_questions",
                "grade": "grade_answer",
                "mistakes": "list_mistakes",
                "redo": "redo_mistake",
                "revision": "decide_plan_revision",
            },
        )
        builder.add_conditional_edges(
            "search_knowledge",
            self._route_after_search,
            {"retry": "retry_search", "finish": "finalize"},
        )
        builder.add_edge("retry_search", "finalize")
        for node in (
            "create_study_plan",
            "get_practice_questions",
            "list_mistakes",
            "redo_mistake",
            "decide_plan_revision",
        ):
            builder.add_edge(node, "finalize")
        builder.add_conditional_edges(
            "grade_answer",
            lambda state: "confirm" if state.get("revision_id") else "finish",
            {"confirm": "await_revision_decision", "finish": "finalize"},
        )
        builder.add_edge("await_revision_decision", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer.saver)

    def _guard(self, node_name: str, function: Callable):
        def guarded(state: AgentGraphState, runtime: Runtime[AgentRunContext]):
            try:
                self._maybe_fail(node_name)
                return function(state, runtime)
            except RecoverableAgentNodeError:
                raise
            except Exception as exc:
                raise RecoverableAgentNodeError(type(exc).__name__) from None

        return guarded

    def _maybe_fail(self, node_name: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(node_name)

    @staticmethod
    def _request(runtime: Runtime[AgentRunContext]) -> AgentRunRequest:
        request = runtime.context.request
        if request is None:
            raise ValueError("original request is required to resume this node")
        return request

    def _route_intent(
        self, state: AgentGraphState, runtime: Runtime[AgentRunContext]
    ) -> AgentGraphState:
        request = self._request(runtime)
        return {
            "intent": LearningAgent._classify(request),
            "current_node": "route_intent",
            "last_status": "success",
        }

    def _call_tool(
        self,
        *,
        state: AgentGraphState,
        runtime: Runtime[AgentRunContext],
        node_name: str,
        tool_name: str,
        payload: dict[str, Any],
        function: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], AgentGraphState]:
        sequence = int(state.get("step_count", 0)) + 1
        if sequence > int(state.get("max_steps", self.MAX_STEPS)):
            raise RuntimeError("agent exceeded maximum step count")
        result, trace = self._trace_helper._call(
            trace_id=runtime.context.trace_id,
            sequence=sequence,
            user_id=runtime.context.user_id,
            tool_name=tool_name,
            payload=payload,
            function=function,
        )
        runtime.context.result_sink["result"] = result
        runtime.context.traces.append(trace)
        update: AgentGraphState = {
            "step_count": sequence,
            "current_node": node_name,
            "last_tool": tool_name,
            "last_status": "success",
            "output_summary": LearningAgent._output_summary(result),
            "error_type": None,
        }
        return result, update

    def _search(
        self,
        state: AgentGraphState,
        runtime: Runtime[AgentRunContext],
        *,
        retry: bool,
    ) -> AgentGraphState:
        request = self._request(runtime)
        payload = dict(request.payload)
        question = str(payload.get("question") or request.message).strip()
        top_k = int(payload.get("top_k", 5))
        if retry:
            question = f"{question} 保研申请 面试 准备 方法"
            top_k = min(20, max(top_k + 2, top_k * 2))
        scope = str(payload.get("scope", "generic"))
        call_payload = {"question": question, "top_k": top_k, "scope": scope}
        result, update = self._call_tool(
            state=state,
            runtime=runtime,
            node_name="retry_search" if retry else "search_knowledge",
            tool_name="search_knowledge",
            payload=call_payload,
            function=lambda: self.tools.search_knowledge(
                question=question,
                top_k=top_k,
                scope=scope,
            ),
        )
        citations = result.get("citations")
        answer = result.get("answer")
        update["evidence_sufficient"] = bool(
            isinstance(citations, list) and citations and isinstance(answer, str) and answer.strip()
        )
        if retry:
            update["retry_count"] = 1
        return update

    def _search_knowledge(self, state, runtime):
        return self._search(state, runtime, retry=False)

    def _retry_search(self, state, runtime):
        return self._search(state, runtime, retry=True)

    @staticmethod
    def _route_after_search(state: AgentGraphState) -> str:
        if state.get("evidence_sufficient") or int(state.get("retry_count", 0)) >= 1:
            return "finish"
        return "retry"

    def _create_plan(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        return self._call_tool(
            state=state,
            runtime=runtime,
            node_name="create_study_plan",
            tool_name="create_study_plan",
            payload=payload,
            function=lambda: self.tools.create_study_plan(
                user_id=runtime.context.user_id, payload=payload
            ),
        )[1]

    def _get_practice(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        return self._call_tool(
            state=state,
            runtime=runtime,
            node_name="get_practice_questions",
            tool_name="get_practice_questions",
            payload=payload,
            function=lambda: self.tools.get_practice_questions(
                user_id=runtime.context.user_id, payload=payload
            ),
        )[1]

    def _grade_answer(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        result, update = self._call_tool(
            state=state,
            runtime=runtime,
            node_name="grade_answer",
            tool_name="grade_answer",
            payload=payload,
            function=lambda: self.tools.grade_answer(
                user_id=runtime.context.user_id, payload=payload
            ),
        )
        revision_id = result.get("revision_id")
        if revision_id:
            update["revision_id"] = str(revision_id)
            update["pending_action"] = {
                "action": "confirm_plan_revision",
                "revision_id": str(revision_id),
            }
        return update

    def _list_mistakes(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        return self._call_tool(
            state=state,
            runtime=runtime,
            node_name="list_mistakes",
            tool_name="list_mistakes",
            payload=payload,
            function=lambda: self.tools.list_mistakes(
                user_id=runtime.context.user_id, payload=payload
            ),
        )[1]

    def _redo_mistake(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        return self._call_tool(
            state=state,
            runtime=runtime,
            node_name="redo_mistake",
            tool_name="redo_mistake",
            payload=payload,
            function=lambda: self.tools.redo_mistake(
                user_id=runtime.context.user_id, payload=payload
            ),
        )[1]

    def _decide_revision(self, state, runtime):
        request = self._request(runtime)
        payload = dict(request.payload)
        return self._call_tool(
            state=state,
            runtime=runtime,
            node_name="decide_plan_revision",
            tool_name="decide_plan_revision",
            payload=payload,
            function=lambda: self.tools.decide_plan_revision(
                user_id=runtime.context.user_id, payload=payload
            ),
        )[1]

    def _await_revision_decision(self, state, runtime):
        revision_id = str(state.get("revision_id", ""))
        decision = interrupt(
            {
                "action": "confirm_plan_revision",
                "trace_id": state["trace_id"],
                "revision_id": revision_id,
            }
        )
        confirm = bool(decision.get("confirm")) if isinstance(decision, dict) else bool(decision)
        payload = {"revision_id": revision_id, "confirm": confirm}
        _, update = self._call_tool(
            state=state,
            runtime=runtime,
            node_name="await_revision_decision",
            tool_name="decide_plan_revision",
            payload=payload,
            function=lambda: self.tools.decide_plan_revision(
                user_id=runtime.context.user_id, payload=payload
            ),
        )
        update["pending_action"] = None
        return update

    def _finalize(self, state, runtime):
        recovery_count = int(state.get("recovery_count", 0))
        if runtime.context.resumed:
            recovery_count += 1
        return {
            "current_node": "finalize",
            "last_status": "completed",
            "pending_action": None,
            "recoverable": False,
            "recovery_count": recovery_count,
        }

    @staticmethod
    def _config(trace_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": trace_id}}

    def _assert_user(self, values: dict[str, Any], user_id: str) -> None:
        if values.get("user_id_hash") != self._user_hash(user_id):
            raise UserScopeError("agent run does not belong to this user")

    def _build_result(
        self,
        *,
        state: dict[str, Any],
        context: AgentRunContext,
        next_nodes: tuple[str, ...] | list[str],
        status: str,
        pending_action: PendingAction | None = None,
    ) -> AgentRunResult:
        result = context.result_sink.get("result", {})
        warnings = list(result.get("warnings", [])) if isinstance(result, dict) else []
        if status == "failed_recoverable" and state.get("error_type"):
            warnings.append(str(state["error_type"]))
        return AgentRunResult(
            trace_id=context.trace_id,
            user_id=context.user_id,
            intent=state.get("intent", "ask"),
            backend=self.backend_name,
            result=result if isinstance(result, dict) else {},
            tool_trace=context.traces,
            steps=max(1, int(state.get("step_count", len(context.traces)))),
            warnings=warnings,
            engine="langgraph",
            status=status,
            checkpointed=True,
            current_node=state.get("current_node"),
            next_nodes=list(next_nodes),
            pending_action=pending_action,
            recovery=RecoveryInfo(
                resumed=context.resumed,
                recovery_count=int(state.get("recovery_count", 0)),
            ),
        )

    def run(self, request: AgentRunRequest, *, trace_id: str | None = None) -> AgentRunResult:
        trace_id = trace_id or uuid.uuid4().hex
        context = AgentRunContext(request=request, user_id=request.user_id, trace_id=trace_id)
        config = self._config(trace_id)
        state = build_checkpoint_seed(request, trace_id=trace_id)
        state["max_steps"] = self.MAX_STEPS
        try:
            output = self.graph.invoke(state, config=config, context=context)
        except Exception as exc:
            snapshot = self.graph.get_state(config)
            values = dict(snapshot.values)
            values["error_type"] = type(exc).__name__
            return self._build_result(
                state=values,
                context=context,
                next_nodes=snapshot.next,
                status="failed_recoverable",
            )
        snapshot = self.graph.get_state(config)
        values = dict(snapshot.values)
        interrupts = output.get("__interrupt__", []) if isinstance(output, dict) else []
        if interrupts:
            revision_id = str(values.get("revision_id", ""))
            pending = PendingAction(
                action="confirm_plan_revision",
                revision_id=revision_id,
            )
            return self._build_result(
                state=values,
                context=context,
                next_nodes=snapshot.next,
                status="waiting_for_confirmation",
                pending_action=pending,
            )
        return self._build_result(
            state=values,
            context=context,
            next_nodes=snapshot.next,
            status="completed",
        )

    def stream(self, request: AgentRunRequest, *, trace_id: str | None = None) -> Iterator[dict[str, object]]:
        """Yield redacted events from the real LangGraph execution, not generated answer chunks."""

        stream_trace_id = trace_id or uuid.uuid4().hex
        context = AgentRunContext(
            request=request,
            user_id=request.user_id,
            trace_id=stream_trace_id,
        )
        config = self._config(stream_trace_id)
        state = build_checkpoint_seed(request, trace_id=stream_trace_id)
        state["max_steps"] = self.MAX_STEPS
        emitted_traces = 0
        yield {"event": "start", "trace_id": stream_trace_id, "engine": self.engine_name}
        try:
            for update in self.graph.stream(
                state,
                config=config,
                context=context,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node_name in update:
                    if node_name == "__interrupt__":
                        continue
                    yield {"event": "node", "trace_id": stream_trace_id, "node": node_name}
                while emitted_traces < len(context.traces):
                    trace = context.traces[emitted_traces]
                    emitted_traces += 1
                    yield {
                        "event": "tool",
                        "trace_id": stream_trace_id,
                        "sequence": trace.sequence,
                        "tool_name": trace.tool_name,
                        "status": trace.status,
                    }
            snapshot = self.graph.get_state(config)
            values = dict(snapshot.values)
            interrupted = any(task.interrupts for task in snapshot.tasks)
            yield {
                "event": "done",
                "trace_id": stream_trace_id,
                "status": "waiting_for_confirmation" if interrupted else "completed",
                "steps": max(1, int(values.get("step_count", len(context.traces)))),
            }
        except Exception:
            yield {
                "event": "error",
                "trace_id": stream_trace_id,
                "code": "agent_run_failed",
                "retryable": True,
            }

    def resume(self, request: AgentResumeRequest) -> AgentRunResult:
        config = self._config(request.trace_id)
        snapshot = self.graph.get_state(config)
        values = dict(snapshot.values)
        if not values:
            raise RecordNotFoundError("agent run not found")
        self._assert_user(values, request.user_id)
        if not snapshot.next:
            context = AgentRunContext(
                request=None,
                user_id=request.user_id,
                trace_id=request.trace_id,
                resumed=True,
            )
            return self._build_result(
                state=values,
                context=context,
                next_nodes=[],
                status="completed",
            )
        has_interrupt = any(task.interrupts for task in snapshot.tasks)
        if has_interrupt:
            if request.confirm is None:
                raise ValueError("confirmation decision is required")
            graph_input: Any = Command(resume={"confirm": request.confirm})
        else:
            if request.request is None:
                raise ValueError("original request is required to recover a failed run")
            graph_input = None
        context = AgentRunContext(
            request=request.request,
            user_id=request.user_id,
            trace_id=request.trace_id,
            resumed=True,
        )
        try:
            output = self.graph.invoke(graph_input, config=config, context=context)
        except Exception as exc:
            current = self.graph.get_state(config)
            current_values = dict(current.values)
            current_values["error_type"] = type(exc).__name__
            return self._build_result(
                state=current_values,
                context=context,
                next_nodes=current.next,
                status="failed_recoverable",
            )
        current = self.graph.get_state(config)
        current_values = dict(current.values)
        interrupts = output.get("__interrupt__", []) if isinstance(output, dict) else []
        if interrupts:
            pending = PendingAction(
                action="confirm_plan_revision",
                revision_id=str(current_values.get("revision_id", "")),
            )
            return self._build_result(
                state=current_values,
                context=context,
                next_nodes=current.next,
                status="waiting_for_confirmation",
                pending_action=pending,
            )
        return self._build_result(
            state=current_values,
            context=context,
            next_nodes=current.next,
            status="completed",
        )

    def get_status(self, trace_id: str, user_id: str) -> AgentRunResult:
        config = self._config(trace_id)
        snapshot = self.graph.get_state(config)
        values = dict(snapshot.values)
        if not values:
            raise RecordNotFoundError("agent run not found")
        self._assert_user(values, user_id)
        has_interrupt = any(task.interrupts for task in snapshot.tasks)
        status = "waiting_for_confirmation" if has_interrupt else (
            "completed" if not snapshot.next else "failed_recoverable"
        )
        pending = None
        if has_interrupt:
            pending = PendingAction(
                action="confirm_plan_revision",
                revision_id=str(values.get("revision_id", "")),
            )
        context = AgentRunContext(request=None, user_id=user_id, trace_id=trace_id)
        return self._build_result(
            state=values,
            context=context,
            next_nodes=snapshot.next,
            status=status,
            pending_action=pending,
        )

    def close(self) -> None:
        self.checkpointer.close()
