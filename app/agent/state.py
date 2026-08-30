from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AgentIntent = Literal["ask", "plan", "practice", "grade", "mistakes", "redo", "revision"]
AgentEngineName = Literal["baseline", "langgraph"]
AgentStatus = Literal["completed", "waiting_for_confirmation", "failed_recoverable"]


class AgentRunRequest(StrictModel):
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    intent: AgentIntent | None = None
    message: str = Field(default="", max_length=12000)
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResumeRequest(StrictModel):
    trace_id: str = Field(pattern=r"^[a-fA-F0-9]{32}$")
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    confirm: bool | None = None
    request: AgentRunRequest | None = None

    @model_validator(mode="after")
    def validate_resume_action(self) -> AgentResumeRequest:
        if self.confirm is None and self.request is None:
            raise ValueError("resume requires a confirmation or the original request")
        if self.request is not None and self.request.user_id != self.user_id:
            raise ValueError("resume request user must match user_id")
        return self


class PendingAction(StrictModel):
    action: Literal["confirm_plan_revision"]
    revision_id: str
    message: str = "请确认是否应用这次学习计划调整。"


class RecoveryInfo(StrictModel):
    resumed: bool = False
    recovery_count: int = Field(default=0, ge=0)


class AgentGraphState(TypedDict, total=False):
    """Checkpoint-safe state. Raw request and tool results must never be stored here."""

    trace_id: str
    user_id_hash: str
    intent: AgentIntent
    scope: str
    step_count: int
    retry_count: int
    evidence_sufficient: bool
    max_steps: int
    current_node: str
    last_tool: str
    last_status: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    revision_id: str
    pending_action: dict[str, Any] | None
    error_type: str | None
    recoverable: bool
    recovery_count: int


def summarize_agent_input(request: AgentRunRequest) -> dict[str, Any]:
    payload = dict(request.payload)
    canonical = json.dumps(
        {"message": request.message, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    text_lengths = {
        key: len(value) for key, value in payload.items() if isinstance(value, str)
    }
    if request.message:
        text_lengths["message"] = len(request.message)
    return {
        "keys": sorted(payload),
        "text_lengths": text_lengths,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def build_checkpoint_seed(request: AgentRunRequest, *, trace_id: str) -> AgentGraphState:
    scope = str(request.payload.get("scope", "generic"))
    state: AgentGraphState = {
        "trace_id": trace_id,
        "user_id_hash": hashlib.sha256(request.user_id.encode("utf-8")).hexdigest()[:16],
        "scope": scope,
        "step_count": 0,
        "retry_count": 0,
        "max_steps": 8,
        "current_node": "route_intent",
        "last_status": "pending",
        "input_summary": summarize_agent_input(request),
        "output_summary": {},
        "pending_action": None,
        "error_type": None,
        "recoverable": True,
        "recovery_count": 0,
    }
    if request.intent is not None:
        state["intent"] = request.intent
    return state


class ToolTrace(StrictModel):
    sequence: int
    tool_name: str
    status: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    error_type: str | None = None
    started_at: datetime
    ended_at: datetime
    elapsed_ms: float = Field(ge=0)
    personal_scope_used: bool = False


class AgentRunResult(StrictModel):
    trace_id: str
    user_id: str
    intent: AgentIntent
    backend: str
    result: dict[str, Any]
    tool_trace: list[ToolTrace]
    steps: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)
    engine: AgentEngineName = "baseline"
    status: AgentStatus = "completed"
    checkpointed: bool = False
    current_node: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    recovery: RecoveryInfo = Field(default_factory=RecoveryInfo)
