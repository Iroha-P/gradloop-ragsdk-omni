from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import LearningAgent
from app.agent.langgraph_agent import LangGraphLearningAgent
from app.agent.state import AgentResumeRequest, AgentRunRequest
from app.agent.tools import LearningTools
from app.domain.plans import StudyIntensity, StudyPlanRequest
from app.domain.practice import Difficulty, PracticeRequest
from app.question_bank.models import QuestionRecord, QuestionReviewStatus
from app.question_bank.repository import QuestionBankRepository
from app.rag.base import DocumentInput, RetrievalScope
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.services.learning import LearningService
from app.storage.sqlite import SQLiteLearningStore

PRIVATE_CASES_PATH = Path("data/local/evaluation/agent_business_cases.jsonl")
PRIVATE_OBSERVATIONS_PATH = Path(
    "data/local/evaluation/agent_business_observations.jsonl"
)
PUBLIC_REPORT_PATH = Path("reports/public/agent_business_comparison_v1.json")

BusinessScenario = Literal["ask", "practice", "plan", "grade", "recovery"]
EngineName = Literal["baseline", "langgraph"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrivateBusinessCase(StrictModel):
    case_id: str = Field(pattern=r"^biz_[a-f0-9]{20}$")
    scenario: BusinessScenario
    scope: Literal["generic", "personal"]
    question: str = Field(min_length=2, max_length=4096)
    topic: str = Field(min_length=2, max_length=256)
    direction: str = Field(min_length=1, max_length=128)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    content_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class BusinessObservation(StrictModel):
    case_id: str
    engine: EngineName
    scenario: BusinessScenario
    scope: Literal["generic", "personal"]
    completion: bool
    task_success: bool
    grounded_result: bool | None = None
    confirmation_success: bool | None = None
    recovery_supported: bool
    recovery_success: bool | None = None
    duplicate_side_effect: bool
    tool_steps: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    final_status: str
    error_type: str | None = None
    tool_sequence: list[str] = Field(default_factory=list)


class PrivateBusinessComparison(StrictModel):
    case_count: int
    scenario_counts: dict[str, int]
    scope_counts: dict[str, int]
    observations: list[BusinessObservation]


class EngineBusinessMetrics(StrictModel):
    observation_count: int
    completion_rate: float
    task_success_rate: float
    grounded_result_rate: float
    confirmation_success_rate: float
    recovery_supported: bool
    recovery_success_rate: float
    duplicate_side_effect_rate: float
    mean_tool_steps: float
    latency_p50_ms: float
    latency_p95_ms: float
    failure_type_counts: dict[str, int]


class PublicBusinessReport(StrictModel):
    report_version: Literal["agent-business-comparison-v1"]
    claim_scope: Literal["local_anonymized_workflow_comparison"]
    case_count: int
    scenario_counts: dict[str, int]
    scope_counts: dict[str, int]
    engines: dict[EngineName, EngineBusinessMetrics]
    langgraph_p50_latency_overhead_pct: float | None
    privacy: dict[str, bool | str]
    recovery_comparison_note: str


def _stable_key(record: QuestionRecord) -> str:
    return hashlib.sha256(record.content_fingerprint.encode("utf-8")).hexdigest()


def _safe_topic(record: QuestionRecord) -> str:
    topic = record.topic.strip()
    if len(topic) >= 2:
        return topic[:256]
    direction = record.direction.strip()
    return direction[:256] if len(direction) >= 2 else "综合面试"


def build_private_business_cases(
    records: list[QuestionRecord], *, count: int = 80
) -> list[PrivateBusinessCase]:
    if count != 80:
        raise ValueError("business evaluation v1 requires exactly 80 cases")
    accepted = [
        item
        for item in records
        if item.review_status == QuestionReviewStatus.ACCEPTED
        and len(item.canonical_text.strip()) >= 2
    ]
    generic = sorted(
        [item for item in accepted if item.privacy_lane.value == "generic"],
        key=_stable_key,
    )
    personal = sorted(
        [item for item in accepted if item.privacy_lane.value == "personal"],
        key=_stable_key,
    )
    if len(generic) < 48 or len(personal) < 32:
        raise ValueError("at least 48 generic and 32 personal accepted questions are required")

    selected: list[QuestionRecord] = []
    generic_index = 0
    personal_index = 0
    lane_pattern = ("generic", "generic", "personal", "generic", "personal")
    for index in range(count):
        if lane_pattern[index % len(lane_pattern)] == "generic":
            selected.append(generic[generic_index])
            generic_index += 1
        else:
            selected.append(personal[personal_index])
            personal_index += 1

    scenario_order: list[BusinessScenario] = (
        ["ask"] * 40
        + ["practice"] * 16
        + ["plan"] * 8
        + ["grade"] * 8
        + ["recovery"] * 8
    )
    cases: list[PrivateBusinessCase] = []
    for record, scenario in zip(selected, scenario_order, strict=True):
        digest = hashlib.sha256(
            f"{scenario}|{record.content_fingerprint}".encode()
        ).hexdigest()
        cases.append(
            PrivateBusinessCase(
                case_id=f"biz_{digest[:20]}",
                scenario=scenario,
                scope=record.privacy_lane.value,
                question=record.canonical_text.strip(),
                topic=_safe_topic(record),
                direction=(record.direction.strip() or "general")[:128],
                difficulty=record.difficulty,
                content_fingerprint=hashlib.sha256(
                    record.content_fingerprint.encode("utf-8")
                ).hexdigest(),
            )
        )
    return cases


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(item.model_dump_json() + "\n" for item in rows)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def write_private_business_cases(
    cases: list[PrivateBusinessCase], path: Path = PRIVATE_CASES_PATH
) -> None:
    _write_jsonl(path, cases)


def load_private_business_cases(path: Path = PRIVATE_CASES_PATH) -> list[PrivateBusinessCase]:
    return [
        PrivateBusinessCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _request_for_case(
    case: PrivateBusinessCase,
    *,
    user_id: str,
    question_id: str | None = None,
) -> AgentRunRequest:
    if case.scenario == "ask":
        return AgentRunRequest(
            user_id=user_id,
            intent="ask",
            payload={"question": case.question, "scope": case.scope, "top_k": 5},
        )
    if case.scenario == "practice":
        return AgentRunRequest(
            user_id=user_id,
            intent="practice",
            payload={
                "topic": case.question[:256],
                "difficulty": case.difficulty,
                "count": 1,
                "scope": case.scope,
            },
        )
    if case.scenario in {"plan", "recovery"}:
        return AgentRunRequest(
            user_id=user_id,
            intent="plan",
            payload={
                "goal": f"{case.topic}面试准备",
                "deadline": (date.today() + timedelta(days=6)).isoformat(),
                "weekly_hours": 14,
                "current_level": "beginner",
                "weak_tags": [case.topic],
                "completed_tasks": [],
                "intensity": "balanced",
                "days": 7,
            },
        )
    if question_id is None:
        raise ValueError("grade case requires a prepared question")
    return AgentRunRequest(
        user_id=user_id,
        intent="grade",
        payload={
            "question_id": question_id,
            "answer": "暂时无法完整回答，我需要重新整理结论、方法、证据和局限。",
            "use_llm_review": False,
        },
    )


def _prepare_grade(
    service: LearningService, case: PrivateBusinessCase, user_id: str
) -> str:
    service.create_plan(
        StudyPlanRequest(
            user_id=user_id,
            goal=f"{case.topic}面试准备",
            deadline=date.today() + timedelta(days=6),
            weekly_hours=14,
            current_level="beginner",
            weak_tags=[case.topic],
            completed_tasks=[],
            intensity=StudyIntensity.BALANCED,
            days=7,
        )
    )
    questions = service.get_questions(
        PracticeRequest(
            user_id=user_id,
            topic=case.question[:256],
            difficulty=Difficulty(case.difficulty),
            count=1,
            scope=RetrievalScope(case.scope),
        )
    )
    if not questions.questions:
        questions = service.get_questions(
            PracticeRequest(
                user_id=user_id,
                topic=case.topic,
                difficulty=Difficulty(case.difficulty),
                count=1,
                scope=RetrievalScope(case.scope),
            )
        )
    if not questions.questions:
        raise ValueError("no grounded question available for grade case")
    return questions.questions[0].question_id


def _task_outcome(
    case: PrivateBusinessCase,
    first_result: dict,
    final_result: dict,
    *,
    confirmation_success: bool | None,
    recovery_success: bool | None,
) -> tuple[bool, bool | None]:
    if case.scenario == "ask":
        grounded = bool(first_result.get("citations"))
        return grounded, grounded
    if case.scenario == "practice":
        grounded = bool(first_result.get("questions"))
        return grounded, grounded
    if case.scenario == "plan":
        return bool(first_result.get("plan_id")), None
    if case.scenario == "grade":
        return bool(first_result.get("attempt_id")) and bool(confirmation_success), None
    return bool(recovery_success), None


def _evaluate_one(
    case: PrivateBusinessCase,
    *,
    engine: EngineName,
    backend: Bm25BaselineBackend,
    question_bank: QuestionBankRepository,
    run_root: Path,
    index: int,
) -> BusinessObservation:
    store = SQLiteLearningStore(run_root / "learning.db")
    service = LearningService(backend, store, question_bank=question_bank)
    tools = LearningTools(service)
    failed_once = False

    def fail_finalize_once(node_name: str) -> None:
        nonlocal failed_once
        if case.scenario == "recovery" and node_name == "finalize" and not failed_once:
            failed_once = True
            raise RuntimeError("anonymous injected failure")

    graph_agent: LangGraphLearningAgent | None = None
    if engine == "langgraph":
        graph_agent = LangGraphLearningAgent(
            tools,
            store,
            backend_name="baseline",
            checkpoint_path=run_root / "checkpoints.sqlite",
            failure_injector=fail_finalize_once,
        )
        agent: LearningAgent | LangGraphLearningAgent = graph_agent
    else:
        agent = LearningAgent(tools, store, backend_name="baseline")

    user_id = f"business_{index:03d}_{'lg' if engine == 'langgraph' else 'bl'}"
    question_id = None
    if case.scenario == "grade":
        question_id = _prepare_grade(service, case, user_id)
    request = _request_for_case(case, user_id=user_id, question_id=question_id)
    traces = []
    confirmation_success: bool | None = None
    recovery_success: bool | None = None
    error_type = None
    first_result: dict = {}
    final_result: dict = {}
    final_status = "failed"
    started = time.perf_counter()
    try:
        first = agent.run(request)
        traces.extend(first.tool_trace)
        first_result = dict(first.result)
        final = first
        if case.scenario == "grade":
            revision_id = str(first.result.get("revision_id") or "")
            if engine == "langgraph":
                if first.status == "waiting_for_confirmation":
                    final = graph_agent.resume(
                        AgentResumeRequest(
                            trace_id=first.trace_id,
                            user_id=user_id,
                            confirm=True,
                        )
                    )
                    traces.extend(final.tool_trace)
            elif revision_id:
                final = agent.run(
                    AgentRunRequest(
                        user_id=user_id,
                        intent="revision",
                        payload={"revision_id": revision_id, "confirm": True},
                    )
                )
                traces.extend(final.tool_trace)
            final_result = dict(final.result)
            confirmation_success = (
                final_result.get("revision", {}).get("status") == "confirmed"
            )
        elif case.scenario == "recovery":
            if engine == "langgraph":
                if first.status == "failed_recoverable":
                    final = graph_agent.resume(
                        AgentResumeRequest(
                            trace_id=first.trace_id,
                            user_id=user_id,
                            request=request,
                        )
                    )
                    traces.extend(final.tool_trace)
                recovery_success = final.status == "completed" and final.recovery.resumed
            else:
                recovery_success = False
            final_result = dict(final.result)
        else:
            final_result = dict(final.result)
        final_status = final.status
        task_success, grounded = _task_outcome(
            case,
            first_result,
            final_result,
            confirmation_success=confirmation_success,
            recovery_success=recovery_success,
        )
        side_effect_tools = Counter(
            item.tool_name
            for item in traces
            if item.tool_name
            in {"create_study_plan", "grade_answer", "decide_plan_revision"}
        )
        duplicate = any(count > 1 for count in side_effect_tools.values())
        return BusinessObservation(
            case_id=case.case_id,
            engine=engine,
            scenario=case.scenario,
            scope=case.scope,
            completion=final_status == "completed",
            task_success=task_success,
            grounded_result=grounded,
            confirmation_success=confirmation_success,
            recovery_supported=engine == "langgraph",
            recovery_success=recovery_success,
            duplicate_side_effect=duplicate,
            tool_steps=len(traces),
            retry_count=max(0, sum(item.tool_name == "search_knowledge" for item in traces) - 1),
            latency_ms=(time.perf_counter() - started) * 1000,
            final_status=final_status,
            error_type=error_type,
            tool_sequence=[item.tool_name for item in traces],
        )
    except Exception as exc:
        error_type = type(exc).__name__
        return BusinessObservation(
            case_id=case.case_id,
            engine=engine,
            scenario=case.scenario,
            scope=case.scope,
            completion=False,
            task_success=False,
            recovery_supported=engine == "langgraph",
            recovery_success=False if case.scenario == "recovery" else None,
            duplicate_side_effect=False,
            tool_steps=len(traces),
            retry_count=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            final_status="failed",
            error_type=error_type,
            tool_sequence=[item.tool_name for item in traces],
        )
    finally:
        if graph_agent is not None:
            graph_agent.close()


def evaluate_business_cases(
    cases: list[PrivateBusinessCase],
    *,
    documents: list[DocumentInput],
    question_bank: QuestionBankRepository,
    workdir: str | Path,
    engine_order: tuple[EngineName, EngineName] = ("baseline", "langgraph"),
) -> PrivateBusinessComparison:
    if len(engine_order) != 2 or set(engine_order) != {"baseline", "langgraph"}:
        raise ValueError("engine_order must contain baseline and langgraph exactly once")
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    backend = Bm25BaselineBackend(documents)
    observations: list[BusinessObservation] = []
    for index, case in enumerate(cases):
        for engine in engine_order:
            observations.append(
                _evaluate_one(
                    case,
                    engine=engine,
                    backend=backend,
                    question_bank=question_bank,
                    run_root=root / case.case_id / engine,
                    index=index,
                )
            )
    return PrivateBusinessComparison(
        case_count=len(cases),
        scenario_counts=dict(Counter(item.scenario for item in cases)),
        scope_counts=dict(Counter(item.scope for item in cases)),
        observations=observations,
    )


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _metrics(rows: list[BusinessObservation]) -> EngineBusinessMetrics:
    grounded = [item.grounded_result for item in rows if item.grounded_result is not None]
    confirmed = [
        item.confirmation_success
        for item in rows
        if item.confirmation_success is not None
    ]
    recovery = [item.recovery_success for item in rows if item.scenario == "recovery"]
    latencies = [item.latency_ms for item in rows]
    failures = Counter(item.error_type for item in rows if item.error_type)
    return EngineBusinessMetrics(
        observation_count=len(rows),
        completion_rate=round(_rate([item.completion for item in rows]), 4),
        task_success_rate=round(_rate([item.task_success for item in rows]), 4),
        grounded_result_rate=round(_rate([bool(value) for value in grounded]), 4),
        confirmation_success_rate=round(_rate([bool(value) for value in confirmed]), 4),
        recovery_supported=all(item.recovery_supported for item in rows),
        recovery_success_rate=round(_rate([bool(value) for value in recovery]), 4),
        duplicate_side_effect_rate=round(
            _rate([item.duplicate_side_effect for item in rows]), 4
        ),
        mean_tool_steps=round(mean(item.tool_steps for item in rows), 4) if rows else 0.0,
        latency_p50_ms=round(median(latencies), 3) if latencies else 0.0,
        latency_p95_ms=round(_percentile_95(latencies), 3),
        failure_type_counts=dict(failures),
    )


def public_business_report(
    comparison: PrivateBusinessComparison,
) -> PublicBusinessReport:
    engines: dict[EngineName, EngineBusinessMetrics] = {
        engine: _metrics(
            [item for item in comparison.observations if item.engine == engine]
        )
        for engine in ("baseline", "langgraph")
    }
    baseline_p50 = engines["baseline"].latency_p50_ms
    graph_p50 = engines["langgraph"].latency_p50_ms
    overhead = (
        ((graph_p50 - baseline_p50) / baseline_p50) * 100 if baseline_p50 > 0 else None
    )
    return PublicBusinessReport(
        report_version="agent-business-comparison-v1",
        claim_scope="local_anonymized_workflow_comparison",
        case_count=comparison.case_count,
        scenario_counts=comparison.scenario_counts,
        scope_counts=comparison.scope_counts,
        engines=engines,
        langgraph_p50_latency_overhead_pct=round(overhead, 2) if overhead is not None else None,
        privacy={
            "raw_cases_public": False,
            "case_ids_public": False,
            "source_ids_public": False,
            "private_outputs_path": "data/local/evaluation",
        },
        recovery_comparison_note=(
            "Baseline has no checkpoint/resume API; LangGraph recovery is measured after an "
            "anonymous one-time finalize-node failure and requires the caller to resend inputs "
            "when the failed node needs them."
        ),
    )


def write_private_business_observations(
    comparison: PrivateBusinessComparison,
    path: Path = PRIVATE_OBSERVATIONS_PATH,
) -> None:
    _write_jsonl(path, comparison.observations)


def write_public_business_report(
    report: PublicBusinessReport, path: Path = PUBLIC_REPORT_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
