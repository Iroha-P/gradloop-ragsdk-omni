from __future__ import annotations

import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.agent.graph import build_innovation_coaching
from app.agent.state import AgentResumeRequest, AgentRunRequest
from app.api.models import AskRequest, EvaluationRequest
from app.core.concurrency import ExpensiveOperationGate
from app.core.config import AppConfig, MultimodalBackend
from app.core.errors import (
    AppError,
    BackendUnavailableError,
    CapacityExceededError,
    PayloadTooLargeError,
    UnsafeInputError,
)
from app.core.runtime import RuntimeContainer, build_runtime, project_root
from app.domain.grading import GradeRequest
from app.domain.mistakes import MistakeListRequest
from app.domain.plans import CompleteTaskRequest, RevisionDecisionRequest, StudyPlanRequest
from app.domain.practice import PracticeRequest
from app.multimodal.base import MultimodalHealth, OmniRequest, build_attachment
from app.question_bank.coverage import load_coverage_report
from app.rag.base import RetrievalRequest
from app.storage.sqlite import RecordNotFoundError, UserScopeError


class LazyRuntime:
    def __init__(self, config: AppConfig, preset: RuntimeContainer | None = None):
        self.config = config
        self._value = preset
        self._lock = threading.Lock()

    def get(self) -> RuntimeContainer:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = build_runtime(self.config)
        return self._value


def _valid_identifier(value: str | None) -> str | None:
    if value is not None and 1 <= len(value) <= 128 and value.replace("-", "").replace("_", "").isalnum():
        return value
    return None


def _valid_trace_id(value: str | None) -> str | None:
    if value is not None and re.fullmatch(r"[a-fA-F0-9]{32}", value):
        return value
    return None


def _sse(event: str, payload: dict[str, Any]) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def create_app(
    config: AppConfig | None = None,
    container: RuntimeContainer | None = None,
) -> FastAPI:
    settings = config or AppConfig.from_env()
    runtime = LazyRuntime(settings, container)
    expensive_gate = ExpensiveOperationGate(
        settings.max_expensive_requests,
        retry_after_seconds=settings.capacity_retry_after_seconds,
    )
    counters: Counter[str] = Counter()
    application = FastAPI(
        title="GradLoop RAGSDK Agent",
        version=__version__,
        description="Privacy-first, evidence-grounded local learning assistant.",
    )
    application.state.runtime = runtime

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = _valid_identifier(request.headers.get("x-request-id")) or uuid.uuid4().hex
        trace_id = _valid_trace_id(request.headers.get("x-trace-id")) or uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            counters["requests_total"] += 1
            counters[f"route:{request.url.path}"] += 1
            counters["request_time_ms_total"] += int((time.perf_counter() - started) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-trace-id"] = trace_id
        response.headers["cache-control"] = "no-store"
        return response

    def correlated_error(
        request: Request,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        error = {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request.state.request_id,
            "trace_id": request.state.trace_id,
        }
        return {"error": error}

    @application.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        headers = None
        if isinstance(exc, CapacityExceededError):
            headers = {"retry-after": str(settings.capacity_retry_after_seconds)}
        return JSONResponse(
            status_code=exc.status_code,
            content=correlated_error(
                request,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
            headers=headers,
        )

    @application.exception_handler(RecordNotFoundError)
    async def not_found_handler(request: Request, exc: RecordNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=correlated_error(
                request,
                code="not_found",
                message="record was not found",
            ),
        )

    @application.exception_handler(UserScopeError)
    async def user_scope_handler(request: Request, _exc: UserScopeError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content=correlated_error(
                request,
                code="user_scope_error",
                message="request is outside the user scope",
            ),
        )

    @application.exception_handler(ValueError)
    @application.exception_handler(ValidationError)
    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, _exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=correlated_error(
                request,
                code="invalid_request",
                message="invalid request",
            ),
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=correlated_error(
                request,
                code="runtime_error",
                message="internal server error",
            ),
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        page = project_root() / "app" / "ui" / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="UI is not installed")
        return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @application.get("/ui-assets/gradloop-icon.png", response_class=FileResponse, include_in_schema=False)
    def project_icon() -> FileResponse:
        icon = project_root() / "app" / "ui" / "assets" / "gradloop-icon.png"
        if not icon.is_file():
            raise HTTPException(status_code=404, detail="Project icon is not installed")
        return FileResponse(icon, media_type="image/png")

    @application.get("/documents/{asset_path:path}", response_class=FileResponse, include_in_schema=False)
    def document_parser_asset(asset_path: str) -> FileResponse:
        allowed = {
            "client.mjs", "policy.mjs", "worker.mjs", "vendor/pdf.mjs",
            "vendor/pdf.worker.mjs", "vendor/word.mjs", "vendor/provenance.json",
            "vendor/THIRD_PARTY_NOTICES.md",
        }
        if asset_path not in allowed:
            raise HTTPException(status_code=404, detail="Document asset is not installed")
        path = project_root() / "app/ui/documents" / asset_path
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Document asset is not installed")
        media_type = "text/javascript" if path.suffix == ".mjs" else "application/json" if path.suffix == ".json" else "text/plain"
        return FileResponse(path, media_type=media_type)

    @application.get("/realtime/{asset_name}", response_class=FileResponse, include_in_schema=False)
    def realtime_browser_asset(asset_name: str) -> FileResponse:
        if asset_name not in {"session.mjs", "protocol.mjs", "audio.mjs"}:
            raise HTTPException(status_code=404, detail="Realtime asset is not installed")
        path = project_root() / "app/ui/realtime" / asset_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Realtime asset is not installed")
        return FileResponse(path, media_type="text/javascript")

    @application.get("/public-demo-fixtures.mjs", response_class=FileResponse, include_in_schema=False)
    def public_browser_fixtures() -> FileResponse:
        path = project_root() / "app/ui/public-demo-fixtures.mjs"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Public fixture asset is not installed")
        return FileResponse(path, media_type="text/javascript")

    @application.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "version": __version__, "profile": settings.profile.value}

    @application.get("/v1/public-config", tags=["system"])
    def public_config() -> dict[str, Any]:
        """Expose browser-safe public demo settings without proxy credentials."""

        realtime = settings.map_realtime
        return {
            "realtime": {
                "enabled": realtime.enabled and realtime.proxy_url is not None,
                "proxy_url": realtime.proxy_url,
                "session_seconds": realtime.session_seconds,
            },
            "privacy": {"public_only": True},
        }

    @application.get("/ready", tags=["system"])
    def ready() -> dict:
        try:
            container_value = runtime.get()
            result = container_value.backend.healthcheck()
            generation = container_value.service.generation_health()
            multimodal = (
                container_value.multimodal.healthcheck()
                if container_value.multimodal is not None
                else MultimodalHealth(
                    backend="none",
                    ready=False,
                    model="none",
                    modalities=[],
                    message="multimodal model is disabled",
                )
            )
            llm_required = settings.llm_backend.value != "none"
            multimodal_required = settings.multimodal_backend != MultimodalBackend.NONE
            status = (
                "ready"
                if result.ready
                and (generation.ready or not llm_required)
                and (multimodal.ready or not multimodal_required)
                else "not_ready"
            )
            return {
                "status": status,
                **result.model_dump(mode="json"),
                "generation": generation.model_dump(mode="json"),
                "multimodal": multimodal.model_dump(mode="json"),
                "agent": {
                    "engine": container_value.agent.engine_name,
                    "checkpointer": container_value.agent.checkpointer_name,
                    "recoverable": container_value.agent.recoverable,
                },
                "components": {
                    "multimodal": {
                        "status": "ready" if multimodal.ready else (
                            "fixture"
                            if container_value.innovation_backend_mode == "fixture"
                            else "unavailable"
                        ),
                        "backend_mode": container_value.innovation_backend_mode,
                    },
                    "retrieval": {"status": "ready" if result.ready else "unavailable"},
                    "agent": {"status": "ready"},
                },
                "privacy": {
                    "mode": "public_demo" if settings.public_demo else "local_private",
                },
            }
        except (AppError, OSError, ValueError):
            return {
                "status": "not_ready",
                "error": {"code": "runtime_unavailable", "retryable": True},
            }

    @application.post("/v1/ask", tags=["learning"])
    def ask(body: AskRequest) -> dict:
        with expensive_gate.lease():
            result = runtime.get().service.ask(
                body.question,
                top_k=body.top_k,
                scope=body.scope,
            )
        return result.model_dump(mode="json")

    @application.post("/v1/omni/analyze", tags=["multimodal"])
    async def analyze_omni(
        prompt: Annotated[str, Form()],
        content_policy: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
    ) -> dict:
        if content_policy != "public_or_synthetic":
            raise UnsafeInputError(
                "Only public or synthetic demo material is accepted",
                details={"accepted_policy": "public_or_synthetic"},
            )
        if not files or len(files) > settings.minicpmo.max_attachments:
            raise UnsafeInputError(
                "Attachment count is outside the allowed range",
                details={"maximum": settings.minicpmo.max_attachments},
            )
        attachments = []
        for upload in files:
            try:
                content = await upload.read(settings.minicpmo.max_upload_bytes + 1)
                attachments.append(
                    build_attachment(
                        filename=upload.filename or "upload",
                        media_type=upload.content_type or "application/octet-stream",
                        content=content,
                        max_bytes=settings.minicpmo.max_upload_bytes,
                    )
                )
            except OverflowError as exc:
                raise PayloadTooLargeError(
                    "An attachment exceeds the configured size limit",
                    details={"maximum_bytes": settings.minicpmo.max_upload_bytes},
                ) from exc
            except ValueError as exc:
                raise UnsafeInputError("An attachment failed media validation") from exc
            finally:
                await upload.close()
        container_value = runtime.get()
        if container_value.multimodal is None and container_value.innovation_backend_mode != "fixture":
            raise BackendUnavailableError(
                "MiniCPM-o multimodal service is not configured",
                details={"backend": "minicpmo-ascend"},
            )
        attachment_modalities = list(dict.fromkeys(item.modality.value for item in attachments))
        # Keep the legacy API shape for existing clients.  The public competition
        # demo opts into the richer evidence/agent trace contract below.
        modalities = (
            ["text", *attachment_modalities]
            if settings.public_demo
            else attachment_modalities
        )
        retrieval = container_value.backend.retrieve(
            RetrievalRequest(query=prompt, top_k=settings.top_k)
        )
        evidence = [
            {
                "source_id": document.source_id,
                "chunk_id": document.chunk_id,
                "title": document.title,
                "excerpt": document.text[:240],
                "score": document.score,
            }
            for document in retrieval.documents
        ]
        if container_value.multimodal is None:
            model_output = {
                "summary": "合成案例已进入离线回放；此结果不是 MiniCPM-o 实时输出。",
                "observations": ["输入媒体通过格式签名校验", "已识别为公开合成素材"],
                "limitations": ["云端模型尚未运行", "不可用于证明真实模型延迟或能力"],
            }
            elapsed_ms = 0.0
            backend = "offline-fixture"
            model = "fixture-v1"
            backend_mode = "fixture"
        else:
            request = OmniRequest(
                prompt=prompt,
                attachments=tuple(attachments),
                max_tokens=settings.minicpmo.max_tokens,
            )
            with expensive_gate.lease():
                result = await run_in_threadpool(container_value.multimodal.generate, request)
            model_output = {"summary": result.content, "observations": [], "limitations": []}
            elapsed_ms = round(result.elapsed_ms, 3)
            backend = result.backend
            model = result.model
            backend_mode = "live"
        if not settings.public_demo:
            return {
                "status": "completed",
                "answer": model_output["summary"],
                "backend": backend,
                "model": model,
                "modalities": modalities,
                "attachment_count": len(attachments),
                "elapsed_ms": elapsed_ms,
                "retention": "request_scoped",
                "content_policy": "public_or_synthetic",
            }
        coaching = build_innovation_coaching(
            prompt=prompt,
            modalities=modalities,
            evidence=evidence,
            backend_mode=backend_mode,
            agent_engine=container_value.agent.engine_name,
        )
        return {
            "status": "completed",
            "backend_mode": backend_mode,
            "backend": backend,
            "model": model,
            "model_output": model_output,
            "modalities": modalities,
            "attachment_count": len(attachments),
            "elapsed_ms": elapsed_ms,
            "evidence": evidence,
            "coach_feedback": coaching["coach_feedback"],
            "next_action": coaching["next_action"],
            "agent_trace": coaching,
            "retention": "request_scoped",
            "content_policy": "public_or_synthetic",
        }

    @application.get("/v1/knowledge/coverage", tags=["knowledge"])
    def knowledge_coverage() -> dict:
        root = project_root().resolve()
        path = (root / settings.coverage_path).resolve()
        if path != root and root not in path.parents:
            raise ValueError("coverage path must stay inside the project directory")
        if not path.exists():
            return {"status": "not_synced", "coverage": None}
        report = load_coverage_report(path)
        return {"status": "ready", "coverage": report.model_dump(mode="json")}

    @application.post("/v1/agent/run", tags=["agent"])
    def run_agent(body: AgentRunRequest, request: Request) -> dict:
        with expensive_gate.lease():
            return runtime.get().agent.run(
                body,
                trace_id=request.state.trace_id,
            ).model_dump(mode="json")

    @application.post("/v1/agent/run/stream", tags=["agent"])
    def stream_agent(body: AgentRunRequest, request: Request) -> StreamingResponse:
        expensive_gate.try_acquire()
        request_id = request.state.request_id
        trace_id = request.state.trace_id

        def generate() -> Iterator[str]:
            try:
                agent = runtime.get().agent
                for event in agent.stream(body, trace_id=trace_id):
                    payload = {key: value for key, value in event.items() if key != "event"}
                    payload["request_id"] = request_id
                    yield _sse(str(event["event"]), payload)
            finally:
                expensive_gate.release()

        return StreamingResponse(generate(), media_type="text/event-stream")

    @application.post("/v1/agent/resume", tags=["agent"])
    def resume_agent(body: AgentResumeRequest) -> dict:
        agent = runtime.get().agent
        if not hasattr(agent, "resume"):
            raise ValueError("the baseline agent does not support checkpoint resume")
        with expensive_gate.lease():
            return agent.resume(body).model_dump(mode="json")

    @application.get("/v1/agent/runs/{trace_id}", tags=["agent"])
    def get_agent_run(
        trace_id: str,
        user_id: str = Query(pattern=r"^[a-zA-Z0-9_-]{6,64}$"),
    ) -> dict:
        agent = runtime.get().agent
        if not hasattr(agent, "get_status"):
            raise ValueError("the baseline agent does not expose checkpoint status")
        return agent.get_status(trace_id, user_id).model_dump(mode="json")

    @application.post("/v1/plans", tags=["plans"])
    def create_plan(body: StudyPlanRequest) -> dict:
        return runtime.get().service.create_plan(body).model_dump(mode="json")

    @application.get("/v1/plans/{plan_id}", tags=["plans"])
    def get_plan(plan_id: str, user_id: str = Query(pattern=r"^[a-zA-Z0-9_-]{6,64}$")) -> dict:
        return runtime.get().store.get_plan(plan_id, user_id).model_dump(mode="json")

    @application.post("/v1/plans/{plan_id}/tasks/{task_id}/complete", tags=["plans"])
    def complete_task(plan_id: str, task_id: str, body: CompleteTaskRequest) -> dict:
        return runtime.get().service.complete_task(plan_id, task_id, body.user_id).model_dump(mode="json")

    @application.post("/v1/plans/{plan_id}/revisions/{revision_id}/decision", tags=["plans"])
    def decide_revision(
        plan_id: str,
        revision_id: str,
        body: RevisionDecisionRequest,
    ) -> dict:
        revision, plan = runtime.get().service.decide_revision(revision_id, body)
        if revision.plan_id != plan_id:
            raise UserScopeError("revision does not belong to this plan")
        return {
            "revision": revision.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json") if plan else None,
        }

    @application.post("/v1/practice/questions", tags=["practice"])
    def questions(body: PracticeRequest) -> dict:
        return runtime.get().service.get_questions(body).model_dump(mode="json")

    @application.post("/v1/practice/grade", tags=["practice"])
    def grade(body: GradeRequest) -> dict:
        return runtime.get().service.grade(body).model_dump(mode="json")

    @application.get("/v1/mistakes", tags=["practice"])
    def mistakes(
        user_id: str = Query(pattern=r"^[a-zA-Z0-9_-]{6,64}$"),
        topic: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        items = runtime.get().service.list_mistakes(
            MistakeListRequest(user_id=user_id, topic=topic, limit=limit)
        )
        return {"mistakes": [item.model_dump(mode="json") for item in items]}

    @application.post("/v1/mistakes/{mistake_id}/redo", tags=["practice"])
    def redo_mistake(mistake_id: str, body: GradeRequest) -> dict:
        return runtime.get().service.redo_mistake(mistake_id, body).model_dump(mode="json")

    @application.delete("/v1/mistakes/{mistake_id}", tags=["practice"])
    def delete_mistake(
        mistake_id: str,
        user_id: str = Query(pattern=r"^[a-zA-Z0-9_-]{6,64}$"),
    ) -> dict:
        deleted = runtime.get().store.delete_mistake(mistake_id, user_id)
        if not deleted:
            raise RecordNotFoundError("mistake not found")
        return {"deleted": True, "mistake_id": mistake_id}

    @application.post("/v1/evaluate", tags=["system"])
    def evaluate(body: EvaluationRequest) -> dict:
        queries = body.queries or ["自我介绍 面试", "项目经历 创新点", "复试 英语表达"]
        backend = runtime.get().backend
        results = [backend.retrieve(RetrievalRequest(query=query, top_k=body.top_k)) for query in queries]
        return {
            "backend": backend.backend_info().model_dump(mode="json"),
            "query_count": len(results),
            "hit_count": sum(bool(item.documents) for item in results),
            "average_selected": round(sum(len(item.documents) for item in results) / len(results), 2),
            "note": "仅报告检索命中，不代表答案正确率；未使用人工标注集时不输出虚假的质量分数。",
        }

    @application.get("/metrics", response_class=PlainTextResponse, tags=["system"])
    def metrics() -> str:
        return (
            "\n".join(f"baoyan_{key.replace(':', '_')} {value}" for key, value in sorted(counters.items()))
            + "\n"
        )

    @application.get("/v1/users/{user_id}/export", tags=["privacy"])
    def export_user(user_id: str) -> dict:
        return runtime.get().store.export_user_data(user_id)

    @application.delete("/v1/users/{user_id}", tags=["privacy"])
    def delete_user(user_id: str) -> dict:
        return {"deleted": runtime.get().store.delete_user_data(user_id)}

    return application


app = create_app()
