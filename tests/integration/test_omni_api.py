from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.graph import LearningAgent
from app.agent.tools import LearningTools
from app.api import main as api_main
from app.api.main import create_app
from app.core.config import AppConfig, MiniCpmoSettings, MultimodalBackend
from app.core.runtime import RuntimeContainer
from app.multimodal.base import MultimodalHealth, OmniRequest, OmniResponse
from app.rag.base import DocumentInput
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.services.learning import LearningService
from app.storage.sqlite import SQLiteLearningStore


class FakeMultimodal:
    def __init__(self) -> None:
        self.last_request: OmniRequest | None = None

    def healthcheck(self) -> MultimodalHealth:
        return MultimodalHealth(
            backend="fake-minicpmo",
            ready=True,
            model="openbmb/MiniCPM-o-4_5",
            modalities=["image", "audio", "video"],
            message="ready",
        )

    def generate(self, request: OmniRequest) -> OmniResponse:
        self.last_request = request
        return OmniResponse(
            backend="fake-minicpmo",
            model="openbmb/MiniCPM-o-4_5",
            content="这是一份基于公开合成输入的结构化反馈。",
            modalities=[item.modality for item in request.attachments],
            attachment_count=len(request.attachments),
            elapsed_ms=12.5,
        )


def _client(tmp_path, *, enabled: bool = True, max_upload_bytes: int = 1024):
    config = AppConfig(
        database_path="data/local/test/omni.db",
        multimodal_backend=(MultimodalBackend.MINICPMO if enabled else MultimodalBackend.NONE),
        minicpmo=MiniCpmoSettings(max_upload_bytes=max_upload_bytes),
    )
    backend = Bm25BaselineBackend(
        [
            DocumentInput(
                source_id="public-guide",
                chunk_id="public-1",
                title="Public guide",
                text="Synthetic public evidence.",
                metadata={"privacy_lane": "generic"},
            )
        ]
    )
    store = SQLiteLearningStore(tmp_path / "omni.db")
    service = LearningService(backend, store)
    agent = LearningAgent(LearningTools(service), store, backend_name="baseline")
    multimodal = FakeMultimodal() if enabled else None
    container = RuntimeContainer(
        config=config,
        backend=backend,
        store=store,
        service=service,
        agent=agent,
        multimodal=multimodal,
    )
    return TestClient(create_app(config, container)), multimodal


def _png(payload: bytes = b"synthetic") -> bytes:
    return b"\x89PNG\r\n\x1a\n" + payload


def test_ready_and_upload_expose_safe_multimodal_status(tmp_path) -> None:
    api, fake = _client(tmp_path)

    ready = api.get("/ready").json()
    response = api.post(
        "/v1/omni/analyze",
        data={
            "prompt": "请分析这份公开合成材料",
            "content_policy": "public_or_synthetic",
        },
        files=[("files", ("public.png", _png(), "image/png"))],
    )

    assert ready["status"] == "ready"
    assert ready["multimodal"]["backend"] == "fake-minicpmo"
    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "answer": "这是一份基于公开合成输入的结构化反馈。",
        "backend": "fake-minicpmo",
        "model": "openbmb/MiniCPM-o-4_5",
        "modalities": ["image"],
        "attachment_count": 1,
        "elapsed_ms": 12.5,
        "retention": "request_scoped",
        "content_policy": "public_or_synthetic",
    }
    assert fake is not None and fake.last_request is not None
    assert not hasattr(fake.last_request.attachments[0], "filename")


def test_upload_rejects_non_public_policy_and_invalid_signature(tmp_path) -> None:
    api, _ = _client(tmp_path)
    private_policy = api.post(
        "/v1/omni/analyze",
        data={"prompt": "analyze", "content_policy": "private"},
        files=[("files", ("public.png", _png(), "image/png"))],
    )
    mismatched = api.post(
        "/v1/omni/analyze",
        data={"prompt": "analyze", "content_policy": "public_or_synthetic"},
        files=[("files", ("fake.png", b"not-a-png", "image/png"))],
    )

    assert private_policy.status_code == 400
    assert private_policy.json()["error"]["code"] == "unsafe_input"
    assert mismatched.status_code == 400
    assert "not-a-png" not in mismatched.text


def test_upload_rejects_oversize_before_model_call(tmp_path) -> None:
    api, fake = _client(tmp_path, max_upload_bytes=1024)
    response = api.post(
        "/v1/omni/analyze",
        data={"prompt": "analyze", "content_policy": "public_or_synthetic"},
        files=[("files", ("large.png", _png(b"x" * 1024), "image/png"))],
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert fake is not None and fake.last_request is None


def test_upload_fails_explicitly_when_backend_is_disabled(tmp_path) -> None:
    api, _ = _client(tmp_path, enabled=False)
    response = api.post(
        "/v1/omni/analyze",
        data={"prompt": "analyze", "content_policy": "public_or_synthetic"},
        files=[("files", ("public.png", _png(), "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_unavailable"


def test_local_ui_exposes_multimodal_flow_without_private_upload_language() -> None:
    html = (api_main.project_root() / "app" / "ui" / "index.html").read_text(encoding="utf-8")

    assert "全模态训练台" in html
    assert "/v1/omni/analyze" in html
    assert "public_or_synthetic" in html
    assert "私人资料禁止发送至云端" in html
    assert "成绩单、证件、私人简历" in html
    assert "FormData" in html


def test_document_parser_route_only_serves_public_allowlisted_assets(tmp_path) -> None:
    api, fake = _client(tmp_path)
    for asset in ("client.mjs", "policy.mjs", "worker.mjs", "vendor/pdf.mjs", "vendor/pdf.worker.mjs", "vendor/word.mjs"):
        response = api.get(f"/documents/{asset}")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]
    for asset in ("unknown.mjs", "vendor/unknown.json", "%2e%2e%2fREADME.md"):
        assert api.get(f"/documents/{asset}").status_code == 404
    assert fake is not None and fake.last_request is None
