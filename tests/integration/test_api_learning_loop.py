from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.agent.langgraph_agent import LangGraphLearningAgent
from app.agent.tools import LearningTools
from app.api.main import create_app
from app.core.config import AppConfig
from app.core.runtime import RuntimeContainer
from app.rag.base import DocumentInput
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.services.learning import LearningService
from app.storage.sqlite import SQLiteLearningStore


def client(tmp_path):
    config = AppConfig(database_path="data/local/test/learning.db")
    backend = Bm25BaselineBackend(
        [
            DocumentInput(
                source_id="synthetic-guide",
                chunk_id="synthetic-1",
                title="合成指南",
                text="混合检索结合关键词检索与向量检索，Agent 使用显式状态和有界工具调用。",
                metadata={"privacy_lane": "generic"},
            )
        ]
    )
    store = SQLiteLearningStore(tmp_path / "api.db")
    service = LearningService(backend, store)
    agent = LangGraphLearningAgent(
        LearningTools(service),
        store,
        backend_name="baseline",
        checkpoint_path=tmp_path / "agent-checkpoints.sqlite",
    )
    container = RuntimeContainer(config=config, backend=backend, store=store, service=service, agent=agent)
    return TestClient(create_app(config, container))


def test_http_minimal_loop(tmp_path):
    api = client(tmp_path)
    user_id = "http_user"
    assert api.get("/health").status_code == 200
    ready = api.get("/ready").json()
    assert ready["status"] == "ready"
    assert ready["agent"] == {
        "engine": "langgraph",
        "checkpointer": "sqlite",
        "recoverable": True,
    }
    plan = api.post(
        "/v1/plans",
        json={
            "user_id": user_id,
            "goal": "大模型应用开发面试准备",
            "deadline": (date.today() + timedelta(days=6)).isoformat(),
            "weekly_hours": 14,
            "current_level": "beginner",
            "weak_tags": ["混合检索", "Agent 状态管理"],
            "completed_tasks": [],
            "intensity": "balanced",
            "days": 7,
        },
    )
    assert plan.status_code == 200
    question = api.post(
        "/v1/practice/questions",
        json={"user_id": user_id, "topic": "混合检索", "difficulty": "medium", "count": 1},
    ).json()["questions"][0]
    grade = api.post(
        "/v1/practice/grade",
        json={"user_id": user_id, "question_id": question["question_id"], "answer": "结合两种检索。"},
    )
    assert grade.status_code == 200
    grade_body = grade.json()
    assert grade_body["mistake_id"] and grade_body["revision_id"]
    revision = api.post(
        f"/v1/plans/{plan.json()['plan_id']}/revisions/{grade_body['revision_id']}/decision",
        json={"user_id": user_id, "confirm": True},
    )
    assert revision.status_code == 200
    assert revision.json()["revision"]["status"] == "confirmed"
    assert len(api.get("/v1/mistakes", params={"user_id": user_id}).json()["mistakes"]) == 1
    assert api.get("/").status_code == 200


def test_public_config_contains_only_non_secret_realtime_fields(tmp_path):
    api = client(tmp_path)

    response = api.get("/v1/public-config")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"realtime", "privacy"}
    assert set(body["realtime"]) == {"enabled", "proxy_url", "session_seconds"}
    assert body["privacy"] == {"public_only": True}
    serialized = response.text.casefold()
    assert all(word not in serialized for word in ("api_key", "token", "secret", "password"))
    assert response.headers["cache-control"] == "no-store"


def test_http_agent_interrupt_status_and_resume(tmp_path):
    api = client(tmp_path)
    user_id = "agent_http_user"
    plan = api.post(
        "/v1/plans",
        json={
            "user_id": user_id,
            "goal": "合成面试准备",
            "deadline": (date.today() + timedelta(days=6)).isoformat(),
            "weekly_hours": 14,
            "current_level": "beginner",
            "weak_tags": ["合成主题"],
            "completed_tasks": [],
            "intensity": "balanced",
            "days": 7,
        },
    ).json()
    question = api.post(
        "/v1/practice/questions",
        json={"user_id": user_id, "topic": "合成主题", "difficulty": "medium", "count": 1},
    ).json()["questions"][0]

    first = api.post(
        "/v1/agent/run",
        json={
            "user_id": user_id,
            "intent": "grade",
            "payload": {"question_id": question["question_id"], "answer": "回答很短"},
        },
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["status"] == "waiting_for_confirmation"
    assert first_body["engine"] == "langgraph"

    status = api.get(
        f"/v1/agent/runs/{first_body['trace_id']}",
        params={"user_id": user_id},
    )
    assert status.status_code == 200
    assert status.json()["next_nodes"] == ["await_revision_decision"]

    resumed = api.post(
        "/v1/agent/resume",
        json={"trace_id": first_body["trace_id"], "user_id": user_id, "confirm": True},
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert resumed.json()["result"]["revision"]["status"] == "confirmed"
    assert api.get("/v1/plans/" + plan["plan_id"], params={"user_id": user_id}).status_code == 200

    forbidden = api.get(
        f"/v1/agent/runs/{first_body['trace_id']}",
        params={"user_id": "other_user"},
    )
    assert forbidden.status_code == 403


def test_api_default_remains_gate_disabled_with_unchanged_response_schema(tmp_path):
    api = client(tmp_path)

    response = api.post(
        "/v1/ask",
        json={
            "user_id": "default_api_user",
            "question": "混合检索是什么？",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "backend",
        "backend_version",
        "query",
        "answer",
        "citations",
        "trace_id",
        "retrieval",
        "abstained",
        "warnings",
        "generation_backend",
        "generator_model",
        "llm_called",
    }
    assert body["generation_backend"] != "boundary-gate"
    assert body["llm_called"] is False


def test_http_mistake_redo_updates_the_existing_record(tmp_path):
    api = client(tmp_path)
    user_id = "redo_http_user"
    question = api.post(
        "/v1/practice/questions",
        json={"user_id": user_id, "topic": "混合检索", "difficulty": "medium", "count": 1},
    ).json()["questions"][0]
    grade = api.post(
        "/v1/practice/grade",
        json={
            "user_id": user_id,
            "question_id": question["question_id"],
            "answer": "回答太短",
            "use_llm_review": False,
        },
    ).json()
    answer = (
        "结论：混合检索结合两路方法。因为关键词检索适合精确术语，因此先召回；"
        "向量检索补充语义候选，再通过融合方法排序。例如我负责实现流程，"
        "数据结果提升 12%。步骤包括召回、去重和重排。局限是语义漂移，"
        "风险是错误证据；下一步改进是加入门控和失败分析。"
    ) * 2

    response = api.post(
        f"/v1/mistakes/{grade['mistake_id']}/redo",
        json={
            "user_id": user_id,
            "question_id": question["question_id"],
            "answer": answer,
            "use_llm_review": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mistake"]["mistake_id"] == grade["mistake_id"]
    assert body["mistake"]["redo_count"] == 1
    assert body["mistake"]["mastery_status"] == "reviewing"
    mistakes = api.get("/v1/mistakes", params={"user_id": user_id}).json()["mistakes"]
    assert len(mistakes) == 1


def test_ui_exposes_mistake_redo_controls(tmp_path):
    html = client(tmp_path).get("/").text

    assert 'id="mistakeSelect"' in html
    assert 'id="redoAnswer"' in html
    assert "/redo" in html
