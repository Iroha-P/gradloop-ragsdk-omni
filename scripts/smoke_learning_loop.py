from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.langgraph_agent import LangGraphLearningAgent
from app.agent.state import AgentResumeRequest, AgentRunRequest
from app.agent.tools import LearningTools
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.rag.corpus_file import load_document_corpus
from app.rag.scoped_corpus import mark_legacy_documents_generic
from app.services.learning import LearningService
from app.storage.sqlite import SQLiteLearningStore


def main() -> int:
    corpus_path = PROJECT_ROOT / "data/local/private/indexes/baseline/corpus.jsonl"
    documents = mark_legacy_documents_generic(load_document_corpus(corpus_path))
    backend = Bm25BaselineBackend(documents)
    user_id = "smoke_user"
    with tempfile.TemporaryDirectory(prefix="baoyan-agent-smoke-") as folder:
        store = SQLiteLearningStore(Path(folder) / "learning.db")
        service = LearningService(backend, store)
        agent = LangGraphLearningAgent(
            LearningTools(service),
            store,
            backend_name="baseline",
            checkpoint_path=Path(folder) / "agent-checkpoints.sqlite",
        )

        plan_run = agent.run(
            AgentRunRequest(
                user_id=user_id,
                intent="plan",
                payload={
                    "goal": "保研面试准备",
                    "deadline": (date.today() + timedelta(days=6)).isoformat(),
                    "weekly_hours": 14,
                    "current_level": "beginner",
                    "weak_tags": ["自我介绍", "项目经历表达"],
                    "completed_tasks": [],
                    "intensity": "balanced",
                    "days": 7,
                },
            )
        )
        plan_id = plan_run.result["plan_id"]
        practice_run = agent.run(
            AgentRunRequest(
                user_id=user_id,
                intent="practice",
                payload={"topic": "自我介绍", "difficulty": "medium", "count": 1},
            )
        )
        question_id = practice_run.result["questions"][0]["question_id"]
        grade_run = agent.run(
            AgentRunRequest(
                user_id=user_id,
                intent="grade",
                payload={"question_id": question_id, "answer": "我会结合两种检索。"},
            )
        )
        mistake_id = grade_run.result.get("mistake_id")
        revision_id = grade_run.result.get("revision_id")
        if not mistake_id or not revision_id:
            raise RuntimeError("low-score answer did not create mistake and revision proposal")
        if grade_run.status != "waiting_for_confirmation":
            raise RuntimeError("low-score answer did not pause for user confirmation")
        revision_run = agent.resume(
            AgentResumeRequest(
                trace_id=grade_run.trace_id,
                user_id=user_id,
                confirm=True,
            )
        )
        mistakes_run = agent.run(AgentRunRequest(user_id=user_id, intent="mistakes", payload={"limit": 10}))
        revised_plan = store.get_plan(plan_id, user_id)
        result = {
            "status": "passed",
            "backend": backend.backend_info().backend,
            "agent_engine": grade_run.engine,
            "checkpoint_resume": revision_run.recovery.resumed,
            "corpus_chunks": len(documents),
            "plan_days": len(revised_plan.tasks),
            "question_count": len(practice_run.result["questions"]),
            "score": grade_run.result["score"],
            "mistake_count": len(mistakes_run.result["mistakes"]),
            "revision_status": revision_run.result["revision"]["status"],
            "max_agent_steps": max(
                plan_run.steps,
                practice_run.steps,
                grade_run.steps,
                revision_run.steps,
                mistakes_run.steps,
            ),
            "raw_private_text_printed": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
