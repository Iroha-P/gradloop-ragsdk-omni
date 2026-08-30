"""Run the privacy-locked public demo against synthetic assets only."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def public_demo_environment() -> dict[str, str]:
    """Return fail-closed paths for a public demo process."""
    return {
        "BAOYAN_PROFILE": "local_baseline",
        "BAOYAN_RAG_BACKEND": "baseline",
        "BAOYAN_LLM_BACKEND": "none",
        "BAOYAN_AGENT_ENGINE": "langgraph",
        "BAOYAN_PUBLIC_DEMO": "true",
        "BAOYAN_CORPUS_PATH": "data/public_eval/portfolio_v1/corpus.jsonl",
        "BAOYAN_GENERIC_CORPUS_PATH": (
            "data/public_eval/portfolio_v1/no-local-generic-corpus.jsonl"
        ),
        "BAOYAN_PERSONAL_CORPUS_PATH": (
            "data/public_eval/portfolio_v1/no-local-personal-corpus.jsonl"
        ),
        "BAOYAN_QUESTION_BANK_PATH": "data/public_demo/question_bank.jsonl",
        "BAOYAN_COVERAGE_PATH": "data/public_demo/no-private-coverage.json",
        "BAOYAN_DATABASE_PATH": "data/local/public_demo_runtime/learning.db",
        "BAOYAN_AGENT_CHECKPOINT_PATH": (
            "data/local/public_demo_runtime/agent_state.sqlite"
        ),
    }


def main() -> None:
    runtime_dir = PROJECT_ROOT / "data" / "local" / "public_demo_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(public_demo_environment())
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
