from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.agent.graph import LearningAgent
from app.agent.langgraph_agent import LangGraphLearningAgent
from app.agent.tools import LearningTools
from app.core.config import AgentEngine, AppConfig, BackendName, LlmBackend, MultimodalBackend
from app.llm.base import LlmClient
from app.llm.llamacpp_client import LlamaCppClient
from app.multimodal.base import MultimodalModel
from app.multimodal.minicpmo_client import MiniCPMOClient
from app.question_bank.repository import QuestionBankRepository
from app.rag.base import RagBackend
from app.rag.corpus_file import load_document_corpus
from app.rag.factory import build_backend
from app.rag.scoped_corpus import load_scoped_corpus, mark_legacy_documents_generic
from app.services.learning import LearningService
from app.storage.sqlite import SQLiteLearningStore


@dataclass(frozen=True)
class RuntimeContainer:
    config: AppConfig
    backend: RagBackend
    store: SQLiteLearningStore
    service: LearningService
    agent: LearningAgent | LangGraphLearningAgent
    llm: LlmClient | None = None
    multimodal: MultimodalModel | None = None

    @property
    def innovation_backend_mode(self) -> str:
        """Return the explicit mode exposed by the public competition demo."""
        if self.multimodal is not None:
            return "live"
        if self.config.public_demo:
            return "fixture"
        return "unavailable"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_local_path(value: str) -> Path:
    project_root_path = project_root()
    path = (project_root_path / value).resolve()
    if path != project_root_path and project_root_path not in path.parents:
        raise ValueError("runtime data path must stay inside the project directory")
    return path


def build_runtime(config: AppConfig) -> RuntimeContainer:
    documents = None
    if config.rag_backend == BackendName.BASELINE:
        generic_path = _project_local_path(config.generic_corpus_path)
        personal_path = _project_local_path(config.personal_corpus_path)
        if generic_path.exists() or personal_path.exists():
            documents = load_scoped_corpus(generic_path, personal_path)
        else:
            documents = mark_legacy_documents_generic(
                load_document_corpus(_project_local_path(config.corpus_path))
            )
    elif config.rag_backend == BackendName.FAKE:
        documents = mark_legacy_documents_generic(
            load_document_corpus(_project_local_path(config.corpus_path))
        )
    backend = build_backend(config, documents)
    llm: LlmClient | None = None
    if config.llm_backend == LlmBackend.LLAMACPP:
        llm = LlamaCppClient(config.llamacpp)
    multimodal: MultimodalModel | None = None
    if config.multimodal_backend == MultimodalBackend.MINICPMO:
        multimodal = MiniCPMOClient(
            config.minicpmo,
            bearer_token=os.getenv("BAOYAN_MINICPMO_API_TOKEN") or None,
        )
    store = SQLiteLearningStore(_project_local_path(config.database_path))
    redaction_terms = tuple(
        term.strip()
        for term in os.getenv("BAOYAN_PRIVATE_REDACTION_TERMS", "").split(";;")
        if len(term.strip()) >= 2
    )
    service = LearningService(
        backend,
        store,
        llm=llm,
        private_redaction_terms=redaction_terms,
        question_bank=(
            QuestionBankRepository(_project_local_path(config.question_bank_path))
            if config.rag_backend == BackendName.BASELINE
            and _project_local_path(config.question_bank_path).is_file()
            else None
        ),
    )
    tools = LearningTools(service)
    if config.agent_engine == AgentEngine.LANGGRAPH:
        agent: LearningAgent | LangGraphLearningAgent = LangGraphLearningAgent(
            tools,
            store,
            backend_name=backend.backend_info().backend,
            checkpoint_path=_project_local_path(config.agent_checkpoint_path),
        )
    else:
        agent = LearningAgent(tools, store, backend_name=backend.backend_info().backend)
    return RuntimeContainer(
        config=config,
        backend=backend,
        llm=llm,
        multimodal=multimodal,
        store=store,
        service=service,
        agent=agent,
    )
