from __future__ import annotations

import os
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeProfile(StrEnum):
    LOCAL_BASELINE = "local_baseline"
    TEST_FAKE = "test_fake"
    RAGSDK_LOCAL = "ragsdk_local"
    RAGSDK_FULL = "ragsdk_full"


class BackendName(StrEnum):
    BASELINE = "baseline"
    FAKE = "fake"
    RAGSDK = "ragsdk"


class LlmBackend(StrEnum):
    NONE = "none"
    LLAMACPP = "llamacpp"


class MultimodalBackend(StrEnum):
    NONE = "none"
    MINICPMO = "minicpmo"


class AgentEngine(StrEnum):
    BASELINE = "baseline"
    LANGGRAPH = "langgraph"


class RagsdkRetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class LlamaCppSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(default="http://127.0.0.1:11435", min_length=1, max_length=2048)
    model: str = Field(default="qwen3-4b-q4", pattern=r"^[a-zA-Z0-9_.:/-]{1,128}$")
    timeout_seconds: int = Field(default=180, ge=5, le=600)
    context_chars: int = Field(default=12000, ge=1000, le=50000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class MiniCpmoSettings(BaseModel):
    """Non-secret connection policy for the separate Ascend inference service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(default="http://127.0.0.1:18080", min_length=1, max_length=2048)
    model: str = Field(default="openbmb/MiniCPM-o-4_5", pattern=r"^[a-zA-Z0-9_.:/-]{1,128}$")
    allowed_hosts: tuple[str, ...] = Field(
        default=("127.0.0.1", "localhost", "::1"),
        min_length=1,
        max_length=16,
    )
    timeout_seconds: int = Field(default=180, ge=5, le=600)
    max_upload_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024)
    max_attachments: int = Field(default=4, ge=1, le=8)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class MapRealtimePublicSettings(BaseModel):
    """Public browser connection settings; credentials never belong here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proxy_url: str | None = Field(default=None, max_length=2048)
    enabled: bool = False
    session_seconds: int = Field(default=300, ge=30, le=600)

    @model_validator(mode="after")
    def validate_proxy_policy(self) -> MapRealtimePublicSettings:
        if self.proxy_url is None:
            if self.enabled:
                raise ValueError("enabled realtime requires proxy_url")
            return self
        parsed = urlparse(self.proxy_url)
        hostname = parsed.hostname or ""
        if parsed.scheme not in {"ws", "wss"} or not hostname or parsed.username or parsed.password:
            raise ValueError("realtime proxy_url must be a websocket URL")
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme == "ws" and hostname.casefold() not in local_hosts:
            raise ValueError("remote realtime proxy_url must use wss")
        return self


class RagsdkSettings(BaseModel):
    """Non-secret RAGSDK runtime settings; credentials remain in service-specific env vars."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: str = Field(default="data/local/ragsdk", min_length=1, max_length=1024)
    milvus_uri: str = Field(default="data/local/ragsdk/milvus.db", min_length=1, max_length=2048)
    embedding_url: str | None = Field(default=None, max_length=2048)
    embedding_dimension: int = Field(default=1024, ge=1, le=1_048_576)
    retrieval_mode: RagsdkRetrievalMode = RagsdkRetrievalMode.DENSE
    reranker_url: str | None = Field(default=None, max_length=2048)
    dense_candidate_k: int = Field(default=20, ge=1, le=10_000)
    sparse_candidate_k: int = Field(default=20, ge=1, le=10_000)
    fusion_candidate_k: int = Field(default=20, ge=1, le=10_000)
    fusion_baseline: float = Field(default=0.4, ge=0.0, le=1.0)
    fusion_amplitude: float = Field(default=0.3, ge=0.0, le=1.0)
    fusion_slope: float = Field(default=1.0, gt=0.0)
    fusion_midpoint: float = Field(default=6.0, gt=0.0)
    knowledge_name: str = Field(default="baoyan_private", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    user_id: str = Field(default="local_user", pattern=r"^[a-zA-Z0-9_-]{6,64}$")
    vector_collection: str = Field(default="baoyan_vectors", pattern=r"^[a-zA-Z0-9_]{1,64}$")
    chunk_collection: str = Field(default="baoyan_chunks", pattern=r"^[a-zA-Z0-9_]{1,64}$")
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)
    llm_url: str | None = Field(default=None, max_length=2048)
    llm_model: str | None = Field(default=None, max_length=256)


class AppConfig(BaseModel):
    """Validated runtime configuration with explicit backend identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: RuntimeProfile = RuntimeProfile.LOCAL_BASELINE
    rag_backend: BackendName = BackendName.BASELINE
    llm_backend: LlmBackend = LlmBackend.NONE
    multimodal_backend: MultimodalBackend = MultimodalBackend.NONE
    agent_engine: AgentEngine = AgentEngine.LANGGRAPH
    public_demo: bool = False
    top_k: int = Field(default=5, ge=1, le=100)
    max_query_chars: int = Field(default=4096, ge=32, le=32768)
    max_expensive_requests: int = Field(default=2, ge=1, le=64)
    capacity_retry_after_seconds: int = Field(default=1, ge=1, le=60)
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    database_path: str = Field(default="data/local/app/learning.db", min_length=1, max_length=1024)
    agent_checkpoint_path: str = Field(
        default="data/local/app/agent_checkpoints.sqlite",
        min_length=1,
        max_length=1024,
    )
    corpus_path: str = Field(
        default="data/local/private/indexes/baseline/corpus.jsonl",
        min_length=1,
        max_length=1024,
    )
    feishu_snapshot_path: str = Field(
        default="data/local/feishu/current", min_length=1, max_length=1024
    )
    question_bank_path: str = Field(
        default="data/local/question_bank/questions.jsonl", min_length=1, max_length=1024
    )
    coverage_path: str = Field(
        default="data/local/question_bank/coverage.json", min_length=1, max_length=1024
    )
    generic_corpus_path: str = Field(
        default="data/local/question_bank/corpora/generic.jsonl",
        min_length=1,
        max_length=1024,
    )
    personal_corpus_path: str = Field(
        default="data/local/question_bank/corpora/personal.jsonl",
        min_length=1,
        max_length=1024,
    )
    llamacpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
    minicpmo: MiniCpmoSettings = Field(default_factory=MiniCpmoSettings)
    ragsdk: RagsdkSettings = Field(default_factory=RagsdkSettings)
    map_realtime: MapRealtimePublicSettings = Field(default_factory=MapRealtimePublicSettings)

    @model_validator(mode="after")
    def validate_profile_backend_pair(self) -> AppConfig:
        if self.rag_backend == BackendName.FAKE and self.profile != RuntimeProfile.TEST_FAKE:
            raise ValueError("FakeBackend is allowed only in the test_fake profile")
        if self.profile == RuntimeProfile.TEST_FAKE and self.rag_backend != BackendName.FAKE:
            raise ValueError("test_fake profile must use the fake backend")
        if self.profile in {RuntimeProfile.RAGSDK_LOCAL, RuntimeProfile.RAGSDK_FULL}:
            if self.rag_backend != BackendName.RAGSDK:
                raise ValueError("ragsdk profiles must explicitly select the ragsdk backend")
        if (
            self.profile == RuntimeProfile.RAGSDK_LOCAL
            and self.ragsdk.retrieval_mode != RagsdkRetrievalMode.DENSE
        ):
            raise ValueError("ragsdk_local supports only dense retrieval mode")
        if self.profile == RuntimeProfile.LOCAL_BASELINE and self.rag_backend != BackendName.BASELINE:
            raise ValueError("local_baseline profile must explicitly select the baseline backend")
        return self

    @classmethod
    def from_env(cls) -> AppConfig:
        embedding_url = os.getenv("BAOYAN_RAGSDK_EMBEDDING_URL") or None
        reranker_url = os.getenv("BAOYAN_RAGSDK_RERANKER_URL") or None
        llm_url = os.getenv("BAOYAN_RAGSDK_LLM_URL") or None
        llm_model = os.getenv("BAOYAN_RAGSDK_LLM_MODEL") or None
        minicpmo_hosts = tuple(
            item.strip()
            for item in os.getenv(
                "BAOYAN_MINICPMO_ALLOWED_HOSTS",
                "127.0.0.1,localhost,::1",
            ).split(",")
            if item.strip()
        )
        return cls(
            profile=os.getenv("BAOYAN_PROFILE", RuntimeProfile.LOCAL_BASELINE.value),
            rag_backend=os.getenv("BAOYAN_RAG_BACKEND", BackendName.BASELINE.value),
            llm_backend=os.getenv("BAOYAN_LLM_BACKEND", LlmBackend.NONE.value),
            multimodal_backend=os.getenv(
                "BAOYAN_MULTIMODAL_BACKEND",
                MultimodalBackend.NONE.value,
            ),
            agent_engine=os.getenv("BAOYAN_AGENT_ENGINE", AgentEngine.LANGGRAPH.value),
            public_demo=os.getenv("BAOYAN_PUBLIC_DEMO", "false"),
            top_k=os.getenv("BAOYAN_TOP_K", "5"),
            max_query_chars=os.getenv("BAOYAN_MAX_QUERY_CHARS", "4096"),
            max_expensive_requests=os.getenv("BAOYAN_MAX_EXPENSIVE_REQUESTS", "2"),
            capacity_retry_after_seconds=os.getenv("BAOYAN_CAPACITY_RETRY_AFTER_SECONDS", "1"),
            log_level=os.getenv("BAOYAN_LOG_LEVEL", "INFO").upper(),
            database_path=os.getenv("BAOYAN_DATABASE_PATH", "data/local/app/learning.db"),
            agent_checkpoint_path=os.getenv(
                "BAOYAN_AGENT_CHECKPOINT_PATH",
                "data/local/app/agent_checkpoints.sqlite",
            ),
            corpus_path=os.getenv(
                "BAOYAN_CORPUS_PATH",
                "data/local/private/indexes/baseline/corpus.jsonl",
            ),
            feishu_snapshot_path=os.getenv(
                "BAOYAN_FEISHU_SNAPSHOT_PATH", "data/local/feishu/current"
            ),
            question_bank_path=os.getenv(
                "BAOYAN_QUESTION_BANK_PATH", "data/local/question_bank/questions.jsonl"
            ),
            coverage_path=os.getenv(
                "BAOYAN_COVERAGE_PATH", "data/local/question_bank/coverage.json"
            ),
            generic_corpus_path=os.getenv(
                "BAOYAN_GENERIC_CORPUS_PATH",
                "data/local/question_bank/corpora/generic.jsonl",
            ),
            personal_corpus_path=os.getenv(
                "BAOYAN_PERSONAL_CORPUS_PATH",
                "data/local/question_bank/corpora/personal.jsonl",
            ),
            llamacpp=LlamaCppSettings(
                base_url=os.getenv("BAOYAN_LLAMACPP_BASE_URL", "http://127.0.0.1:11435"),
                model=os.getenv("BAOYAN_LLAMACPP_MODEL", "qwen3-4b-q4"),
                timeout_seconds=os.getenv("BAOYAN_LLAMACPP_TIMEOUT_SECONDS", "180"),
                context_chars=os.getenv("BAOYAN_LLAMACPP_CONTEXT_CHARS", "12000"),
                temperature=os.getenv("BAOYAN_LLAMACPP_TEMPERATURE", "0.2"),
            ),
            minicpmo=MiniCpmoSettings(
                base_url=os.getenv("BAOYAN_MINICPMO_BASE_URL", "http://127.0.0.1:18080"),
                model=os.getenv("BAOYAN_MINICPMO_MODEL", "openbmb/MiniCPM-o-4_5"),
                allowed_hosts=minicpmo_hosts,
                timeout_seconds=os.getenv("BAOYAN_MINICPMO_TIMEOUT_SECONDS", "180"),
                max_upload_bytes=os.getenv(
                    "BAOYAN_MINICPMO_MAX_UPLOAD_BYTES",
                    str(8 * 1024 * 1024),
                ),
                max_attachments=os.getenv("BAOYAN_MINICPMO_MAX_ATTACHMENTS", "4"),
                max_tokens=os.getenv("BAOYAN_MINICPMO_MAX_TOKENS", "512"),
            ),
            ragsdk=RagsdkSettings(
                data_root=os.getenv("BAOYAN_RAGSDK_DATA_ROOT", "data/local/ragsdk"),
                milvus_uri=os.getenv("BAOYAN_RAGSDK_MILVUS_URI", "data/local/ragsdk/milvus.db"),
                embedding_url=embedding_url,
                embedding_dimension=os.getenv("BAOYAN_RAGSDK_EMBEDDING_DIMENSION", "1024"),
                retrieval_mode=os.getenv(
                    "BAOYAN_RAGSDK_RETRIEVAL_MODE",
                    RagsdkRetrievalMode.DENSE.value,
                ),
                reranker_url=reranker_url,
                dense_candidate_k=os.getenv("BAOYAN_RAGSDK_DENSE_CANDIDATE_K", "20"),
                sparse_candidate_k=os.getenv("BAOYAN_RAGSDK_SPARSE_CANDIDATE_K", "20"),
                fusion_candidate_k=os.getenv("BAOYAN_RAGSDK_FUSION_CANDIDATE_K", "20"),
                fusion_baseline=os.getenv("BAOYAN_RAGSDK_FUSION_BASELINE", "0.4"),
                fusion_amplitude=os.getenv("BAOYAN_RAGSDK_FUSION_AMPLITUDE", "0.3"),
                fusion_slope=os.getenv("BAOYAN_RAGSDK_FUSION_SLOPE", "1.0"),
                fusion_midpoint=os.getenv("BAOYAN_RAGSDK_FUSION_MIDPOINT", "6.0"),
                knowledge_name=os.getenv("BAOYAN_RAGSDK_KNOWLEDGE_NAME", "baoyan_private"),
                user_id=os.getenv("BAOYAN_RAGSDK_USER_ID", "local_user"),
                vector_collection=os.getenv("BAOYAN_RAGSDK_VECTOR_COLLECTION", "baoyan_vectors"),
                chunk_collection=os.getenv("BAOYAN_RAGSDK_CHUNK_COLLECTION", "baoyan_chunks"),
                score_threshold=os.getenv("BAOYAN_RAGSDK_SCORE_THRESHOLD") or None,
                request_timeout_seconds=os.getenv("BAOYAN_RAGSDK_TIMEOUT_SECONDS", "60"),
                llm_url=llm_url,
                llm_model=llm_model,
            ),
            map_realtime=MapRealtimePublicSettings(
                proxy_url=os.getenv("GRADLOOP_MAP_PROXY_URL") or None,
                enabled=os.getenv("GRADLOOP_MAP_REALTIME_ENABLED", "false"),
                session_seconds=os.getenv("GRADLOOP_MAP_SESSION_SECONDS", "300"),
            ),
        )
