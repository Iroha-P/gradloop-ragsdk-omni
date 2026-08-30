from __future__ import annotations

import hashlib
import importlib.util
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.config import RagsdkRetrievalMode, RagsdkSettings
from app.core.errors import BackendUnavailableError, ConfigurationError

from .base import DocumentInput


@dataclass(frozen=True)
class RagsdkHit:
    source_id: str
    chunk_id: str
    text: str
    score: float
    retrieval_type: str
    title: str = ""
    section: str = ""
    page: int | None = None
    published_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagsdkGeneratedAnswer:
    text: str
    hits: list[RagsdkHit]


@dataclass(frozen=True)
class RagsdkStage:
    name: str
    candidate_count: int
    selected_count: int
    elapsed_ms: float


@dataclass(frozen=True)
class RagsdkRetrievalResponse:
    hits: list[RagsdkHit]
    stages: list[RagsdkStage]
    mode: RagsdkRetrievalMode


@runtime_checkable
class RagsdkRuntime(Protocol):
    version: str
    capabilities: tuple[str, ...]
    runtime_verified: bool

    def index_documents(
        self, documents: list[DocumentInput]
    ) -> tuple[list[str], list[str], list[str]]: ...

    def delete_documents(
        self, source_ids: list[str]
    ) -> tuple[list[str], list[str], list[str]]: ...

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, Any]
    ) -> RagsdkRetrievalResponse: ...

    def generate(self, query: str, top_k: int) -> RagsdkGeneratedAnswer | None: ...

    def healthcheck(self) -> tuple[dict[str, bool], list[str], str]: ...


@dataclass(frozen=True)
class RuntimeProbe:
    checks: dict[str, bool]
    missing: list[str]

    @property
    def ready(self) -> bool:
        return not self.missing


def _safe_local_root(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def probe_ragsdk_runtime(
    settings: RagsdkSettings,
    *,
    require_llm: bool = False,
    require_full: bool = False,
) -> RuntimeProbe:
    checks = {
        "linux": platform.system() == "Linux",
        "python_3_11": sys.version_info[:2] == (3, 11),
        "mx_rag_installed": importlib.util.find_spec("mx_rag") is not None,
        "pymilvus_installed": importlib.util.find_spec("pymilvus") is not None,
        "embedding_configured": bool(settings.embedding_url),
        "data_root_safe": _safe_local_root(settings.data_root),
        "llm_configured": bool(settings.llm_url and settings.llm_model) if require_llm else True,
        "reranker_configured": (
            bool(settings.reranker_url)
            if settings.retrieval_mode == RagsdkRetrievalMode.HYBRID_RERANK
            else True
        ),
    }
    missing = sorted(name for name, passed in checks.items() if not passed)
    return RuntimeProbe(checks=checks, missing=missing)


def _safe_metadata(document: DocumentInput) -> dict[str, Any]:
    return {
        **document.metadata,
        "source_id": document.source_id,
        "chunk_id": document.chunk_id,
        "title": document.title,
        "section": document.section,
        "page": document.page,
        "published_at": document.published_at.isoformat() if document.published_at else None,
        "authority_level": document.authority_level,
    }


class NativeRagsdkRuntime:
    """Thin adapter around installed RAGSDK 26.x components.

    Imports live inside ``create`` so unsupported developer machines can still run all contract tests.
    """

    def __init__(
        self,
        *,
        settings: RagsdkSettings,
        knowledge: Any,
        dense_retriever: Any | None,
        sparse_retriever: Any | None = None,
        fusion_reranker: Any | None = None,
        final_reranker: Any | None = None,
        dense_embed: Any,
        chain: Any | None,
        version: str,
        staging_root: Path,
    ):
        self.settings = settings
        self.knowledge = knowledge
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_reranker = fusion_reranker
        self.final_reranker = final_reranker
        self.retriever = dense_retriever
        self.dense_embed = dense_embed
        self.chain = chain
        self.version = version
        self.staging_root = staging_root
        self._index_verified = False
        self._retrieve_verified = False
        mode_capability = f"{settings.retrieval_mode.value}_retrieve"
        capabilities = [
            "index",
            "delete_source",
            mode_capability,
        ]
        if settings.retrieval_mode == RagsdkRetrievalMode.DENSE:
            capabilities.append("optional_generate")
        self.capabilities = tuple(capabilities)

    @property
    def runtime_verified(self) -> bool:
        return self._index_verified and self._retrieve_verified

    @classmethod
    def create(
        cls,
        settings: RagsdkSettings,
        *,
        require_llm: bool = False,
        require_full: bool = False,
    ) -> NativeRagsdkRuntime:
        probe = probe_ragsdk_runtime(
            settings,
            require_llm=require_llm,
            require_full=require_full,
        )
        if not probe.ready:
            raise BackendUnavailableError(
                "RAGSDK runtime prerequisites are not satisfied",
                details={
                    "selected_backend": "ragsdk",
                    "checks": probe.checks,
                    "missing": probe.missing,
                    "fallback": "disabled",
                },
            )

        try:
            from importlib.metadata import PackageNotFoundError
            from importlib.metadata import version as package_version

            from mx_rag.chain import SingleText2TextChain
            from mx_rag.embedding.service import TEIEmbedding
            from mx_rag.knowledge import KnowledgeDB
            from mx_rag.knowledge.knowledge import KnowledgeStore
            from mx_rag.llm import Text2TextLLM
            from mx_rag.reranker.local import MixRetrieveReranker
            from mx_rag.reranker.reranker_factory import RerankerFactory
            from mx_rag.retrievers import FullTextRetriever, Retriever
            from mx_rag.storage.document_store import MilvusDocstore
            from mx_rag.storage.vectorstore import MilvusDB
            from mx_rag.utils import ClientParam
            from pymilvus import MilvusClient

            project_root = Path.cwd().resolve()
            data_root = (project_root / settings.data_root).resolve()
            if project_root not in data_root.parents:
                raise ConfigurationError("RAGSDK data root must stay inside the project directory")
            data_root.mkdir(parents=True, exist_ok=True)
            staging_root = data_root / "staging"
            staging_root.mkdir(parents=True, exist_ok=True)

            milvus_uri = settings.milvus_uri
            if "://" not in milvus_uri:
                milvus_path = (project_root / milvus_uri).resolve()
                if project_root not in milvus_path.parents:
                    raise ConfigurationError("Local Milvus path must stay inside the project directory")
                milvus_path.parent.mkdir(parents=True, exist_ok=True)
                milvus_uri = str(milvus_path)

            client_param = ClientParam(use_http=True, timeout=settings.request_timeout_seconds)
            embedding = TEIEmbedding(url=settings.embedding_url, client_param=client_param)
            milvus_client = MilvusClient(milvus_uri)
            vector_store = MilvusDB.create(
                client=milvus_client,
                x_dim=settings.embedding_dimension,
                collection_name=settings.vector_collection,
            )
            if vector_store is None:
                raise RuntimeError("RAGSDK failed to create the vector store")
            chunk_store = MilvusDocstore(
                milvus_client,
                collection_name=settings.chunk_collection,
                enable_bm25=True,
            )
            knowledge_store = KnowledgeStore(db_path=str(data_root / "knowledge.db"))
            knowledge_store.add_knowledge(settings.knowledge_name, settings.user_id, "admin")
            knowledge = KnowledgeDB(
                knowledge_store=knowledge_store,
                chunk_store=chunk_store,
                vector_store=vector_store,
                knowledge_name=settings.knowledge_name,
                white_paths=[str(staging_root)],
                user_id=settings.user_id,
            )
            dense_retriever = Retriever(
                vector_store=vector_store,
                document_store=chunk_store,
                embed_func=embedding.embed_documents,
                k=5,
                score_threshold=settings.score_threshold,
            )
            sparse_retriever = None
            fusion_reranker = None
            final_reranker = None
            if settings.retrieval_mode != RagsdkRetrievalMode.DENSE:
                sparse_retriever = FullTextRetriever(
                    document_store=chunk_store,
                    k=settings.sparse_candidate_k,
                )
            if settings.retrieval_mode in {
                RagsdkRetrievalMode.HYBRID,
                RagsdkRetrievalMode.HYBRID_RERANK,
            }:
                fusion_reranker = MixRetrieveReranker(
                    k=settings.fusion_candidate_k,
                    baseline=settings.fusion_baseline,
                    amplitude=settings.fusion_amplitude,
                    slope=settings.fusion_slope,
                    midpoint=settings.fusion_midpoint,
                )
            if settings.retrieval_mode == RagsdkRetrievalMode.HYBRID_RERANK:
                final_reranker = RerankerFactory.create_reranker(
                    similarity_type="tei_reranker",
                    url=settings.reranker_url,
                    client_param=client_param,
                    k=settings.fusion_candidate_k,
                )
                if final_reranker is None:
                    raise RuntimeError("RAGSDK failed to create the TEI reranker")
            chain = None
            if (
                settings.retrieval_mode == RagsdkRetrievalMode.DENSE
                and settings.llm_url
                and settings.llm_model
            ):
                llm = Text2TextLLM(
                    base_url=settings.llm_url,
                    model_name=settings.llm_model,
                    client_param=client_param,
                )
                chain = SingleText2TextChain(retriever=dense_retriever, llm=llm)
            try:
                installed_version = package_version("mx-rag")
            except PackageNotFoundError:
                from mx_rag.version import __version__ as installed_version
        except (BackendUnavailableError, ConfigurationError):
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                "RAGSDK components could not be initialized",
                details={
                    "selected_backend": "ragsdk",
                    "failure_type": type(exc).__name__,
                    "fallback": "disabled",
                },
            ) from exc

        return cls(
            settings=settings,
            knowledge=knowledge,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion_reranker=fusion_reranker,
            final_reranker=final_reranker,
            dense_embed=embedding.embed_documents,
            chain=chain,
            version=installed_version,
            staging_root=staging_root,
        )

    @staticmethod
    def _document_name(source_id: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in source_id
        )
        if not safe:
            raise ValueError("source_id cannot produce an empty RAGSDK document name")
        fingerprint = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
        return f"{safe[:96]}_{fingerprint}.md"

    def index_documents(
        self, documents: list[DocumentInput]
    ) -> tuple[list[str], list[str], list[str]]:
        grouped: dict[str, list[DocumentInput]] = defaultdict(list)
        for document in documents:
            grouped[document.source_id].append(document)

        indexed: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        for source_id, source_documents in sorted(grouped.items()):
            document_name = self._document_name(source_id)
            staging_path = self.staging_root / document_name
            staging_path.write_text("# privacy-gated staged document\n", encoding="utf-8")
            try:
                if self.knowledge.check_document_exist(document_name):
                    self.knowledge.delete_file(document_name)
                self.knowledge.add_file(
                    staging_path,
                    [item.text for item in source_documents],
                    {"dense": self.dense_embed, "sparse": None},
                    [_safe_metadata(item) for item in source_documents],
                )
                indexed.extend(item.chunk_id for item in source_documents)
                self._index_verified = True
            except Exception as exc:
                failed.extend(item.chunk_id for item in source_documents)
                warnings.append(f"source {source_id} failed with {type(exc).__name__}")
        return indexed, failed, warnings

    def delete_documents(
        self, source_ids: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        deleted: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []
        for source_id in dict.fromkeys(source_ids):
            document_name = self._document_name(source_id)
            try:
                if not self.knowledge.check_document_exist(document_name):
                    missing.append(source_id)
                    continue
                self.knowledge.delete_file(document_name)
                deleted.append(source_id)
            except Exception as exc:
                warnings.append(f"source {source_id} delete failed with {type(exc).__name__}")
        return deleted, missing, warnings

    @staticmethod
    def _convert_document(document: Any) -> RagsdkHit:
        if isinstance(document, dict):
            metadata = dict(document.get("metadata", {}) or {})
            page_content = document.get("page_content", "")
        else:
            metadata = dict(getattr(document, "metadata", {}) or {})
            page_content = getattr(document, "page_content", "")
        return RagsdkHit(
            source_id=str(metadata.pop("source_id", "unknown")),
            chunk_id=str(metadata.pop("chunk_id", "unknown")),
            title=str(metadata.pop("title", "")),
            section=str(metadata.pop("section", "")),
            page=metadata.pop("page", None),
            published_at=metadata.pop("published_at", None),
            text=str(page_content),
            score=float(metadata.pop("score", 0.0)),
            retrieval_type=str(metadata.pop("retrieval_type", "dense")),
            metadata=metadata,
        )

    @staticmethod
    def _set_retriever_filter(retriever: Any, filters: dict[str, Any]) -> None:
        if filters:
            retriever.set_filter(filters)
        elif hasattr(retriever, "filter_dict"):
            retriever.filter_dict = {}

    def _invoke(
        self,
        *,
        retriever: Any | None,
        query: str,
        candidate_k: int,
        filters: dict[str, Any],
        name: str,
    ) -> tuple[list[Any], RagsdkStage]:
        if retriever is None:
            raise RuntimeError(f"RAGSDK {name} component is not configured")
        retriever.k = candidate_k
        self._set_retriever_filter(retriever, filters)
        started = time.perf_counter()
        documents = list(retriever.invoke(query))
        return documents, RagsdkStage(
            name=name,
            candidate_count=len(documents),
            selected_count=len(documents),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, Any]
    ) -> RagsdkRetrievalResponse:
        invalid_filters = set(filters) - {"document_id"}
        if invalid_filters:
            raise ValueError("RAGSDK supports only the document_id filter")
        mode = self.settings.retrieval_mode
        stages: list[RagsdkStage] = []
        if mode == RagsdkRetrievalMode.DENSE:
            documents, stage = self._invoke(
                retriever=self.dense_retriever,
                query=query,
                candidate_k=top_k,
                filters=filters,
                name="dense",
            )
            stages.append(stage)
        elif mode == RagsdkRetrievalMode.SPARSE:
            documents, stage = self._invoke(
                retriever=self.sparse_retriever,
                query=query,
                candidate_k=top_k,
                filters=filters,
                name="sparse",
            )
            stages.append(stage)
        else:
            dense_documents, dense_stage = self._invoke(
                retriever=self.dense_retriever,
                query=query,
                candidate_k=self.settings.dense_candidate_k,
                filters=filters,
                name="dense",
            )
            sparse_documents, sparse_stage = self._invoke(
                retriever=self.sparse_retriever,
                query=query,
                candidate_k=self.settings.sparse_candidate_k,
                filters=filters,
                name="sparse",
            )
            stages.extend((dense_stage, sparse_stage))
            if self.fusion_reranker is None:
                raise RuntimeError("RAGSDK fusion component is not configured")
            combined = [*dense_documents, *sparse_documents]
            started = time.perf_counter()
            documents = list(self.fusion_reranker.rerank(query, combined))
            stages.append(
                RagsdkStage(
                    name="fusion",
                    candidate_count=len(combined),
                    selected_count=len(documents),
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            )
            if mode == RagsdkRetrievalMode.HYBRID_RERANK:
                if self.final_reranker is None:
                    raise RuntimeError("RAGSDK final reranker component is not configured")
                started = time.perf_counter()
                scores = list(
                    self.final_reranker.rerank(
                        query,
                        [str(document.page_content) for document in documents],
                    )
                )
                if not scores or len(scores) != len(documents):
                    raise RuntimeError(
                        "RAGSDK reranker score count must equal the candidate count"
                    )
                ranked = sorted(
                    zip(documents, scores, strict=True),
                    key=lambda pair: float(pair[1]),
                    reverse=True,
                )
                documents = []
                for document, score in ranked[:top_k]:
                    document.metadata["rerank_score"] = float(score)
                    documents.append(document)
                stages.append(
                    RagsdkStage(
                        name="rerank",
                        candidate_count=len(scores),
                        selected_count=len(documents),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                )
        documents = documents[:top_k]
        self._retrieve_verified = True
        return RagsdkRetrievalResponse(
            hits=[self._convert_document(document) for document in documents],
            stages=stages,
            mode=mode,
        )

    def generate(self, query: str, top_k: int) -> RagsdkGeneratedAnswer | None:
        if self.settings.retrieval_mode != RagsdkRetrievalMode.DENSE:
            return None
        if self.chain is None:
            return None
        if self.dense_retriever is None:
            raise RuntimeError("RAGSDK dense retriever is required for generation")
        self.dense_retriever.k = top_k
        result = self.chain.query(query)
        hits = [self._convert_document(item) for item in result.get("source_documents", [])]
        return RagsdkGeneratedAnswer(text=str(result.get("result", "")), hits=hits)

    def healthcheck(self) -> tuple[dict[str, bool], list[str], str]:
        checks = {
            "mx_rag": True,
            "knowledge_store": False,
            "dense_retriever": self.dense_retriever is not None,
            "sparse_retriever": (
                self.sparse_retriever is not None
                if self.settings.retrieval_mode != RagsdkRetrievalMode.DENSE
                else True
            ),
            "fusion": (
                self.fusion_reranker is not None
                if self.settings.retrieval_mode
                in {RagsdkRetrievalMode.HYBRID, RagsdkRetrievalMode.HYBRID_RERANK}
                else True
            ),
            "reranker": (
                self.final_reranker is not None
                if self.settings.retrieval_mode == RagsdkRetrievalMode.HYBRID_RERANK
                else True
            ),
            "embedding": self.dense_embed is not None,
            "llm": self.chain is not None,
        }
        missing: list[str] = []
        try:
            self.knowledge.get_all_documents()
            checks["knowledge_store"] = True
        except Exception:
            missing.append("knowledge_store")
        for component in (
            "dense_retriever",
            "sparse_retriever",
            "fusion",
            "reranker",
        ):
            if not checks[component]:
                missing.append(component)
        if not checks["embedding"]:
            missing.append("embedding")
        return checks, missing, "RAGSDK runtime ready" if not missing else "RAGSDK runtime degraded"
