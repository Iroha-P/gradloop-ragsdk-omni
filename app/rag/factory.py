from __future__ import annotations

from app.core.config import AppConfig, BackendName, RuntimeProfile
from app.core.errors import ConfigurationError

from .base import DocumentInput, RagBackend
from .bm25_baseline import Bm25BaselineBackend
from .fake_backend import FakeBackend
from .ragsdk_backend import RagsdkBackend
from .ragsdk_runtime import NativeRagsdkRuntime, RagsdkRuntime


def build_backend(
    config: AppConfig,
    documents: list[DocumentInput] | None = None,
    *,
    ragsdk_runtime: RagsdkRuntime | None = None,
) -> RagBackend:
    """Create exactly the configured backend; never perform an implicit fallback."""

    if config.rag_backend == BackendName.BASELINE:
        return Bm25BaselineBackend(documents)

    if config.rag_backend == BackendName.FAKE:
        if config.profile != RuntimeProfile.TEST_FAKE:
            raise ConfigurationError("FakeBackend cannot be enabled outside test_fake")
        return FakeBackend(documents)

    if config.rag_backend == BackendName.RAGSDK:
        runtime = ragsdk_runtime or NativeRagsdkRuntime.create(
            config.ragsdk,
            require_llm=config.profile == RuntimeProfile.RAGSDK_FULL,
            require_full=config.profile == RuntimeProfile.RAGSDK_FULL,
        )
        return RagsdkBackend(runtime)

    raise ConfigurationError(f"Unsupported backend: {config.rag_backend}")
