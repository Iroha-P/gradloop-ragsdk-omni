from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import RagsdkRetrievalMode, RagsdkSettings
from app.core.errors import BackendUnavailableError
from app.evaluation.retrieval import RetrievalBenchmarkCase, evaluate_retrieval
from app.rag.base import DocumentInput, RagBackend, RetrievalRequest, RetrievalResult
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.rag.ragsdk_backend import RagsdkBackend
from app.rag.ragsdk_runtime import (
    NativeRagsdkRuntime,
    RagsdkRuntime,
    RuntimeProbe,
    probe_ragsdk_runtime,
)

PUBLIC_MATRIX_REPORT_PATH = Path(
    "reports/public/ragsdk_retrieval_matrix_status_v1.json"
)

VariantName = Literal[
    "bm25",
    "ragsdk_dense",
    "ragsdk_sparse",
    "ragsdk_hybrid",
    "ragsdk_hybrid_rerank",
]
EXPECTED_VARIANTS = {
    "bm25",
    "ragsdk_dense",
    "ragsdk_sparse",
    "ragsdk_hybrid",
    "ragsdk_hybrid_rerank",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatrixVariantStatus(StrEnum):
    COMPLETED = "completed"
    NOT_RUN = "not_run"
    FAILED = "failed"


class RetrievalAggregateMetrics(StrictModel):
    query_count: int = Field(ge=1)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    no_answer_accuracy: float | None = Field(default=None, ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)


class RetrievalVariantReport(StrictModel):
    variant: VariantName
    status: MatrixVariantStatus
    backend: Literal["baseline", "ragsdk"]
    mode: RagsdkRetrievalMode | None = None
    runtime_verified: bool | None = None
    metrics: RetrievalAggregateMetrics | None = None
    stage_average_candidates: dict[str, float] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    failure_type: str | None = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$"
    )

    @field_validator("missing")
    @classmethod
    def validate_missing_names(cls, values: list[str]) -> list[str]:
        cleaned = sorted(set(values))
        if any(
            not value
            or len(value) > 128
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
            for value in cleaned
        ):
            raise ValueError("missing dependencies must use safe lowercase names")
        return cleaned

    @model_validator(mode="after")
    def validate_status_payload(self) -> RetrievalVariantReport:
        if self.variant == "bm25":
            if self.backend != "baseline" or self.mode is not None:
                raise ValueError("bm25 must use baseline without a RAGSDK mode")
        elif self.backend != "ragsdk" or self.mode is None:
            raise ValueError("RAGSDK variants require backend=ragsdk and an explicit mode")
        if self.status == MatrixVariantStatus.COMPLETED:
            if self.metrics is None:
                raise ValueError("completed variants require metrics")
            if self.variant != "bm25" and self.runtime_verified is not True:
                raise ValueError("completed RAGSDK variants require runtime_verified=true")
            if self.missing or self.failure_type:
                raise ValueError("completed variants cannot contain failure state")
        else:
            if self.metrics is not None or self.stage_average_candidates:
                raise ValueError("not_run/failed variants must not contain metrics")
            if self.runtime_verified is True:
                raise ValueError("incomplete variants cannot be runtime verified")
        if self.status == MatrixVariantStatus.NOT_RUN and not self.missing:
            raise ValueError("not_run variants require at least one missing dependency")
        if self.status == MatrixVariantStatus.FAILED and not self.failure_type:
            raise ValueError("failed variants require a safe failure_type")
        return self


class RagsdkMatrixReport(StrictModel):
    report_version: Literal["ragsdk-retrieval-matrix-status-v1"]
    benchmark_version: Literal["public-synthetic-v1"]
    top_k: int = Field(ge=1, le=100)
    variants: list[RetrievalVariantReport]
    runtime_claim: Literal[
        "ragsdk_not_run_on_current_machine",
        "ragsdk_runtime_partially_verified",
        "ragsdk_runtime_verified",
        "ragsdk_run_incomplete",
    ]
    privacy: dict[str, bool | str]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_variant_matrix(self) -> RagsdkMatrixReport:
        names = [row.variant for row in self.variants]
        if len(names) != len(EXPECTED_VARIANTS) or set(names) != EXPECTED_VARIANTS:
            raise ValueError("matrix requires exactly one report for every expected variant")
        return self


class _TraceCapturingBackend:
    def __init__(self, backend: RagBackend):
        self.backend = backend
        self.results: list[RetrievalResult] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        result = self.backend.retrieve(request)
        self.results.append(result)
        return result

    def backend_info(self):
        return self.backend.backend_info()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _stage_averages(results: list[RetrievalResult]) -> dict[str, float]:
    candidates: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for stage in result.retrieval.stages:
            name = str(stage.get("name", "")).strip()
            candidate_count = stage.get("candidate_count")
            if name and isinstance(candidate_count, int | float) and candidate_count >= 0:
                candidates[name].append(float(candidate_count))
    return {
        name: round(mean(values), 4)
        for name, values in sorted(candidates.items())
    }


def build_completed_variant(
    *,
    variant: VariantName,
    backend: RagBackend,
    cases: list[RetrievalBenchmarkCase],
    top_k: int,
    runtime_verified: bool | None,
    mode: RagsdkRetrievalMode | None = None,
) -> RetrievalVariantReport:
    capture = _TraceCapturingBackend(backend)
    report = evaluate_retrieval(capture, cases, top_k=top_k)
    latencies = [row.elapsed_ms for row in report.cases]
    metrics = RetrievalAggregateMetrics(
        query_count=report.query_count,
        answerable_count=report.answerable_count,
        unanswerable_count=report.unanswerable_count,
        recall_at_k=report.recall_at_k,
        mrr=report.mrr,
        ndcg_at_k=report.ndcg_at_k,
        no_answer_accuracy=report.no_answer_accuracy,
        latency_p50_ms=round(median(latencies), 6),
        latency_p95_ms=round(_p95(latencies), 6),
    )
    verified = (
        backend.backend_info().runtime_verified
        if variant != "bm25"
        else runtime_verified
    )
    return RetrievalVariantReport(
        variant=variant,
        status=MatrixVariantStatus.COMPLETED,
        backend="baseline" if variant == "bm25" else "ragsdk",
        mode=mode,
        runtime_verified=verified,
        metrics=metrics,
        stage_average_candidates=_stage_averages(capture.results),
    )


def build_not_run_variant(
    *,
    variant: VariantName,
    mode: RagsdkRetrievalMode,
    missing: list[str],
) -> RetrievalVariantReport:
    return RetrievalVariantReport(
        variant=variant,
        status=MatrixVariantStatus.NOT_RUN,
        backend="ragsdk",
        mode=mode,
        runtime_verified=False,
        missing=missing,
    )


def build_failed_variant(
    *,
    variant: VariantName,
    mode: RagsdkRetrievalMode,
    failure_type: str,
) -> RetrievalVariantReport:
    return RetrievalVariantReport(
        variant=variant,
        status=MatrixVariantStatus.FAILED,
        backend="ragsdk",
        mode=mode,
        runtime_verified=False,
        failure_type=failure_type,
    )


def build_matrix_report(
    variants: list[RetrievalVariantReport], *, top_k: int
) -> RagsdkMatrixReport:
    names = [row.variant for row in variants]
    if len(names) != len(EXPECTED_VARIANTS) or set(names) != EXPECTED_VARIANTS:
        raise ValueError("matrix requires exactly one report for every expected variant")
    ragsdk_rows = [row for row in variants if row.variant != "bm25"]
    completed = sum(row.status == MatrixVariantStatus.COMPLETED for row in ragsdk_rows)
    if completed == len(ragsdk_rows):
        claim = "ragsdk_runtime_verified"
    elif completed:
        claim = "ragsdk_runtime_partially_verified"
    elif all(row.status == MatrixVariantStatus.NOT_RUN for row in ragsdk_rows):
        claim = "ragsdk_not_run_on_current_machine"
    else:
        claim = "ragsdk_run_incomplete"
    ordered = sorted(
        variants,
        key=lambda row: (
            "bm25",
            "ragsdk_dense",
            "ragsdk_sparse",
            "ragsdk_hybrid",
            "ragsdk_hybrid_rerank",
        ).index(row.variant),
    )
    return RagsdkMatrixReport(
        report_version="ragsdk-retrieval-matrix-status-v1",
        benchmark_version="public-synthetic-v1",
        top_k=top_k,
        variants=ordered,
        runtime_claim=claim,
        privacy={
            "public_synthetic_data_only": True,
            "service_configuration_public": False,
            "absolute_paths_public": False,
        },
        limitations=[
            "Synthetic retrieval sanity set; not a real-user quality claim.",
            "A RAGSDK variant has metrics only after real runtime index and query completion.",
            "Indexing latency and external service resource usage are not included.",
        ],
    )


def _isolated_settings(
    settings: RagsdkSettings, mode: RagsdkRetrievalMode
) -> RagsdkSettings:
    suffix = f"eval_{mode.value}"
    return settings.model_copy(
        update={
            "retrieval_mode": mode,
            "knowledge_name": f"{settings.knowledge_name[:44]}_{suffix}"[:64],
            "vector_collection": f"{settings.vector_collection[:44]}_{suffix}"[:64],
            "chunk_collection": f"{settings.chunk_collection[:44]}_{suffix}"[:64],
        }
    )


def _safe_missing(values: object) -> list[str]:
    if not isinstance(values, list):
        return ["runtime_prerequisites"]
    safe = sorted(
        {
            value
            for value in values
            if isinstance(value, str)
            and value
            and len(value) <= 128
            and all(
                character in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in value
            )
        }
    )
    return safe or ["runtime_prerequisites"]


def run_ragsdk_matrix(
    *,
    documents: list[DocumentInput],
    cases: list[RetrievalBenchmarkCase],
    settings: RagsdkSettings,
    top_k: int = 5,
    runtime_factory: Callable[[RagsdkSettings], RagsdkRuntime] = (
        NativeRagsdkRuntime.create
    ),
    probe_func: Callable[[RagsdkSettings], RuntimeProbe] = probe_ragsdk_runtime,
) -> RagsdkMatrixReport:
    variants = [
        build_completed_variant(
            variant="bm25",
            backend=Bm25BaselineBackend(documents),
            cases=cases,
            top_k=top_k,
            runtime_verified=None,
        )
    ]
    variant_by_mode: dict[RagsdkRetrievalMode, VariantName] = {
        RagsdkRetrievalMode.DENSE: "ragsdk_dense",
        RagsdkRetrievalMode.SPARSE: "ragsdk_sparse",
        RagsdkRetrievalMode.HYBRID: "ragsdk_hybrid",
        RagsdkRetrievalMode.HYBRID_RERANK: "ragsdk_hybrid_rerank",
    }
    for mode in RagsdkRetrievalMode:
        variant = variant_by_mode[mode]
        mode_settings = _isolated_settings(settings, mode)
        probe = probe_func(mode_settings)
        if not probe.ready:
            variants.append(
                build_not_run_variant(
                    variant=variant,
                    mode=mode,
                    missing=_safe_missing(probe.missing),
                )
            )
            continue
        try:
            runtime = runtime_factory(mode_settings)
        except BackendUnavailableError as exc:
            variants.append(
                build_not_run_variant(
                    variant=variant,
                    mode=mode,
                    missing=_safe_missing(exc.details.get("missing")),
                )
            )
            continue
        except Exception as exc:
            variants.append(
                build_failed_variant(
                    variant=variant,
                    mode=mode,
                    failure_type=type(exc).__name__,
                )
            )
            continue
        try:
            backend = RagsdkBackend(runtime)
            indexed = backend.index_documents(documents)
            if indexed.failed_ids or len(indexed.indexed_ids) != len(documents):
                raise RuntimeError("RAGSDK public evaluation index was incomplete")
            health = backend.healthcheck()
            if not health.ready:
                raise RuntimeError("RAGSDK runtime healthcheck failed after initialization")
            variants.append(
                build_completed_variant(
                    variant=variant,
                    backend=backend,
                    cases=cases,
                    top_k=top_k,
                    runtime_verified=None,
                    mode=mode,
                )
            )
        except Exception as exc:
            variants.append(
                build_failed_variant(
                    variant=variant,
                    mode=mode,
                    failure_type=type(exc).__name__,
                )
            )
    return build_matrix_report(variants, top_k=top_k)


def write_public_matrix_report(
    report: RagsdkMatrixReport,
    path: Path = PUBLIC_MATRIX_REPORT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
