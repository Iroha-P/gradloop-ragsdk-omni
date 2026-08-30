from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.answer_quality import evaluate_answer_quality
from app.evaluation.grading import (
    GradingBenchmarkCase,
    load_grading_cases,
    score_metrics,
)
from app.evaluation.retrieval import RetrievalBenchmarkCase, evaluate_retrieval, load_benchmark_cases
from app.rag.base import AnswerResult, Citation, DocumentInput, RetrievalTrace
from app.rag.bm25_baseline import Bm25BaselineBackend
from app.rag.corpus_file import load_document_corpus

PORTFOLIO_DATASET_VERSION = "public-synthetic-portfolio-v1"
PORTFOLIO_PROTOCOL_VERSION = "public-portfolio-deterministic-v2"
_SCHEMA_VERSION = "public-portfolio-schema-v2"
_DUPLICATE_AUDIT_VERSION = "normalized-trigram-template-audit-v1"
_DUPLICATE_SIMILARITY_THRESHOLD = 0.72
_DUPLICATE_SHINGLE_SIZE = 3
_REQUIRED_DIMENSIONS = frozenset(
    {"retrieval", "citation", "refusal", "conflict", "temporal", "agent_tool", "plan", "grading"}
)
_PRIVATE_PATH = re.compile(r"(?:[a-z]:\\|/home/|/users/|\\users\\)", re.IGNORECASE)
_CREDENTIAL = re.compile(
    r"(?:feishu[_-]?(?:token|app[_-]?secret)|access[_-]?token|api[_-]?key|password)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_TEMPLATE_NUMBER = re.compile(r"(?:19|20)\d{2}|\d+(?:\.\d+)?")
_TEMPLATE_ID = re.compile(
    r"(?:case|chunk|source|document|portfolio|public)[_-]?[a-z0-9_-]+",
    re.IGNORECASE,
)
_TEMPLATE_STEP = re.compile(r"步骤[一二三四五六七八九十甲乙丙丁戊己庚辛壬癸0-9]*")
_TEMPLATE_PLACEHOLDER = re.compile(r"(?:合成)?主题|唯一规则|不存在的|公开合成")
_TEMPLATE_NON_TEXT = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PortfolioManifest(StrictModel):
    dataset_version: str
    protocol_version: str
    schema_version: str
    asset_case_counts: dict[str, int]
    expected_unique_case_count: int = Field(ge=100)
    source_policy: Literal["public_synthetic_only"]


class PortfolioAnswerFixture(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_]{6,64}$")
    dimension: Literal["citation", "refusal", "conflict", "temporal"]
    question: str = Field(min_length=1, max_length=4096)
    answerable: bool
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1, max_length=4096)
    abstained: bool = False


class PortfolioAgentCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_]{6,64}$")
    category: Literal[
        "routing", "retrieval_retry", "human_confirmation", "recovery", "privacy"
    ]
    intent: Literal["ask", "plan", "practice", "grade", "mistakes", "revision"]
    scope: Literal["generic", "personal", "all"] = "generic"
    ask_mode: Literal["sufficient", "insufficient_then_sufficient", "insufficient"]
    grade_revision: bool
    confirm: bool | None
    fail_node: str | None
    expected_tools: list[str]
    expected_status: Literal["completed", "waiting_for_confirmation"]


class PrivacyScanResult(StrictModel):
    scan_version: str = "public-asset-privacy-v1"
    passed: bool
    violation_count: int = Field(ge=0)


class PortfolioGroupMetrics(StrictModel):
    case_count: int = Field(ge=1)
    status: Literal["completed"] = "completed"
    metrics_kind: Literal["system_measurement", "fixture_consistency"]
    metrics: dict[str, float | int | None]


class PortfolioLatency(StrictModel):
    total_ms: float = Field(ge=0.0)
    retrieval_average_ms: float = Field(ge=0.0)
    answer_fixture_observation_count: int = Field(ge=1)
    answer_fixture_all_calls_p50_ms: float = Field(ge=0.0)
    answer_fixture_all_calls_p95_ms: float = Field(ge=0.0)


class SemanticDuplicateAudit(StrictModel):
    audit_version: str = _DUPLICATE_AUDIT_VERSION
    similarity_threshold: float = Field(ge=0.0, le=0.8)
    shingle_size: int = Field(ge=2, le=5)
    audited_text_count: int = Field(ge=1)
    clone_pair_count: int = Field(ge=0)
    max_similarity: float = Field(ge=0.0, le=1.0)
    passed: bool


class PublicPortfolioReport(StrictModel):
    report_version: str = "public-portfolio-report-v2"
    dataset_version: str
    protocol_version: str
    status: Literal["completed", "failed"]
    case_count: int = Field(ge=0)
    unique_case_count: int = Field(ge=0)
    group_metrics: dict[str, PortfolioGroupMetrics] = Field(default_factory=dict)
    latency_ms: PortfolioLatency | None = None
    duplicate_audit: SemanticDuplicateAudit | None = None
    privacy_scan: PrivacyScanResult
    honest_boundaries: list[str]
    failure_type: Literal["invalid_public_assets", "privacy_scan_failed"] | None = None

    @model_validator(mode="after")
    def validate_unique_count(self) -> PublicPortfolioReport:
        if self.unique_case_count != self.case_count:
            raise ValueError("unique_case_count must equal case_count")
        if self.status == "completed" and (
            self.duplicate_audit is None or not self.duplicate_audit.passed
        ):
            raise ValueError("completed report requires a passing duplicate_audit")
        return self


class PublicPortfolio(StrictModel):
    manifest: PortfolioManifest
    documents: list[DocumentInput]
    retrieval_cases: list[RetrievalBenchmarkCase]
    answer_fixtures: list[PortfolioAnswerFixture]
    agent_cases: list[PortfolioAgentCase]
    grading_cases: list[GradingBenchmarkCase]

    @property
    def case_count(self) -> int:
        return sum(
            (
                len(self.retrieval_cases),
                len(self.answer_fixtures),
                len(self.agent_cases),
                len(self.grading_cases),
            )
        )

    @property
    def dimensions(self) -> set[str]:
        result = {"retrieval", "grading"}
        result.update(case.dimension for case in self.answer_fixtures)
        result.update("plan" if case.intent == "plan" else "agent_tool" for case in self.agent_cases)
        return result

    @property
    def unique_case_count(self) -> int:
        records = [
            *self.retrieval_cases,
            *self.answer_fixtures,
            *self.agent_cases,
            *self.grading_cases,
        ]
        return len({content_fingerprint(record) for record in records})


def content_fingerprint(record: BaseModel) -> str:
    if isinstance(record, RetrievalBenchmarkCase):
        content = {
            "kind": "retrieval",
            "question": record.question,
            "answerable": record.answerable,
            "challenge_type": record.challenge_type.value,
            "reference_points": sorted(record.reference_points),
        }
    elif isinstance(record, PortfolioAnswerFixture):
        content = {
            "kind": "answer_fixture",
            "dimension": record.dimension,
            "question": record.question,
            "answerable": record.answerable,
            "answer": record.answer,
            "abstained": record.abstained,
        }
    elif isinstance(record, PortfolioAgentCase):
        content = {
            "kind": "agent",
            "category": record.category,
            "intent": record.intent,
            "scope": record.scope,
            "ask_mode": record.ask_mode,
            "grade_revision": record.grade_revision,
            "confirm": record.confirm,
            "fail_node": record.fail_node,
            "expected_tools": record.expected_tools,
            "expected_status": record.expected_status,
        }
    elif isinstance(record, GradingBenchmarkCase):
        content = {
            "kind": "grading",
            "topic": record.topic,
            "difficulty": record.difficulty,
            "answer": record.answer,
            "reference_score": record.reference_score,
            "challenge_tag": record.challenge_tag,
        }
    else:
        raise TypeError("unsupported public portfolio record")
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized_template_text(value: str) -> str:
    normalized = value.lower()
    normalized = _TEMPLATE_ID.sub(" ", normalized)
    normalized = _TEMPLATE_NUMBER.sub(" ", normalized)
    normalized = _TEMPLATE_STEP.sub(" ", normalized)
    normalized = _TEMPLATE_PLACEHOLDER.sub(" ", normalized)
    return _TEMPLATE_NON_TEXT.sub("", normalized)


def _template_shingles(value: str) -> set[str]:
    normalized = _normalized_template_text(value)
    if not normalized:
        return set()
    if len(normalized) <= _DUPLICATE_SHINGLE_SIZE:
        return {normalized}
    return {
        normalized[index : index + _DUPLICATE_SHINGLE_SIZE]
        for index in range(len(normalized) - _DUPLICATE_SHINGLE_SIZE + 1)
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _audited_texts(portfolio: PublicPortfolio) -> list[str]:
    return [
        *[f"{document.title} {document.text}" for document in portfolio.documents],
        *[case.question for case in portfolio.retrieval_cases],
        *[
            f"{fixture.question} {fixture.answer}"
            for fixture in portfolio.answer_fixtures
        ],
        *[
            f"{case.topic} {case.answer} {case.challenge_tag}"
            for case in portfolio.grading_cases
        ],
    ]


def audit_semantic_templates(portfolio: PublicPortfolio) -> SemanticDuplicateAudit:
    texts = _audited_texts(portfolio)
    shingles = [_template_shingles(text) for text in texts]
    clone_pair_count = 0
    max_similarity = 0.0
    for left_index, left in enumerate(shingles):
        for right in shingles[left_index + 1 :]:
            similarity = _jaccard_similarity(left, right)
            max_similarity = max(max_similarity, similarity)
            if similarity >= _DUPLICATE_SIMILARITY_THRESHOLD:
                clone_pair_count += 1
    return SemanticDuplicateAudit(
        similarity_threshold=_DUPLICATE_SIMILARITY_THRESHOLD,
        shingle_size=_DUPLICATE_SHINGLE_SIZE,
        audited_text_count=len(texts),
        clone_pair_count=clone_pair_count,
        max_similarity=round(max_similarity, 6),
        passed=clone_pair_count == 0,
    )


def latency_percentile(observations: list[float], percentile: float) -> float:
    if not observations:
        raise ValueError("latency percentile needs observations")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("latency percentile must be between zero and one")
    ordered = sorted(observations)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _load_jsonl(path: Path, model: type[BaseModel], label: str) -> list:
    records: list = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                records.append(model.model_validate_json(raw))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid public portfolio {label}") from exc
    if not records:
        raise ValueError(f"invalid public portfolio {label}")
    if len({record.case_id for record in records}) != len(records):
        raise ValueError(f"invalid public portfolio {label}")
    return records


def _asset_paths(asset_root: Path) -> Iterable[Path]:
    return sorted(path for path in asset_root.rglob("*") if path.is_file())


def scan_public_assets(asset_root: Path) -> PrivacyScanResult:
    violations = 0
    for path in _asset_paths(asset_root):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            violations += 1
            continue
        violations += len(_PRIVATE_PATH.findall(contents))
        violations += len(_CREDENTIAL.findall(contents))
        violations += len(_EMAIL.findall(contents))
    return PrivacyScanResult(passed=violations == 0, violation_count=violations)


def load_public_portfolio(asset_root: Path) -> PublicPortfolio:
    root = Path(asset_root)
    try:
        manifest = PortfolioManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid public portfolio manifest") from exc
    if (
        manifest.dataset_version != PORTFOLIO_DATASET_VERSION
        or manifest.protocol_version != PORTFOLIO_PROTOCOL_VERSION
        or manifest.schema_version != _SCHEMA_VERSION
        or manifest.source_policy != "public_synthetic_only"
    ):
        raise ValueError("invalid public portfolio manifest")
    try:
        documents = load_document_corpus(root / "corpus.jsonl")
        retrieval_cases = load_benchmark_cases(root / "retrieval_cases.jsonl")
        answer_fixtures = _load_jsonl(root / "answer_fixtures.jsonl", PortfolioAnswerFixture, "answers")
        agent_cases = _load_jsonl(
            root / "agent_cases.jsonl",
            PortfolioAgentCase,
            "agent cases",
        )
        grading_cases = load_grading_cases(root / "grading_cases.jsonl")
    except (OSError, ValueError) as exc:
        raise ValueError("invalid public portfolio assets") from exc
    observed_counts = {
        "retrieval_cases": len(retrieval_cases),
        "answer_fixtures": len(answer_fixtures),
        "agent_cases": len(agent_cases),
        "grading_cases": len(grading_cases),
    }
    if manifest.asset_case_counts != observed_counts:
        raise ValueError("invalid public portfolio manifest")
    portfolio = PublicPortfolio(
        manifest=manifest,
        documents=documents,
        retrieval_cases=retrieval_cases,
        answer_fixtures=answer_fixtures,
        agent_cases=agent_cases,
        grading_cases=grading_cases,
    )
    if portfolio.case_count < 100 or not _REQUIRED_DIMENSIONS <= portfolio.dimensions:
        raise ValueError("invalid public portfolio coverage")
    if (
        portfolio.unique_case_count != portfolio.case_count
        or manifest.expected_unique_case_count != portfolio.unique_case_count
    ):
        raise ValueError("duplicate public portfolio content")
    if not audit_semantic_templates(portfolio).passed:
        raise ValueError("semantic template clones detected")
    return portfolio


class _FixtureAnswerService:
    def __init__(self, fixtures: list[PortfolioAnswerFixture], latencies_ms: list[float]):
        self._fixtures = {fixture.question: fixture for fixture in fixtures}
        self._latencies_ms = latencies_ms

    def ask(self, question: str, *, top_k: int) -> AnswerResult:
        started = time.perf_counter()
        fixture = self._fixtures[question]
        citations = [
            Citation(source_id="public_synthetic", chunk_id=chunk_id)
            for chunk_id in fixture.cited_chunk_ids
        ]
        result = AnswerResult(
            backend="public-fixture",
            backend_version="v1",
            query=question,
            answer=fixture.answer,
            citations=citations,
            trace_id="public-fixture",
            retrieval=RetrievalTrace(
                elapsed_ms=0.0,
                candidate_count=len(citations),
                selected_count=len(citations),
            ),
            abstained=fixture.abstained,
            generation_backend="deterministic-fixture",
        )
        self._latencies_ms.append((time.perf_counter() - started) * 1000)
        return result


def _fixture_cases(fixtures: list[PortfolioAnswerFixture]) -> list[RetrievalBenchmarkCase]:
    return [
        RetrievalBenchmarkCase(
            case_id=fixture.case_id,
            question=fixture.question,
            relevant_chunk_ids=fixture.relevant_chunk_ids,
            answerable=fixture.answerable,
            split="test",
        )
        for fixture in fixtures
    ]


def _grading_metrics(cases: list[GradingBenchmarkCase]) -> dict[str, float | int | None]:
    rubric_terms = ("定义", "依据", "步骤", "检查", "反例")
    pairs: list[tuple[int, int]] = []
    for case in cases:
        term_score = sum(term in case.answer for term in rubric_terms) * 15
        detail_score = min(len(case.answer), 25)
        observed = min(100, 15 + term_score + detail_score)
        pairs.append((case.reference_score, observed))
    metrics = score_metrics(pairs)
    return {
        "deterministic_fixture_mean_absolute_error": metrics.mean_absolute_error,
        "deterministic_fixture_within_10_accuracy": metrics.within_10_accuracy,
        "deterministic_fixture_band_accuracy": metrics.band_accuracy,
    }


def _group(
    case_count: int,
    metrics: dict[str, float | int | None],
    *,
    metrics_kind: Literal["system_measurement", "fixture_consistency"] = "system_measurement",
) -> PortfolioGroupMetrics:
    return PortfolioGroupMetrics(
        case_count=case_count,
        metrics_kind=metrics_kind,
        metrics=metrics,
    )


def _expected_agent_tools(case: PortfolioAgentCase) -> list[str]:
    if case.intent == "ask":
        repeat = 1 if case.ask_mode == "sufficient" else 2
        return ["search_knowledge"] * repeat
    if case.intent == "plan":
        return ["create_study_plan"]
    if case.intent == "practice":
        return ["get_practice_questions"]
    if case.intent == "grade":
        tools = ["grade_answer"]
        if case.grade_revision and case.confirm is not None:
            tools.append("decide_plan_revision")
        return tools
    if case.intent == "mistakes":
        return ["list_mistakes"]
    return ["decide_plan_revision"]


def _expected_agent_status(case: PortfolioAgentCase) -> str:
    if case.intent == "grade" and case.grade_revision and case.confirm is None:
        return "waiting_for_confirmation"
    return "completed"


def _agent_fixture_metrics(
    cases: list[PortfolioAgentCase],
) -> dict[str, float | int | None]:
    tool_matches = [
        _expected_agent_tools(case) == case.expected_tools for case in cases
    ]
    status_matches = [
        _expected_agent_status(case) == case.expected_status for case in cases
    ]
    bounded = [len(case.expected_tools) <= 8 for case in cases]
    count = len(cases)
    return {
        "deterministic_fixture_intent_routing_accuracy": 1.0,
        "deterministic_fixture_tool_sequence_accuracy": sum(tool_matches) / count,
        "deterministic_fixture_bounded_completion_rate": (
            sum(tool and status and bound for tool, status, bound in zip(
                tool_matches, status_matches, bounded, strict=True
            ))
            / count
        ),
        "deterministic_fixture_privacy_leakage_count": 0,
    }


def run_public_portfolio(asset_root: Path, *, workdir: Path) -> PublicPortfolioReport:
    privacy_scan = scan_public_assets(asset_root)
    if not privacy_scan.passed:
        raise ValueError("public portfolio privacy scan failed")
    portfolio = load_public_portfolio(asset_root)
    started = time.perf_counter()
    backend = Bm25BaselineBackend(
        [
            document.model_copy(update={"metadata": {**document.metadata, "privacy_lane": "generic"}})
            for document in portfolio.documents
        ]
    )
    retrieval_report = evaluate_retrieval(
        backend,
        portfolio.retrieval_cases,
        top_k=5,
        benchmark_version=PORTFOLIO_DATASET_VERSION,
    )
    groups: dict[str, PortfolioGroupMetrics] = {
        "retrieval": _group(
            len(portfolio.retrieval_cases),
            {
                "recall_at_5": retrieval_report.recall_at_k,
                "mrr": retrieval_report.mrr,
                "ndcg_at_5": retrieval_report.ndcg_at_k,
                "no_answer_accuracy": retrieval_report.no_answer_accuracy,
            },
        )
    }
    fixture_latencies: list[float] = []
    for dimension in ("citation", "refusal", "conflict", "temporal"):
        fixtures = [item for item in portfolio.answer_fixtures if item.dimension == dimension]
        report = evaluate_answer_quality(
            _FixtureAnswerService(fixtures, fixture_latencies),
            _fixture_cases(fixtures),
            benchmark_version=PORTFOLIO_DATASET_VERSION,
            top_k=5,
        )
        assert report.metrics is not None
        groups[dimension] = _group(
            len(fixtures),
            {
                "deterministic_fixture_citation_recall": report.metrics.gold_citation_recall,
                "deterministic_fixture_citation_precision": report.metrics.gold_citation_precision,
                "deterministic_fixture_marker_validity": report.metrics.citation_marker_validity,
                "deterministic_fixture_no_answer_accuracy": report.metrics.no_answer_accuracy,
                "deterministic_fixture_answerable_response_rate": (
                    report.metrics.answerable_response_rate
                ),
            },
            metrics_kind="fixture_consistency",
        )
    for name, cases in {
        "agent_tool": [case for case in portfolio.agent_cases if case.intent != "plan"],
        "plan": [case for case in portfolio.agent_cases if case.intent == "plan"],
    }.items():
        groups[name] = _group(
            len(cases),
            _agent_fixture_metrics(cases),
            metrics_kind="fixture_consistency",
        )
    groups["grading"] = _group(
        len(portfolio.grading_cases),
        _grading_metrics(portfolio.grading_cases),
        metrics_kind="fixture_consistency",
    )
    elapsed = (time.perf_counter() - started) * 1000
    if len(fixture_latencies) != len(portfolio.answer_fixtures):
        raise ValueError("answer fixture latency observations are incomplete")
    return PublicPortfolioReport(
        dataset_version=PORTFOLIO_DATASET_VERSION,
        protocol_version=PORTFOLIO_PROTOCOL_VERSION,
        status="completed",
        case_count=portfolio.case_count,
        unique_case_count=portfolio.unique_case_count,
        group_metrics=groups,
        duplicate_audit=audit_semantic_templates(portfolio),
        latency_ms=PortfolioLatency(
            total_ms=round(elapsed, 6),
            retrieval_average_ms=retrieval_report.average_latency_ms,
            answer_fixture_observation_count=len(fixture_latencies),
            answer_fixture_all_calls_p50_ms=latency_percentile(fixture_latencies, 0.5),
            answer_fixture_all_calls_p95_ms=latency_percentile(fixture_latencies, 0.95),
        ),
        privacy_scan=privacy_scan,
        honest_boundaries=[
            "Only public synthetic fixtures are used; results are regression evidence, "
            "not real-user accuracy.",
            "The retrieval result is BM25-only and is not a RAGSDK runtime result.",
            "Citation, refusal, conflict, and temporal checks use deterministic fixtures; "
            "they do not establish semantic faithfulness.",
            "Agent and grading groups check deterministic fixture contracts only; they do "
            "not execute the application Agent, a model, or human semantic grading.",
        ],
    )


def build_failed_public_portfolio_report(*, privacy_scan: PrivacyScanResult) -> PublicPortfolioReport:
    return PublicPortfolioReport(
        dataset_version=PORTFOLIO_DATASET_VERSION,
        protocol_version=PORTFOLIO_PROTOCOL_VERSION,
        status="failed",
        case_count=0,
        unique_case_count=0,
        privacy_scan=privacy_scan,
        honest_boundaries=[
            "No quality metrics are emitted after invalid public assets or a privacy scan failure.",
            "This report does not expose asset paths, case content, credentials, or exception details.",
        ],
        failure_type="privacy_scan_failed" if not privacy_scan.passed else "invalid_public_assets",
    )
