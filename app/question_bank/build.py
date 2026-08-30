from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.ingestion.chunking import build_chunks
from app.ingestion.feishu_snapshot import PrivacyLane, SnapshotManifest, SnapshotStatus
from app.ingestion.models import ChunkRecord, ExtractionBlock, Sensitivity, SourceSpec
from app.llm.base import LlmClient
from app.rag.base import DocumentInput
from app.rag.scoped_corpus import load_scoped_corpus

from .coverage import (
    CoverageBuildInput,
    CoverageReport,
    SourceCoverage,
    build_coverage_report,
    load_coverage_report,
    write_coverage_report,
)
from .dedupe import canonicalize_questions
from .extractor import extract_questions
from .llm_enrichment import QuestionEnricher, QuestionEnrichment
from .models import (
    AnswerConflict,
    CanonicalizationResult,
    DuplicateGroup,
    QuestionCandidate,
    QuestionModel,
    QuestionReviewStatus,
    QuestionSource,
    SimilarityCandidate,
)
from .repository import QuestionBankRepository, write_question_bank


class QuestionBankBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage: CoverageReport
    question_count: int
    generic_document_count: int
    personal_document_count: int


def sources_from_local_chunks(
    directories: Iterable[Path],
    *,
    lane: PrivacyLane,
) -> list[QuestionSource]:
    sources: list[QuestionSource] = []
    for directory in directories:
        if not directory.exists():
            continue
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("local chunk path must be a regular directory")
        for path in sorted(directory.glob("*.jsonl")):
            if path.is_symlink():
                raise ValueError("symlinked local chunks are not accepted")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = ChunkRecord.model_validate_json(line)
                if chunk.sensitivity == Sensitivity.HIGH:
                    continue
                sources.append(
                    QuestionSource(
                        source_id=chunk.source_id,
                        title=chunk.title,
                        section=chunk.section,
                        page=chunk.page,
                        text=chunk.text,
                        privacy_lane=lane,
                    )
                )
    return sources


def _load_inventory(snapshot_root: Path) -> dict:
    path = snapshot_root / "inventory.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("snapshot inventory is missing or unsafe")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory.get("complete") is not True:
        raise ValueError("snapshot inventory is incomplete")
    return inventory


def sources_from_snapshot(snapshot_root: Path) -> list[QuestionSource]:
    inventory = _load_inventory(snapshot_root)
    sources: list[QuestionSource] = []
    for item in inventory.get("sources", []):
        manifest = SnapshotManifest.model_validate(item)
        if manifest.status not in {SnapshotStatus.INDEXED, SnapshotStatus.UNCHANGED}:
            continue
        path = snapshot_root / "extracted_redacted" / f"{manifest.source_id}.md"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"snapshot content is missing for {manifest.source_id}")
        sources.append(
            QuestionSource(
                source_id=manifest.source_id,
                title=manifest.node_path[-1] if manifest.node_path else manifest.source_id,
                revision_id=manifest.remote_revision_id,
                text=path.read_text(encoding="utf-8"),
                privacy_lane=manifest.privacy_lane,
            )
        )
    return sources


def build_scoped_documents(sources: Iterable[QuestionSource]) -> list[DocumentInput]:
    documents: list[DocumentInput] = []
    for source in sources:
        source_checksum = hashlib.sha256(source.text.encode("utf-8")).hexdigest()
        spec = SourceSpec(
            source_id=source.source_id,
            path=f"private://{source.source_id}",
            title=source.title or source.source_id,
            category="question_bank_source",
            authority_level="private",
            declared_sensitivity=Sensitivity.LOW,
        )
        chunks = build_chunks(
            spec,
            [
                ExtractionBlock(
                    text=source.text,
                    section=source.section,
                    page=source.page,
                    block_type="question_source",
                )
            ],
            source_checksum=source_checksum,
            sensitivity=Sensitivity.LOW,
        )
        documents.extend(
            DocumentInput(
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                title=chunk.title,
                section=chunk.section,
                page=chunk.page,
                authority_level=chunk.authority_level,
                metadata={**chunk.metadata, "privacy_lane": source.privacy_lane.value},
            )
            for chunk in chunks
        )
    ids = [item.chunk_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("derived corpus contains duplicate chunk IDs")
    return sorted(documents, key=lambda item: item.chunk_id)


def _anonymous_id(source_id: str) -> str:
    return "source_" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]


def _corpus_checksum(documents: list[DocumentInput]) -> str:
    payload = "".join(
        item.model_dump_json() + "\n" for item in sorted(documents, key=lambda row: row.chunk_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_from_build(
    *,
    snapshot_root: Path,
    local_sources: list[QuestionSource],
    candidates: list[QuestionCandidate],
    canonical: CanonicalizationResult,
    documents: list[DocumentInput],
) -> CoverageBuildInput:
    inventory = _load_inventory(snapshot_root)
    candidate_counts = Counter(item.source_ref.source_id for item in candidates)
    sections: dict[str, set[str]] = defaultdict(set)
    for source in local_sources:
        sections[source.source_id].add(source.section)

    source_rows: list[SourceCoverage] = []
    seen_sources: set[str] = set()
    for raw in inventory.get("sources", []):
        manifest = SnapshotManifest.model_validate(raw)
        seen_sources.add(manifest.source_id)
        count = candidate_counts[manifest.source_id]
        unresolved = int(
            manifest.status in {SnapshotStatus.INDEXED, SnapshotStatus.UNCHANGED} and count == 0
        )
        source_rows.append(
            SourceCoverage(
                source_id=_anonymous_id(manifest.source_id),
                status=manifest.status.value,
                section_count=int(manifest.status in {SnapshotStatus.INDEXED, SnapshotStatus.UNCHANGED}),
                candidate_count=count,
                extracted_count=count,
                unparsed_block_count=unresolved,
            )
        )
    for source_id in sorted({item.source_id for item in local_sources}):
        if source_id in seen_sources:
            continue
        count = candidate_counts[source_id]
        source_rows.append(
            SourceCoverage(
                source_id=_anonymous_id(source_id),
                status="indexed",
                section_count=len(sections[source_id]),
                candidate_count=count,
                extracted_count=count,
                unparsed_block_count=int(count == 0),
            )
        )

    statuses = Counter(item.review_status for item in canonical.questions)
    snapshot_checksum = str(inventory.get("snapshot_checksum", ""))
    if not snapshot_checksum:
        snapshot_checksum = hashlib.sha256(
            (snapshot_root / "inventory.json").read_bytes()
        ).hexdigest()
    return CoverageBuildInput(
        snapshot_checksum=snapshot_checksum,
        corpus_checksum=_corpus_checksum(documents),
        sources=source_rows,
        candidate_count=len(candidates),
        canonical_count=len(canonical.questions),
        duplicate_group_count=len(canonical.duplicate_groups),
        conflict_group_count=len(canonical.conflicts),
        generic_count=sum(
            item.privacy_lane == PrivacyLane.GENERIC for item in canonical.questions
        ),
        personal_count=sum(
            item.privacy_lane == PrivacyLane.PERSONAL for item in canonical.questions
        ),
        has_answer_count=sum(bool(item.answer_points) for item in canonical.questions),
        needs_answer_count=statuses[QuestionReviewStatus.NEEDS_ANSWER],
        needs_review_count=statuses[QuestionReviewStatus.NEEDS_REVIEW],
        conflict_count=statuses[QuestionReviewStatus.CONFLICT],
        unparsed_block_count=sum(item.unparsed_block_count for item in source_rows),
    )


def _write_models(path: Path, rows: Iterable[QuestionModel]) -> None:
    values = sorted(rows, key=lambda item: item.model_dump_json())
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in values),
        encoding="utf-8",
        newline="\n",
    )


def _write_documents(path: Path, documents: list[DocumentInput]) -> None:
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in documents),
        encoding="utf-8",
        newline="\n",
    )


def validate_published_bank(root: Path) -> None:
    QuestionBankRepository(root / "questions.jsonl").load()
    load_coverage_report(root / "coverage.json")
    load_scoped_corpus(
        root / "corpora" / "generic.jsonl",
        root / "corpora" / "personal.jsonl",
    )
    for path, model in (
        (root / "duplicate_groups.jsonl", DuplicateGroup),
        (root / "conflicts.jsonl", AnswerConflict),
        (root / "similarity_candidates.jsonl", SimilarityCandidate),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                model.model_validate_json(line)


class QuestionBankBuilder:
    def __init__(self, output_root: Path, *, llm: LlmClient | None = None):
        self.output_root = output_root
        self.llm = llm

    def build(
        self,
        *,
        snapshot_root: Path,
        local_chunk_directories: list[Path],
        local_default_lane: str,
    ) -> QuestionBankBuildResult:
        local_sources = sources_from_local_chunks(
            local_chunk_directories,
            lane=PrivacyLane(local_default_lane),
        )
        sources = [*local_sources, *sources_from_snapshot(snapshot_root)]
        candidates = extract_questions(sources)
        enrichments: dict[str, QuestionEnrichment] = {}
        reviewer = QuestionEnricher(self.llm) if self.llm else None
        if reviewer:
            enrichments = {
                item.candidate_id: reviewer.enrich(item)
                for item in candidates
            }
        canonical = canonicalize_questions(candidates, enrichments, reviewer)
        documents = build_scoped_documents(sources)
        generic = [
            item for item in documents if item.metadata["privacy_lane"] == "generic"
        ]
        personal = [
            item for item in documents if item.metadata["privacy_lane"] == "personal"
        ]
        coverage = build_coverage_report(
            coverage_from_build(
                snapshot_root=snapshot_root,
                local_sources=local_sources,
                candidates=candidates,
                canonical=canonical,
                documents=documents,
            )
        )
        self._publish(canonical, coverage, generic, personal)
        return QuestionBankBuildResult(
            coverage=coverage,
            question_count=len(canonical.questions),
            generic_document_count=len(generic),
            personal_document_count=len(personal),
        )

    def _publish(
        self,
        canonical: CanonicalizationResult,
        coverage: CoverageReport,
        generic: list[DocumentInput],
        personal: list[DocumentInput],
    ) -> None:
        parent = self.output_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / ".question-bank-staging"
        rollback = parent / ".question-bank-rollback"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        (staging / "corpora").mkdir()
        try:
            write_question_bank(staging / "questions.jsonl", canonical.questions)
            _write_models(staging / "duplicate_groups.jsonl", canonical.duplicate_groups)
            _write_models(staging / "conflicts.jsonl", canonical.conflicts)
            _write_models(
                staging / "similarity_candidates.jsonl",
                canonical.similarity_candidates,
            )
            _write_documents(staging / "corpora" / "generic.jsonl", generic)
            _write_documents(staging / "corpora" / "personal.jsonl", personal)
            write_coverage_report(staging / "coverage.json", coverage)
            validate_published_bank(staging)
            if rollback.exists():
                shutil.rmtree(rollback)
            if self.output_root.exists():
                os.replace(self.output_root, rollback)
            try:
                os.replace(staging, self.output_root)
            except Exception:
                if rollback.exists() and not self.output_root.exists():
                    os.replace(rollback, self.output_root)
                raise
            if rollback.exists():
                shutil.rmtree(rollback)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
