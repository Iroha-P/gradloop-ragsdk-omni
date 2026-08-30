from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import Field

from .models import QuestionModel


class SourceCoverage(QuestionModel):
    source_id: str
    status: str
    section_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    extracted_count: int = Field(default=0, ge=0)
    unparsed_block_count: int = Field(default=0, ge=0)


class CoverageBuildInput(QuestionModel):
    snapshot_checksum: str
    corpus_checksum: str
    sources: list[SourceCoverage]
    candidate_count: int = Field(ge=0)
    canonical_count: int = Field(ge=0)
    duplicate_group_count: int = Field(ge=0)
    conflict_group_count: int = Field(ge=0)
    generic_count: int = Field(ge=0)
    personal_count: int = Field(ge=0)
    has_answer_count: int = Field(ge=0)
    needs_answer_count: int = Field(ge=0)
    needs_review_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    unparsed_block_count: int = Field(ge=0)


class CoverageReport(CoverageBuildInput):
    builder_version: str = "question-bank-v1"
    source_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    extracted_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)


def build_coverage_report(value: CoverageBuildInput) -> CoverageReport:
    if value.candidate_count != sum(item.candidate_count for item in value.sources):
        raise ValueError("candidate coverage is incomplete")
    if any(item.extracted_count > item.candidate_count for item in value.sources):
        raise ValueError("source extracted count exceeds candidate count")
    unresolved = (
        value.needs_answer_count
        + value.needs_review_count
        + value.conflict_count
        + value.unparsed_block_count
    )
    return CoverageReport(
        **value.model_dump(),
        source_count=len(value.sources),
        section_count=sum(item.section_count for item in value.sources),
        extracted_count=sum(item.extracted_count for item in value.sources),
        unresolved_count=unresolved,
    )


def write_coverage_report(path: Path, report: CoverageReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_coverage_report(path: Path) -> CoverageReport:
    if not path.is_file() or path.is_symlink():
        raise ValueError("coverage report is not a regular local file")
    return CoverageReport.model_validate_json(path.read_text(encoding="utf-8"))
