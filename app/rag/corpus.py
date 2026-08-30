from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from app.ingestion.models import ChunkRecord, Sensitivity
from app.ingestion.privacy import scan_privacy

from .base import DocumentInput


class UnsafeCorpusError(ValueError):
    """Raised when derived private chunks fail the final privacy boundary."""


def _is_within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def load_private_corpus(
    chunk_directories: Iterable[Path],
    *,
    allowed_roots: Iterable[Path],
    forbidden_terms: Iterable[str] = (),
) -> list[DocumentInput]:
    roots = [root.resolve(strict=True) for root in allowed_roots]
    blocked_terms = tuple(term for term in forbidden_terms if len(term.strip()) >= 2)
    documents: dict[str, DocumentInput] = {}

    for directory in chunk_directories:
        resolved = directory.resolve(strict=True)
        if directory.is_symlink() or not resolved.is_dir() or not _is_within(resolved, roots):
            raise UnsafeCorpusError("Chunk directory is outside the allowed local roots")
        for path in sorted(resolved.glob("*.jsonl")):
            if path.is_symlink():
                raise UnsafeCorpusError("Symlinked chunk files are not accepted")
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    chunk = ChunkRecord.model_validate_json(line)
                except Exception as exc:
                    raise UnsafeCorpusError(
                        f"Invalid chunk record in {path.name}:{line_number}"
                    ) from exc
                if chunk.sensitivity == Sensitivity.HIGH:
                    raise UnsafeCorpusError(f"High-sensitivity chunk rejected: {chunk.chunk_id}")
                combined_text = "\n".join((chunk.title, chunk.section, chunk.text))
                privacy = scan_privacy(combined_text)
                if privacy.finding_counts or privacy.prompt_injection_detected:
                    raise UnsafeCorpusError(f"Residual private or injected content: {chunk.chunk_id}")
                if any(term in combined_text for term in blocked_terms):
                    raise UnsafeCorpusError(f"Known private identity term remains: {chunk.chunk_id}")
                privacy_lane = chunk.metadata.get("privacy_lane")
                if privacy_lane not in {"generic", "personal"}:
                    raise UnsafeCorpusError(f"Missing privacy lane: {chunk.chunk_id}")

                document = DocumentInput(
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    title=chunk.title,
                    section=chunk.section,
                    page=chunk.page,
                    published_at=chunk.published_at,
                    authority_level=chunk.authority_level,
                    metadata={
                        "category": chunk.category,
                        "sensitivity": chunk.sensitivity.value,
                        "effective_at": chunk.effective_at.isoformat() if chunk.effective_at else None,
                        "privacy_lane": privacy_lane,
                    },
                )
                existing = documents.get(document.chunk_id)
                if existing and existing != document:
                    raise UnsafeCorpusError(f"Conflicting duplicate chunk ID: {document.chunk_id}")
                documents[document.chunk_id] = document

    return [documents[chunk_id] for chunk_id in sorted(documents)]


def write_private_corpus(path: Path, documents: Iterable[DocumentInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for document in documents
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
