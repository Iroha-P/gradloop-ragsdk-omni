from __future__ import annotations

from pathlib import Path

from .base import DocumentInput, RetrievalScope
from .corpus_file import load_document_corpus


def document_matches_scope(document: DocumentInput, scope: RetrievalScope) -> bool:
    lane = document.metadata.get("privacy_lane")
    if lane not in {"generic", "personal"}:
        raise ValueError(f"document {document.chunk_id} has no valid privacy_lane")
    return scope == RetrievalScope.ALL or lane == scope.value


def load_scoped_corpus(generic_path: Path, personal_path: Path) -> list[DocumentInput]:
    generic = load_document_corpus(generic_path)
    personal = load_document_corpus(personal_path)
    if any(item.metadata.get("privacy_lane") != "generic" for item in generic):
        raise ValueError("generic corpus contains a non-generic document")
    if any(item.metadata.get("privacy_lane") != "personal" for item in personal):
        raise ValueError("personal corpus contains a non-personal document")
    ids = [item.chunk_id for item in [*generic, *personal]]
    if len(ids) != len(set(ids)):
        raise ValueError("scoped corpora contain duplicate chunk IDs")
    return [*generic, *personal]


def mark_legacy_documents_generic(documents: list[DocumentInput]) -> list[DocumentInput]:
    """One-way migration for the already approved pre-lane local corpus."""
    return [
        item.model_copy(
            update={"metadata": {**item.metadata, "privacy_lane": "generic"}}
        )
        for item in documents
    ]
