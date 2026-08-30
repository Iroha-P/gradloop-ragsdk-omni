from __future__ import annotations

from pathlib import Path

from .base import DocumentInput


def load_document_corpus(path: Path) -> list[DocumentInput]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise ValueError("corpus path must be a regular non-symlink file")
    documents: list[DocumentInput] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = DocumentInput.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"invalid corpus record at line {line_number}") from exc
        if document.chunk_id in seen:
            raise ValueError(f"duplicate chunk ID at line {line_number}")
        seen.add(document.chunk_id)
        documents.append(document)
    return documents
