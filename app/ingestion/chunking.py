from __future__ import annotations

import hashlib
import re

from .models import CHUNKER_VERSION, ChunkRecord, ExtractionBlock, Sensitivity, SourceSpec


def _stable_chunk_id(source_id: str, source_checksum: str, position: int, text: str) -> str:
    payload = f"{CHUNKER_VERSION}\n{source_id}\n{source_checksum}\n{position}\n{text}".encode()
    return f"{source_id}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        boundary = text.rfind("。", start, end)
        if boundary <= start + chunk_size // 2:
            boundary = text.rfind("\n", start, end)
        if boundary > start + chunk_size // 2:
            end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def build_chunks(
    spec: SourceSpec,
    blocks: list[ExtractionBlock],
    *,
    source_checksum: str,
    sensitivity: Sensitivity,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[ChunkRecord]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[ChunkRecord] = []
    seen: set[str] = set()
    position = 0
    for block in blocks:
        text = re.sub(r"\n{3,}", "\n\n", block.text).strip()
        if not text:
            continue
        for piece in _split_long_text(text, chunk_size, overlap):
            content_fingerprint = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            if content_fingerprint in seen:
                continue
            seen.add(content_fingerprint)
            position += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id(spec.source_id, source_checksum, position, piece),
                    source_id=spec.source_id,
                    title=spec.title,
                    category=spec.category,
                    authority_level=spec.authority_level,
                    published_at=spec.published_at,
                    effective_at=spec.effective_at,
                    section=block.section,
                    page=block.page,
                    text=piece,
                    checksum=content_fingerprint,
                    sensitivity=sensitivity,
                    metadata={
                        "block_type": block.block_type,
                        "position": position,
                        "privacy_lane": spec.privacy_lane,
                    },
                )
            )
    return chunks
