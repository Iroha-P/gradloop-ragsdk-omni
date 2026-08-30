from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .models import ChunkRecord, SourceManifest


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


class ManifestStore:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.manifest_dir = output_root / "manifests"
        self.chunk_dir = output_root / "chunks"
        self.extracted_dir = output_root / "extracted_redacted"

    def manifest_path(self, source_id: str) -> Path:
        return self.manifest_dir / f"{source_id}.json"

    def chunks_path(self, source_id: str) -> Path:
        return self.chunk_dir / f"{source_id}.jsonl"

    def extracted_path(self, source_id: str) -> Path:
        return self.extracted_dir / f"{source_id}.txt"

    def load_manifest(self, source_id: str) -> SourceManifest | None:
        path = self.manifest_path(source_id)
        if not path.exists():
            return None
        return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def load_chunks(self, source_id: str) -> list[ChunkRecord]:
        path = self.chunks_path(source_id)
        if not path.exists():
            return []
        return [
            ChunkRecord.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def save_manifest(self, manifest: SourceManifest) -> None:
        payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write(self.manifest_path(manifest.source_id), payload + "\n")

    def save_chunks(self, source_id: str, chunks: Iterable[ChunkRecord]) -> None:
        payload = "".join(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for chunk in chunks
        )
        _atomic_write(self.chunks_path(source_id), payload)

    def save_extracted(self, source_id: str, text: str) -> Path:
        path = self.extracted_path(source_id)
        _atomic_write(path, text.rstrip() + "\n")
        return path

