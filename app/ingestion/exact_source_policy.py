from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .source_scope import SourcePolicyError

_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_EXTENSION_PATTERN = re.compile(r"^\.[A-Za-z0-9]+$")
_PATH_METACHARACTERS = frozenset('*?[]{}<>|"')
_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class ExactSourceEntry:
    source_id: str
    path: str


def _is_unc_or_device_path(raw_path: str) -> bool:
    normalized = raw_path.replace("/", "\\")
    return normalized.startswith(("\\\\", "\\\\?\\", "\\\\.\\"))


def _has_path_metacharacters(raw_path: str) -> bool:
    return any(character in raw_path for character in _PATH_METACHARACTERS)


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SourcePolicyError("source does not exist or cannot be inspected") from exc
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & _REPARSE_POINT_FLAG)


def _reject_reparse_components(candidate: Path, approved_root: Path) -> None:
    current = candidate
    while True:
        if _is_reparse_point(current):
            raise SourcePolicyError("source path must not contain links or reparse points")
        if current == approved_root:
            return
        parent = current.parent
        if parent == current:
            raise SourcePolicyError("source is outside the configured approved roots")
        current = parent


class ExactSourcePolicy:
    def __init__(self, payload: Mapping[str, object]):
        if payload.get("schema_version") != 1:
            raise SourcePolicyError("unsupported exact-source policy schema")

        max_file_bytes = payload.get("max_file_bytes")
        if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes < 1:
            raise SourcePolicyError("max_file_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes

        raw_extensions = payload.get("allowed_extensions")
        if not isinstance(raw_extensions, list) or not raw_extensions:
            raise SourcePolicyError("allowed_extensions must be a non-empty list")
        allowed_extensions: set[str] = set()
        for value in raw_extensions:
            extension = str(value)
            if not _EXTENSION_PATTERN.fullmatch(extension):
                raise SourcePolicyError("allowed_extensions contains an invalid suffix")
            allowed_extensions.add(extension.casefold())
        self.allowed_extensions = frozenset(allowed_extensions)

        raw_roots = payload.get("approved_roots")
        if not isinstance(raw_roots, list) or not raw_roots:
            raise SourcePolicyError("approved_roots must be a non-empty list")
        approved_roots: list[Path] = []
        for value in raw_roots:
            raw_root = str(value)
            root = Path(raw_root)
            if (
                _is_unc_or_device_path(raw_root)
                or _has_path_metacharacters(raw_root)
                or ".." in root.parts
                or not root.is_absolute()
            ):
                raise SourcePolicyError("approved root must be an exact local absolute path")
            approved_roots.append(root)
        self.approved_roots = tuple(approved_roots)

        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            raise SourcePolicyError("sources must be a list")
        entries: dict[str, ExactSourceEntry] = {}
        casefolded_ids: set[str] = set()
        for value in raw_sources:
            if not isinstance(value, Mapping):
                raise SourcePolicyError("each source entry must be an object")
            source_id = str(value.get("source_id", ""))
            raw_path = str(value.get("path", ""))
            if not _SOURCE_ID_PATTERN.fullmatch(source_id):
                raise SourcePolicyError("source_id contains disallowed characters")
            folded_id = source_id.casefold()
            if source_id in entries or folded_id in casefolded_ids:
                raise SourcePolicyError("source IDs must be unique without case ambiguity")
            entries[source_id] = ExactSourceEntry(source_id=source_id, path=raw_path)
            casefolded_ids.add(folded_id)
        self.entries = entries

    @classmethod
    def from_file(cls, path: str | Path) -> ExactSourcePolicy:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SourcePolicyError("exact-source policy cannot be loaded") from exc
        if not isinstance(payload, Mapping):
            raise SourcePolicyError("exact-source policy must contain an object")
        return cls(payload)

    def resolve(self, source_id: str) -> Path:
        entry = self.entries.get(source_id)
        if entry is None or entry.source_id != source_id:
            raise SourcePolicyError("source_id is not allowlisted")

        raw_path = entry.path
        candidate = Path(raw_path)
        if (
            not raw_path
            or _is_unc_or_device_path(raw_path)
            or _has_path_metacharacters(raw_path)
            or ".." in candidate.parts
            or not candidate.is_absolute()
        ):
            raise SourcePolicyError("source path must be an exact local absolute path")

        lexical_roots = [root for root in self.approved_roots if candidate.is_relative_to(root)]
        if not lexical_roots:
            raise SourcePolicyError("source is outside the configured approved roots")

        for root in lexical_roots:
            _reject_reparse_components(candidate, root)

        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise SourcePolicyError("source does not exist or cannot be resolved") from exc

        resolved_roots: list[Path] = []
        for root in lexical_roots:
            try:
                resolved_root = root.resolve(strict=True)
            except OSError as exc:
                raise SourcePolicyError("approved root does not exist or cannot be resolved") from exc
            if not resolved_root.is_dir():
                raise SourcePolicyError("approved root must be a directory")
            if resolved.is_relative_to(resolved_root):
                resolved_roots.append(resolved_root)
        if not resolved_roots:
            raise SourcePolicyError("resolved source escapes the configured approved roots")

        try:
            metadata = os.lstat(resolved)
        except OSError as exc:
            raise SourcePolicyError("source does not exist or cannot be inspected") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise SourcePolicyError("source must be a regular allowlisted file")

        if resolved.suffix.casefold() not in self.allowed_extensions:
            raise SourcePolicyError("source extension is not allowlisted")
        if metadata.st_size > self.max_file_bytes:
            raise SourcePolicyError("source exceeds the configured file-size limit")
        return resolved
