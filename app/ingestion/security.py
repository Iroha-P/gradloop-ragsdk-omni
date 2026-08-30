from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import UnsafeInputError

ALLOWED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class AuthorizedFile:
    resolved_path: Path
    relative_path: str
    size_bytes: int
    suffix: str


def _validate_file_signature(path: Path, suffix: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(8192)
    if suffix in {".md", ".txt"}:
        if b"\x00" in header:
            raise UnsafeInputError("Text source contains binary null bytes")
        return
    if suffix == ".pdf":
        if not header.startswith(b"%PDF-"):
            raise UnsafeInputError("PDF extension does not match the file signature")
        return
    if suffix == ".docx":
        if not zipfile.is_zipfile(path):
            raise UnsafeInputError("DOCX extension does not match a ZIP-based Office document")
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise UnsafeInputError("DOCX container is damaged") from exc
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise UnsafeInputError("DOCX container is missing required document entries")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def authorize_file(
    path: str | Path,
    allowed_roots: list[Path],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> AuthorizedFile:
    raw = str(path)
    if raw.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        raise UnsafeInputError("UNC and device paths are not allowed")

    candidate = Path(path)
    if ".." in candidate.parts:
        raise UnsafeInputError("Path traversal is not allowed")
    if candidate.is_symlink():
        raise UnsafeInputError("Symbolic-link sources are not allowed")

    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise UnsafeInputError("Source file does not exist or cannot be resolved") from exc

    if not resolved.is_file():
        raise UnsafeInputError("Source must be a regular file")

    authorized_root: Path | None = None
    root_index = -1
    for index, root in enumerate(allowed_roots, start=1):
        try:
            resolved_root = root.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise UnsafeInputError("Configured source root cannot be resolved") from exc
        if resolved.is_relative_to(resolved_root):
            authorized_root = resolved_root
            root_index = index
            break
    if authorized_root is None:
        raise UnsafeInputError("Source is outside the configured read-only roots")

    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsafeInputError(f"Unsupported source type: {suffix or '<none>'}")

    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise UnsafeInputError("Empty files are not accepted")
    if size_bytes > max_file_bytes:
        raise UnsafeInputError("Source exceeds the configured file-size limit")

    _validate_file_signature(resolved, suffix)

    relative = resolved.relative_to(authorized_root).as_posix()
    return AuthorizedFile(
        resolved_path=resolved,
        relative_path=f"source_root_{root_index}/{relative}",
        size_bytes=size_bytes,
        suffix=suffix,
    )
