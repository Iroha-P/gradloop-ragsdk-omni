from __future__ import annotations

import getpass
import hashlib
import logging
import os
import stat
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path

from app.ingestion.exact_source_policy import ExactSourcePolicy

from .models import ExtractedBlock
from .secure_write import (
    DirectoryIdentity,
    SecureWriteError,
    atomic_write_batch,
    directory_identity,
)

_PASSWORD_ENVIRONMENT_VARIABLE = "PRIVATE_PDF_PASSWORD"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGGER = logging.getLogger(__name__)
_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class LocalExtractionError(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def _anonymous_source_id(source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return f"anon-{digest[:24]}"


def _anonymous_block_id(anonymous_source_id: str, block_index: int, text: str) -> str:
    seed = f"{anonymous_source_id}\0{block_index}\0{text}".encode()
    return f"block-{hashlib.sha256(seed).hexdigest()[:24]}"


def _reject_reparse_write_components(path: Path, project_root: Path, error_category: str) -> None:
    current = path
    while current != project_root:
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            pass
        except OSError:
            raise LocalExtractionError(error_category) from None
        else:
            attributes = getattr(metadata, "st_file_attributes", 0)
            if stat.S_ISLNK(metadata.st_mode) or attributes & _REPARSE_POINT_FLAG:
                raise LocalExtractionError(error_category)
        parent = current.parent
        if parent == current:
            raise LocalExtractionError(error_category)
        current = parent


def validate_local_write_root(
    selected_root: Path,
    *,
    project_root: Path,
    allowed_relative_root: Path,
    exact: bool,
    error_category: str,
) -> Path:
    try:
        resolved_project = project_root.resolve(strict=True)
    except OSError:
        raise LocalExtractionError(error_category) from None

    lexical_allowed = resolved_project / allowed_relative_root
    lexical_selected = Path(os.path.abspath(selected_root))
    allowed_by_scope = (
        lexical_selected == lexical_allowed
        if exact
        else lexical_selected != lexical_allowed and lexical_selected.is_relative_to(lexical_allowed)
    )
    if not allowed_by_scope:
        raise LocalExtractionError(error_category)

    _reject_reparse_write_components(lexical_selected, resolved_project, error_category)
    resolved_allowed = lexical_allowed.resolve(strict=False)
    resolved_selected = lexical_selected.resolve(strict=False)
    still_allowed = (
        resolved_selected == resolved_allowed
        if exact
        else resolved_selected != resolved_allowed and resolved_selected.is_relative_to(resolved_allowed)
    )
    if not still_allowed:
        raise LocalExtractionError(error_category)
    return resolved_selected


def validate_and_bind_local_write_root(
    selected_root: Path,
    *,
    project_root: Path,
    allowed_relative_root: Path,
    exact: bool,
    error_category: str,
) -> tuple[Path, DirectoryIdentity]:
    validated_root = validate_local_write_root(
        selected_root,
        project_root=project_root,
        allowed_relative_root=allowed_relative_root,
        exact=exact,
        error_category=error_category,
    )
    try:
        expected_identity = directory_identity(validated_root)
    except SecureWriteError:
        raise LocalExtractionError(error_category) from None
    confirmed_root = validate_local_write_root(
        validated_root,
        project_root=project_root,
        allowed_relative_root=allowed_relative_root,
        exact=exact,
        error_category=error_category,
    )
    try:
        confirmed_identity = directory_identity(confirmed_root)
    except SecureWriteError:
        raise LocalExtractionError(error_category) from None
    if confirmed_root != validated_root or confirmed_identity != expected_identity:
        raise LocalExtractionError(error_category)
    return confirmed_root, expected_identity


def read_password_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    prompt: Callable[[str], str] | None = None,
    allow_prompt: bool = True,
) -> str | None:
    environment = os.environ if environ is None else environ
    password = environment.get(_PASSWORD_ENVIRONMENT_VARIABLE)
    if password:
        return password
    if not allow_prompt:
        return None

    prompt_function = getpass.getpass if prompt is None else prompt
    password = prompt_function("Private PDF password (input hidden): ")
    return password or None


def write_ocr_intermediate(
    image_bytes: bytes,
    *,
    anonymous_id: str,
    project_root: Path = _PROJECT_ROOT,
    cache_root: Path | None = None,
    media_type: str,
) -> Path:
    selected_cache = validate_local_write_root(
        (project_root / "data" / "local" / "extraction-cache" if cache_root is None else cache_root),
        project_root=project_root,
        allowed_relative_root=Path("data") / "local" / "extraction-cache",
        exact=True,
        error_category="invalid_ocr_cache_root",
    )
    if media_type != "image/png" or not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise LocalExtractionError("unsupported_ocr_media")

    selected_cache.mkdir(parents=True, exist_ok=True)
    selected_cache, expected_identity = validate_and_bind_local_write_root(
        selected_cache,
        project_root=project_root,
        allowed_relative_root=Path("data") / "local" / "extraction-cache",
        exact=True,
        error_category="invalid_ocr_cache_root",
    )
    digest = hashlib.sha256(anonymous_id.encode("utf-8") + b"\0" + image_bytes).hexdigest()
    destination = selected_cache / f"ocr-{digest[:32]}.png"
    try:
        atomic_write_batch(
            selected_cache,
            {destination.name: image_bytes},
            expected_identity=expected_identity,
        )
    except SecureWriteError as exc:
        raise LocalExtractionError(exc.category) from None
    return destination


def extract_pdf_in_memory(
    path: Path,
    *,
    password: str | None = None,
    password_provider: Callable[[], str | None] | None = None,
    anonymous_source_id: str,
) -> list[ExtractedBlock]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise LocalExtractionError("pdf_dependency_unavailable") from None

    try:
        reader = PdfReader(BytesIO(path.read_bytes()))
    except Exception:
        raise LocalExtractionError("pdf_parse_failed") from None

    if reader.is_encrypted:
        selected_password = password
        if selected_password is None:
            provider = password_provider or read_password_from_environment
            try:
                selected_password = provider()
            except (EOFError, KeyboardInterrupt):
                raise LocalExtractionError("password_unavailable") from None
        if not selected_password:
            raise LocalExtractionError("password_unavailable")
        try:
            decrypt_status = reader.decrypt(selected_password)
        except Exception:
            raise LocalExtractionError("password_invalid") from None
        if int(decrypt_status) == 0:
            raise LocalExtractionError("password_invalid")

    blocks: list[ExtractedBlock] = []
    try:
        for pdf_page in reader.pages:
            text = " ".join((pdf_page.extract_text() or "").split())
            if not text:
                continue
            block_index = len(blocks)
            blocks.append(
                ExtractedBlock(
                    source_id=anonymous_source_id,
                    block_id=_anonymous_block_id(anonymous_source_id, block_index, text),
                    text=text,
                    ocr_confidence=None,
                )
            )
    except Exception:
        raise LocalExtractionError("pdf_parse_failed") from None
    if not blocks:
        raise LocalExtractionError("no_extractable_text")
    return blocks


def extract_allowlisted_source(
    source_id: str,
    policy: ExactSourcePolicy,
    *,
    password_provider: Callable[[], str | None] | None = None,
) -> list[ExtractedBlock]:
    anonymous_id = _anonymous_source_id(source_id)
    try:
        path = policy.resolve(source_id)
        if path.suffix.casefold() != ".pdf":
            raise LocalExtractionError("unsupported_source_type")
        blocks = extract_pdf_in_memory(
            path,
            password_provider=password_provider,
            anonymous_source_id=anonymous_id,
        )
    except LocalExtractionError as exc:
        _LOGGER.warning(
            "local_extraction_blocked anonymous_id=%s block_count=0 error_category=%s",
            anonymous_id,
            exc.category,
        )
        raise
    _LOGGER.info(
        "local_extraction_complete anonymous_id=%s block_count=%d error_category=none",
        anonymous_id,
        len(blocks),
    )
    return blocks
