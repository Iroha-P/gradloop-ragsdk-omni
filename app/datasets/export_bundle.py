from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import ipaddress
import json
import os
import re
import secrets
import stat
import tarfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from . import secure_write
from .anonymize import scan_sensitive_text
from .leakage import validate_candidate_records
from .secure_write import DirectoryIdentity, SecureWriteError

if os.name == "nt":
    import msvcrt

SOURCE_DATA_NAMES = (
    "train.jsonl",
    "validation.jsonl",
    "frozen_test.jsonl",
    "provenance.jsonl",
    "DATA_CARD.md",
)
SOURCE_INPUT_NAMES = (*SOURCE_DATA_NAMES, "human_review.json")
BUNDLE_MEMBER_NAMES = (*SOURCE_DATA_NAMES, "scan-report.json", "SHA256SUMS.json")
EXPECTED_SPLITS = {
    "train.jsonl": "train",
    "validation.jsonl": "dev",
    "frozen_test.jsonl": "test",
}
RIGHTS_CLASSES = frozenset({"user_owned_derived", "open_licensed", "synthetic"})
OPEN_LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "PDM-1.0": "https://creativecommons.org/publicdomain/mark/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}
ALLOWED_USES = (
    "research",
    "model_training",
    "redistribution",
    "adaptation",
)
RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "task",
        "question",
        "answer",
        "rubric",
        "rights_class",
        "group_id",
        "split",
        "reference_texts",
    }
)
PROVENANCE_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "rights_class",
        "record_sha256",
    }
)
OPEN_PROVENANCE_FIELDS = PROVENANCE_BASE_FIELDS | frozenset(
    {
        "canonical_url",
        "retrieved_at",
        "content_sha256",
        "license_id",
        "license_url",
        "attribution",
        "allowed_uses",
    }
)
PRIVATE_PROVENANCE_FIELDS = PROVENANCE_BASE_FIELDS | frozenset({"license_id"})
PROVENANCE_FIELDS_BY_RIGHTS = {
    "open_licensed": OPEN_PROVENANCE_FIELDS,
    "synthetic": PRIVATE_PROVENANCE_FIELDS,
    "user_owned_derived": PRIVATE_PROVENANCE_FIELDS,
}
PRIVATE_LICENSES_BY_RIGHTS = {
    "synthetic": "synthetic",
    "user_owned_derived": "user-owned-derived",
}
REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "reviewer_id",
        "seed",
        "sample_size",
        "sample_ids",
        "dataset_digest",
        "sample_digest",
        "reviewed_at",
        "expires_at",
        "blocking_findings",
        "binding_digest",
    }
)
MAX_SOURCE_FILE_BYTES = 32 * 1024 * 1024
MAX_TAR_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 20_000
MAX_TEXT_CHARS = 500_000
MAX_JSON_BYTES = 512 * 1024
MAX_JSON_LINE_CHARS = 256 * 1024
MAX_JSON_NESTING = 32
MAX_JSON_NODES = 100_000
MAX_PUBLIC_URL_CHARS = 2048
MAX_REVIEW_AGE = timedelta(days=30)
_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REVIEWER_PATTERN = re.compile(r"anon-reviewer-[0-9a-f]{24}\Z")
_CANONICAL_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")
_LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}\Z")
_CANONICAL_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
CANONICAL_ID_SCHEMA_PATTERN = r"^[a-z0-9][a-z0-9._:-]{0,127}$"
REVIEWER_SCHEMA_PATTERN = r"^anon-reviewer-[0-9a-f]{24}$"
CANONICAL_TIMESTAMP_SCHEMA_PATTERN = (
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
ATTRIBUTION_SCHEMA_PATTERN = r"^\S(?:[\s\S]*\S)?$"
PUBLIC_HTTPS_SCHEMA_PATTERN = (
    r"^https://(?!(?:[^/]*\.)?(?:internal|local|localhost|home|lan)(?:/|$))"
    r"(?:(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]?|0)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]?|0)"
    r"|(?!(?:(?:0x[0-9a-f]+|[0-9]+)\.)*(?:0x[0-9a-f]+|[0-9]+)(?:/|$))"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)"
    r"(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))+|\[[0-9a-f:]+\])"
    r"(?:/(?:[^\s?#\\%]|%(?!(?:0[0-9A-F]|1[0-9A-F]|25|2[D-F]|3[0-9]"
    r"|4[1-9A-F]|5[0-9ACF]|6[1-9A-F]|7[0-9AEF]))[0-9A-F]{2})*)?$"
)
_ATTRIBUTION_PATTERN = re.compile(ATTRIBUTION_SCHEMA_PATTERN)
_PRIVATE_HOST_SUFFIXES = (".internal", ".local", ".localhost", ".home", ".lan")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_LEGACY_NUMERIC_LABEL = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)", re.IGNORECASE)
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password|passcode|access[_-]?key)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
)
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|\\\\[^\\\s]+\\[^\\\s]+"
    r"|(?<![A-Za-z0-9_.:/-])/(?:home|Users|private|mnt)/"
)
_CERTIFICATE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:\d{17}[0-9Xx]|[A-Z][0-9]{8}|[EG]\d{8})(?![A-Za-z0-9])")
_URL_PATTERN = re.compile(r"(?i)\b(?:https?|file)://\S+")
_SOURCE_MAPPING_PATTERN = re.compile(
    r"(?i)(?:source[_ -]?(?:path|url|title|filename)|original[_ -]?source)"
    r"\s*[:=：]"
)
_WINDOWS_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CLOCK$",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_WINDOWS_INVALID_NAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_FILE_RENAME_INFO_EX = 22
_WINDOWS_FILE_RENAME_INFO = 3
_WINDOWS_FILE_DISPOSITION_INFO_EX = 21
_WINDOWS_FILE_DISPOSITION_INFO = 4
_WINDOWS_FILE_DISPOSITION_DELETE = 0x00000001
_WINDOWS_TARGET_EXISTS_ERRORS = frozenset({80, 183})
_WINDOWS_EXTENDED_INFO_UNAVAILABLE_ERRORS = frozenset({50, 87, 120})
_AT_EMPTY_PATH = 0x1000
_O_TMPFILE = getattr(os, "O_TMPFILE", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_LIBC_LINKAT = None
if os.name != "nt":
    try:
        _LIBC_LINKAT = ctypes.CDLL(None, use_errno=True).linkat
        _LIBC_LINKAT.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        _LIBC_LINKAT.restype = ctypes.c_int
    except (AttributeError, OSError):
        _LIBC_LINKAT = None


class BundleSafetyError(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class TransferManifest:
    safe: bool
    member_count: int
    sample_count: int
    split_counts: dict[str, int]
    dataset_digest: str
    bundle_sha256: str
    schema_version: str = field(default="secure-transfer-manifest.v1", init=False)

    def model_dump(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "safe": self.safe,
            "member_count": self.member_count,
            "sample_count": self.sample_count,
            "split_counts": dict(self.split_counts),
            "dataset_digest": self.dataset_digest,
            "bundle_sha256": self.bundle_sha256,
        }


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _DuplicateJsonKey
        result[name] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise BundleSafetyError("invalid_json_value") from None
    return (payload + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_dataset_digest(payloads: Mapping[str, bytes]) -> str:
    if set(payloads) != set(SOURCE_DATA_NAMES):
        raise BundleSafetyError("invalid_dataset_payload_set")
    digest = hashlib.sha256()
    digest.update(b"gradloop-secure-transfer-dataset-v1\0")
    for name in SOURCE_DATA_NAMES:
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise BundleSafetyError("invalid_dataset_payload")
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def deterministic_review_sample(
    sample_ids: Sequence[str],
    *,
    dataset_digest: str,
    seed: str,
    sample_size: int,
) -> list[str]:
    if type(sample_ids) not in {list, tuple} or any(
        type(sample_id) is not str or not _CANONICAL_ID_PATTERN.fullmatch(sample_id)
        for sample_id in sample_ids
    ):
        raise BundleSafetyError("invalid_review_sampling")
    if (
        type(dataset_digest) is not str
        or not _SHA256_PATTERN.fullmatch(dataset_digest)
        or type(seed) is not str
        or not _CANONICAL_ID_PATTERN.fullmatch(seed)
        or isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 0
        or sample_size > len(sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise BundleSafetyError("invalid_review_sampling")
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}\0{dataset_digest}\0{sample_id}".encode()).digest(),
            sample_id,
        ),
    )
    return ranked[:sample_size]


def review_binding_digest(review: Mapping[str, object]) -> str:
    projection = {name: value for name, value in review.items() if name != "binding_digest"}
    return _sha256(_canonical_json_bytes(projection))


def _sample_digest(sample_ids: Sequence[str]) -> str:
    return _sha256(_canonical_json_bytes({"sample_ids": list(sample_ids)}))


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_FLAG)


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _assert_regular_unique(metadata: os.stat_result, category: str) -> None:
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BundleSafetyError(category)


@contextmanager
def _bound_directory(path: Path, *, category: str) -> Iterator[tuple[DirectoryIdentity, int | None]]:
    try:
        metadata = os.lstat(path)
    except OSError:
        raise BundleSafetyError(category) from None
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise BundleSafetyError(category)

    if os.name == "nt":
        try:
            locked_directory = secure_write._locked_windows_directory  # type: ignore[attr-defined]
            with locked_directory(path) as (_, identity):
                yield identity, None
        except (AttributeError, SecureWriteError, OSError):
            raise BundleSafetyError(category) from None
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise BundleSafetyError(category) from None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise BundleSafetyError(category)
        yield DirectoryIdentity(opened.st_dev, opened.st_ino), descriptor
    finally:
        try:
            os.close(descriptor)
        except OSError:
            raise BundleSafetyError("secure_cleanup_failed") from None


def _read_bound_regular_file(
    path: Path,
    *,
    max_bytes: int = MAX_SOURCE_FILE_BYTES,
    directory_fd: int | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> bytes:
    try:
        before = (
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            if directory_fd is not None
            else os.lstat(path)
        )
    except OSError:
        raise BundleSafetyError("unsafe_input_leaf") from None
    _assert_regular_unique(before, "unsafe_input_leaf")
    if before.st_size > max_bytes:
        raise BundleSafetyError("input_too_large")

    if expected_parent_identity is not None and directory_fd is None:
        try:
            if secure_write.directory_identity(path.parent) != expected_parent_identity:
                raise BundleSafetyError("input_directory_changed")
        except SecureWriteError:
            raise BundleSafetyError("input_directory_changed") from None

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = (
            os.open(path.name, flags, dir_fd=directory_fd)
            if directory_fd is not None
            else os.open(path, flags)
        )
    except OSError:
        raise BundleSafetyError("unsafe_input_leaf") from None
    try:
        opened = os.fstat(descriptor)
        _assert_regular_unique(opened, "unsafe_input_leaf")
        if _identity(opened) != _identity(before) or opened.st_size != before.st_size:
            raise BundleSafetyError("input_leaf_changed")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise BundleSafetyError("input_too_large")
        after = os.fstat(descriptor)
        if (
            _identity(after) != _identity(opened)
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise BundleSafetyError("input_leaf_changed")
    finally:
        try:
            os.close(descriptor)
        except OSError:
            raise BundleSafetyError("secure_cleanup_failed") from None

    if expected_parent_identity is not None and directory_fd is None:
        try:
            if secure_write.directory_identity(path.parent) != expected_parent_identity:
                raise BundleSafetyError("input_directory_changed")
        except SecureWriteError:
            raise BundleSafetyError("input_directory_changed") from None
    return payload


def _directory_names(path: Path, directory_fd: int | None) -> list[str]:
    try:
        values = os.listdir(directory_fd) if directory_fd is not None else os.listdir(path)
    except OSError:
        raise BundleSafetyError("invalid_input_root") from None
    if any(not isinstance(name, str) or Path(name).name != name for name in values):
        raise BundleSafetyError("unregistered_input")
    return sorted(values)


def _read_source_tree(input_root: Path) -> dict[str, bytes]:
    with _bound_directory(input_root, category="invalid_input_root") as (
        root_identity,
        directory_fd,
    ):
        names = _directory_names(input_root, directory_fd)
        if set(names) != set(SOURCE_INPUT_NAMES):
            raise BundleSafetyError("unregistered_input")
        payloads: dict[str, bytes] = {}
        for name in SOURCE_INPUT_NAMES:
            payloads[name] = _read_bound_regular_file(
                input_root / name,
                directory_fd=directory_fd,
                expected_parent_identity=root_identity,
            )
        if directory_fd is None:
            try:
                if secure_write.directory_identity(input_root) != root_identity:
                    raise BundleSafetyError("input_directory_changed")
            except SecureWriteError:
                raise BundleSafetyError("input_directory_changed") from None
        return payloads


def _decode_scannable(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise BundleSafetyError("unscannable_content") from None
    if "\x00" in text or len(text) > MAX_TEXT_CHARS:
        raise BundleSafetyError("unscannable_content")
    return text


def _scan_sensitive_payload(payload: bytes) -> str:
    text = _decode_scannable(payload)
    _scan_sensitive_text(text)
    return text


def _scan_sensitive_text(text: str) -> None:
    scan = scan_sensitive_text(text)
    if (
        scan.hits
        or _SECRET_PATTERN.search(text)
        or _PRIVATE_PATH_PATTERN.search(text)
        or _CERTIFICATE_PATTERN.search(text)
        or _URL_PATTERN.search(text)
        or _SOURCE_MAPPING_PATTERN.search(text)
    ):
        raise BundleSafetyError("sensitive_content")


def _scan_public_url_privacy(hostname: str, raw_path: str, decoded_path: str) -> None:
    for text in (raw_path, decoded_path):
        if (
            _SECRET_PATTERN.search(text)
            or _PRIVATE_PATH_PATTERN.search(text)
            or _CERTIFICATE_PATTERN.search(text)
            or _SOURCE_MAPPING_PATTERN.search(text)
        ):
            raise BundleSafetyError("sensitive_content")
    for component in (
        hostname,
        *raw_path.split("/"),
        *decoded_path.split("/"),
    ):
        if not component:
            continue
        scan = scan_sensitive_text(component)
        if (
            scan.hits
            or _SECRET_PATTERN.search(component)
            or _PRIVATE_PATH_PATTERN.search(component)
            or _CERTIFICATE_PATTERN.search(component)
            or _SOURCE_MAPPING_PATTERN.search(component)
        ):
            raise BundleSafetyError("sensitive_content")


def _decode_canonical_url_path(raw_path: str) -> str:
    uppercase_hex = frozenset("0123456789ABCDEF")
    unreserved = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
    cursor = 0
    while cursor < len(raw_path):
        if raw_path[cursor] != "%":
            cursor += 1
            continue
        if cursor + 2 >= len(raw_path):
            raise BundleSafetyError("invalid_provenance_url")
        encoded = raw_path[cursor + 1 : cursor + 3]
        if any(character not in uppercase_hex for character in encoded):
            raise BundleSafetyError("invalid_provenance_url")
        decoded_byte = int(encoded, 16)
        decoded_ascii = chr(decoded_byte)
        if (
            decoded_byte in {*range(0x20), 0x7F}
            or decoded_ascii in unreserved
            or decoded_ascii in {"/", "\\", "%"}
        ):
            raise BundleSafetyError("invalid_provenance_url")
        cursor += 3
    try:
        decoded_path = unquote(raw_path, errors="strict")
    except (UnicodeDecodeError, ValueError):
        raise BundleSafetyError("invalid_provenance_url") from None
    if "%" in decoded_path:
        raise BundleSafetyError("invalid_provenance_url")
    return decoded_path


def _canonical_public_host(hostname: str) -> str | None:
    if not hostname or hostname.endswith("."):
        return None
    lowered = hostname.casefold()
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        labels = lowered.split(".")
        if all(_LEGACY_NUMERIC_LABEL.fullmatch(label) for label in labels):
            return None
        try:
            ascii_host = hostname.encode("idna").decode("ascii").casefold()
        except UnicodeError:
            return None
        ascii_labels = ascii_host.split(".")
        if (
            len(ascii_host) > 253
            or len(ascii_labels) < 2
            or any(not label for label in ascii_labels)
            or all(_LEGACY_NUMERIC_LABEL.fullmatch(label) for label in ascii_labels)
            or ascii_host == "localhost"
            or ascii_host.endswith(_PRIVATE_HOST_SUFFIXES)
        ):
            return None
        for label in ascii_labels:
            if _HOST_LABEL.fullmatch(label) is None:
                return None
            if label.startswith("xn--"):
                try:
                    decoded = label.encode("ascii").decode("idna")
                    round_trip = decoded.encode("idna").decode("ascii").casefold()
                except UnicodeError:
                    return None
                if round_trip != label:
                    return None
        return ascii_host
    canonical = address.compressed.casefold()
    if address.is_multicast or not address.is_global or canonical != lowered:
        return None
    return canonical


def _validate_public_https_url(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PUBLIC_URL_CHARS:
        raise BundleSafetyError("invalid_provenance_url")
    if (
        value != value.strip()
        or "\\" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise BundleSafetyError("invalid_provenance_url")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise BundleSafetyError("invalid_provenance_url") from None
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not hostname
        or hostname.endswith(".")
        or hostname != hostname.lower()
        or "%" in parsed.netloc
    ):
        raise BundleSafetyError("invalid_provenance_url")
    if _canonical_public_host(hostname) != hostname:
        raise BundleSafetyError("invalid_provenance_url")
    decoded_path = _decode_canonical_url_path(parsed.path)
    if (
        any(ord(character) < 0x20 or ord(character) == 0x7F for character in decoded_path)
        or "\\" in decoded_path
        or any(part in {".", ".."} for part in decoded_path.split("/"))
        or unicodedata.normalize("NFC", decoded_path) != decoded_path
    ):
        raise BundleSafetyError("invalid_provenance_url")
    _scan_public_url_privacy(hostname, parsed.path, decoded_path)
    canonical_netloc = f"[{hostname}]" if ":" in hostname else hostname
    if value != f"https://{canonical_netloc}{parsed.path}":
        raise BundleSafetyError("invalid_provenance_url")
    return value


def _scan_structured_value(
    value: object,
    *,
    digest_fields: frozenset[str] = frozenset(),
    raw_digest_fields: frozenset[str] = frozenset(),
    anonymous_fields: frozenset[str] = frozenset(),
    url_fields: frozenset[str] = frozenset(),
    parent_field: str | None = None,
) -> None:
    if isinstance(value, str):
        if parent_field in digest_fields and _SHA256_PATTERN.fullmatch(value):
            return
        if parent_field in raw_digest_fields and _RAW_SHA256_PATTERN.fullmatch(value):
            return
        if parent_field in anonymous_fields and _REVIEWER_PATTERN.fullmatch(value):
            return
        if parent_field in url_fields:
            _validate_public_https_url(value)
            return
        _scan_sensitive_text(value)
        return
    if isinstance(value, list):
        for item in value:
            _scan_structured_value(
                item,
                digest_fields=digest_fields,
                raw_digest_fields=raw_digest_fields,
                anonymous_fields=anonymous_fields,
                url_fields=url_fields,
            )
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _scan_sensitive_text(str(name))
            _scan_structured_value(
                item,
                digest_fields=digest_fields,
                raw_digest_fields=raw_digest_fields,
                anonymous_fields=anonymous_fields,
                url_fields=url_fields,
                parent_field=str(name),
            )


def _validate_json_shape(value: object, category: str) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_NESTING:
            raise BundleSafetyError(category)
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _parse_json_bytes(
    payload: bytes,
    *,
    category: str,
    digest_fields: frozenset[str] = frozenset(),
    raw_digest_fields: frozenset[str] = frozenset(),
    anonymous_fields: frozenset[str] = frozenset(),
    url_fields: frozenset[str] = frozenset(),
) -> object:
    if len(payload) > MAX_JSON_BYTES:
        raise BundleSafetyError(category)
    text = _decode_scannable(payload)
    if any(len(line) > MAX_JSON_LINE_CHARS for line in text.splitlines()):
        raise BundleSafetyError(category)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        _validate_json_shape(value, category)
        _scan_structured_value(
            value,
            digest_fields=digest_fields,
            raw_digest_fields=raw_digest_fields,
            anonymous_fields=anonymous_fields,
            url_fields=url_fields,
        )
    except BundleSafetyError:
        raise
    except (
        json.JSONDecodeError,
        UnicodeError,
        ValueError,
        TypeError,
        RecursionError,
        OverflowError,
        _DuplicateJsonKey,
    ):
        raise BundleSafetyError(category) from None
    return value


def _parse_jsonl(
    payload: bytes,
    *,
    category: str,
    digest_fields: frozenset[str] = frozenset(),
    raw_digest_fields: frozenset[str] = frozenset(),
    url_fields: frozenset[str] = frozenset(),
    raw_scan: bool = True,
) -> list[dict[str, object]]:
    if len(payload) > MAX_JSON_BYTES:
        raise BundleSafetyError(category)
    text = _decode_scannable(payload)
    if raw_scan:
        _scan_sensitive_text(text)
    lines = text.splitlines()
    if (
        not lines
        or len(lines) > MAX_RECORDS
        or any(not line.strip() or len(line) > MAX_JSON_LINE_CHARS for line in lines)
    ):
        raise BundleSafetyError(category)
    records: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
            )
            _validate_json_shape(value, category)
            if not isinstance(value, dict):
                raise BundleSafetyError(category)
            _scan_structured_value(
                value,
                digest_fields=digest_fields,
                raw_digest_fields=raw_digest_fields,
                url_fields=url_fields,
            )
        except BundleSafetyError:
            raise
        except (
            json.JSONDecodeError,
            UnicodeError,
            ValueError,
            TypeError,
            RecursionError,
            OverflowError,
            _DuplicateJsonKey,
        ):
            raise BundleSafetyError(category) from None
        records.append(value)
    return records


def _canonical_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_CANONICAL_ID_PATTERN.fullmatch(value))


def _validate_record(record: Mapping[str, object], expected_split: str) -> None:
    if set(record) != RECORD_FIELDS:
        raise BundleSafetyError("invalid_dataset_record")
    if not _canonical_identifier(record.get("sample_id")):
        raise BundleSafetyError("invalid_dataset_record")
    if not _canonical_identifier(record.get("group_id")):
        raise BundleSafetyError("invalid_dataset_record")
    if record.get("split") != expected_split or record.get("rights_class") not in RIGHTS_CLASSES:
        raise BundleSafetyError("invalid_dataset_record")


def _canonicalize_and_validate_dataset(
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, bytes], list[dict[str, object]], dict[str, int], str]:
    if set(payloads) != set(SOURCE_DATA_NAMES):
        raise BundleSafetyError("invalid_dataset_payload_set")

    canonical: dict[str, bytes] = {}
    all_records: list[dict[str, object]] = []
    record_by_id: dict[str, dict[str, object]] = {}
    split_counts: dict[str, int] = {}
    for name, expected_split in EXPECTED_SPLITS.items():
        records = _parse_jsonl(payloads[name], category="invalid_dataset_jsonl")
        for record in records:
            _validate_record(record, expected_split)
            sample_id = str(record["sample_id"])
            if sample_id in record_by_id:
                raise BundleSafetyError("duplicate_sample_id")
            record_by_id[sample_id] = record
        all_records.extend(records)
        split_counts[expected_split] = len(records)
        canonical[name] = b"".join(_canonical_json_bytes(record) for record in records)

    if len(all_records) < 50:
        raise BundleSafetyError("insufficient_dataset_size")

    provenance = _parse_jsonl(
        payloads["provenance.jsonl"],
        category="invalid_provenance",
        digest_fields=frozenset({"record_sha256"}),
        raw_digest_fields=frozenset({"content_sha256"}),
        url_fields=frozenset({"canonical_url", "license_url"}),
        raw_scan=False,
    )
    provenance_ids: set[str] = set()
    canonical_provenance: list[dict[str, object]] = []
    for item in provenance:
        if item.get("schema_version") != "transfer-provenance.v1":
            raise BundleSafetyError("invalid_provenance")
        sample_id = item.get("sample_id")
        rights_class = item.get("rights_class")
        if (
            not isinstance(rights_class, str)
            or rights_class not in PROVENANCE_FIELDS_BY_RIGHTS
            or set(item) != PROVENANCE_FIELDS_BY_RIGHTS[rights_class]
        ):
            raise BundleSafetyError("invalid_provenance")
        license_id = item.get("license_id")
        record_digest = item.get("record_sha256")
        if (
            not _canonical_identifier(sample_id)
            or sample_id in provenance_ids
            or not isinstance(license_id, str)
            or not _LICENSE_PATTERN.fullmatch(license_id)
            or not isinstance(record_digest, str)
            or not _SHA256_PATTERN.fullmatch(record_digest)
        ):
            raise BundleSafetyError("invalid_provenance")
        record = record_by_id.get(str(sample_id))
        if record is None or record["rights_class"] != rights_class:
            raise BundleSafetyError("invalid_provenance")
        if record_digest != _sha256(_canonical_json_bytes(record)):
            raise BundleSafetyError("provenance_record_mismatch")

        normalized = dict(item)
        if rights_class == "open_licensed":
            content_digest = item.get("content_sha256")
            license_url = item.get("license_url")
            attribution = item.get("attribution")
            allowed_uses = item.get("allowed_uses")
            if (
                not isinstance(content_digest, str)
                or not _RAW_SHA256_PATTERN.fullmatch(content_digest)
                or not isinstance(attribution, str)
                or not _ATTRIBUTION_PATTERN.fullmatch(attribution)
                or len(attribution) > 500
            ):
                raise BundleSafetyError("invalid_provenance")
            _validate_public_https_url(item.get("canonical_url"))
            _validate_public_https_url(license_url)
            if (
                license_id not in OPEN_LICENSES
                or license_url != OPEN_LICENSES[license_id]
                or type(allowed_uses) is not list
                or not allowed_uses
                or any(type(use) is not str or use not in ALLOWED_USES for use in allowed_uses)
                or len(set(allowed_uses)) != len(allowed_uses)
            ):
                raise BundleSafetyError("invalid_provenance_license")
            retrieved_at = item.get("retrieved_at")
            if not isinstance(retrieved_at, str) or not _CANONICAL_TIMESTAMP_PATTERN.fullmatch(retrieved_at):
                raise BundleSafetyError("invalid_provenance_time")
            _parse_timestamp(retrieved_at, "invalid_provenance_time")
            allowed_use_set = set(allowed_uses)
            normalized["allowed_uses"] = [use for use in ALLOWED_USES if use in allowed_use_set]
        elif license_id != PRIVATE_LICENSES_BY_RIGHTS[rights_class]:
            raise BundleSafetyError("invalid_provenance_license")

        provenance_ids.add(str(sample_id))
        canonical_provenance.append(normalized)
    if provenance_ids != set(record_by_id):
        raise BundleSafetyError("invalid_provenance")
    canonical["provenance.jsonl"] = b"".join(
        _canonical_json_bytes(item)
        for item in sorted(canonical_provenance, key=lambda item: str(item["sample_id"]))
    )

    data_card = _scan_sensitive_payload(payloads["DATA_CARD.md"])
    if not data_card.strip():
        raise BundleSafetyError("invalid_data_card")
    canonical["DATA_CARD.md"] = (data_card.rstrip() + "\n").encode("utf-8")

    leakage_report = validate_candidate_records(all_records)
    if (
        leakage_report.status != "passed"
        or leakage_report.schema_validity < 0.99
        or leakage_report.finding_count != 0
    ):
        raise BundleSafetyError("dataset_gate_failed")

    dataset_digest = canonical_dataset_digest(canonical)
    return canonical, all_records, split_counts, dataset_digest


def _parse_timestamp(value: object, category: str) -> datetime:
    if not isinstance(value, str) or not _CANONICAL_TIMESTAMP_PATTERN.fullmatch(value):
        raise BundleSafetyError(category)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise BundleSafetyError(category) from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise BundleSafetyError(category)
    return parsed.astimezone(UTC)


def _validate_human_review(
    payload: bytes,
    *,
    records: Sequence[Mapping[str, object]],
    dataset_digest: str,
) -> None:
    review = _parse_json_bytes(
        payload,
        category="invalid_human_review",
        digest_fields=frozenset(
            {
                "dataset_digest",
                "sample_digest",
                "binding_digest",
            }
        ),
        anonymous_fields=frozenset({"reviewer_id"}),
    )
    if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
        raise BundleSafetyError("invalid_human_review")
    if (
        type(review.get("schema_version")) is not str
        or review.get("schema_version") != "transfer-human-review.v1"
    ):
        raise BundleSafetyError("invalid_human_review")
    reviewer_id = review.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not _REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise BundleSafetyError("reviewer_not_anonymous")
    sample_size = review.get("sample_size")
    sample_ids = review.get("sample_ids")
    if (
        type(sample_size) is not int
        or sample_size < 50
        or type(sample_ids) is not list
        or len(sample_ids) != sample_size
    ):
        raise BundleSafetyError("review_sample_size")
    if any(
        type(sample_id) is not str or not _CANONICAL_ID_PATTERN.fullmatch(sample_id)
        for sample_id in sample_ids
    ):
        raise BundleSafetyError("review_sample_size")
    if len(set(sample_ids)) != len(sample_ids):
        raise BundleSafetyError("review_sample_size")
    blocking_findings = review.get("blocking_findings")
    if type(blocking_findings) is not int:
        raise BundleSafetyError("invalid_human_review")
    if blocking_findings != 0:
        raise BundleSafetyError("review_blocked")
    if review.get("dataset_digest") != dataset_digest:
        raise BundleSafetyError("review_dataset_mismatch")
    seed = review.get("seed")
    if type(seed) is not str:
        raise BundleSafetyError("invalid_human_review")
    all_sample_ids = [str(record["sample_id"]) for record in records]
    expected_sample = deterministic_review_sample(
        all_sample_ids,
        dataset_digest=dataset_digest,
        seed=seed,
        sample_size=sample_size,
    )
    if sample_ids != expected_sample or review.get("sample_digest") != _sample_digest(expected_sample):
        raise BundleSafetyError("review_sample_mismatch")
    if review.get("binding_digest") != review_binding_digest(review):
        raise BundleSafetyError("review_binding_mismatch")

    reviewed_at = _parse_timestamp(review.get("reviewed_at"), "invalid_review_time")
    expires_at = _parse_timestamp(review.get("expires_at"), "invalid_review_time")
    now = datetime.now(UTC)
    if reviewed_at > now + timedelta(minutes=5):
        raise BundleSafetyError("invalid_review_time")
    if expires_at <= now:
        raise BundleSafetyError("review_expired")
    if expires_at <= reviewed_at or expires_at - reviewed_at > MAX_REVIEW_AGE:
        raise BundleSafetyError("invalid_review_time")


def _scan_report(
    *,
    sample_count: int,
    split_counts: Mapping[str, int],
    dataset_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": "secure-transfer-scan-report.v1",
        "status": "passed",
        "sample_count": sample_count,
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_digest": dataset_digest,
        "schema_validity": 1.0,
        "privacy_finding_count": 0,
        "leakage_finding_count": 0,
        "human_review_gate": "passed",
    }


def _checksum_payload(payloads: Mapping[str, bytes]) -> bytes:
    members = {name: _sha256(payloads[name]) for name in BUNDLE_MEMBER_NAMES if name != "SHA256SUMS.json"}
    members["SHA256SUMS.json"] = "sha256:SELF"
    manifest: dict[str, object] = {
        "schema_version": "secure-transfer-checksums.v1",
        "self_hash_scheme": "canonical-json-self-sentinel-v1",
        "members": {name: members[name] for name in BUNDLE_MEMBER_NAMES},
    }
    manifest_members = manifest["members"]
    if not isinstance(manifest_members, dict):
        raise BundleSafetyError("invalid_checksum_manifest")
    manifest_members["SHA256SUMS.json"] = _sha256(_canonical_json_bytes(manifest))
    return _canonical_json_bytes(manifest)


def _deterministic_tar(payloads: Mapping[str, bytes]) -> bytes:
    if set(payloads) != set(BUNDLE_MEMBER_NAMES):
        raise BundleSafetyError("invalid_bundle_payload_set")
    buffer = io.BytesIO()
    try:
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in BUNDLE_MEMBER_NAMES:
                payload = payloads[name]
                member = tarfile.TarInfo(name=name)
                member.size = len(payload)
                member.mtime = 0
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mode = 0o600
                member.type = tarfile.REGTYPE
                archive.addfile(member, io.BytesIO(payload))
    except (OSError, tarfile.TarError):
        raise BundleSafetyError("tar_build_failed") from None
    result = buffer.getvalue()
    if len(result) > MAX_TAR_BYTES:
        raise BundleSafetyError("bundle_too_large")
    return result


def rescan_built_tar_bytes(tar_bytes: bytes) -> dict[str, bytes]:
    if not isinstance(tar_bytes, bytes) or len(tar_bytes) > MAX_TAR_BYTES:
        raise BundleSafetyError("invalid_tar")
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(BUNDLE_MEMBER_NAMES):
                if len({member.name for member in members}) != len(members):
                    raise BundleSafetyError("duplicate_tar_member")
                raise BundleSafetyError("invalid_tar_member_count")
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) != 1
                    or member.name not in BUNDLE_MEMBER_NAMES
                ):
                    raise BundleSafetyError("invalid_tar_member")
                if member.name in payloads:
                    raise BundleSafetyError("duplicate_tar_member")
                if (
                    not member.isfile()
                    or member.linkname
                    or member.pax_headers
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mode != 0o600
                    or member.size < 0
                    or member.size > MAX_MEMBER_BYTES
                ):
                    raise BundleSafetyError("invalid_tar_member")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BundleSafetyError("invalid_tar_member")
                payload = extracted.read(MAX_MEMBER_BYTES + 1)
                if len(payload) != member.size or len(payload) > MAX_MEMBER_BYTES:
                    raise BundleSafetyError("invalid_tar_member")
                payloads[member.name] = payload
    except BundleSafetyError:
        raise
    except (OSError, tarfile.TarError, EOFError):
        raise BundleSafetyError("invalid_tar") from None
    if tuple(payloads) != BUNDLE_MEMBER_NAMES:
        raise BundleSafetyError("invalid_tar_member_order")
    if _deterministic_tar(payloads) != tar_bytes:
        raise BundleSafetyError("noncanonical_tar")
    return payloads


def verify_bundle_payloads(
    payloads: Mapping[str, bytes],
    *,
    expected_dataset_digest: str | None = None,
) -> TransferManifest:
    if set(payloads) != set(BUNDLE_MEMBER_NAMES):
        raise BundleSafetyError("invalid_bundle_payload_set")

    checksums = _parse_json_bytes(
        payloads["SHA256SUMS.json"],
        category="invalid_checksum_manifest",
        digest_fields=frozenset(BUNDLE_MEMBER_NAMES),
    )
    if (
        not isinstance(checksums, dict)
        or set(checksums) != {"schema_version", "self_hash_scheme", "members"}
        or checksums.get("schema_version") != "secure-transfer-checksums.v1"
        or checksums.get("self_hash_scheme") != "canonical-json-self-sentinel-v1"
        or not isinstance(checksums.get("members"), dict)
        or set(checksums["members"]) != set(BUNDLE_MEMBER_NAMES)
    ):
        raise BundleSafetyError("invalid_checksum_manifest")
    for name in BUNDLE_MEMBER_NAMES:
        value = checksums["members"].get(name)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise BundleSafetyError("invalid_checksum_manifest")
        if name != "SHA256SUMS.json" and value != _sha256(payloads[name]):
            raise BundleSafetyError("checksum_mismatch")
    projection = {
        "schema_version": checksums["schema_version"],
        "self_hash_scheme": checksums["self_hash_scheme"],
        "members": dict(checksums["members"]),
    }
    projection["members"]["SHA256SUMS.json"] = "sha256:SELF"
    if checksums["members"]["SHA256SUMS.json"] != _sha256(_canonical_json_bytes(projection)):
        raise BundleSafetyError("checksum_mismatch")

    dataset_payloads = {name: payloads[name] for name in SOURCE_DATA_NAMES}
    canonical, records, split_counts, dataset_digest = _canonicalize_and_validate_dataset(dataset_payloads)
    if canonical != dataset_payloads:
        raise BundleSafetyError("noncanonical_bundle_content")
    if expected_dataset_digest is not None and dataset_digest != expected_dataset_digest:
        raise BundleSafetyError("dataset_digest_mismatch")

    report = _parse_json_bytes(
        payloads["scan-report.json"],
        category="invalid_scan_report",
        digest_fields=frozenset({"dataset_digest"}),
    )
    expected_report = _scan_report(
        sample_count=len(records),
        split_counts=split_counts,
        dataset_digest=dataset_digest,
    )
    if report != expected_report:
        raise BundleSafetyError("invalid_scan_report")
    return TransferManifest(
        safe=True,
        member_count=len(payloads),
        sample_count=len(records),
        split_counts=split_counts,
        dataset_digest=dataset_digest,
        bundle_sha256="",
    )


def _validate_output_leaf_name(name: str) -> None:
    normalized = unicodedata.normalize("NFKC", name)
    stem = name.split(".", 1)[0].upper()
    if (
        not name
        or name in {".", ".."}
        or name != normalized
        or name != Path(name).name
        or name[-1] in {".", " "}
        or stem in _WINDOWS_RESERVED_STEMS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name)
        or any(character in _WINDOWS_INVALID_NAME_CHARS for character in name)
    ):
        raise BundleSafetyError("unsafe_output")


def _validate_output_parent(output_tar: Path) -> tuple[Path, str]:
    _validate_output_leaf_name(output_tar.name)
    if not output_tar.name or output_tar.name != Path(output_tar.name).name:
        raise BundleSafetyError("unsafe_output")
    parent = output_tar.parent
    try:
        metadata = os.lstat(parent)
    except OSError:
        raise BundleSafetyError("unsafe_output") from None
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise BundleSafetyError("unsafe_output")
    try:
        leaf = os.lstat(output_tar)
    except FileNotFoundError:
        pass
    except OSError:
        raise BundleSafetyError("unsafe_output") from None
    else:
        _assert_regular_unique(leaf, "unsafe_output")
        raise BundleSafetyError("unsafe_output")
    return parent, output_tar.name


def _paths_overlap(input_root: Path, output_tar: Path) -> bool:
    lexical_input = Path(os.path.abspath(input_root))
    lexical_output = Path(os.path.abspath(output_tar))
    if lexical_output == lexical_input or lexical_output.is_relative_to(lexical_input):
        return True
    try:
        resolved_input = input_root.resolve(strict=True)
        resolved_parent = output_tar.parent.resolve(strict=True)
    except OSError:
        return False
    return resolved_parent == resolved_input or resolved_parent.is_relative_to(resolved_input)


@dataclass
class _OpenTransferTemp:
    parent: Path
    name: str
    descriptor: int
    directory_fd: int | None
    parent_identity: DirectoryIdentity
    file_identity: tuple[int, int]
    final_name: str
    windows_directory_handle: object | None = None
    final_created: bool = False
    published: bool = False
    unnamed: bool = False
    delete_registered: bool = False

    @property
    def path(self) -> Path:
        return self.parent / self.name


def _assert_output_parent_bound(
    parent: Path,
    expected_identity: DirectoryIdentity,
    directory_fd: int | None,
) -> None:
    try:
        if directory_fd is not None:
            held_metadata = os.fstat(directory_fd)
            path_metadata = os.stat(parent, follow_symlinks=False)
            held_identity = DirectoryIdentity(held_metadata.st_dev, held_metadata.st_ino)
            path_identity = DirectoryIdentity(path_metadata.st_dev, path_metadata.st_ino)
            path_attributes = getattr(path_metadata, "st_file_attributes", 0)
            if (
                not stat.S_ISDIR(held_metadata.st_mode)
                or not stat.S_ISDIR(path_metadata.st_mode)
                or path_attributes & _REPARSE_POINT_FLAG
                or held_identity != path_identity
                or held_metadata.st_mode != path_metadata.st_mode
            ):
                raise BundleSafetyError("unsafe_output")
            identity = held_identity
        else:
            identity = secure_write.directory_identity(parent)
    except (OSError, SecureWriteError):
        raise BundleSafetyError("unsafe_output") from None
    if identity != expected_identity:
        raise BundleSafetyError("unsafe_output")


def _create_open_transfer_temp(
    parent: Path,
    *,
    final_name: str,
    directory_fd: int | None,
    parent_identity: DirectoryIdentity,
) -> _OpenTransferTemp:
    descriptor = -1
    if os.name != "nt":
        if directory_fd is None or not _O_TMPFILE:
            raise BundleSafetyError("secure_write_failed")
        flags = os.O_RDWR | _O_TMPFILE | _O_CLOEXEC
        try:
            descriptor = os.open(".", flags, 0o600, dir_fd=directory_fd)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 0 or metadata.st_size != 0:
                raise BundleSafetyError("secure_write_failed")
        except BundleSafetyError:
            try:
                if descriptor >= 0:
                    os.close(descriptor)
            except OSError:
                raise BundleSafetyError("secure_cleanup_failed") from None
            raise
        except OSError:
            try:
                if descriptor >= 0:
                    os.close(descriptor)
            except OSError:
                raise BundleSafetyError("secure_cleanup_failed") from None
            raise BundleSafetyError("secure_write_failed") from None
        return _OpenTransferTemp(
            parent=parent,
            name="",
            descriptor=descriptor,
            directory_fd=directory_fd,
            parent_identity=parent_identity,
            file_identity=_identity(metadata),
            final_name=final_name,
            unnamed=True,
        )

    temporary_name = ""
    windows_directory_handle: object | None = None
    if os.name == "nt" and not callable(getattr(secure_write, "_SET_FILE_INFORMATION", None)):
        raise BundleSafetyError("secure_write_failed")
    if os.name == "nt":
        try:
            windows_directory_handle = secure_write._open_windows_directory(  # type: ignore[attr-defined]
                parent
            )
            if (
                secure_write._windows_identity(  # type: ignore[attr-defined]
                    windows_directory_handle
                )
                != parent_identity
            ):
                raise BundleSafetyError("unsafe_output")
        except (OSError, SecureWriteError, BundleSafetyError):
            if windows_directory_handle is not None:
                _close_windows_handle(windows_directory_handle)
            raise BundleSafetyError("unsafe_output") from None
    for _ in range(32):
        temporary_name = f".private-transfer-{secrets.token_hex(16)}.tmp"
        try:
            if os.name == "nt":
                create_file = secure_write._CREATE_FILE  # type: ignore[attr-defined]
                invalid_handle = secure_write._INVALID_HANDLE_VALUE  # type: ignore[attr-defined]
                raw_handle = create_file(
                    str(parent / temporary_name),
                    0x80000000 | 0x40000000 | 0x00010000,
                    0x00000001,
                    None,
                    1,
                    0x00000080 | 0x00200000,
                    None,
                )
                if raw_handle == invalid_handle:
                    error_code = ctypes.get_last_error()
                    if error_code in {80, 183}:
                        continue
                    raise BundleSafetyError("secure_write_failed")
                try:
                    _discard_windows_native_handle(raw_handle)
                except BundleSafetyError:
                    _close_windows_handle(raw_handle)
                    raise
                try:
                    descriptor = msvcrt.open_osfhandle(
                        raw_handle,
                        os.O_RDWR | getattr(os, "O_BINARY", 0),
                    )
                except OSError:
                    _close_windows_handle(raw_handle)
                    raise BundleSafetyError("secure_write_failed") from None
            break
        except FileExistsError:
            continue
        except BundleSafetyError as exc:
            if windows_directory_handle is not None:
                _close_windows_handle(windows_directory_handle)
            raise exc
        except OSError:
            if windows_directory_handle is not None:
                _close_windows_handle(windows_directory_handle)
            raise BundleSafetyError("secure_write_failed") from None
    if descriptor < 0:
        if windows_directory_handle is not None:
            _close_windows_handle(windows_directory_handle)
        raise BundleSafetyError("secure_write_failed")
    try:
        metadata = os.fstat(descriptor)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink not in {0, 1}
            or metadata.st_size != 0
        ):
            raise BundleSafetyError("secure_write_failed")
        return _OpenTransferTemp(
            parent=parent,
            name=temporary_name,
            descriptor=descriptor,
            directory_fd=directory_fd,
            parent_identity=parent_identity,
            file_identity=_identity(metadata),
            final_name=final_name,
            windows_directory_handle=windows_directory_handle,
            delete_registered=True,
        )
    except Exception as exc:
        cleanup_failed = False
        try:
            os.close(descriptor)
        except OSError:
            cleanup_failed = True
        if windows_directory_handle is not None:
            try:
                _close_windows_handle(windows_directory_handle)
            except BundleSafetyError:
                cleanup_failed = True
        if cleanup_failed:
            raise BundleSafetyError("secure_cleanup_failed") from None
        if isinstance(exc, BundleSafetyError):
            raise
        raise BundleSafetyError("secure_write_failed") from None


def _write_open_transfer_temp(temporary: _OpenTransferTemp, payload: bytes) -> None:
    try:
        os.lseek(temporary.descriptor, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            written = os.write(temporary.descriptor, view)
            if written <= 0:
                raise BundleSafetyError("secure_write_failed")
            view = view[written:]
        os.fsync(temporary.descriptor)
    except BundleSafetyError:
        raise
    except OSError:
        raise BundleSafetyError("secure_write_failed") from None


def _read_open_transfer_temp(
    temporary: _OpenTransferTemp,
    *,
    max_bytes: int = MAX_TAR_BYTES,
) -> bytes:
    try:
        before = os.fstat(temporary.descriptor)
        if (
            _identity(before) != temporary.file_identity
            or not stat.S_ISREG(before.st_mode)
            or before.st_size > max_bytes
        ):
            raise BundleSafetyError("second_pass_failed")
        os.lseek(temporary.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(temporary.descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(temporary.descriptor)
    except BundleSafetyError:
        raise
    except OSError:
        raise BundleSafetyError("second_pass_failed") from None
    if (
        len(payload) > max_bytes
        or len(payload) != before.st_size
        or _identity(after) != temporary.file_identity
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise BundleSafetyError("second_pass_failed")
    return payload


def _stat_transfer_name(temporary: _OpenTransferTemp, name: str) -> os.stat_result:
    try:
        return (
            os.stat(name, dir_fd=temporary.directory_fd, follow_symlinks=False)
            if temporary.directory_fd is not None
            else os.lstat(temporary.parent / name)
        )
    except (OSError, SecureWriteError):
        raise BundleSafetyError("second_pass_failed") from None


def _assert_open_temp_path_bound(temporary: _OpenTransferTemp) -> None:
    if temporary.unnamed or temporary.delete_registered:
        try:
            metadata = os.fstat(temporary.descriptor)
        except OSError:
            raise BundleSafetyError("second_pass_failed") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or _identity(metadata) != temporary.file_identity
        ):
            raise BundleSafetyError("second_pass_failed")
        return
    metadata = _stat_transfer_name(temporary, temporary.name)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or _identity(metadata) != temporary.file_identity
    ):
        raise BundleSafetyError("second_pass_failed")


def _unlink_transfer_name(temporary: _OpenTransferTemp, name: str) -> None:
    if temporary.directory_fd is not None:
        os.unlink(name, dir_fd=temporary.directory_fd)
    else:
        os.unlink(temporary.parent / name)


def _link_posix_unnamed_inode(temporary: _OpenTransferTemp) -> None:
    if temporary.directory_fd is None or not callable(_LIBC_LINKAT):
        raise BundleSafetyError("secure_write_failed")
    ctypes.set_errno(0)
    result = _LIBC_LINKAT(
        temporary.descriptor,
        b"",
        temporary.directory_fd,
        os.fsencode(temporary.final_name),
        _AT_EMPTY_PATH,
    )
    if result == 0:
        return
    error_code = ctypes.get_errno()
    if error_code == errno.EEXIST:
        raise BundleSafetyError("unsafe_output") from None
    raise BundleSafetyError("secure_write_failed") from None


def _assert_published_final_bound(
    temporary: _OpenTransferTemp,
    *,
    expected_size: int | None = None,
) -> None:
    metadata = _stat_transfer_name(temporary, temporary.final_name)
    _assert_regular_unique(metadata, "second_pass_failed")
    if _identity(metadata) != temporary.file_identity or (
        expected_size is not None and metadata.st_size != expected_size
    ):
        raise BundleSafetyError("second_pass_failed")


def _publish_verified_temp(temporary: _OpenTransferTemp) -> None:
    _assert_output_parent_bound(
        temporary.parent,
        temporary.parent_identity,
        temporary.directory_fd,
    )
    _assert_open_temp_path_bound(temporary)
    try:
        if os.name == "nt":
            _retain_windows_transfer_file(temporary)
            _rename_windows_temp_no_replace(temporary)
            temporary.final_created = True
        elif temporary.directory_fd is not None:
            _link_posix_unnamed_inode(temporary)
            temporary.final_created = True
            os.fsync(temporary.directory_fd)
        else:
            raise BundleSafetyError("secure_write_failed")
    except FileExistsError:
        raise BundleSafetyError("unsafe_output") from None
    except OSError:
        raise BundleSafetyError("secure_write_failed") from None


def _close_windows_handle(handle: object) -> None:
    try:
        closed = secure_write._CLOSE_HANDLE(handle)  # type: ignore[attr-defined]
    except Exception:
        raise BundleSafetyError("secure_cleanup_failed") from None
    if not closed:
        raise BundleSafetyError("secure_cleanup_failed")


def _rename_windows_temp_no_replace(temporary: _OpenTransferTemp) -> None:
    if os.name != "nt" or temporary.windows_directory_handle is None:
        raise BundleSafetyError("secure_write_failed")
    set_information = getattr(secure_write, "_SET_FILE_INFORMATION", None)
    if not callable(set_information):
        raise BundleSafetyError("secure_write_failed")
    information_type = secure_write._FileRenameInformation  # type: ignore[attr-defined]
    encoded_name = os.path.abspath(temporary.parent / temporary.final_name).encode("utf-16-le")
    file_name_offset = information_type.file_name.offset
    buffer_size = file_name_offset + len(encoded_name) + ctypes.sizeof(ctypes.c_wchar)
    buffer = ctypes.create_string_buffer(buffer_size)
    ctypes.c_uint32.from_buffer(buffer).value = 0
    information = ctypes.cast(buffer, ctypes.POINTER(information_type)).contents
    information.root_directory = None
    information.file_name_length = len(encoded_name)
    ctypes.memmove(ctypes.addressof(buffer) + file_name_offset, encoded_name, len(encoded_name))
    native_handle = msvcrt.get_osfhandle(temporary.descriptor)
    ctypes.set_last_error(0)
    if set_information(
        native_handle,
        _WINDOWS_FILE_RENAME_INFO_EX,
        buffer,
        buffer_size,
    ):
        return
    error_code = ctypes.get_last_error()
    if error_code in _WINDOWS_TARGET_EXISTS_ERRORS:
        raise BundleSafetyError("unsafe_output") from None
    if error_code not in _WINDOWS_EXTENDED_INFO_UNAVAILABLE_ERRORS:
        raise BundleSafetyError("secure_write_failed") from None
    ctypes.c_uint8.from_buffer(buffer).value = 0
    ctypes.set_last_error(0)
    if set_information(
        native_handle,
        _WINDOWS_FILE_RENAME_INFO,
        buffer,
        buffer_size,
    ):
        return
    error_code = ctypes.get_last_error()
    if error_code in _WINDOWS_TARGET_EXISTS_ERRORS:
        raise BundleSafetyError("unsafe_output") from None
    raise BundleSafetyError("secure_write_failed") from None


def _discard_windows_native_handle(native_handle: object) -> None:
    set_information = getattr(secure_write, "_SET_FILE_INFORMATION", None)
    if not callable(set_information):
        raise BundleSafetyError("secure_cleanup_failed")
    flags = ctypes.c_uint32(_WINDOWS_FILE_DISPOSITION_DELETE)
    if set_information(
        native_handle,
        _WINDOWS_FILE_DISPOSITION_INFO_EX,
        ctypes.byref(flags),
        ctypes.sizeof(flags),
    ):
        return
    try:
        secure_write._discard_windows_file(native_handle)  # type: ignore[attr-defined]
    except SecureWriteError:
        raise BundleSafetyError("secure_cleanup_failed") from None


def _discard_windows_transfer_file(temporary: _OpenTransferTemp) -> None:
    _discard_windows_native_handle(msvcrt.get_osfhandle(temporary.descriptor))
    temporary.delete_registered = True


def _retain_windows_transfer_file(temporary: _OpenTransferTemp) -> None:
    set_information = getattr(secure_write, "_SET_FILE_INFORMATION", None)
    if not callable(set_information):
        raise BundleSafetyError("secure_write_failed")
    native_handle = msvcrt.get_osfhandle(temporary.descriptor)
    flags = ctypes.c_uint32(0)
    ctypes.set_last_error(0)
    if set_information(
        native_handle,
        _WINDOWS_FILE_DISPOSITION_INFO_EX,
        ctypes.byref(flags),
        ctypes.sizeof(flags),
    ):
        temporary.delete_registered = False
        return
    if ctypes.get_last_error() not in _WINDOWS_EXTENDED_INFO_UNAVAILABLE_ERRORS:
        raise BundleSafetyError("secure_write_failed")
    disposition = secure_write._FileDispositionInformation(  # type: ignore[attr-defined]
        delete_file=0
    )
    if not set_information(
        native_handle,
        _WINDOWS_FILE_DISPOSITION_INFO,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise BundleSafetyError("secure_write_failed")
    temporary.delete_registered = False


def _cleanup_posix_transfer_file(temporary: _OpenTransferTemp) -> None:
    if not temporary.final_created or temporary.published:
        return
    try:
        metadata = _stat_transfer_name(temporary, temporary.final_name)
    except BundleSafetyError:
        raise BundleSafetyError("secure_cleanup_failed") from None
    if _identity(metadata) != temporary.file_identity:
        raise BundleSafetyError("secure_cleanup_failed")
    try:
        _unlink_transfer_name(temporary, temporary.final_name)
    except OSError:
        raise BundleSafetyError("secure_cleanup_failed") from None


def _cleanup_transfer_temp(temporary: _OpenTransferTemp) -> None:
    cleanup_failed = False
    if os.name == "nt":
        try:
            if not temporary.published:
                _discard_windows_transfer_file(temporary)
        except (OSError, BundleSafetyError):
            cleanup_failed = True
    else:
        try:
            _cleanup_posix_transfer_file(temporary)
        except BundleSafetyError:
            cleanup_failed = True
    try:
        os.close(temporary.descriptor)
    except OSError:
        cleanup_failed = True
    if temporary.windows_directory_handle is not None:
        try:
            _close_windows_handle(temporary.windows_directory_handle)
        except BundleSafetyError:
            cleanup_failed = True
    if cleanup_failed:
        raise BundleSafetyError("secure_cleanup_failed")


def _commit_verified_output(
    *,
    output_parent: Path,
    output_name: str,
    output_directory_fd: int | None,
    output_identity: DirectoryIdentity,
    tar_bytes: bytes,
    dataset_digest: str,
    first_manifest: TransferManifest,
) -> TransferManifest:
    temporary = _create_open_transfer_temp(
        output_parent,
        final_name=output_name,
        directory_fd=output_directory_fd,
        parent_identity=output_identity,
    )
    result: TransferManifest | None = None
    try:
        _write_open_transfer_temp(temporary, tar_bytes)
        reopened = _read_open_transfer_temp(temporary)
        if _sha256(reopened) != _sha256(tar_bytes):
            raise BundleSafetyError("second_pass_failed")
        second_manifest = verify_bundle_payloads(
            rescan_built_tar_bytes(reopened),
            expected_dataset_digest=dataset_digest,
        )
        if second_manifest.model_dump() != first_manifest.model_dump():
            raise BundleSafetyError("second_pass_failed")

        _assert_open_temp_path_bound(temporary)
        prepublish = _read_open_transfer_temp(temporary)
        if _sha256(prepublish) != _sha256(tar_bytes):
            raise BundleSafetyError("second_pass_failed")
        _publish_verified_temp(temporary)

        published_payload = _read_open_transfer_temp(temporary)
        if _sha256(published_payload) != _sha256(tar_bytes):
            raise BundleSafetyError("second_pass_failed")
        _assert_published_final_bound(temporary, expected_size=len(tar_bytes))
        _assert_output_parent_bound(
            output_parent,
            output_identity,
            output_directory_fd,
        )
        _assert_published_final_bound(temporary, expected_size=len(tar_bytes))
        result = TransferManifest(
            safe=True,
            member_count=len(BUNDLE_MEMBER_NAMES),
            sample_count=first_manifest.sample_count,
            split_counts=first_manifest.split_counts,
            dataset_digest=dataset_digest,
            bundle_sha256=_sha256(tar_bytes),
        )
        _assert_output_parent_bound(
            output_parent,
            output_identity,
            output_directory_fd,
        )
        _assert_published_final_bound(temporary, expected_size=len(tar_bytes))
        temporary.published = True
    except BundleSafetyError as exc:
        if exc.category in {"unsafe_output", "secure_write_failed", "secure_cleanup_failed"}:
            raise
        raise BundleSafetyError("second_pass_failed") from None
    finally:
        _cleanup_transfer_temp(temporary)
    if result is None:
        raise BundleSafetyError("second_pass_failed")
    return result


def build_secure_transfer(input_root: Path, output_tar: Path) -> TransferManifest:
    input_root = Path(input_root)
    output_tar = Path(output_tar)
    _validate_output_leaf_name(output_tar.name)
    if _paths_overlap(input_root, output_tar):
        raise BundleSafetyError("input_output_overlap")

    source_payloads = _read_source_tree(input_root)
    if _paths_overlap(input_root, output_tar):
        raise BundleSafetyError("input_output_overlap")
    output_parent, output_name = _validate_output_parent(output_tar)
    dataset_payloads = {name: source_payloads[name] for name in SOURCE_DATA_NAMES}
    canonical, records, split_counts, dataset_digest = _canonicalize_and_validate_dataset(dataset_payloads)
    _validate_human_review(
        source_payloads["human_review.json"],
        records=records,
        dataset_digest=dataset_digest,
    )

    output_payloads = dict(canonical)
    output_payloads["scan-report.json"] = _canonical_json_bytes(
        _scan_report(
            sample_count=len(records),
            split_counts=split_counts,
            dataset_digest=dataset_digest,
        )
    )
    output_payloads["SHA256SUMS.json"] = _checksum_payload(output_payloads)
    tar_bytes = _deterministic_tar(output_payloads)
    first_manifest = verify_bundle_payloads(
        rescan_built_tar_bytes(tar_bytes),
        expected_dataset_digest=dataset_digest,
    )

    with _bound_directory(output_parent, category="unsafe_output") as (
        output_identity,
        output_directory_fd,
    ):
        return _commit_verified_output(
            output_parent=output_parent,
            output_name=output_name,
            output_directory_fd=output_directory_fd,
            output_identity=output_identity,
            tar_bytes=tar_bytes,
            dataset_digest=dataset_digest,
            first_manifest=first_manifest,
        )
