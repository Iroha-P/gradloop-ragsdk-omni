from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

_EMAIL = re.compile(r"(?i)(?<![\w.-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)\b\s*[:=]\s*([^\s,;]+)"
)
_SENSITIVE_KEYS = {
    "answer",
    "api_key",
    "authorization",
    "content",
    "context",
    "cookie",
    "document_text",
    "password",
    "prompt",
    "query",
    "secret",
    "system_prompt",
    "token",
    "user_input",
}


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def redact_text(value: str) -> str:
    value = _EMAIL.sub("<redacted:email>", value)
    value = _PHONE.sub("<redacted:phone>", value)
    value = _ID_CARD.sub("<redacted:id>", value)
    return _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted:secret>", value)


def _safe_value(key: str, value: Any) -> Any:
    normalized_key = key.lower()
    if normalized_key in _SENSITIVE_KEYS:
        text = str(value)
        return {"redacted": True, "length": len(text), "sha256_12": stable_digest(text)}
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return sanitize_fields(value)
    if isinstance(value, list):
        return [_safe_value("item", item) for item in value[:50]]
    return value


def sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: _safe_value(key, value) for key, value in fields.items()}


def json_log(event: str, **fields: Any) -> str:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **sanitize_fields(fields),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(message)s", force=True)

