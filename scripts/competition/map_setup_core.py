"""Pure helpers for the one-key MAP deployment wizard.

This module deliberately does not execute subprocesses or read user data.  It
only validates the committed public defaults and redacts operator summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REQUIRED_CONFIG_KEYS = {"name", "main", "compatibility_date", "vars"}
_REQUIRED_VAR_KEYS = {
    "ALLOWED_ORIGINS",
    "MAP_REALTIME_URL",
    "MAP_AUTH_HEADER",
    "MAP_AUTH_PREFIX",
    "MAP_SESSION_SECONDS",
}
_SUMMARY_KEYS = {
    "schema_version",
    "worker_url",
    "pages_url",
    "health_status",
    "chat_smoke",
    "started_at",
    "finished_at",
    "close_reason",
}
_FORBIDDEN_KEY_TERMS = {"api_key", "authorization", "token", "secret", "password"}
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]")


@dataclass(frozen=True)
class OneKeyDefaults:
    """Public, non-secret values used by the setup wizard."""

    worker_name: str
    pages_project: str
    pages_origin: str
    map_realtime_url: str
    session_seconds: int
    auth_header: str
    auth_prefix: str


def _strip_full_line_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _require_exact_keys(mapping: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(mapping)
    extra = set(mapping) - required
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def load_defaults(config_path: Path) -> OneKeyDefaults:
    """Load the committed JSONC config while rejecting unexpected fields."""

    config = json.loads(_strip_full_line_comments(Path(config_path).read_text(encoding="utf-8")))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    _require_exact_keys(config, _REQUIRED_CONFIG_KEYS, "config")
    variables = config["vars"]
    if not isinstance(variables, dict):
        raise ValueError("vars must be an object")
    _require_exact_keys(variables, _REQUIRED_VAR_KEYS, "vars")
    defaults = OneKeyDefaults(
        worker_name=str(config["name"]),
        pages_project="gradloop-ragsdk-omni",
        pages_origin=str(variables["ALLOWED_ORIGINS"]),
        map_realtime_url=str(variables["MAP_REALTIME_URL"]),
        session_seconds=int(variables["MAP_SESSION_SECONDS"]),
        auth_header=str(variables["MAP_AUTH_HEADER"]),
        auth_prefix=str(variables["MAP_AUTH_PREFIX"]),
    )
    validate_defaults(defaults)
    return defaults


def validate_defaults(defaults: OneKeyDefaults) -> None:
    """Reject credentials, loose origins and unsupported auth settings."""

    map_url = urlparse(defaults.map_realtime_url)
    if (
        map_url.scheme != "wss"
        or not map_url.netloc
        or map_url.username
        or map_url.password
        or map_url.query
        or map_url.fragment
    ):
        raise ValueError("map_realtime_url must be a credential-free query-free wss URL")
    pages = urlparse(defaults.pages_origin)
    if (
        pages.scheme != "https"
        or not pages.netloc
        or pages.username
        or pages.password
        or pages.path not in {"", "/"}
        or pages.params
        or pages.query
        or pages.fragment
    ):
        raise ValueError("pages_origin must be an exact HTTPS origin")
    if not 30 <= defaults.session_seconds <= 600:
        raise ValueError("session_seconds must be between 30 and 600")
    if defaults.auth_header != "Authorization" or defaults.auth_prefix != "Bearer":
        raise ValueError("one-key setup supports only Authorization Bearer")
    if not defaults.worker_name or not defaults.pages_project:
        raise ValueError("public project names must not be empty")


def build_public_plan(root: Path, pages_output: Path, defaults: OneKeyDefaults) -> dict[str, object]:
    """Return a deterministic, secret-free ordered action plan."""

    del root, pages_output
    validate_defaults(defaults)
    return {
        "schema_version": "one-key-map-setup-plan.v1",
        "worker_name": defaults.worker_name,
        "pages_project": defaults.pages_project,
        "pages_origin": defaults.pages_origin,
        "map_realtime_url": defaults.map_realtime_url,
        "session_seconds": defaults.session_seconds,
        "actions": [
            "preflight",
            "build_validation_pages",
            "cloudflare_login",
            "deploy_worker",
            "set_worker_secret",
            "build_final_pages",
            "deploy_pages",
            "health",
            "chat_smoke",
        ],
    }


def _contains_forbidden_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).casefold()
            if any(term in key_text for term in _FORBIDDEN_KEY_TERMS):
                return True
            if _contains_forbidden_value(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_value(item) for item in value)
    if isinstance(value, str):
        return bool(_WINDOWS_PATH.search(value) or value.startswith(("/", "\\")) or "?" in value)
    return False


def sanitize_public_summary(payload: Mapping[str, object]) -> dict[str, object]:
    """Allow only aggregate public fields and reject content-bearing fields."""

    unknown = set(payload) - _SUMMARY_KEYS
    if unknown:
        raise ValueError(f"summary contains unsupported fields: {sorted(unknown)}")
    if _contains_forbidden_value(payload):
        raise ValueError("summary contains forbidden credential, path or query data")
    sanitized = dict(payload)
    for key in ("worker_url", "pages_url"):
        if key in sanitized:
            parsed = urlparse(str(sanitized[key]))
            if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"{key} must be a query-free HTTPS URL")
    return sanitized


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the public one-key MAP setup plan")
    parser.add_argument("plan", choices=["plan"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    defaults = load_defaults(args.config)
    print(json.dumps(build_public_plan(args.root, args.output, defaults), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
