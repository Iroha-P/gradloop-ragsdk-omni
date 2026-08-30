from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

READ_ONLY_PREFIXES = {
    ("lark-cli", "auth", "status"),
    ("lark-cli", "wiki", "+space-list"),
    ("lark-cli", "wiki", "+node-list"),
    ("lark-cli", "docs", "+fetch"),
}
REQUIRED_READ_SCOPES = frozenset(
    {"wiki:space:retrieve", "wiki:node:retrieve", "docx:document:readonly"}
)


class FeishuReadError(RuntimeError):
    """A sanitized failure at the read-only Feishu boundary."""


class FeishuModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeishuAuthState(FeishuModel):
    verified: bool
    user_active: bool
    scopes: frozenset[str] = Field(default_factory=frozenset)


class FeishuNode(FeishuModel):
    node_token: str
    obj_token: str
    obj_type: str
    parent_node_token: str = ""
    node_type: str
    title: str
    has_child: bool = False


class FeishuSpace(FeishuModel):
    space_id: str
    name: str


class FeishuDocument(FeishuModel):
    obj_token: str
    revision_id: int
    markdown: str


class CliJsonRunner(Protocol):
    def run_json(self, argv: tuple[str, ...]) -> dict: ...


class FeishuReadClient(Protocol):
    def verify_auth(self) -> FeishuAuthState: ...

    def list_spaces(self) -> list[FeishuSpace]: ...

    def list_children(
        self, space_id: str, parent_node_token: str = ""
    ) -> list[FeishuNode]: ...

    def fetch_docx(self, obj_token: str) -> FeishuDocument: ...


_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_SECRET_FIELD = re.compile(
    r"(?i)(access_token|refresh_token|device_code|disposable_login_token|token|content)"
    r"\s*[=:]\s*(?:\"[^\"]*\"|\S+)"
)
_REMOTE_ID = re.compile(r"(?i)\b(?:wikcn|doccn|doxcn|blkcn)[A-Za-z0-9_-]+\b")


def sanitize_cli_error(value: str) -> str:
    cleaned = _URL.sub("<redacted:url>", value)
    cleaned = _SECRET_FIELD.sub(lambda match: f"{match.group(1)}=<redacted>", cleaned)
    cleaned = _REMOTE_ID.sub("<redacted:remote_id>", cleaned)
    return cleaned[:1000]


def windows_safe_command(
    argv: tuple[str, ...],
    *,
    resolved_executable: str | None = None,
    platform_name: str | None = None,
) -> tuple[str, ...]:
    platform = platform_name or os.name
    resolved = resolved_executable or shutil.which(argv[0]) or argv[0]
    if platform == "nt" and resolved.casefold().endswith(".ps1"):
        return (
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            resolved,
            *argv[1:],
        )
    return (resolved, *argv[1:])


@dataclass(frozen=True)
class SubprocessJsonRunner:
    timeout_seconds: int = 120

    def run_json(self, argv: tuple[str, ...]) -> dict:
        env = {
            **os.environ,
            "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
            "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
        }
        completed = subprocess.run(
            windows_safe_command(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode:
            raise FeishuReadError(sanitize_cli_error(completed.stderr or completed.stdout))
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise FeishuReadError("lark-cli returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise FeishuReadError("lark-cli returned a non-object JSON response")
        return payload


class LarkCliFeishuClient:
    """Minimal lark-cli adapter that cannot construct a write command."""

    def __init__(self, runner: CliJsonRunner | None = None):
        self.runner = runner or SubprocessJsonRunner()

    def _run(self, argv: tuple[str, ...]) -> dict:
        if argv[:3] not in READ_ONLY_PREFIXES:
            raise FeishuReadError("command is outside the read-only allowlist")
        payload = self.runner.run_json(argv)
        if payload.get("ok") is False:
            error = json.dumps(payload.get("error", {}), ensure_ascii=False)
            raise FeishuReadError(sanitize_cli_error(error))
        return payload

    def verify_auth(self) -> FeishuAuthState:
        payload = self._run(("lark-cli", "auth", "status", "--json", "--verify"))
        data = payload.get("data", {})
        identities = payload.get("identities") or data.get("identities", {})
        user = identities.get("user", {}) if isinstance(identities, dict) else {}
        raw_scopes = user.get("scope", user.get("scopes", []))
        scopes = (
            frozenset(raw_scopes.split())
            if isinstance(raw_scopes, str)
            else frozenset(raw_scopes or [])
        )
        verified = bool(payload.get("verified", data.get("verified", user.get("verified", False))))
        status = str(user.get("status", "")).casefold()
        state = FeishuAuthState(
            verified=verified,
            user_active=status in {"active", "ready", "valid"},
            scopes=scopes,
        )
        if not state.verified or not state.user_active:
            raise FeishuReadError("feishu_auth_required")
        missing = sorted(REQUIRED_READ_SCOPES - state.scopes)
        if missing:
            raise FeishuReadError("missing_read_scope:" + ",".join(missing))
        return state

    def list_spaces(self) -> list[FeishuSpace]:
        spaces: list[FeishuSpace] = []
        page_token = ""
        while True:
            argv = [
                "lark-cli",
                "wiki",
                "+space-list",
                "--page-size",
                "50",
                "--as",
                "user",
                "--format",
                "json",
            ]
            if page_token:
                argv.extend(("--page-token", page_token))
            payload = self._run(tuple(argv))
            data = payload.get("data", {})
            items = data.get("items", data.get("spaces", []))
            spaces.extend(
                FeishuSpace(space_id=str(item["space_id"]), name=str(item["name"]))
                for item in items
            )
            if not data.get("has_more"):
                return spaces
            page_token = str(data.get("page_token", ""))
            if not page_token:
                raise FeishuReadError("inventory_incomplete: has_more without page_token")

    def list_children(self, space_id: str, parent_node_token: str = "") -> list[FeishuNode]:
        nodes: list[FeishuNode] = []
        page_token = ""
        while True:
            argv = [
                "lark-cli",
                "wiki",
                "+node-list",
                "--space-id",
                space_id,
                "--page-size",
                "50",
                "--as",
                "user",
                "--format",
                "json",
            ]
            if parent_node_token:
                argv.extend(("--parent-node-token", parent_node_token))
            if page_token:
                argv.extend(("--page-token", page_token))
            payload = self._run(tuple(argv))
            data = payload.get("data", {})
            nodes.extend(
                FeishuNode(
                    node_token=str(item["node_token"]),
                    obj_token=str(item["obj_token"]),
                    obj_type=str(item["obj_type"]),
                    parent_node_token=str(item.get("parent_node_token", "")),
                    node_type=str(item["node_type"]),
                    title=str(item.get("title", "")),
                    has_child=bool(item.get("has_child", False)),
                )
                for item in data.get("nodes", [])
            )
            if not data.get("has_more"):
                return nodes
            page_token = str(data.get("page_token", ""))
            if not page_token:
                raise FeishuReadError("inventory_incomplete: has_more without page_token")

    def fetch_docx(self, obj_token: str) -> FeishuDocument:
        payload = self._run(
            (
                "lark-cli",
                "docs",
                "+fetch",
                "--doc",
                obj_token,
                "--doc-format",
                "markdown",
                "--detail",
                "simple",
                "--as",
                "user",
                "--format",
                "json",
            )
        )
        document = payload.get("data", {}).get("document", {})
        try:
            revision_id = int(document["revision_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeishuReadError("document response is missing a valid revision") from exc
        return FeishuDocument(
            obj_token=obj_token,
            revision_id=revision_id,
            markdown=str(document.get("content", "")),
        )
