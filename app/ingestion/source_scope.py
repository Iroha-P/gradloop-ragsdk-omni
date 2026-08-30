from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class SourcePolicyError(ValueError):
    """Raised when a source is not authorized by an exact source policy."""


@dataclass(frozen=True)
class ScopeDecision:
    source_id: str | None = None
    rights_class: str | None = None
    extract_local: bool = False
    local_rag: bool = False
    direct_training: bool = False
    synthetic_derivation: bool = False
    remote_upload: bool = False
    requires_document_review: bool = False
    reason: str = "default_deny"


def _normalized(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _decision_from_permissions(
    permissions: Mapping[str, object],
    *,
    source_id: str | None,
    rights_class: str | None,
    reason: str,
    requires_document_review: bool = False,
) -> ScopeDecision:
    return ScopeDecision(
        source_id=source_id,
        rights_class=rights_class,
        extract_local=bool(permissions.get("extract_local", False)),
        local_rag=bool(permissions.get("local_rag", False)),
        direct_training=bool(permissions.get("direct_training", False)),
        synthetic_derivation=bool(permissions.get("synthetic_derivation", False)),
        remote_upload=bool(permissions.get("remote_upload", False)),
        requires_document_review=requires_document_review,
        reason=reason,
    )


class SourceScopePolicy:
    def __init__(self, payload: Mapping[str, object]):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported source-scope schema version")
        if payload.get("default_action") != "deny":
            raise ValueError("Source-scope policy must use default deny")
        self.payload = payload
        self.protected_roots = tuple(_normalized(value) for value in payload.get("protected_roots", []))
        self.protected_name_terms = tuple(
            str(value).casefold() for value in payload.get("protected_name_terms", [])
        )
        self.sources = tuple(payload.get("sources", []))
        self.feishu = payload.get("feishu", {})

    @classmethod
    def from_file(cls, path: str | Path) -> SourceScopePolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload)

    def decide_path(self, path: str | Path) -> ScopeDecision:
        candidate = _normalized(path)
        if any(_is_within(candidate, root) for root in self.protected_roots):
            return ScopeDecision(reason="protected_root")
        candidate_name = candidate.name.casefold()
        if any(term in candidate_name for term in self.protected_name_terms):
            return ScopeDecision(reason="protected_name_term")

        for source in self.sources:
            if not isinstance(source, Mapping):
                continue
            source_path = _normalized(str(source.get("path", "")))
            path_type = str(source.get("path_type", "file"))
            matches = candidate == source_path
            if path_type == "directory":
                matches = _is_within(candidate, source_path)
            if not matches:
                continue
            return _decision_from_permissions(
                source.get("permissions", {}),
                source_id=str(source.get("source_id", "")) or None,
                rights_class=str(source.get("rights_class", "")) or None,
                reason="allowlisted_source",
            )
        return ScopeDecision()

    def decide_feishu(
        self,
        *,
        space_name: str,
        node_type: str,
        environ: Mapping[str, str] | None = None,
    ) -> ScopeDecision:
        if not isinstance(self.feishu, Mapping) or not self.feishu.get("enabled", False):
            return ScopeDecision(reason="feishu_disabled")
        env_name = str(self.feishu.get("space_name_env", ""))
        runtime_env = os.environ if environ is None else environ
        expected_space = runtime_env.get(env_name, "") if env_name else ""
        if not expected_space or space_name != expected_space:
            return ScopeDecision(reason="feishu_space_not_allowlisted")
        allowed_types = {str(value) for value in self.feishu.get("allowed_node_types", [])}
        if node_type not in allowed_types:
            return ScopeDecision(reason="feishu_node_type_not_allowlisted")
        return _decision_from_permissions(
            self.feishu.get("permissions", {}),
            source_id="feishu_allowlisted_space",
            rights_class="document_level_review",
            reason="allowlisted_feishu_space",
            requires_document_review=bool(self.feishu.get("requires_document_review", True)),
        )
