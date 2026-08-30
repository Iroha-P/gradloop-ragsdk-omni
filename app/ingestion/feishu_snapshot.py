from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, deque
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.ingestion.models import CHUNKER_VERSION, PARSER_VERSION
from app.ingestion.privacy import redact_text, scan_privacy
from app.integrations.feishu.client import FeishuNode, FeishuReadClient


class PrivacyLane(StrEnum):
    GENERIC = "generic"
    PERSONAL = "personal"


class SnapshotStatus(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    EMPTY = "empty"
    READ_FAILED = "read_failed"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"


class SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SnapshotManifest(SnapshotModel):
    source_id: str
    source_kind: str = "feishu_docx"
    node_path: tuple[str, ...]
    node_token: str
    obj_token: str
    remote_revision_id: int | None = None
    content_checksum: str = ""
    privacy_lane: PrivacyLane
    status: SnapshotStatus
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    synced_at: datetime
    warning_codes: tuple[str, ...] = ()


class SnapshotDocument(SnapshotModel):
    source_id: str
    title: str
    markdown: str
    privacy_lane: PrivacyLane
    revision_id: int
    checksum: str


class SnapshotSyncResult(SnapshotModel):
    published: bool
    unchanged: bool = False
    source_count: int
    status_counts: dict[str, int]
    snapshot_checksum: str


class SnapshotSyncError(RuntimeError):
    """A staging sync failed and the previous current snapshot remains active."""


def anonymous_source_id(node_token: str) -> str:
    return "feishu_" + hashlib.sha256(node_token.encode("utf-8")).hexdigest()[:20]


def strict_lane_resolver(
    node_path: tuple[str, ...],
    explicit_policy: dict[str, PrivacyLane] | None = None,
) -> PrivacyLane:
    root = node_path[0].strip() if node_path else ""
    if explicit_policy and root in explicit_policy:
        return explicit_policy[root]
    if any(marker in root for marker in ("经历", "故事", "自我介绍", "申请叙事", "历史面试")):
        return PrivacyLane.PERSONAL
    if any(marker in root for marker in ("使用说明", "总览", "专业知识", "英文表达")):
        return PrivacyLane.GENERIC
    return PrivacyLane.PERSONAL


def _atomic_publish(staged_current: Path, current: Path) -> None:
    rollback = current.with_name(".current-rollback")
    if rollback.exists():
        shutil.rmtree(rollback)
    if current.exists():
        os.replace(current, rollback)
    try:
        os.replace(staged_current, current)
    except Exception:
        if rollback.exists() and not current.exists():
            os.replace(rollback, current)
        raise
    if rollback.exists():
        shutil.rmtree(rollback)


def _snapshot_checksum(manifests: list[SnapshotManifest]) -> str:
    stable = [
        {
            "source_id": item.source_id,
            "source_kind": item.source_kind,
            "node_path": item.node_path,
            "remote_revision_id": item.remote_revision_id,
            "content_checksum": item.content_checksum,
            "privacy_lane": item.privacy_lane.value,
            "status": (
                SnapshotStatus.INDEXED.value
                if item.status == SnapshotStatus.UNCHANGED
                else item.status.value
            ),
            "parser_version": item.parser_version,
            "chunker_version": item.chunker_version,
            "warning_codes": item.warning_codes,
        }
        for item in sorted(manifests, key=lambda value: value.source_id)
    ]
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SnapshotSyncService:
    def __init__(
        self,
        client: FeishuReadClient,
        root: Path,
        *,
        lane_resolver: Callable[[tuple[str, ...]], PrivacyLane],
    ):
        self.client = client
        self.root = root
        self.lane_resolver = lane_resolver

    def sync(self, space_id: str) -> SnapshotSyncResult:
        self.client.verify_auth()
        self.root.mkdir(parents=True, exist_ok=True)
        staging_root = self.root / "staging"
        staging_root.mkdir(exist_ok=True)
        run_root = Path(tempfile.mkdtemp(prefix="sync-", dir=staging_root))
        staged_current = run_root / "current"
        manifests_dir = staged_current / "manifests"
        extracted_dir = staged_current / "extracted_redacted"
        manifests_dir.mkdir(parents=True)
        extracted_dir.mkdir(parents=True)
        manifests: list[SnapshotManifest] = []
        try:
            roots = self.client.list_children(space_id)
            queue = deque((node, (node.title,)) for node in roots)
            seen_nodes: set[str] = set()
            while queue:
                node, path = queue.popleft()
                if node.node_token in seen_nodes:
                    raise SnapshotSyncError("inventory_incomplete: duplicate node traversal")
                seen_nodes.add(node.node_token)
                if node.has_child:
                    children = self.client.list_children(space_id, node.node_token)
                    queue.extend((child, (*path, child.title)) for child in children)
                manifests.append(
                    self._snapshot_node(node, path, manifests_dir, extracted_dir)
                )

            hard_failures = [
                item
                for item in manifests
                if item.status in {SnapshotStatus.READ_FAILED, SnapshotStatus.REJECTED}
            ]
            if hard_failures:
                raise SnapshotSyncError(hard_failures[0].status.value)

            checksum = _snapshot_checksum(manifests)
            inventory = {
                "complete": True,
                "snapshot_checksum": checksum,
                "sources": [item.model_dump(mode="json") for item in manifests],
            }
            inventory_path = staged_current / "inventory.json"
            inventory_path.write_text(
                json.dumps(inventory, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            _atomic_publish(staged_current, self.root / "current")
            counts = Counter(item.status.value for item in manifests)
            result = SnapshotSyncResult(
                published=True,
                source_count=len(manifests),
                status_counts=dict(counts),
                snapshot_checksum=checksum,
            )
            self._write_sync_report(result, manifests)
            return result
        except SnapshotSyncError:
            counts = Counter(item.status.value for item in manifests)
            self._write_sync_report(
                SnapshotSyncResult(
                    published=False,
                    source_count=len(manifests),
                    status_counts=dict(counts),
                    snapshot_checksum="",
                ),
                manifests,
            )
            raise
        except Exception as exc:
            counts = Counter(item.status.value for item in manifests)
            self._write_sync_report(
                SnapshotSyncResult(
                    published=False,
                    source_count=len(manifests),
                    status_counts=dict(counts),
                    snapshot_checksum="",
                ),
                manifests,
            )
            raise SnapshotSyncError(type(exc).__name__) from exc
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def _snapshot_node(
        self,
        node: FeishuNode,
        path: tuple[str, ...],
        manifests_dir: Path,
        extracted_dir: Path,
    ) -> SnapshotManifest:
        source_id = anonymous_source_id(node.node_token)
        lane = self.lane_resolver(path)
        now = datetime.now(UTC)
        if node.obj_type != "docx":
            manifest = SnapshotManifest(
                source_id=source_id,
                source_kind=f"feishu_{node.obj_type}",
                node_path=path,
                node_token=node.node_token,
                obj_token=node.obj_token,
                privacy_lane=lane,
                status=SnapshotStatus.UNSUPPORTED,
                synced_at=now,
                warning_codes=(f"unsupported:{node.obj_type}",),
            )
            self._write_manifest(manifests_dir, manifest)
            return manifest

        try:
            remote = self.client.fetch_docx(node.obj_token)
        except Exception:
            manifest = SnapshotManifest(
                source_id=source_id,
                node_path=path,
                node_token=node.node_token,
                obj_token=node.obj_token,
                privacy_lane=lane,
                status=SnapshotStatus.READ_FAILED,
                synced_at=now,
                warning_codes=("read_failed",),
            )
            self._write_manifest(manifests_dir, manifest)
            return manifest

        redacted = redact_text(remote.markdown).strip()
        checksum = hashlib.sha256(redacted.encode("utf-8")).hexdigest()
        if not redacted:
            status = SnapshotStatus.EMPTY
        else:
            privacy = scan_privacy(redacted)
            if privacy.finding_counts or privacy.prompt_injection_detected:
                manifest = SnapshotManifest(
                    source_id=source_id,
                    node_path=path,
                    node_token=node.node_token,
                    obj_token=node.obj_token,
                    remote_revision_id=remote.revision_id,
                    content_checksum=checksum,
                    privacy_lane=lane,
                    status=SnapshotStatus.REJECTED,
                    synced_at=now,
                    warning_codes=("privacy_gate_failed",),
                )
                self._write_manifest(manifests_dir, manifest)
                return manifest
            status = SnapshotStatus.INDEXED

        previous_path = self.root / "current" / "manifests" / f"{source_id}.json"
        if previous_path.is_file() and status == SnapshotStatus.INDEXED:
            previous = SnapshotManifest.model_validate_json(
                previous_path.read_text(encoding="utf-8")
            )
            same = (
                previous.remote_revision_id == remote.revision_id
                and previous.content_checksum == checksum
                and previous.parser_version == PARSER_VERSION
                and previous.chunker_version == CHUNKER_VERSION
            )
            if same:
                status = SnapshotStatus.UNCHANGED

        if redacted:
            (extracted_dir / f"{source_id}.md").write_text(
                redacted + "\n", encoding="utf-8", newline="\n"
            )
        manifest = SnapshotManifest(
            source_id=source_id,
            node_path=path,
            node_token=node.node_token,
            obj_token=node.obj_token,
            remote_revision_id=remote.revision_id,
            content_checksum=checksum,
            privacy_lane=lane,
            status=status,
            synced_at=now,
        )
        self._write_manifest(manifests_dir, manifest)
        return manifest

    @staticmethod
    def _write_manifest(directory: Path, manifest: SnapshotManifest) -> None:
        (directory / f"{manifest.source_id}.json").write_text(
            manifest.model_dump_json(), encoding="utf-8", newline="\n"
        )

    def _write_sync_report(
        self, result: SnapshotSyncResult, manifests: list[SnapshotManifest]
    ) -> None:
        payload = {
            "published": result.published,
            "source_count": result.source_count,
            "status_counts": result.status_counts,
            "sources": [
                {
                    "source_id": item.source_id,
                    "status": item.status.value,
                    "warning_codes": list(item.warning_codes),
                }
                for item in manifests
            ],
        }
        temporary = self.root / ".sync_report.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.root / "sync_report.json")
