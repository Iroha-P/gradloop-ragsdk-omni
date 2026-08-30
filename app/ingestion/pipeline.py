from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import UnsafeInputError

from .chunking import build_chunks
from .extractors import ExtractionError, extract_file
from .manifest import ManifestStore
from .models import (
    CHUNKER_VERSION,
    PARSER_VERSION,
    ExtractionBlock,
    IngestionResult,
    IngestionStatus,
    PrivacyFindingSummary,
    Sensitivity,
    SourceManifest,
    SourceSpec,
)
from .privacy import decide_ingestion, redact_text, scan_privacy
from .security import DEFAULT_MAX_FILE_BYTES, authorize_file, sha256_file
from .source_scope import SourceScopePolicy


@dataclass(frozen=True)
class PipelineConfig:
    allowed_roots: list[Path]
    output_root: Path
    source_scope_policy: SourceScopePolicy | None = None
    public_mode: bool = False
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    chunk_size: int = 800
    chunk_overlap: int = 120
    private_redaction_terms: tuple[str, ...] = ()


class IngestionPipeline:
    def __init__(self, config: PipelineConfig):
        if not config.allowed_roots:
            raise ValueError("At least one allowed source root is required")
        self.config = config
        self.store = ManifestStore(config.output_root)

    def _redact(self, text: str) -> str:
        redacted = redact_text(text)
        for term in self.config.private_redaction_terms:
            if len(term.strip()) >= 2:
                redacted = redacted.replace(term, "<redacted:identity>")
        return redacted

    def _failed_manifest(
        self,
        spec: SourceSpec,
        *,
        relative_path: str,
        checksum: str,
        status: IngestionStatus,
        privacy: PrivacyFindingSummary | None = None,
        warnings: list[str] | None = None,
        parser: str = "",
    ) -> SourceManifest:
        return SourceManifest(
            source_id=spec.source_id,
            relative_path=relative_path,
            title=self._redact(spec.title),
            category=spec.category,
            authority_level=spec.authority_level,
            published_at=spec.published_at,
            effective_at=spec.effective_at,
            checksum=checksum,
            sensitivity=(privacy.recommended_sensitivity if privacy else spec.declared_sensitivity),
            status=status,
            parser=parser,
            privacy=privacy or PrivacyFindingSummary(),
            warnings=warnings or [],
        )

    def ingest(
        self,
        spec: SourceSpec,
        *,
        confirm_medium: bool = False,
        confirm_secret_redaction: bool = False,
    ) -> IngestionResult:
        if self.config.source_scope_policy is not None:
            scope = self.config.source_scope_policy.decide_path(spec.path)
            if not scope.extract_local:
                manifest = self._failed_manifest(
                    spec,
                    relative_path="unavailable",
                    checksum="unavailable",
                    status=IngestionStatus.REJECTED,
                    warnings=[f"source_scope:{scope.reason}"],
                )
                self.store.save_manifest(manifest)
                return IngestionResult(manifest=manifest)
        try:
            authorized = authorize_file(
                spec.path,
                self.config.allowed_roots,
                max_file_bytes=self.config.max_file_bytes,
            )
        except UnsafeInputError as exc:
            manifest = self._failed_manifest(
                spec,
                relative_path="unavailable",
                checksum="unavailable",
                status=IngestionStatus.REJECTED,
                warnings=[exc.message],
            )
            self.store.save_manifest(manifest)
            return IngestionResult(manifest=manifest)

        checksum = sha256_file(authorized.resolved_path)
        safe_relative_path = self._redact(authorized.relative_path)
        existing = self.store.load_manifest(spec.source_id)
        if (
            existing
            and existing.checksum == checksum
            and existing.parser_version == PARSER_VERSION
            and existing.chunker_version == CHUNKER_VERSION
            and existing.status == IngestionStatus.INDEXED
        ):
            unchanged = existing.model_copy(update={"status": IngestionStatus.UNCHANGED})
            return IngestionResult(
                manifest=unchanged,
                chunks=self.store.load_chunks(spec.source_id),
                extracted_output=str(self.store.extracted_path(spec.source_id)),
            )

        try:
            password = os.getenv(spec.password_env) if spec.password_env else None
            extraction = extract_file(authorized.resolved_path, password=password)
        except (ExtractionError, UnicodeError, OSError) as exc:
            manifest = self._failed_manifest(
                spec,
                relative_path=safe_relative_path,
                checksum=checksum,
                status=IngestionStatus.FAILED,
                warnings=[str(exc)],
            )
            self.store.save_manifest(manifest)
            return IngestionResult(manifest=manifest)

        full_text = "\n\n".join(
            value
            for block in extraction.blocks
            for value in (block.section, block.text)
            if value
        )
        privacy = scan_privacy(full_text, path_hint=authorized.relative_path)
        decision = decide_ingestion(
            spec.declared_sensitivity,
            privacy,
            public_mode=self.config.public_mode,
        )
        warnings = [*extraction.warnings, *privacy.warnings, decision.reason]
        if self.config.private_redaction_terms:
            warnings.append("explicit private identity redaction terms applied")
        confirmed_secret_only = (
            confirm_medium
            and confirm_secret_redaction
            and not self.config.public_mode
            and spec.declared_sensitivity == Sensitivity.MEDIUM
            and set(privacy.finding_counts) == {"secret"}
            and not privacy.path_risk_terms
            and not privacy.prompt_injection_detected
        )
        if confirmed_secret_only:
            warnings.append("explicit local confirmation permits secret-value redaction before chunking")
        effective_sensitivity = (
            Sensitivity.MEDIUM if confirmed_secret_only else decision.sensitivity
        )
        if decision.reject and not confirmed_secret_only:
            manifest = self._failed_manifest(
                spec,
                relative_path=safe_relative_path,
                checksum=checksum,
                status=IngestionStatus.REJECTED,
                privacy=privacy,
                warnings=warnings,
                parser=extraction.parser,
            ).model_copy(update={"sensitivity": decision.sensitivity})
            self.store.save_manifest(manifest)
            return IngestionResult(manifest=manifest)

        if decision.requires_confirmation and not confirm_medium:
            preview = self._redact(full_text)[:2000]
            preview_path = self.store.save_extracted(f"{spec.source_id}.preview", preview)
            manifest = self._failed_manifest(
                spec,
                relative_path=safe_relative_path,
                checksum=checksum,
                status=IngestionStatus.PENDING_CONFIRMATION,
                privacy=privacy,
                warnings=warnings,
                parser=extraction.parser,
            ).model_copy(update={"sensitivity": decision.sensitivity})
            self.store.save_manifest(manifest)
            return IngestionResult(manifest=manifest, extracted_output=str(preview_path))

        redacted_blocks = [
            ExtractionBlock(
                text=self._redact(block.text),
                section=self._redact(block.section),
                page=block.page,
                block_type=block.block_type,
            )
            for block in extraction.blocks
        ]
        output_spec = spec.model_copy(update={"title": self._redact(spec.title)})
        chunks = build_chunks(
            output_spec,
            redacted_blocks,
            source_checksum=checksum,
            sensitivity=effective_sensitivity,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        extracted_text = "\n\n".join(block.text for block in redacted_blocks)
        extracted_path = self.store.save_extracted(spec.source_id, extracted_text)
        self.store.save_chunks(spec.source_id, chunks)
        manifest = SourceManifest(
            source_id=spec.source_id,
            relative_path=safe_relative_path,
            title=self._redact(spec.title),
            category=spec.category,
            authority_level=spec.authority_level,
            published_at=spec.published_at,
            effective_at=spec.effective_at,
            checksum=checksum,
            sensitivity=effective_sensitivity,
            status=IngestionStatus.INDEXED,
            parser=extraction.parser,
            chunk_count=len(chunks),
            privacy=privacy,
            warnings=warnings,
        )
        self.store.save_manifest(manifest)
        return IngestionResult(manifest=manifest, chunks=chunks, extracted_output=str(extracted_path))
