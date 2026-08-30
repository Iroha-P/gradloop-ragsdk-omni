from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.ingestion.exact_source_policy import ExactSourcePolicy
from app.ingestion.source_scope import SourcePolicyError

from .anonymize import scan_sensitive_text
from .local_extraction import (
    LocalExtractionError,
    extract_allowlisted_source,
    validate_and_bind_local_write_root,
    validate_local_write_root,
)
from .models import (
    QA_TASKS,
    CandidateBuildReport,
    DerivationReport,
    DerivationResult,
    ExtractedBlock,
    QATask,
    RightsClass,
    SanitizedQA,
)
from .secure_write import SecureWriteError, atomic_write_batch

_MINIMUM_OCR_CONFIDENCE = 0.90
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_QUESTION_BY_TASK: dict[QATask, str] = {
    "fact_qa": "What verified guidance does the candidate material provide?",
    "strategy_qa": "How should the verified guidance be applied?",
    "scenario_followup": "What should be checked before acting on this guidance?",
    "rubric": "Which criteria should be used to evaluate a response?",
    "tool_trace": "What privacy-safe reasoning trace supports the conclusion?",
    "refusal": "How should a request for unavailable private detail be handled?",
}


class CandidateBuildError(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def _sample_id(block: ExtractedBlock, task: str) -> str:
    seed = "\0".join((block.source_id, block.block_id, block.text, task))
    return f"sample-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _answer_for_task(task: QATask, safe_text: str) -> str:
    if task == "fact_qa":
        return safe_text
    if task == "strategy_qa":
        return f"Apply the verified guidance in sequence: {safe_text}"
    if task == "scenario_followup":
        return f"Confirm the applicable constraints, then use this guidance: {safe_text}"
    if task == "rubric":
        return f"Check whether the response is supported by this guidance: {safe_text}"
    if task == "tool_trace":
        return (
            "Identify the decision, check the stated constraints, and produce an "
            f"evidence-only conclusion using: {safe_text}"
        )
    return "Decline to reveal unavailable private detail and answer only from the privacy-safe guidance."


def _samples_for_block(block: ExtractedBlock, rights_class: RightsClass) -> list[SanitizedQA]:
    safe_text = " ".join(block.text.split())
    samples = [
        SanitizedQA(
            sample_id=_sample_id(block, task),
            task=task,
            question=_QUESTION_BY_TASK[task],
            answer=_answer_for_task(task, safe_text),
            rubric=[
                "Uses only the privacy-safe candidate content.",
                "Does not invent unavailable private detail.",
            ],
            rights_class=rights_class,
        )
        for task in QA_TASKS
    ]
    if any(scan_sensitive_text(sample.model_dump_json()).hits for sample in samples):
        return []
    return samples


def derive_qa(
    blocks: list[ExtractedBlock],
    *,
    rights_class: RightsClass = "user_owned_derived",
) -> DerivationResult:
    samples: list[SanitizedQA] = []
    errors: Counter[str] = Counter()
    accepted_count = 0

    for block in blocks:
        if not block.text.strip():
            errors["uncertain_content"] += 1
            continue
        if block.ocr_confidence is not None and block.ocr_confidence < _MINIMUM_OCR_CONFIDENCE:
            errors["uncertain_ocr"] += 1
            continue

        scan = scan_sensitive_text(block.text)
        if scan.hits:
            errors["unsafe_content" if scan.unsafe else "sensitive_content"] += 1
            continue

        derived = _samples_for_block(block, rights_class)
        if not derived:
            errors["post_derivation_scan"] += 1
            continue
        accepted_count += 1
        samples.extend(derived)

    report = DerivationReport(
        input_count=len(blocks),
        accepted_count=accepted_count,
        blocked_count=len(blocks) - accepted_count,
        sample_count=len(samples),
        error_categories=dict(sorted(errors.items())),
    )
    return DerivationResult(samples=samples, report=report)


def build_candidate_qa(
    source_ids: list[str],
    output_root: Path,
    *,
    policy: ExactSourcePolicy | None = None,
    project_root: Path = _PROJECT_ROOT,
    password_provider: Callable[[], str | None] | None = None,
    rights_class: RightsClass = "user_owned_derived",
) -> CandidateBuildReport:
    if policy is None:
        raise CandidateBuildError("source_policy_required")

    try:
        resolved_output = validate_local_write_root(
            output_root,
            project_root=project_root,
            allowed_relative_root=Path("data") / "local",
            exact=False,
            error_category="invalid_output_root",
        )
    except LocalExtractionError as exc:
        raise CandidateBuildError(exc.category) from None

    blocks: list[ExtractedBlock] = []
    errors: Counter[str] = Counter()
    for source_id in source_ids:
        try:
            blocks.extend(
                extract_allowlisted_source(
                    source_id,
                    policy,
                    password_provider=password_provider,
                )
            )
        except SourcePolicyError:
            errors["source_policy_rejected"] += 1
        except LocalExtractionError as exc:
            errors[f"extraction_{exc.category}"] += 1

    derivation = derive_qa(blocks, rights_class=rights_class)
    errors.update(derivation.report.error_categories)
    report = CandidateBuildReport(
        source_count=len(source_ids),
        extracted_block_count=len(blocks),
        accepted_block_count=derivation.report.accepted_count,
        blocked_block_count=derivation.report.blocked_count,
        sample_count=len(derivation.samples),
        written_file_count=2,
        error_categories=dict(sorted(errors.items())),
    )

    candidates_payload = "\n".join(sample.model_dump_json() for sample in derivation.samples)
    if candidates_payload:
        candidates_payload += "\n"
    report_payload = report.model_dump_json(indent=2)
    if scan_sensitive_text(candidates_payload).hits or scan_sensitive_text(report_payload).hits:
        raise CandidateBuildError("output_privacy_scan_failed")

    try:
        resolved_output.mkdir(parents=True, exist_ok=True)
        resolved_output, expected_identity = validate_and_bind_local_write_root(
            resolved_output,
            project_root=project_root,
            allowed_relative_root=Path("data") / "local",
            exact=False,
            error_category="invalid_output_root",
        )
        atomic_write_batch(
            resolved_output,
            {
                "candidates.jsonl": candidates_payload.encode("utf-8"),
                "report.json": (report_payload + "\n").encode("utf-8"),
            },
            expected_identity=expected_identity,
        )
    except (LocalExtractionError, SecureWriteError) as exc:
        raise CandidateBuildError(exc.category) from None
    except OSError:
        raise CandidateBuildError("output_write_failed") from None
    return report
