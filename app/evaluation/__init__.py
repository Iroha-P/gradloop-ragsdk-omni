from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnswerQualityMetrics": ("answer_quality", "AnswerQualityMetrics"),
    "AnswerQualityReport": ("answer_quality", "AnswerQualityReport"),
    "AnswerQualityStatus": ("answer_quality", "AnswerQualityStatus"),
    "BoundaryGateAbDelta": ("boundary_gate_ab", "BoundaryGateAbDelta"),
    "BoundaryGateAbReport": ("boundary_gate_ab", "BoundaryGateAbReport"),
    "BoundaryGateDevReport": ("boundary_gate", "BoundaryGateDevReport"),
    "BoundaryGateMetrics": ("boundary_gate", "BoundaryGateMetrics"),
    "BoundaryGateV3Report": ("boundary_gate_v3", "BoundaryGateV3Report"),
    "BoundaryPolicyCalibration": ("boundary_gate", "BoundaryPolicyCalibration"),
    "BoundaryPolicyResult": ("boundary_gate", "BoundaryPolicyResult"),
    "BoundaryV3Metrics": ("boundary_gate_v3", "BoundaryV3Metrics"),
    "BoundaryV3Observation": ("boundary_gate_v3", "BoundaryV3Observation"),
    "EvidenceGateAbDelta": ("evidence_gate_ab", "EvidenceGateAbDelta"),
    "EvidenceGateAbReport": ("evidence_gate_ab", "EvidenceGateAbReport"),
    "GradingBenchmarkCase": ("grading", "GradingBenchmarkCase"),
    "GradingBenchmarkReport": ("grading", "GradingBenchmarkReport"),
    "GradingObservation": ("grading", "GradingObservation"),
    "NoSurvivingConfigurationError": ("boundary_gate", "NoSurvivingConfigurationError"),
    "PublicHumanAgreementReport": (
        "public_human_grading",
        "PublicHumanAgreementReport",
    ),
    "PublicHumanAnnotationStatus": (
        "public_human_grading",
        "PublicHumanAnnotationStatus",
    ),
    "PublicHumanGradingItem": (
        "public_human_grading",
        "PublicHumanGradingItem",
    ),
    "RetrievalBenchmarkCase": ("retrieval", "RetrievalBenchmarkCase"),
    "RetrievalBenchmarkReport": ("retrieval", "RetrievalBenchmarkReport"),
    "RetrievalChallengeType": ("retrieval", "RetrievalChallengeType"),
    "RetrievalSliceMetrics": ("retrieval", "RetrievalSliceMetrics"),
    "ScoreMetrics": ("grading", "ScoreMetrics"),
    "ThresholdCalibration": ("retrieval", "ThresholdCalibration"),
    "aggregate_boundary_metrics": ("boundary_gate", "aggregate_boundary_metrics"),
    "build_awaiting_public_status": (
        "public_human_grading",
        "build_awaiting_public_status",
    ),
    "build_boundary_gate_dev_report": ("boundary_gate", "build_boundary_gate_dev_report"),
    "build_completed_boundary_gate_ab_report": (
        "boundary_gate_ab",
        "build_completed_boundary_gate_ab_report",
    ),
    "build_completed_evidence_gate_ab_report": (
        "evidence_gate_ab",
        "build_completed_evidence_gate_ab_report",
    ),
    "build_failed_boundary_gate_ab_report": (
        "boundary_gate_ab",
        "build_failed_boundary_gate_ab_report",
    ),
    "build_failed_boundary_gate_dev_report": (
        "boundary_gate",
        "build_failed_boundary_gate_dev_report",
    ),
    "build_failed_evidence_gate_ab_report": (
        "evidence_gate_ab",
        "build_failed_evidence_gate_ab_report",
    ),
    "build_grading_report": ("grading", "build_grading_report"),
    "build_public_items": ("public_human_grading", "build_public_items"),
    "build_not_run_answer_quality_report": (
        "answer_quality",
        "build_not_run_answer_quality_report",
    ),
    "build_not_run_boundary_gate_ab_report": (
        "boundary_gate_ab",
        "build_not_run_boundary_gate_ab_report",
    ),
    "build_not_run_boundary_gate_dev_report": (
        "boundary_gate",
        "build_not_run_boundary_gate_dev_report",
    ),
    "build_not_run_evidence_gate_ab_report": (
        "evidence_gate_ab",
        "build_not_run_evidence_gate_ab_report",
    ),
    "calibrate_policy": ("boundary_gate", "calibrate_policy"),
    "calibrate_score_threshold": ("retrieval", "calibrate_score_threshold"),
    "calculate_public_agreement": (
        "public_human_grading",
        "calculate_public_agreement",
    ),
    "candidate_thresholds": ("boundary_gate", "candidate_thresholds"),
    "canonical_dev_report_bytes": ("boundary_gate_ab", "canonical_dev_report_bytes"),
    "evaluate_answer_quality": ("answer_quality", "evaluate_answer_quality"),
    "evaluate_policy": ("boundary_gate", "evaluate_policy"),
    "evaluate_retrieval": ("retrieval", "evaluate_retrieval"),
    "evaluate_v3": ("boundary_gate_v3", "evaluate_v3"),
    "dependency_versions": ("boundary_gate_v3", "dependency_versions"),
    "load_completed_public_ratings": (
        "public_human_grading",
        "load_completed_public_ratings",
    ),
    "load_benchmark_cases": ("retrieval", "load_benchmark_cases"),
    "load_grading_cases": ("grading", "load_grading_cases"),
    "score_metrics": ("grading", "score_metrics"),
    "select_candidate": ("boundary_gate", "select_candidate"),
    "primary_acceptance": ("boundary_gate_v3", "primary_acceptance"),
    "validate_benchmark_gold": ("retrieval", "validate_benchmark_gold"),
    "validate_passing_dev_report_binding": (
        "boundary_gate_ab",
        "validate_passing_dev_report_binding",
    ),
    "validate_passing_dev_report_bytes": (
        "boundary_gate_ab",
        "validate_passing_dev_report_bytes",
    ),
    "write_blind_pack": ("public_human_grading", "write_blind_pack"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
