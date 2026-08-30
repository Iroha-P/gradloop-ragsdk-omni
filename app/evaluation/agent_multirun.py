from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.agent_business import BusinessObservation, EngineName

PRIVATE_MULTIRUN_OBSERVATIONS_PATH = Path(
    "data/local/evaluation/agent_business_multirun_observations.jsonl"
)
PUBLIC_MULTIRUN_REPORT_PATH = Path(
    "reports/public/agent_business_multirun_v1.json"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatedBusinessObservation(StrictModel):
    repeat_index: int = Field(ge=1)
    observation: BusinessObservation


class ConfidenceInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    confidence_level: Literal[0.95] = 0.95


class RepeatedEngineMetrics(StrictModel):
    observation_count: int
    repeat_count: int
    completion_rate: float
    task_success_rate: ConfidenceInterval
    grounded_result_rate: float
    confirmation_success_rate: float
    recovery_supported: bool
    recovery_success_rate: ConfidenceInterval
    duplicate_side_effect_rate: float
    mean_tool_steps: float
    latency_p50_ms: ConfidenceInterval
    latency_p95_ms: ConfidenceInterval
    failure_type_counts: dict[str, int]


class PublicMultirunReport(StrictModel):
    report_version: Literal["agent-business-multirun-v1"]
    claim_scope: Literal["local_anonymized_workflow_multirun"]
    case_count: int
    repeat_count: int
    observation_count: int
    scenario_counts: dict[str, int]
    scope_counts: dict[str, int]
    engines: dict[EngineName, RepeatedEngineMetrics]
    execution_order: str
    confidence_interval_method: str
    bootstrap_samples: int
    bootstrap_seed: int
    privacy: dict[str, bool | str]
    limitations: list[str]


def engine_order_for_repeat(repeat_index: int) -> tuple[EngineName, EngineName]:
    if repeat_index < 1:
        raise ValueError("repeat_index must be at least 1")
    return (
        ("baseline", "langgraph")
        if repeat_index % 2
        else ("langgraph", "baseline")
    )


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _metric(rows: list[BusinessObservation], name: str) -> float:
    if name == "task_success_rate":
        return _rate([row.task_success for row in rows])
    if name == "recovery_success_rate":
        recovery = [
            bool(row.recovery_success) for row in rows if row.scenario == "recovery"
        ]
        return _rate(recovery)
    latencies = [row.latency_ms for row in rows]
    if name == "latency_p50_ms":
        return median(latencies) if latencies else 0.0
    if name == "latency_p95_ms":
        return _p95(latencies)
    raise ValueError(f"unknown metric: {name}")


def _cluster_bootstrap_interval(
    rows: list[BusinessObservation],
    *,
    name: str,
    samples: int,
    random_source: random.Random,
) -> ConfidenceInterval:
    by_case: dict[str, list[BusinessObservation]] = {}
    for row in rows:
        by_case.setdefault(row.case_id, []).append(row)
    case_ids = sorted(by_case)
    if not case_ids:
        raise ValueError("at least one case is required")
    estimate = _metric(rows, name)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_rows: list[BusinessObservation] = []
        for _case in case_ids:
            sampled_id = random_source.choice(case_ids)
            sampled_rows.extend(by_case[sampled_id])
        estimates.append(_metric(sampled_rows, name))
    lower = min(estimate, _quantile(estimates, 0.025))
    upper = max(estimate, _quantile(estimates, 0.975))
    return ConfidenceInterval(
        estimate=round(estimate, 4 if "rate" in name else 3),
        lower=round(lower, 4 if "rate" in name else 3),
        upper=round(upper, 4 if "rate" in name else 3),
    )


def _validate_matrix(
    rows: list[RepeatedBusinessObservation], *, case_count: int, repeats: int
) -> list[str]:
    case_ids = sorted({row.observation.case_id for row in rows})
    expected = {
        (case_id, repeat_index, engine)
        for case_id in case_ids
        for repeat_index in range(1, repeats + 1)
        for engine in ("baseline", "langgraph")
    }
    actual = {
        (row.observation.case_id, row.repeat_index, row.observation.engine)
        for row in rows
    }
    if (
        len(case_ids) != case_count
        or len(actual) != len(rows)
        or actual != expected
    ):
        raise ValueError("observations must form a complete engine-case-repeat matrix")
    return case_ids


def build_multirun_report(
    rows: list[RepeatedBusinessObservation],
    *,
    case_count: int,
    repeats: int,
    bootstrap_samples: int = 1000,
    seed: int = 20260722,
) -> PublicMultirunReport:
    if repeats < 2:
        raise ValueError("multirun benchmark requires at least two repeats")
    if bootstrap_samples < 20:
        raise ValueError("bootstrap_samples must be at least 20")
    case_ids = _validate_matrix(rows, case_count=case_count, repeats=repeats)
    scenario_by_case = {
        case_id: next(
            row.observation.scenario
            for row in rows
            if row.observation.case_id == case_id
        )
        for case_id in case_ids
    }
    scope_by_case = {
        case_id: next(
            row.observation.scope
            for row in rows
            if row.observation.case_id == case_id
        )
        for case_id in case_ids
    }
    engines: dict[EngineName, RepeatedEngineMetrics] = {}
    for engine_index, engine in enumerate(("baseline", "langgraph")):
        engine_rows = [
            row.observation for row in rows if row.observation.engine == engine
        ]
        grounded = [
            bool(row.grounded_result)
            for row in engine_rows
            if row.grounded_result is not None
        ]
        confirmed = [
            bool(row.confirmation_success)
            for row in engine_rows
            if row.confirmation_success is not None
        ]
        random_source = random.Random(seed + engine_index)
        engines[engine] = RepeatedEngineMetrics(
            observation_count=len(engine_rows),
            repeat_count=repeats,
            completion_rate=round(_rate([row.completion for row in engine_rows]), 4),
            task_success_rate=_cluster_bootstrap_interval(
                engine_rows,
                name="task_success_rate",
                samples=bootstrap_samples,
                random_source=random_source,
            ),
            grounded_result_rate=round(_rate(grounded), 4),
            confirmation_success_rate=round(_rate(confirmed), 4),
            recovery_supported=all(row.recovery_supported for row in engine_rows),
            recovery_success_rate=_cluster_bootstrap_interval(
                engine_rows,
                name="recovery_success_rate",
                samples=bootstrap_samples,
                random_source=random_source,
            ),
            duplicate_side_effect_rate=round(
                _rate([row.duplicate_side_effect for row in engine_rows]), 4
            ),
            mean_tool_steps=round(mean(row.tool_steps for row in engine_rows), 4),
            latency_p50_ms=_cluster_bootstrap_interval(
                engine_rows,
                name="latency_p50_ms",
                samples=bootstrap_samples,
                random_source=random_source,
            ),
            latency_p95_ms=_cluster_bootstrap_interval(
                engine_rows,
                name="latency_p95_ms",
                samples=bootstrap_samples,
                random_source=random_source,
            ),
            failure_type_counts=dict(
                Counter(row.error_type for row in engine_rows if row.error_type)
            ),
        )
    return PublicMultirunReport(
        report_version="agent-business-multirun-v1",
        claim_scope="local_anonymized_workflow_multirun",
        case_count=case_count,
        repeat_count=repeats,
        observation_count=len(rows),
        scenario_counts=dict(Counter(scenario_by_case.values())),
        scope_counts=dict(Counter(scope_by_case.values())),
        engines=engines,
        execution_order="alternating baseline/langgraph by repeat",
        confidence_interval_method=(
            "95% percentile cluster bootstrap by case; all repeats retained"
        ),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
        privacy={
            "raw_details_public": False,
            "identifiers_public": False,
            "private_outputs_path": "data/local/evaluation",
        },
        limitations=[
            "Single-machine local workflow benchmark; not an online throughput test.",
            "LLM generation is disabled, so results do not measure model answer quality.",
            "Recovery compares checkpoint/resume capability; baseline has no equivalent API.",
        ],
    )


def write_private_multirun_observations(
    rows: list[RepeatedBusinessObservation],
    path: Path = PRIVATE_MULTIRUN_OBSERVATIONS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(row.model_dump_json() + "\n" for row in rows)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_public_multirun_report(
    report: PublicMultirunReport,
    path: Path = PUBLIC_MULTIRUN_REPORT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
