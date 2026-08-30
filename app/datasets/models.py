from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QA_TASKS = (
    "fact_qa",
    "strategy_qa",
    "scenario_followup",
    "rubric",
    "tool_trace",
    "refusal",
)

QATask = Literal[
    "fact_qa",
    "strategy_qa",
    "scenario_followup",
    "rubric",
    "tool_trace",
    "refusal",
]
RightsClass = Literal["user_owned_derived", "open_licensed", "synthetic"]


class ExtractedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    block_id: str
    text: str
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SanitizedQA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: QATask
    question: str
    answer: str
    rubric: list[str]
    rights_class: RightsClass


class DerivationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    error_categories: dict[str, int]


class DerivationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[SanitizedQA]
    report: DerivationReport


class CandidateBuildReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_count: int = Field(ge=0)
    extracted_block_count: int = Field(ge=0)
    accepted_block_count: int = Field(ge=0)
    blocked_block_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    written_file_count: int = Field(ge=0)
    error_categories: dict[str, int]
