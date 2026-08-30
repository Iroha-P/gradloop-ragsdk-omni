from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.question_bank.models import QuestionRecord, QuestionReviewStatus

HUMAN_GRADING_DIR = Path("data/local/evaluation/human_grading")
PUBLIC_ANNOTATION_STATUS_PATH = Path(
    "reports/public/human_grading_annotation_status_v1.json"
)

AnswerStrength = Literal["weak", "partial", "strong"]
RUBRIC_LIMITS = {
    "clarity": 15,
    "coverage": 30,
    "reasoning": 20,
    "evidence": 20,
    "reflection": 15,
}
CSV_FIELDS = [
    "item_id",
    "clarity",
    "coverage",
    "reasoning",
    "evidence",
    "reflection",
    "confidence",
    "notes",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HumanGradingItem(StrictModel):
    item_id: str = Field(pattern=r"^hgi_[a-f0-9]{20}$")
    scope: Literal["generic", "personal"]
    question: str = Field(min_length=2, max_length=4096)
    answer: str = Field(min_length=1, max_length=12000)
    answer_strength: AnswerStrength
    rubric: dict[str, int]


class HumanAnnotation(StrictModel):
    item_id: str = Field(min_length=1, max_length=64)
    clarity: int = Field(ge=0, le=15)
    coverage: int = Field(ge=0, le=30)
    reasoning: int = Field(ge=0, le=20)
    evidence: int = Field(ge=0, le=20)
    reflection: int = Field(ge=0, le=15)
    confidence: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)

    @property
    def total(self) -> int:
        return (
            self.clarity
            + self.coverage
            + self.reasoning
            + self.evidence
            + self.reflection
        )


class HumanAgreementReport(StrictModel):
    report_version: Literal["human-grading-agreement-v1"]
    status: Literal["completed"]
    rater_count: int = Field(ge=2)
    item_count: int = Field(ge=1)
    pair_count: int = Field(ge=1)
    mean_absolute_total_difference: float = Field(ge=0)
    within_10_proportion: float = Field(ge=0, le=1)
    band_agreement: float = Field(ge=0, le=1)
    cohen_kappa: float = Field(ge=-1, le=1)
    quadratic_weighted_kappa: float = Field(ge=-1, le=1)
    pearson_correlation: float = Field(ge=-1, le=1)
    privacy: dict[str, bool]


class HumanAnnotationStatus(StrictModel):
    report_version: Literal["human-grading-annotation-status-v1"]
    status: Literal["awaiting_human_labels", "completed"]
    item_count: int = Field(ge=1)
    rater_slots: int = Field(ge=2)
    metrics: HumanAgreementReport | None = None
    note: str


def _stable_key(record: QuestionRecord) -> str:
    return hashlib.sha256(record.content_fingerprint.encode("utf-8")).hexdigest()


def _answer_for(record: QuestionRecord, strength: AnswerStrength) -> str:
    points = [point.strip() for point in record.answer_points if point.strip()]
    if strength == "weak" or not points:
        return "我会先给出结论，再结合具体情况补充说明。"
    if strength == "partial":
        return "；".join(points[:2]) + "。"
    return "\n".join(
        ["结论：围绕问题建立清晰主线。"]
        + [f"要点 {index}：{point}" for index, point in enumerate(points, start=1)]
        + ["复盘：补充适用边界、证据来源和可改进之处。"]
    )


def build_human_grading_items(
    records: list[QuestionRecord], *, count: int = 30
) -> list[HumanGradingItem]:
    if count != 30:
        raise ValueError("human grading v1 requires exactly 30 items")
    accepted = [
        record
        for record in records
        if record.review_status == QuestionReviewStatus.ACCEPTED
        and len(record.canonical_text.strip()) >= 2
    ]
    generic = sorted(
        [record for record in accepted if record.privacy_lane.value == "generic"],
        key=_stable_key,
    )
    personal = sorted(
        [record for record in accepted if record.privacy_lane.value == "personal"],
        key=_stable_key,
    )
    if len(generic) < 18 or len(personal) < 12:
        raise ValueError("at least 18 generic and 12 personal accepted questions are required")
    lane_pattern = ("generic", "generic", "personal", "generic", "personal")
    selected: list[QuestionRecord] = []
    lane_indexes = {"generic": 0, "personal": 0}
    lane_records = {"generic": generic, "personal": personal}
    for index in range(count):
        lane = lane_pattern[index % len(lane_pattern)]
        selected.append(lane_records[lane][lane_indexes[lane]])
        lane_indexes[lane] += 1
    strengths: tuple[AnswerStrength, ...] = ("weak", "partial", "strong")
    items: list[HumanGradingItem] = []
    for index, record in enumerate(selected):
        strength = strengths[index % len(strengths)]
        digest = hashlib.sha256(
            f"{record.content_fingerprint}|{strength}".encode()
        ).hexdigest()
        items.append(
            HumanGradingItem(
                item_id=f"hgi_{digest[:20]}",
                scope=record.privacy_lane.value,
                question=record.canonical_text.strip(),
                answer=_answer_for(record, strength),
                answer_strength=strength,
                rubric=dict(RUBRIC_LIMITS),
            )
        )
    return items


def _atomic_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _blank_csv(items: list[HumanGradingItem]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in items:
        writer.writerow({field: item.item_id if field == "item_id" else "" for field in CSV_FIELDS})
    return stream.getvalue()


def write_annotation_pack(
    items: list[HumanGradingItem], output_dir: Path = HUMAN_GRADING_DIR
) -> None:
    if len(items) != 30 or len({item.item_id for item in items}) != 30:
        raise ValueError("annotation pack requires 30 unique items")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "items.jsonl",
        "".join(item.model_dump_json() + "\n" for item in items),
    )
    template = _blank_csv(items)
    _atomic_text(output_dir / "rater_a.csv", template, encoding="utf-8-sig")
    _atomic_text(output_dir / "rater_b.csv", template, encoding="utf-8-sig")
    _atomic_text(
        output_dir / "GUIDE.md",
        """# 保研面试回答人工评分说明

请根据 `items.jsonl` 中同一 `item_id` 的问题和待评回答，在自己的 CSV 中独立评分。

- 清晰度 clarity：0–15
- 覆盖度 coverage：0–30
- 推理 reasoning：0–20
- 证据 evidence：0–20
- 反思 reflection：0–15

标注期间不要查看另一名评分者的 CSV，也不要查看系统评分。
`confidence` 可填 1–5，`notes` 可记录疑问；两者均不计入总分。
五个维度必须全部填写，否则评分文件不会被接受。

冲突处理：先保留两份原始独立评分，计算一致性后再另行讨论；不回改原始标注来人为提高一致性。
""",
    )


def load_human_grading_items(path: Path) -> list[HumanGradingItem]:
    items = [
        HumanGradingItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items or len({item.item_id for item in items}) != len(items):
        raise ValueError("human grading items must be non-empty and unique")
    return items


def load_completed_annotations(
    path: Path, expected_ids: list[str]
) -> list[HumanAnnotation]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("annotation file is incomplete: no rows")
    if set(rows[0]) != set(CSV_FIELDS):
        raise ValueError("annotation columns do not match the template")
    annotations: list[HumanAnnotation] = []
    for row_number, row in enumerate(rows, start=2):
        required = [row.get(field, "").strip() for field in RUBRIC_LIMITS]
        if not all(required):
            raise ValueError(f"annotation file is incomplete at row {row_number}")
        payload = {
            "item_id": row.get("item_id", "").strip(),
            **{field: row.get(field, "").strip() for field in RUBRIC_LIMITS},
            "confidence": row.get("confidence", "").strip() or None,
            "notes": row.get("notes", "").strip(),
        }
        try:
            annotations.append(HumanAnnotation.model_validate(payload))
        except ValidationError as exc:
            raise ValueError(f"invalid annotation at row {row_number}: {exc}") from exc
    actual_ids = [annotation.item_id for annotation in annotations]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("annotation item_id values must be unique")
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError("annotation file is incomplete or contains unexpected item_id values")
    return annotations


def _band(score: int) -> int:
    if score < 60:
        return 0
    if score < 80:
        return 1
    return 2


def _cohen_kappa(left: list[int], right: list[int]) -> float:
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[category] / len(left)) * (right_counts[category] / len(right))
        for category in (0, 1, 2)
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def _quadratic_weighted_kappa(left: list[int], right: list[int]) -> float:
    scale = 100
    observed = mean(((a - b) / scale) ** 2 for a, b in zip(left, right, strict=True))
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_count
        * right_count
        * (((left_score - right_score) / scale) ** 2)
        for left_score, left_count in left_counts.items()
        for right_score, right_count in right_counts.items()
    ) / (len(left) ** 2)
    if math.isclose(expected, 0.0):
        return 1.0 if math.isclose(observed, 0.0) else 0.0
    return 1 - (observed / expected)


def _pearson(left: list[int], right: list[int]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if math.isclose(left_scale * right_scale, 0.0):
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def calculate_agreement(
    rater_annotations: list[list[HumanAnnotation]],
) -> HumanAgreementReport:
    if len(rater_annotations) < 2:
        raise ValueError("at least two completed raters are required")
    expected_ids = {item.item_id for item in rater_annotations[0]}
    if not expected_ids:
        raise ValueError("at least one annotation item is required")
    normalized: list[dict[str, HumanAnnotation]] = []
    for ratings in rater_annotations:
        by_id = {item.item_id: item for item in ratings}
        if len(by_id) != len(ratings) or set(by_id) != expected_ids:
            raise ValueError("all raters must contain the same unique item_id values")
        normalized.append(by_id)
    pair_metrics: list[tuple[float, float, float, float, float, float]] = []
    ordered_ids = sorted(expected_ids)
    for left, right in itertools.combinations(normalized, 2):
        left_totals = [left[item_id].total for item_id in ordered_ids]
        right_totals = [right[item_id].total for item_id in ordered_ids]
        differences = [
            abs(a - b) for a, b in zip(left_totals, right_totals, strict=True)
        ]
        left_bands = [_band(score) for score in left_totals]
        right_bands = [_band(score) for score in right_totals]
        pair_metrics.append(
            (
                mean(differences),
                _rate([difference <= 10 for difference in differences]),
                _rate([a == b for a, b in zip(left_bands, right_bands, strict=True)]),
                _cohen_kappa(left_bands, right_bands),
                _quadratic_weighted_kappa(left_totals, right_totals),
                _pearson(left_totals, right_totals),
            )
        )
    averages = [mean(values) for values in zip(*pair_metrics, strict=True)]
    return HumanAgreementReport(
        report_version="human-grading-agreement-v1",
        status="completed",
        rater_count=len(normalized),
        item_count=len(expected_ids),
        pair_count=len(pair_metrics),
        mean_absolute_total_difference=round(averages[0], 4),
        within_10_proportion=round(averages[1], 4),
        band_agreement=round(averages[2], 4),
        cohen_kappa=round(averages[3], 4),
        quadratic_weighted_kappa=round(averages[4], 4),
        pearson_correlation=round(averages[5], 4),
        privacy={
            "item_level_scores_public": False,
            "rater_names_public": False,
        },
    )


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_annotation_status(
    *, item_count: int, rater_slots: int
) -> HumanAnnotationStatus:
    return HumanAnnotationStatus(
        report_version="human-grading-annotation-status-v1",
        status="awaiting_human_labels",
        item_count=item_count,
        rater_slots=rater_slots,
        metrics=None,
        note=(
            "Annotation templates are ready, but no human agreement metric is claimed "
            "until at least two real raters complete all items independently."
        ),
    )


def write_public_annotation_status(
    status: HumanAnnotationStatus,
    path: Path = PUBLIC_ANNOTATION_STATUS_PATH,
) -> None:
    _atomic_text(
        path,
        json.dumps(status.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )


def write_public_agreement_report(
    report: HumanAgreementReport,
    path: Path = PUBLIC_ANNOTATION_STATUS_PATH,
) -> None:
    status = HumanAnnotationStatus(
        report_version="human-grading-annotation-status-v1",
        status="completed",
        item_count=report.item_count,
        rater_slots=report.rater_count,
        metrics=report,
        note="Agreement metrics are aggregated from completed independent human ratings.",
    )
    write_public_annotation_status(status, path)
