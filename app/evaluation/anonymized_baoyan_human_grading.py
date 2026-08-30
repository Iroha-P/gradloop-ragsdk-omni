# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

V3Category = Literal[
    "self_intro_story",
    "project_research",
    "professional_explanation",
    "english_expression",
    "pressure_followup",
    "application_decision",
]
V3Difficulty = Literal["basic", "intermediate", "advanced"]
V3AnswerStrength = Literal["weak", "partial", "strong"]

DATASET_ID_V3 = "public-synthetic-anonymized-baoyan-human-grading-v3"
RUBRIC_VERSION_V3 = "human-grading-five-dimension-v3"
MANIFEST_VERSION_V3 = "anonymized-baoyan-human-grading-manifest-v3"
PACK_VERSION_V3 = "anonymized-baoyan-human-grading-pack-v3"

# These canonical identities are pinned independently from the manifest.
# Fixture mode never treats regenerated hashes as authorization for scoring.
CANONICAL_V3_ITEMS_SHA256 = (
    "99799eac8d4b7a2578d4a5e66f31be0d5fd63d3899160f66d91320e71e3abb54"
)
CANONICAL_V3_MANIFEST_SHA256 = (
    "260aed2953341280681a69c281dab87d7947e16c3669ae94f552cc154a2e8f5e"
)

RUBRIC_LIMITS_V3: dict[str, int] = {
    "clarity": 15,
    "coverage": 30,
    "reasoning": 20,
    "evidence": 20,
    "reflection": 15,
}

RUBRIC_BANDS_V3: dict[str, dict[str, object]] = {
    "clarity": {
        "label": "表达清晰",
        "max": RUBRIC_LIMITS_V3["clarity"],
        "bands": (
            {
                "key": "absent",
                "label": "完全缺失",
                "min": 0,
                "max": 0,
                "representative": 0,
                "description": "无有效回答，或表达无法理解。",
            },
            {
                "key": "low",
                "label": "低档",
                "min": 1,
                "max": 4,
                "representative": 3,
                "description": "观点零散、主结论不清，术语没有解释，存在明显语病或自相矛盾。",
            },
            {
                "key": "medium",
                "label": "中档",
                "min": 5,
                "max": 8,
                "representative": 7,
                "description": "能识别主结论，但层次跳跃、重复较多，关键术语或对象指代仍有歧义。",
            },
            {
                "key": "good",
                "label": "良好",
                "min": 9,
                "max": 12,
                "representative": 11,
                "description": "结构基本清楚，术语大多准确，能够顺畅理解；仅有局部冗余或轻微歧义。",
            },
            {
                "key": "excellent",
                "label": "优秀",
                "min": 13,
                "max": 15,
                "representative": 14,
                "description": "结论先行、层次分明、表述简洁准确，术语和对象清楚，便于面试官继续追问。",
            },
        ),
    },
    "coverage": {
        "label": "要点覆盖",
        "max": RUBRIC_LIMITS_V3["coverage"],
        "bands": (
            {
                "key": "absent",
                "label": "完全缺失",
                "min": 0,
                "max": 0,
                "representative": 0,
                "description": "回答与问题无关，或没有任何可用要点。",
            },
            {
                "key": "low",
                "label": "低档",
                "min": 1,
                "max": 7,
                "representative": 4,
                "description": "只触及一个零散点，核心任务、过程或结论大部分缺失。",
            },
            {
                "key": "medium",
                "label": "中档",
                "min": 8,
                "max": 15,
                "representative": 12,
                "description": "覆盖若干相关点，但遗漏至少一个决定回答是否成立的核心部分。",
            },
            {
                "key": "good",
                "label": "良好",
                "min": 16,
                "max": 23,
                "representative": 20,
                "description": "覆盖大多数关键点，能够回应问题主干；仍缺少局部细节、优先级或必要展开。",
            },
            {
                "key": "excellent",
                "label": "优秀",
                "min": 24,
                "max": 30,
                "representative": 27,
                "description": "关键点完整且有取舍，覆盖问题所需深度；不以无关扩写或堆砌术语冒充完整。",
            },
        ),
    },
    "reasoning": {
        "label": "推理逻辑",
        "max": RUBRIC_LIMITS_V3["reasoning"],
        "bands": (
            {
                "key": "absent",
                "label": "完全缺失",
                "min": 0,
                "max": 0,
                "representative": 0,
                "description": "没有推理过程，或前后结论直接冲突。",
            },
            {
                "key": "low",
                "label": "低档",
                "min": 1,
                "max": 5,
                "representative": 3,
                "description": "主要是结论和口号，缺少因果链、判断依据或步骤关系。",
            },
            {
                "key": "medium",
                "label": "中档",
                "min": 6,
                "max": 10,
                "representative": 8,
                "description": "存在部分推理链，但有明显跳步、隐含假设或结论超出前提。",
            },
            {
                "key": "good",
                "label": "良好",
                "min": 11,
                "max": 15,
                "representative": 13,
                "description": "推理链总体连贯，主要假设和步骤能够对应结论；仅有局部缺口。",
            },
            {
                "key": "excellent",
                "label": "优秀",
                "min": 16,
                "max": 20,
                "representative": 18,
                "description": "前提、过程和结论严密连接，能比较替代解释或权衡，并说明为何选择当前判断。",
            },
        ),
    },
    "evidence": {
        "label": "证据意识",
        "max": RUBRIC_LIMITS_V3["evidence"],
        "bands": (
            {
                "key": "absent",
                "label": "完全缺失",
                "min": 0,
                "max": 0,
                "representative": 0,
                "description": "没有证据，使用无法核验的断言，或把猜测当作事实。",
            },
            {
                "key": "low",
                "label": "低档",
                "min": 1,
                "max": 5,
                "representative": 3,
                "description": "只使用“效果很好”等笼统描述，没有数据、案例、方法或可观察指标。",
            },
            {
                "key": "medium",
                "label": "中档",
                "min": 6,
                "max": 10,
                "representative": 8,
                "description": "提到数据、案例、实验或方法，但没有说明来源、对照、测量方式或如何支撑结论。",
            },
            {
                "key": "good",
                "label": "良好",
                "min": 11,
                "max": 15,
                "representative": 13,
                "description": "使用了具体证据并能连接到结论，验证方法或对照大体清楚；仍有局部可复核性缺口。",
            },
            {
                "key": "excellent",
                "label": "优秀",
                "min": 16,
                "max": 20,
                "representative": 18,
                "description": "证据链完整，来源、指标、对照和结论关系清楚，同时避免用有限证据作过度推断。",
            },
        ),
    },
    "reflection": {
        "label": "反思边界",
        "max": RUBRIC_LIMITS_V3["reflection"],
        "bands": (
            {
                "key": "absent",
                "label": "完全缺失",
                "min": 0,
                "max": 0,
                "representative": 0,
                "description": "完全不承认限制，给出明显不合理的绝对化结论。",
            },
            {
                "key": "low",
                "label": "低档",
                "min": 1,
                "max": 4,
                "representative": 3,
                "description": "只泛泛表示“还需改进”，没有指出具体限制、风险或适用条件。",
            },
            {
                "key": "medium",
                "label": "中档",
                "min": 5,
                "max": 8,
                "representative": 7,
                "description": "能指出至少一个具体限制，但没有解释其影响，也没有可执行的改进方向。",
            },
            {
                "key": "good",
                "label": "良好",
                "min": 9,
                "max": 12,
                "representative": 11,
                "description": "说明限制会怎样影响结论，并提出基本可行的验证、补救或下一步。",
            },
            {
                "key": "excellent",
                "label": "优秀",
                "min": 13,
                "max": 15,
                "representative": 14,
                "description": "主动区分事实、推断与未知，说明适用边界、风险和备选方案，并给出可验证的下一步。",
            },
        ),
    },
}

V3_PACK_FILE_NAMES = frozenset(
    {
        "评分者A_离线评分.html",
        "评分者B_离线评分.html",
        "README.txt",
        "SHA256SUMS.txt",
    }
)

V3_IMMUTABLE_FIELDS = [
    "item_id",
    "category",
    "difficulty",
    "question",
    "candidate_answer",
    "reference_points",
    "immutable_sha256",
]
V3_RATER_FIELDS = [
    "rater_slot",
    *V3_IMMUTABLE_FIELDS,
    *RUBRIC_LIMITS_V3,
    "confidence",
    "notes",
]

CATEGORY_QUOTAS_V3: dict[str, int] = {
    "self_intro_story": 4,
    "project_research": 7,
    "professional_explanation": 8,
    "english_expression": 3,
    "pressure_followup": 4,
    "application_decision": 4,
}
STRENGTH_QUOTAS_V3: dict[str, int] = {
    "weak": 10,
    "partial": 10,
    "strong": 10,
}
DIFFICULTY_QUOTAS_V3: dict[str, int] = {
    "basic": 8,
    "intermediate": 14,
    "advanced": 8,
}

_FORBIDDEN_PUBLIC_MARKERS = (
    "http://",
    "https://",
    "feishu",
    "source_refs",
    "question_id",
    "content_fingerprint",
)
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class V3HumanGradingItem(StrictModel):
    item_id: str = Field(pattern=r"^BHG3-\d{3}$")
    category: V3Category
    difficulty: V3Difficulty
    question: str = Field(min_length=8)
    candidate_answer: str = Field(min_length=8)
    reference_points: list[str] = Field(min_length=3, max_length=5)
    answer_strength: V3AnswerStrength
    immutable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("reference_points")
    @classmethod
    def _unique_reference_points(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reference points must be unique")
        if any(not point.strip() for point in value):
            raise ValueError("reference points must be non-empty")
        return value


class V3PrivateTaxonomyProfile(StrictModel):
    accepted_count: int = Field(ge=100)
    generic_count: int = Field(ge=0)
    personal_count: int = Field(ge=0)
    category_counts: dict[str, int]


class V3PublicHumanGradingItem(StrictModel):
    item_id: str = Field(pattern=r"^BHG3-\d{3}$")
    category: V3Category
    difficulty: V3Difficulty
    question: str = Field(min_length=8)
    candidate_answer: str = Field(min_length=8)
    reference_points: list[str] = Field(min_length=3, max_length=5)
    immutable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V3HumanGradingManifest(StrictModel):
    manifest_version: Literal["anonymized-baoyan-human-grading-manifest-v3"]
    dataset_id: Literal["public-synthetic-anonymized-baoyan-human-grading-v3"]
    rubric_version: Literal["human-grading-five-dimension-v3"]
    item_count: Literal[30]
    category_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    items_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    immutable_row_sha256: dict[str, str]
    private_taxonomy_summary: V3PrivateTaxonomyProfile | None
    privacy_scan_passed: Literal[True]


class V3HumanAnnotation(StrictModel):
    item_id: str = Field(pattern=r"^BHG3-\d{3}$")
    category: V3Category
    clarity: int = Field(ge=0, le=15)
    coverage: int = Field(ge=0, le=30)
    reasoning: int = Field(ge=0, le=20)
    evidence: int = Field(ge=0, le=20)
    reflection: int = Field(ge=0, le=15)
    confidence: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)

    @property
    def total(self) -> int:
        return sum(getattr(self, field) for field in RUBRIC_LIMITS_V3)


class V3CategoryAgreement(StrictModel):
    item_count: int = Field(ge=1)
    pair_count: int = Field(ge=1)
    total_mae: float = Field(ge=0)


class V3HumanAgreementReport(StrictModel):
    report_version: Literal["anonymized-baoyan-human-grading-agreement-v3"]
    status: Literal["completed"]
    rater_count: int = Field(ge=2)
    item_count: Literal[30]
    pair_count: int = Field(ge=1)
    dimension_mae: dict[str, float]
    total_mae: float = Field(ge=0)
    within_10_proportion: float = Field(ge=0, le=1)
    band_agreement: float = Field(ge=0, le=1)
    cohen_kappa: float = Field(ge=-1, le=1)
    quadratic_weighted_kappa: float = Field(ge=-1, le=1)
    pearson_correlation: float = Field(ge=-1, le=1)
    category_total_mae: dict[str, V3CategoryAgreement]
    privacy: dict[str, bool]


class V3AnnotationStatus(StrictModel):
    report_version: Literal["anonymized-baoyan-human-grading-status-v3"]
    dataset_id: Literal["public-synthetic-anonymized-baoyan-human-grading-v3"]
    status: Literal["awaiting_human_labels", "completed"]
    item_count: Literal[30]
    rater_count: int = Field(ge=0)
    required_rater_count: Literal[2]
    items_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: V3HumanAgreementReport | None
    privacy_scan_passed: Literal[True]
    note: str

    @model_validator(mode="after")
    def _metrics_match_status(self) -> V3AnnotationStatus:
        if self.status == "completed" and self.metrics is None:
            raise ValueError("completed v3 status requires metrics")
        if self.status == "awaiting_human_labels" and self.metrics is not None:
            raise ValueError("awaiting v3 status cannot contain metrics")
        return self


_V3_CASES: tuple[dict[str, object], ...] = (
    {
        "category": "self_intro_story",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "请用一分钟介绍自己，并说明希望继续深造的主要原因。",
        "candidate_answer": "我学习比较认真，也参加过一些课程和活动，希望以后继续提升自己，所以想继续深造。",
        "reference_points": ["形成清晰的个人主线", "给出具体能力证据", "说明深造动机", "自然留下可追问方向"],
    },
    {
        "category": "self_intro_story",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "如果只能保留两段经历，你会如何组织一段有重点的自我介绍？",
        "candidate_answer": (
            "我会选择一段课程研究和一段团队项目，分别说明我的分析能力与协作能力。"
            "介绍时先交代任务，再说明我做了什么，最后连接到后续学习方向；但还没有设计两段经历之间的统一主线。"
        ),
        "reference_points": ["说明经历筛选标准", "突出个人行动", "连接目标方向", "形成统一叙事线索"],
    },
    {
        "category": "self_intro_story",
        "difficulty": "intermediate",
        "answer_strength": "strong",
        "question": "请说明一项能够代表你学习方式的优势，并给出可验证证据。",
        "candidate_answer": (
            "我的优势是把模糊任务转化为可以验证的小问题。一次公开合成数据分析练习中，我先固定数据划分和评价指标，"
            "再按数据清洗、基线复现和误差分析三个阶段推进。首次结果不稳定后，我通过按来源分组发现了重复样本问题。"
            "修正后五次重复实验的波动明显收窄。这说明我的优势不只是推进速度，也包括主动检查结论是否可靠；"
            "但该证据来自小规模练习，还需要在更复杂任务中继续验证。"
        ),
        "reference_points": ["明确单一核心优势", "提供具体行动过程", "给出可核验结果", "说明能力边界", "引导后续追问"],
    },
    {
        "category": "self_intro_story",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "怎样把未来规划讲得具体，同时避免把尚未完成的目标说成既有成果？",
        "candidate_answer": (
            "我会把规划拆成问题、能力缺口、近期行动和验证标准。比如我希望研究可靠的知识检索，不会直接说自己已经具备完整研究能力，"
            "而会说明目前完成了可复现基线，接下来计划比较不同检索策略，并以新数据上的召回、引用正确性和拒答表现作为验收。"
            "如果实验未达到预设门槛，我会保留失败结果并调整假设。这样既说明方向，也区分了事实、计划和条件性判断。"
        ),
        "reference_points": ["区分已完成事实与未来计划", "说明能力缺口", "给出阶段行动", "定义验证标准", "保留失败分支"],
    },
    {
        "category": "project_research",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "请介绍一个团队项目中你个人负责的核心工作。",
        "candidate_answer": "我们一起完成了数据处理和模型训练，我主要参与模型部分，最后取得了不错的效果。",
        "reference_points": ["界定个人职责", "说明关键技术动作", "区分个人与团队贡献", "提供结果证据"],
    },
    {
        "category": "project_research",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "你认为项目中最主要的创新点是什么？",
        "candidate_answer": "创新点是采用了新的模型和新的处理流程，因此效果比原来更好。",
        "reference_points": ["说明相对哪一基线创新", "解释具体机制", "给出对照证据", "限定创新范围"],
    },
    {
        "category": "project_research",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "如果新方法只比简单基线提高少量指标，你会怎样判断改进是否有效？",
        "candidate_answer": (
            "我会使用相同数据划分重复实验，比较平均值和波动，并检查提升是否集中在少数样本。"
            "同时记录额外计算成本。如果波动大于提升幅度，就暂时不宣称有效；但我还需要预先确定重复次数和接受门槛。"
        ),
        "reference_points": ["保持比较条件一致", "报告重复实验与不确定性", "分析样本分布", "比较效果和成本", "预设验收门槛"],
    },
    {
        "category": "project_research",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "实验结果异常时，你会按照什么顺序定位数据、实现和模型问题？",
        "candidate_answer": (
            "我会先检查输入数据范围、缺失值和划分，再用少量样例逐步核对中间结果，最后检查训练参数和模型输出。"
            "修复后重新运行对照实验，并保存日志。不过我还没有说明如何冻结环境和避免同时修改多个变量。"
        ),
        "reference_points": ["给出分层排查顺序", "使用最小样例定位", "保存可观察证据", "控制单一变量", "验证修复可复现"],
    },
    {
        "category": "project_research",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "请设计一个能够验证项目核心模块真实贡献的消融实验。",
        "candidate_answer": (
            "我会先把主张写成：重排模块能在不超过既定时延预算的条件下提升困难查询的前五位命中率。"
            "随后固定数据、随机种子和检索候选，比较完整系统、移除重排、替换为简单规则以及仅扩大召回数量四组。"
            "每组重复五次，报告整体和困难子集指标、时延与置信区间。如果移除重排后收益消失且替代方案不能复现，"
            "才支持该模块在当前系统中的边际贡献；这仍不能证明所有分布上都有效，因此还需独立冻结集复验。"
        ),
        "reference_points": ["把创新主张写成可检验假设", "设置合理对照与替代实现", "固定其他条件", "同时报告效果与成本", "限定因果结论"],
    },
    {
        "category": "project_research",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "复现者得到与项目报告相反的结果时，你会如何处理？",
        "candidate_answer": (
            "我会先停止扩张原结论，并分别冻结双方的数据版本、依赖、配置和随机种子。"
            "随后用同一组公开样例逐层比较预处理、中间张量和最终指标，定位最早分叉点。"
            "如果问题来自我遗漏的默认参数，就更正说明并保留原报告；如果来自数据分布差异，就收缩结论适用范围。"
            "最后把复现步骤自动化，记录两次结果，而不是只保留支持原观点的一次。"
        ),
        "reference_points": ["暂停未经确认的主张", "冻结复现条件", "定位最早分叉点", "按原因修正结论", "保留相反证据"],
    },
    {
        "category": "project_research",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "请完整说明一个研究项目从问题定义到结论边界的证据链。",
        "candidate_answer": (
            "先把目标限定为在公开合成问答中减少无证据回答，同时保持可回答问题的覆盖。"
            "我会预先划分校准集和冻结测试集，比较固定规则、单一分数阈值和多特征分类器。"
            "阈值只在校准集选择，冻结集只运行一次，并同时报告拒答率、回答覆盖、检索召回、调用成本和失败案例。"
            "如果分类器只在合成数据通过，我只声称流程和边界控制在该分布有效，不外推为真实用户准确率；"
            "接下来还需匿名真实样本和人工语义评价。"
        ),
        "reference_points": ["明确问题和双侧约束", "设置基线与数据隔离", "预声明选择规则", "报告多维指标和失败", "限定外推范围"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "请解释什么是过拟合，以及它为什么会影响模型应用。",
        "candidate_answer": "过拟合就是模型把训练数据学得太好，换到新数据后效果会下降。",
        "reference_points": ["区分训练表现与泛化表现", "解释偶然模式或噪声", "给出诊断方法", "说明缓解思路"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "精确率与召回率分别反映什么？",
        "candidate_answer": "精确率表示结果准不准，召回率表示找得全不全，具体使用哪个要看任务。",
        "reference_points": ["准确给出两个定义", "说明假阳性与假阴性", "结合代价选择指标", "指出二者可能存在权衡"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "为什么实验需要设置对照组？",
        "candidate_answer": "有对照组才能比较实验前后是否有变化，也更容易说明方法有效。",
        "reference_points": ["说明反事实比较作用", "控制其他变量", "区分相关与因果", "解释对照组选择"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "请向非专业听众解释向量检索的基本流程和一个主要风险。",
        "candidate_answer": (
            "可以把每段文字转成一组表示含义的数字，再寻找与问题最接近的数字组合，"
            "因此即使措辞不同也可能找到相关内容。风险是数字上的接近不等于事实真的支持答案，"
            "所以需要引用和复核；但我还没有解释索引建立和如何评价检索质量。"
        ),
        "reference_points": ["使用易懂类比解释编码", "说明相似度召回", "解释语义匹配优势", "指出相关不等于支持", "提出评价方法"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "怎样判断两个变量之间的相关关系是否可能来自混杂因素？",
        "candidate_answer": (
            "我会先列出同时影响两个变量的候选因素，再通过分层比较或回归控制观察关系是否仍存在。"
            "如果控制后关系明显减弱，就说明原相关可能被混杂影响。不过观察数据仍难以排除未测量因素，"
            "还需要更强的实验设计。"
        ),
        "reference_points": ["定义混杂因素", "提出分层或控制方法", "比较控制前后关系", "承认未测量混杂", "提出实验验证"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "工程方案同时受到性能、成本和稳定性约束时，应如何比较？",
        "candidate_answer": (
            "我会先区分硬约束和可优化目标，排除不满足稳定性底线的方案，再比较性能与成本。"
            "可以绘制帕累托前沿并选择符合使用规模的点，同时进行压力测试。"
            "但我还需要说明指标权重如何由实际使用方确认。"
        ),
        "reference_points": ["识别硬约束", "量化多目标指标", "使用帕累托比较", "进行稳定性测试", "说明决策权重来源"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "请解释模型置信度校准，并说明为什么校准良好不等于预测准确。",
        "candidate_answer": (
            "如果一组预测都给出约百分之七十的概率，而长期看其中约七成确实发生，这组概率可以说较为校准。"
            "准确率只看最终类别是否判断正确，校准比较的是置信度与实际频率。"
            "一个模型可能分类正确很多，却对所有结果都过度自信；也可能分组概率吻合，但排序区分能力较弱。"
            "因此应同时报告任务指标、可靠性曲线和校准误差，并检查小样本分箱的不稳定性。"
        ),
        "reference_points": ["用频率解释校准", "区分校准与准确率", "提供反例", "提出联合评价", "说明统计局限"],
    },
    {
        "category": "professional_explanation",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "面对跨学科问题，如何把专业机理、可测变量和实验过程连接起来？",
        "candidate_answer": (
            "我会先把专业判断写成可观察的因果链，例如结构变化影响传输路径，进而改变宏观性能。"
            "然后为每一环选择可测变量和测量方法，设置只改变一个关键因素的对照，并记录可能的替代解释。"
            "分析时不只看最终相关性，还检查中间变量是否按预期变化。"
            "如果中间证据缺失，就把结论写成现象关联而不是完整机理，并提出补充实验。"
        ),
        "reference_points": ["建立机理链条", "映射到可测变量", "设计控制实验", "检查中间证据", "按证据强度限定结论"],
    },
    {
        "category": "english_expression",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "Please introduce your academic interests in about thirty seconds.",
        "candidate_answer": (
            "I am interested in artificial intelligence and data analysis. I have taken some courses and joined projects. "
            "I hope to learn more in graduate school."
        ),
        "reference_points": ["state a focused interest", "connect prior preparation", "give one concrete example", "present a clear next goal"],
    },
    {
        "category": "english_expression",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "Please summarize a project contribution and one limitation in English.",
        "candidate_answer": (
            "I built the evaluation pipeline for a synthetic retrieval project and compared a lexical baseline with a semantic method. "
            "The semantic method performed better on paraphrased questions. However, the dataset was small, so the conclusion may not generalize. "
            "I would add a frozen test set, but I have not explained how I would control computational cost."
        ),
        "reference_points": ["define the personal contribution", "describe the comparison", "report a bounded result", "state a limitation", "propose a concrete next test"],
    },
    {
        "category": "english_expression",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "An interviewer says your explanation is unclear. Please clarify the core claim and evidence in English.",
        "candidate_answer": (
            "Thank you for pointing that out. My claim is not that the whole system is universally better. "
            "The narrower claim is that, on a frozen synthetic test set, the evidence gate reduced unsupported answers while preserving retrieval recall. "
            "The threshold was selected only on a separate calibration set, and the test set was evaluated once. "
            "This evidence does not establish real-user accuracy, so the next step is an anonymized human evaluation."
        ),
        "reference_points": ["acknowledge the clarification request", "state a narrower claim", "name the supporting evidence", "separate calibration and testing", "limit external validity"],
    },
    {
        "category": "pressure_followup",
        "difficulty": "basic",
        "answer_strength": "weak",
        "question": "评审认为你的贡献很少，你会怎样回应？",
        "candidate_answer": "我确实做了很多工作，只是刚才没有全部说出来，我可以再详细介绍。",
        "reference_points": ["正面接受质疑", "明确个人责任边界", "提供可验证产出", "承认团队贡献"],
    },
    {
        "category": "pressure_followup",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "当被连续追问到不熟悉的知识点时，你会如何处理？",
        "candidate_answer": (
            "我会先说明自己目前确定的部分，再明确指出不确定的假设，尝试从基本原理推导。"
            "如果无法得到可靠结论，就坦诚需要进一步核实，并说明会查阅哪些资料。"
            "但我还需要避免用过长的推导拖延回答。"
        ),
        "reference_points": ["区分已知与未知", "从基础原理推导", "避免编造结论", "给出核实路径", "保持回答节奏"],
    },
    {
        "category": "pressure_followup",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "项目结果没有达到预期时，你会怎样承担责任并说明下一步？",
        "candidate_answer": (
            "我会先报告未达标的事实和受影响范围，再区分是目标设定、数据还是实现问题。"
            "随后给出一个最小排查计划和新的检查点，同时保留失败结果。"
            "我不会把原因全部归给外部条件，但还需要说明如何提前设置风险预警。"
        ),
        "reference_points": ["如实报告偏差", "界定责任与影响", "提出最小排查计划", "保留失败证据", "建立预警机制"],
    },
    {
        "category": "pressure_followup",
        "difficulty": "advanced",
        "answer_strength": "strong",
        "question": "如果评审指出你的核心假设可能错误，请给出一套可执行回应。",
        "candidate_answer": (
            "我会先复述质疑，确认争点是样本独立性假设，而不是实现细节。"
            "接着预先登记两项反证测试：按来源重新分组划分，以及提高近重复比例观察指标变化。"
            "如果分组后提升从明显优势降到接近零，我会撤回普遍有效的表述；如果结果稳定，也只说明这两项测试未推翻假设。"
            "原始报告和新结果都会保留，并继续增加时间顺序划分，因为单次反证不能穷尽所有泄漏途径。"
        ),
        "reference_points": ["准确界定被质疑假设", "设计可证伪测试", "预先规定结果解释", "愿意收缩结论", "保留完整证据"],
    },
    {
        "category": "application_decision",
        "difficulty": "intermediate",
        "answer_strength": "weak",
        "question": "联系研究团队前，你会如何判断自己的方向是否匹配？",
        "candidate_answer": "我会查看团队主页和研究方向，如果感觉比较感兴趣，就准备材料进行联系。",
        "reference_points": ["阅读近期工作而非只看标签", "比较能力与问题匹配", "识别可贡献切入点", "准备针对性问题"],
    },
    {
        "category": "application_decision",
        "difficulty": "intermediate",
        "answer_strength": "weak",
        "question": "面对多个学习机会时，你会依据什么做选择？",
        "candidate_answer": "我会综合考虑平台、方向和未来发展，选择总体条件更好、自己也更喜欢的机会。",
        "reference_points": ["明确个人优先级", "比较培养与研究匹配", "识别不确定信息", "设置决策截止时间"],
    },
    {
        "category": "application_decision",
        "difficulty": "intermediate",
        "answer_strength": "partial",
        "question": "申请准备、课程任务和项目工作同时发生时，你会如何安排？",
        "candidate_answer": (
            "我会先列出不可移动的截止时间和最低交付标准，再按影响与紧急程度分配每周时间块。"
            "申请材料采用版本清单，项目保留固定深度工作时段，每周复盘一次偏差。"
            "不过我还需要预留突发面试和材料返工的缓冲。"
        ),
        "reference_points": ["识别硬截止时间", "定义最低交付标准", "进行时间分块", "建立版本与复盘机制", "预留缓冲"],
    },
    {
        "category": "application_decision",
        "difficulty": "intermediate",
        "answer_strength": "strong",
        "question": "如何在信息不完整时比较两个方向，并避免只凭短期印象决定？",
        "candidate_answer": (
            "我会先把选择拆成研究问题、指导方式、能力成长、资源条件和风险五个维度，并为各维度写出证据来源。"
            "对于缺失信息，不用想象补齐，而是通过公开成果、在读成员交流和一次小任务验证。"
            "随后设置必须满足的底线与可权衡项，做敏感性分析：如果权重小幅变化就改变结论，说明选择仍不稳。"
            "最后在预设日期根据当时证据决策，并记录放弃选项的理由，避免结果出来后倒推标准。"
        ),
        "reference_points": ["建立多维决策框架", "区分事实与未知", "主动获取验证证据", "检查权重敏感性", "预设决策规则"],
    },
)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def _immutable_payload(
    *,
    item_id: str,
    category: str,
    difficulty: str,
    question: str,
    candidate_answer: str,
    reference_points: list[str],
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "candidate_answer": candidate_answer,
        "reference_points": reference_points,
    }


def _immutable_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_item(index: int, payload: dict[str, object]) -> V3HumanGradingItem:
    item_id = f"BHG3-{index:03d}"
    reference_points = [str(point) for point in payload["reference_points"]]
    immutable = _immutable_payload(
        item_id=item_id,
        category=str(payload["category"]),
        difficulty=str(payload["difficulty"]),
        question=str(payload["question"]),
        candidate_answer=str(payload["candidate_answer"]),
        reference_points=reference_points,
    )
    return V3HumanGradingItem(
        **immutable,
        answer_strength=str(payload["answer_strength"]),
        immutable_sha256=_immutable_hash(immutable),
    )


def _assert_public_shareable_text(text: str) -> None:
    lowered = text.lower()
    for marker in _FORBIDDEN_PUBLIC_MARKERS:
        if marker in lowered:
            raise ValueError("public v3 text contains a forbidden private or provider marker")
    if _WINDOWS_ABSOLUTE_PATH_PATTERN.search(text):
        raise ValueError("public v3 text contains a forbidden private or provider marker")


def validate_v3_items(
    items: list[V3HumanGradingItem],
    *,
    private_questions: list[str],
) -> None:
    if len(items) != 30:
        raise ValueError("v3 dataset must contain exactly 30 items")
    if Counter(item.category for item in items) != CATEGORY_QUOTAS_V3:
        raise ValueError("v3 category quotas do not match the frozen design")
    if Counter(item.answer_strength for item in items) != STRENGTH_QUOTAS_V3:
        raise ValueError("v3 answer-strength quotas do not match the frozen design")
    if Counter(item.difficulty for item in items) != DIFFICULTY_QUOTAS_V3:
        raise ValueError("v3 difficulty quotas do not match the frozen design")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("v3 item IDs must be unique")
    if len({item.question for item in items}) != len(items):
        raise ValueError("v3 questions must be unique")

    normalized_private = [
        normalized
        for question in private_questions
        if (normalized := _normalize_text(question))
    ]
    for item in items:
        public_text = "\n".join(
            [item.question, item.candidate_answer, *item.reference_points]
        )
        _assert_public_shareable_text(public_text)
        normalized_question = _normalize_text(item.question)
        for private_question in normalized_private:
            ratio = SequenceMatcher(
                None,
                normalized_question,
                private_question,
                autojunk=False,
            ).ratio()
            if ratio >= 0.82:
                raise ValueError("private-question similarity gate rejected a public v3 item")


def build_v3_items() -> list[V3HumanGradingItem]:
    items = [_build_item(index, payload) for index, payload in enumerate(_V3_CASES, start=1)]
    validate_v3_items(items, private_questions=[])
    return items


def _public_item_payload(item: V3HumanGradingItem) -> dict[str, object]:
    return _immutable_payload(
        item_id=item.item_id,
        category=item.category,
        difficulty=item.difficulty,
        question=item.question,
        candidate_answer=item.candidate_answer,
        reference_points=item.reference_points,
    ) | {"immutable_sha256": item.immutable_sha256}


def serialize_v3_public_items(items: list[V3HumanGradingItem]) -> str:
    validate_v3_items(items, private_questions=[])
    return "".join(
        json.dumps(_public_item_payload(item), ensure_ascii=False, sort_keys=True)
        + "\n"
        for item in items
    )


def _read_question_bank(path: Path) -> list[dict[str, object]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("question bank is unavailable") from exc
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"question bank contains invalid JSON at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"question bank row {line_number} must be an object")
        rows.append(payload)
    return rows


def _classify_private_intent(row: dict[str, object]) -> str:
    searchable = " ".join(
        str(row.get(field, ""))
        for field in ("topic", "direction", "canonical_text", "normalized_text")
    ).lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("english_expression", ("english", "英文", "英语")),
        (
            "application_decision",
            ("申请", "导师", "联系", "选择", "offer", "时间管理", "夏令营", "预推免"),
        ),
        (
            "pressure_followup",
            ("压力", "质疑", "冲突", "责任", "困难", "失败", "不会", "追问"),
        ),
        (
            "self_intro_story",
            ("自我介绍", "动机", "优势", "未来规划", "个人成长", "专业选择"),
        ),
        (
            "project_research",
            ("项目", "科研", "研究", "实验", "创新", "复现", "基线", "消融", "贡献"),
        ),
    )
    for category, keywords in rules:
        if any(keyword in searchable for keyword in keywords):
            return category
    return "professional_explanation"


def profile_accepted_question_bank(path: Path) -> V3PrivateTaxonomyProfile:
    accepted = [
        row
        for row in _read_question_bank(path)
        if row.get("review_status") == "accepted"
    ]
    if len(accepted) < 100:
        raise ValueError("accepted question bank is too small for v3 profiling")
    generic_count = sum(row.get("privacy_lane") == "generic" for row in accepted)
    personal_count = sum(row.get("privacy_lane") == "personal" for row in accepted)
    if generic_count + personal_count != len(accepted):
        raise ValueError("accepted question bank contains an unknown privacy lane")
    category_counts = Counter(_classify_private_intent(row) for row in accepted)
    return V3PrivateTaxonomyProfile(
        accepted_count=len(accepted),
        generic_count=generic_count,
        personal_count=personal_count,
        category_counts=dict(sorted(category_counts.items())),
    )


def load_private_profile_inputs(
    path: Path,
) -> tuple[V3PrivateTaxonomyProfile, list[str]]:
    rows = [
        row
        for row in _read_question_bank(path)
        if row.get("review_status") == "accepted"
    ]
    profile = profile_accepted_question_bank(path)
    questions = [
        str(row.get("canonical_text") or row.get("normalized_text") or "")
        for row in rows
        if str(row.get("canonical_text") or row.get("normalized_text") or "").strip()
    ]
    return profile, questions


def _format_band_range(minimum: int, maximum: int) -> str:
    return str(minimum) if minimum == maximum else f"{minimum}–{maximum}"


def _render_rubric_rows() -> str:
    rows: list[str] = []
    for field, rubric in RUBRIC_BANDS_V3.items():
        label = escape(str(rubric["label"]))
        maximum = int(rubric["max"])
        bands = tuple(rubric["bands"])
        buttons = "".join(
            (
                f'<button class="band-button" type="button" '
                f'data-band-field="{field}" '
                f'data-band-key="{escape(str(band["key"]))}" '
                f'data-representative="{int(band["representative"])}" '
                f'aria-pressed="false">'
                f'<span>{escape(str(band["label"]))}</span>'
                f'<small>{_format_band_range(int(band["min"]), int(band["max"]))}</small>'
                "</button>"
            )
            for band in bands
        )
        references = "".join(
            (
                '<div class="band-reference-item">'
                '<div class="band-reference-meta">'
                f'<strong>{escape(str(band["label"]))}</strong>'
                f'<span>{_format_band_range(int(band["min"]), int(band["max"]))}</span>'
                "</div>"
                f'<p>{escape(str(band["description"]))}</p>'
                "</div>"
            )
            for band in bands
        )
        rows.append(
            f"""
        <div class="rubric-row" data-field="{field}">
          <div class="rubric-head"><span class="rubric-name">{label}</span><span id="{field}-value" class="rubric-value">未评分 / {maximum}</span></div>
          <div class="scale-line"><button class="step-button" data-action="minus" type="button">−</button><input data-field="{field}" type="range" min="0" max="{maximum}" value="0"><button class="step-button" data-action="plus" type="button">+</button></div>
          <p id="{field}-band-current" class="band-current" aria-live="polite">尚未评分</p>
          <div class="band-buttons" data-band-controls="{field}">{buttons}</div>
          <details class="band-reference">
            <summary>查看五档完整标准</summary>
            <div class="band-reference-list" data-band-reference="{field}">{references}</div>
          </details>
        </div>"""
        )
    return "".join(rows)


_OFFLINE_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>匿名双盲评分 · 评分者__RATER_SLOT__</title>
  <style>
    :root {
      --canvas: #F3F6FA;
      --paper: #FFFFFF;
      --ink: #142033;
      --muted: #607089;
      --line: #D8E0EA;
      --blue: #2859D6;
      --blue-soft: #EAF0FF;
      --green: #11866F;
      --green-soft: #E7F5F1;
      --amber: #B56B12;
      --amber-soft: #FFF3DF;
      --shadow: 0 14px 38px rgba(20, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    html { background: var(--canvas); color: var(--ink); }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background: var(--canvas);
    }
    button, input, textarea { font: inherit; }
    button { cursor: pointer; }
    .app-shell {
      width: min(1480px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: end;
      padding: 22px 26px;
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    h1 { margin: 0; font-size: clamp(24px, 3vw, 36px); letter-spacing: -0.03em; }
    .subtitle { margin: 9px 0 0; color: var(--muted); line-height: 1.65; }
    .progress-block { min-width: 260px; text-align: right; }
    .progress-count { font-weight: 800; }
    .progress-track {
      height: 8px;
      margin-top: 10px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #EEF2F7;
    }
    .progress-fill { width: 0; height: 100%; background: var(--green); transition: width 160ms ease; }
    .notice {
      margin-top: 16px;
      padding: 14px 18px;
      border: 1px solid #E8D5B7;
      background: var(--amber-soft);
      color: #6C430F;
      line-height: 1.65;
    }
    .notice strong { color: var(--amber); }
    .workspace {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr) 390px;
      gap: 16px;
      margin-top: 16px;
      align-items: start;
    }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .sidebar { position: sticky; top: 16px; padding: 18px; }
    .panel-title {
      margin: 0 0 12px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-transform: uppercase;
    }
    .category-progress { display: grid; gap: 8px; margin-bottom: 16px; }
    .category-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .category-row strong { color: var(--ink); font-family: Consolas, monospace; }
    .filter-toggle {
      display: flex;
      gap: 9px;
      align-items: center;
      padding: 12px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    .item-nav {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 7px;
      margin-top: 14px;
      max-height: 330px;
      overflow: auto;
    }
    .item-nav button {
      min-height: 38px;
      border: 1px solid var(--line);
      background: #F8FAFC;
      color: var(--muted);
      font-family: Consolas, monospace;
      font-size: 12px;
    }
    .item-nav button.active { border-color: var(--blue); background: var(--blue); color: #fff; }
    .item-nav button.done:not(.active) { border-color: #9ED7C8; background: var(--green-soft); color: var(--green); }
    .item-nav button.hidden { display: none; }
    .reset-button {
      width: 100%;
      margin-top: 16px;
      padding: 10px 12px;
      border: 1px solid #D7B886;
      background: transparent;
      color: var(--amber);
      font-weight: 700;
    }
    .content-panel { padding: 24px; }
    .item-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
    .tag {
      padding: 5px 9px;
      border: 1px solid var(--line);
      background: #F8FAFC;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .tag.primary { border-color: #B8C8F7; background: var(--blue-soft); color: var(--blue); }
    .content-section {
      padding: 18px 0;
      border-top: 1px solid var(--line);
    }
    .content-section:first-of-type { border-top: 0; padding-top: 0; }
    .section-label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
    }
    .question {
      margin: 0;
      font-size: clamp(20px, 2.2vw, 28px);
      line-height: 1.55;
      letter-spacing: -0.015em;
    }
    .answer {
      margin: 0;
      padding: 18px;
      border-left: 4px solid var(--blue);
      background: #F8FAFD;
      line-height: 1.85;
      white-space: pre-wrap;
    }
    .reference-list { margin: 0; padding-left: 22px; color: #304158; line-height: 1.8; }
    .content-nav { display: flex; justify-content: space-between; gap: 12px; margin-top: 18px; }
    .secondary-button, .primary-button {
      min-height: 44px;
      padding: 0 18px;
      border: 1px solid var(--line);
      font-weight: 800;
    }
    .secondary-button { background: #fff; color: var(--ink); }
    .primary-button { border-color: var(--blue); background: var(--blue); color: #fff; }
    .secondary-button:disabled, .primary-button:disabled { cursor: not-allowed; opacity: 0.45; }
    .rubric-panel { position: sticky; top: 16px; padding: 20px; }
    .rubric-intro { margin: -4px 0 16px; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .rubric-row {
      padding: 15px 0;
      border-top: 1px solid var(--line);
    }
    .rubric-row:first-of-type { border-top: 0; }
    .rubric-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
    .rubric-name { font-weight: 800; }
    .rubric-value {
      color: var(--blue);
      font-family: Consolas, monospace;
      font-size: 16px;
      font-weight: 800;
    }
    .scale-line { display: grid; grid-template-columns: 34px minmax(0, 1fr) 34px; gap: 8px; align-items: center; margin-top: 10px; }
    .step-button { width: 34px; height: 34px; border: 1px solid var(--line); background: #fff; color: var(--blue); font-weight: 900; }
    input[type="range"] { width: 100%; accent-color: var(--blue); }
    .band-current {
      min-height: 46px;
      margin: 10px 0 9px;
      padding: 9px 11px;
      border: 1px solid #D6E1FF;
      background: #F5F8FF;
      color: #334A78;
      font-size: 12px;
      line-height: 1.55;
    }
    .band-buttons { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; }
    .band-button {
      display: grid;
      gap: 3px;
      min-height: 48px;
      padding: 7px 4px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      text-align: center;
    }
    .band-button span { font-size: 11px; font-weight: 800; }
    .band-button small { font-family: Consolas, monospace; font-size: 10px; }
    .band-button.active {
      border-color: var(--blue);
      background: var(--blue-soft);
      color: var(--blue);
      box-shadow: inset 0 0 0 1px var(--blue);
    }
    .band-reference { margin-top: 9px; color: var(--muted); font-size: 12px; }
    .band-reference summary { cursor: pointer; color: #425576; font-weight: 800; }
    .band-reference-list { display: grid; gap: 7px; margin-top: 9px; }
    .band-reference-item {
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr);
      gap: 9px;
      padding: 8px;
      border: 1px solid #E1E7F0;
      background: #FBFCFE;
    }
    .band-reference-meta { display: grid; align-content: start; gap: 2px; }
    .band-reference-meta strong { color: #304158; font-size: 11px; }
    .band-reference-meta span { font-family: Consolas, monospace; font-size: 10px; }
    .band-reference-item p { margin: 0; color: #4E5E74; line-height: 1.55; }
    .confidence-block { padding-top: 16px; border-top: 1px solid var(--line); }
    .confidence-buttons { display: grid; grid-template-columns: repeat(5, 1fr); gap: 7px; margin-top: 9px; }
    .confidence-buttons button { height: 36px; border: 1px solid var(--line); background: #fff; color: var(--muted); font-family: Consolas, monospace; }
    .confidence-buttons button.active { border-color: var(--green); background: var(--green-soft); color: var(--green); font-weight: 800; }
    textarea {
      width: 100%;
      min-height: 84px;
      margin-top: 10px;
      padding: 11px;
      resize: vertical;
      border: 1px solid var(--line);
      color: var(--ink);
      background: #FBFCFE;
      line-height: 1.55;
    }
    .save-state { min-height: 22px; margin: 10px 0 0; color: var(--green); font-size: 12px; }
    .export-area {
      display: grid;
      gap: 8px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }
    .export-hint { color: var(--muted); font-size: 12px; line-height: 1.55; }
    .export-button { min-height: 46px; border: 1px solid var(--green); background: var(--green); color: #fff; font-weight: 900; }
    .export-button:disabled { cursor: not-allowed; border-color: var(--line); background: #E7EBF0; color: #8B97A8; }
    @media (max-width: 1120px) {
      .workspace { grid-template-columns: 230px minmax(0, 1fr); }
      .rubric-panel { position: static; grid-column: 1 / -1; }
    }
    @media (max-width: 760px) {
      .app-shell { width: min(100% - 20px, 720px); padding-top: 10px; }
      .topbar { grid-template-columns: 1fr; align-items: start; padding: 18px; }
      .progress-block { min-width: 0; text-align: left; }
      .workspace { grid-template-columns: 1fr; }
      .sidebar, .rubric-panel { position: static; }
      .sidebar { order: 2; }
      .content-panel { order: 1; padding: 18px; }
      .rubric-panel { order: 3; }
      .item-nav { grid-template-columns: repeat(10, 1fr); }
      .band-buttons { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .band-button:last-child { grid-column: 1 / -1; }
      .band-reference-item { grid-template-columns: 68px minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">Independent Human Review · Slot __RATER_SLOT__</p>
        <h1>匿名双盲回答质量评分</h1>
        <p class="subtitle">请独立判断候选回答本身，不推测回答来源，不与另一位评分者讨论。预计用时35—50分钟。</p>
      </div>
      <div class="progress-block">
        <div class="progress-count">已完成 <span id="completed-count">0</span> / 30</div>
        <div class="progress-track" aria-label="总进度"><div id="progress-fill" class="progress-fill"></div></div>
      </div>
    </header>

    <div id="storage-notice" class="notice" hidden>
      <strong>浏览器存储不可用。</strong> 当前会话仍可评分，但刷新或关闭页面会丢失进度，请完成后立即导出。
    </div>

    <main class="workspace">
      <aside class="panel sidebar">
        <h2 class="panel-title">章节进度</h2>
        <div id="category-progress" class="category-progress"></div>
        <label class="filter-toggle">
          <input id="unfinished-only" type="checkbox">
          只显示未完成题目
        </label>
        <div id="item-nav" class="item-nav" aria-label="题目导航"></div>
        <button id="reset-button" class="reset-button" type="button">清空本地进度</button>
      </aside>

      <article class="panel content-panel">
        <div class="item-meta">
          <span id="item-number" class="tag primary"></span>
          <span id="item-category" class="tag"></span>
          <span id="item-difficulty" class="tag"></span>
        </div>
        <section class="content-section">
          <span class="section-label">面试问题</span>
          <h2 id="question" class="question"></h2>
        </section>
        <section class="content-section">
          <span class="section-label">候选回答</span>
          <p id="candidate-answer" class="answer"></p>
        </section>
        <section class="content-section">
          <span class="section-label">参考观察点 · 用于校准覆盖范围，不是标准答案</span>
          <ol id="reference-points" class="reference-list"></ol>
        </section>
        <div class="content-nav">
          <button id="previous-button" class="secondary-button" type="button">上一题</button>
          <button id="next-button" class="primary-button" type="button">下一题</button>
        </div>
      </article>

      <aside class="panel rubric-panel">
        <h2 class="panel-title">量表尺</h2>
        <p class="rubric-intro"><strong>先选档、再微调；五个维度独立判断。</strong><br>不要先凭总印象给总分。参考观察点用于校准覆盖范围，不是唯一标准答案。</p>

__RUBRIC_ROWS__

        <div class="confidence-block">
          <span class="section-label">评分把握度（可选）</span>
          <div id="confidence-buttons" class="confidence-buttons">
            <button type="button" data-confidence="1">1</button><button type="button" data-confidence="2">2</button>
            <button type="button" data-confidence="3">3</button><button type="button" data-confidence="4">4</button>
            <button type="button" data-confidence="5">5</button>
          </div>
          <label class="section-label" for="notes" style="margin-top:14px">备注（可选）</label>
          <textarea id="notes" maxlength="1000" placeholder="记录拿不准的地方；不要填写姓名或联系方式。"></textarea>
          <p id="save-state" class="save-state" aria-live="polite"></p>
        </div>

        <div class="export-area">
          <span class="export-hint">全部30题的五个维度完成后才能导出。导出文件请原样返回，不要用表格软件改写。</span>
          <button id="export-button" class="export-button" type="button" disabled>导出评分 CSV</button>
        </div>
      </aside>
    </main>
  </div>

  <script>
    "use strict";
    const RATER_SLOT = "__RATER_SLOT__";
    const ITEMS = __ITEMS_JSON__;
    const RUBRICS = __RUBRICS_JSON__;
    const STORAGE_KEY = "anonymized-baoyan-human-grading-v3-rater-" + RATER_SLOT;
    const DIMENSIONS = Object.entries(RUBRICS).map(([field, rubric]) => ({
      field,
      label: rubric.label,
      max: rubric.max,
      bands: rubric.bands
    }));
    const CATEGORY_LABELS = {
      self_intro_story: "自我介绍与申请叙事",
      project_research: "项目与科研深挖",
      professional_explanation: "专业知识与跨学科解释",
      english_expression: "英文表达",
      pressure_followup: "压力面与连续追问",
      application_decision: "申请决策与流程"
    };
    const DIFFICULTY_LABELS = { basic: "基础", intermediate: "进阶", advanced: "高阶" };
    let currentIndex = 0;
    let storageAvailable = true;
    let state = { ratings: {} };

    function blankRating() {
      return { clarity: null, coverage: null, reasoning: null, evidence: null, reflection: null, confidence: null, notes: "" };
    }
    function ensureRatings() {
      ITEMS.forEach((item) => {
        if (!state.ratings[item.item_id]) state.ratings[item.item_id] = blankRating();
      });
    }
    function loadState() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && parsed.ratings && typeof parsed.ratings === "object") state = parsed;
        }
      } catch (error) {
        storageAvailable = false;
        document.getElementById("storage-notice").hidden = false;
      }
      ensureRatings();
    }
    function persistState() {
      const saveState = document.getElementById("save-state");
      if (!storageAvailable) {
        saveState.textContent = "仅保存在当前会话，请及时导出。";
        return;
      }
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        saveState.textContent = "已自动保存在本机浏览器。";
      } catch (error) {
        storageAvailable = false;
        document.getElementById("storage-notice").hidden = false;
        saveState.textContent = "无法写入浏览器存储，请及时导出。";
      }
    }
    function isComplete(itemId) {
      const rating = state.ratings[itemId] || blankRating();
      return DIMENSIONS.every((dimension) => Number.isInteger(rating[dimension.field]));
    }
    function completedCount() {
      return ITEMS.filter((item) => isComplete(item.item_id)).length;
    }
    function currentRating() {
      return state.ratings[ITEMS[currentIndex].item_id];
    }
    function setDimension(field, rawValue) {
      const dimension = DIMENSIONS.find((entry) => entry.field === field);
      if (!dimension) return;
      const value = Math.max(0, Math.min(dimension.max, Number(rawValue)));
      currentRating()[field] = value;
      persistState();
      renderRubric();
      renderProgress();
    }
    function bandForScore(dimension, score) {
      if (score === null) return null;
      return dimension.bands.find((band) => score >= band.min && score <= band.max) || null;
    }
    function renderRubric() {
      const rating = currentRating();
      DIMENSIONS.forEach((dimension) => {
        const input = document.querySelector('input[data-field="' + dimension.field + '"]');
        const row = input.closest(".rubric-row");
        const value = rating[dimension.field];
        const band = bandForScore(dimension, value);
        input.value = value === null ? "0" : String(value);
        document.getElementById(dimension.field + "-value").textContent =
          value === null ? "未评分 / " + dimension.max : value + " / " + dimension.max;
        document.getElementById(dimension.field + "-band-current").textContent =
          band === null ? "尚未评分" : band.label + " · " + band.description;
        row.querySelectorAll("[data-band-key]").forEach((button) => {
          button.classList.toggle("active", value !== null && band !== null && button.dataset.bandKey === band.key);
          button.setAttribute("aria-pressed", String(value !== null && band !== null && button.dataset.bandKey === band.key));
        });
      });
      document.querySelectorAll("[data-confidence]").forEach((button) => {
        button.classList.toggle("active", Number(button.dataset.confidence) === rating.confidence);
      });
      document.getElementById("notes").value = rating.notes || "";
    }
    function renderItem() {
      const item = ITEMS[currentIndex];
      document.getElementById("item-number").textContent = "第 " + (currentIndex + 1) + " / " + ITEMS.length + " 题";
      document.getElementById("item-category").textContent = CATEGORY_LABELS[item.category];
      document.getElementById("item-difficulty").textContent = DIFFICULTY_LABELS[item.difficulty];
      document.getElementById("question").textContent = item.question;
      document.getElementById("candidate-answer").textContent = item.candidate_answer;
      const list = document.getElementById("reference-points");
      list.replaceChildren();
      item.reference_points.forEach((point) => {
        const li = document.createElement("li");
        li.textContent = point;
        list.appendChild(li);
      });
      document.getElementById("previous-button").disabled = currentIndex === 0;
      document.getElementById("next-button").disabled = currentIndex === ITEMS.length - 1;
      renderRubric();
      renderProgress();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    function renderProgress() {
      const completed = completedCount();
      document.getElementById("completed-count").textContent = String(completed);
      document.getElementById("progress-fill").style.width = (completed / ITEMS.length * 100) + "%";
      document.getElementById("export-button").disabled = completed !== ITEMS.length;
      const unfinishedOnly = document.getElementById("unfinished-only").checked;
      document.querySelectorAll("#item-nav button").forEach((button, index) => {
        const done = isComplete(ITEMS[index].item_id);
        button.classList.toggle("active", index === currentIndex);
        button.classList.toggle("done", done);
        button.classList.toggle("hidden", unfinishedOnly && done && index !== currentIndex);
      });
      const counts = {};
      ITEMS.forEach((item) => {
        if (!counts[item.category]) counts[item.category] = { completed: 0, total: 0 };
        counts[item.category].total += 1;
        if (isComplete(item.item_id)) counts[item.category].completed += 1;
      });
      const categoryProgress = document.getElementById("category-progress");
      categoryProgress.replaceChildren();
      Object.entries(CATEGORY_LABELS).forEach(([category, label]) => {
        const row = document.createElement("div");
        row.className = "category-row";
        const name = document.createElement("span");
        const value = document.createElement("strong");
        name.textContent = label;
        value.textContent = counts[category].completed + "/" + counts[category].total;
        row.append(name, value);
        categoryProgress.appendChild(row);
      });
    }
    function buildNavigation() {
      const nav = document.getElementById("item-nav");
      ITEMS.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = String(index + 1).padStart(2, "0");
        button.title = CATEGORY_LABELS[item.category];
        button.addEventListener("click", () => {
          currentIndex = index;
          renderItem();
        });
        nav.appendChild(button);
      });
    }
    function csvCell(value) {
      const text = value === null || value === undefined ? "" : String(value);
      return '"' + text.replaceAll('"', '""') + '"';
    }
    function exportCsv() {
      if (completedCount() !== ITEMS.length) {
        alert("全部30题的五个维度完成后才能导出。");
        return;
      }
      const fields = [
        "rater_slot", "item_id", "category", "difficulty", "question", "candidate_answer",
        "reference_points", "immutable_sha256", "clarity", "coverage", "reasoning",
        "evidence", "reflection", "confidence", "notes"
      ];
      const rows = [fields.map(csvCell).join(",")];
      ITEMS.forEach((item) => {
        const rating = state.ratings[item.item_id];
        const row = {
          rater_slot: RATER_SLOT,
          item_id: item.item_id,
          category: item.category,
          difficulty: item.difficulty,
          question: item.question,
          candidate_answer: item.candidate_answer,
          reference_points: JSON.stringify(item.reference_points),
          immutable_sha256: item.immutable_sha256,
          clarity: rating.clarity,
          coverage: rating.coverage,
          reasoning: rating.reasoning,
          evidence: rating.evidence,
          reflection: rating.reflection,
          confidence: rating.confidence,
          notes: rating.notes || ""
        };
        rows.push(fields.map((field) => csvCell(row[field])).join(","));
      });
      const blob = new Blob(["\ufeff" + rows.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const stamp = new Date().toISOString().replaceAll(":", "-").replace("T", "_").slice(0, 19);
      link.href = url;
      link.download = "评分者" + RATER_SLOT + "_v3_" + stamp + ".csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }
    function bindEvents() {
      DIMENSIONS.forEach((dimension) => {
        const input = document.querySelector('input[data-field="' + dimension.field + '"]');
        input.addEventListener("input", () => setDimension(dimension.field, input.value));
        const row = input.closest(".rubric-row");
        row.querySelector('[data-action="minus"]').addEventListener("click", () => {
          const value = currentRating()[dimension.field];
          setDimension(dimension.field, value === null ? 0 : value - 1);
        });
        row.querySelector('[data-action="plus"]').addEventListener("click", () => {
          const value = currentRating()[dimension.field];
          setDimension(dimension.field, value === null ? 1 : value + 1);
        });
        row.querySelectorAll("[data-representative]").forEach((button) => {
          button.addEventListener("click", () => setDimension(dimension.field, button.dataset.representative));
        });
      });
      document.querySelectorAll("[data-confidence]").forEach((button) => {
        button.addEventListener("click", () => {
          currentRating().confidence = Number(button.dataset.confidence);
          persistState();
          renderRubric();
        });
      });
      document.getElementById("notes").addEventListener("input", (event) => {
        currentRating().notes = event.target.value;
        persistState();
      });
      document.getElementById("previous-button").addEventListener("click", () => {
        if (currentIndex > 0) { currentIndex -= 1; renderItem(); }
      });
      document.getElementById("next-button").addEventListener("click", () => {
        if (currentIndex < ITEMS.length - 1) { currentIndex += 1; renderItem(); }
      });
      document.getElementById("unfinished-only").addEventListener("change", renderProgress);
      document.getElementById("export-button").addEventListener("click", exportCsv);
      document.getElementById("reset-button").addEventListener("click", () => {
        if (!confirm("确定要清空评分者" + RATER_SLOT + "在本机保存的全部进度吗？")) return;
        if (prompt("请在下方输入“清空”完成二次确认。") !== "清空") return;
        state = { ratings: {} };
        ensureRatings();
        persistState();
        currentIndex = 0;
        renderItem();
      });
    }
    loadState();
    buildNavigation();
    bindEvents();
    renderItem();
  </script>
</body>
</html>
"""


def render_offline_grading_page(
    items: list[V3HumanGradingItem],
    rater_slot: Literal["A", "B"],
) -> str:
    validate_v3_items(items, private_questions=[])
    if rater_slot not in {"A", "B"}:
        raise ValueError("rater slot must be A or B")
    public_payload = [_public_item_payload(item) for item in items]
    payload_json = json.dumps(
        public_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    rubric_json = json.dumps(
        RUBRIC_BANDS_V3,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return (
        _OFFLINE_HTML_TEMPLATE.replace("__RATER_SLOT__", rater_slot)
        .replace("__ITEMS_JSON__", payload_json)
        .replace("__RUBRICS_JSON__", rubric_json)
        .replace("__RUBRIC_ROWS__", _render_rubric_rows())
        .strip()
        + "\n"
    )


def _atomic_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_v3_blind_pack(
    items: list[V3HumanGradingItem],
    output_dir: Path,
) -> None:
    validate_v3_items(items, private_questions=[])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {path.name for path in output_dir.iterdir()}
    unexpected = existing_names - V3_PACK_FILE_NAMES
    if unexpected:
        raise ValueError("v3 pack target contains unexpected files")

    page_a = render_offline_grading_page(items, "A")
    page_b = render_offline_grading_page(items, "B")
    readme = (
        "匿名双盲回答质量评分 v3\r\n"
        "========================\r\n\r\n"
        "1. 将“评分者A_离线评分.html”和“评分者B_离线评分.html”分别交给两名真实评分者。\r\n"
        "2. 两名评分者应独立完成，不讨论分数，也不要交换页面或导出文件。\r\n"
        "3. 双击页面即可离线使用；页面不会连接网络，进度仅保存在当前浏览器。\r\n"
        "4. 五个维度独立判断：先选档、再微调；选择最符合表现的行为档位，再在区间内微调整数分。\r\n"
        "5. 参考观察点只用于校准覆盖范围，不是唯一标准答案；不要先凭总印象给总分再反推维度分。\r\n"
        "6. 全部30题完成后导出CSV，并将两个原始CSV交回汇总者。\r\n"
        "7. 不要使用表格软件改写导出文件，不要在备注中填写姓名、联系方式或单位。\r\n"
        "8. 该评分只衡量量表一致性，不代表真实用户准确率或底层检索系统性能。\r\n"
    )
    contents = {
        "评分者A_离线评分.html": page_a,
        "评分者B_离线评分.html": page_b,
        "README.txt": readme,
    }
    checksums = "".join(
        f"{_sha256_text(content)}  {name}\r\n"
        for name, content in sorted(contents.items())
    )
    contents["SHA256SUMS.txt"] = checksums
    for name, content in contents.items():
        _atomic_text(output_dir / name, content)

    final_names = {path.name for path in output_dir.iterdir()}
    if final_names != V3_PACK_FILE_NAMES:
        raise ValueError("v3 pack inventory does not match the frozen design")


def build_v3_manifest(
    items: list[V3HumanGradingItem],
    *,
    profile: V3PrivateTaxonomyProfile | None = None,
) -> V3HumanGradingManifest:
    serialized = serialize_v3_public_items(items)
    return V3HumanGradingManifest(
        manifest_version=MANIFEST_VERSION_V3,
        dataset_id=DATASET_ID_V3,
        rubric_version=RUBRIC_VERSION_V3,
        item_count=30,
        category_counts=dict(Counter(item.category for item in items)),
        difficulty_counts=dict(Counter(item.difficulty for item in items)),
        items_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        immutable_row_sha256={
            item.item_id: item.immutable_sha256 for item in items
        },
        private_taxonomy_summary=profile,
        privacy_scan_passed=True,
    )


def write_v3_dataset(
    items: list[V3HumanGradingItem],
    output_dir: Path,
    *,
    profile: V3PrivateTaxonomyProfile | None = None,
) -> V3HumanGradingManifest:
    manifest = build_v3_manifest(items, profile=profile)
    _atomic_text(output_dir / "items.jsonl", serialize_v3_public_items(items))
    _atomic_text(
        output_dir / "manifest.json",
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return manifest


def load_v3_public_items(path: Path) -> list[V3PublicHumanGradingItem]:
    try:
        items = [
            V3PublicHumanGradingItem.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValidationError) as exc:
        raise ValueError("invalid v3 public human-grading dataset") from exc
    if len(items) != 30:
        raise ValueError("v3 public dataset must contain exactly 30 items")
    if Counter(item.category for item in items) != CATEGORY_QUOTAS_V3:
        raise ValueError("v3 public dataset category quotas do not match")
    if Counter(item.difficulty for item in items) != DIFFICULTY_QUOTAS_V3:
        raise ValueError("v3 public dataset difficulty quotas do not match")
    if len({item.item_id for item in items}) != 30:
        raise ValueError("v3 public dataset item IDs must be unique")
    for item in items:
        immutable = _immutable_payload(
            item_id=item.item_id,
            category=item.category,
            difficulty=item.difficulty,
            question=item.question,
            candidate_answer=item.candidate_answer,
            reference_points=item.reference_points,
        )
        if _immutable_hash(immutable) != item.immutable_sha256:
            raise ValueError("v3 public dataset immutable hash mismatch")
        _assert_public_shareable_text(
            "\n".join([item.question, item.candidate_answer, *item.reference_points])
        )
    return items


def load_v3_manifest(path: Path) -> V3HumanGradingManifest:
    try:
        manifest = V3HumanGradingManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        raw_items = path.with_name("items.jsonl").read_bytes()
    except (OSError, ValidationError) as exc:
        raise ValueError("invalid v3 public human-grading manifest") from exc
    if hashlib.sha256(raw_items).hexdigest() != manifest.items_sha256:
        raise ValueError("v3 public dataset hash does not match the manifest")
    items = load_v3_public_items(path.with_name("items.jsonl"))
    if dict(Counter(item.category for item in items)) != manifest.category_counts:
        raise ValueError("v3 manifest category counts do not match the dataset")
    if dict(Counter(item.difficulty for item in items)) != manifest.difficulty_counts:
        raise ValueError("v3 manifest difficulty counts do not match the dataset")
    if {
        item.item_id: item.immutable_sha256 for item in items
    } != manifest.immutable_row_sha256:
        raise ValueError("v3 manifest immutable hashes do not match the dataset")
    return manifest


def read_v3_rater_csv(path: Path) -> list[dict[str, str]]:
    try:
        stream = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError("v3 rating CSV is unavailable") from exc
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != V3_RATER_FIELDS:
            raise ValueError("rater columns do not match the public v3 export")
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if (
                None in row
                or set(row) != set(V3_RATER_FIELDS)
                or any(
                    not isinstance(row.get(field), str)
                    for field in V3_RATER_FIELDS
                )
            ):
                raise ValueError(f"malformed CSV cell count at row {row_number}")
            rows.append({field: row[field] for field in V3_RATER_FIELDS})
    return rows


def _parse_bounded_integer(
    raw: str,
    *,
    field: str,
    minimum: int,
    maximum: int,
    row_number: int,
) -> int:
    value = raw.strip()
    if not value:
        raise ValueError(f"rating file is incomplete at row {row_number}: {field}")
    if not re.fullmatch(r"(?:0|[1-9]\d*)", value):
        raise ValueError(f"invalid {field} at row {row_number}")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field} is out of range at row {row_number}")
    return parsed


def _parse_reference_points(raw: str, *, row_number: int) -> list[str]:
    try:
        points = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid reference_points at row {row_number}"
        ) from exc
    if (
        not isinstance(points, list)
        or not 3 <= len(points) <= 5
        or any(not isinstance(point, str) or not point for point in points)
    ):
        raise ValueError(f"invalid reference_points at row {row_number}")
    return points


def load_completed_v3_ratings(
    path: Path,
    manifest: V3HumanGradingManifest,
) -> list[V3HumanAnnotation]:
    rows = read_v3_rater_csv(path)
    if len(rows) != manifest.item_count:
        raise ValueError("v3 rating file must contain exactly 30 rows")
    actual_ids = [row["item_id"].strip() for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("v3 rating file item IDs must be unique")
    if set(actual_ids) != set(manifest.immutable_row_sha256):
        raise ValueError("v3 rating file contains missing or unexpected item IDs")
    slots = {row["rater_slot"].strip() for row in rows}
    if len(slots) != 1 or slots.pop() not in {"A", "B"}:
        raise ValueError("v3 rating file must contain one consistent rater slot")

    annotations: list[V3HumanAnnotation] = []
    for row_number, row in enumerate(rows, start=2):
        item_id = row["item_id"].strip()
        reference_points = _parse_reference_points(
            row["reference_points"],
            row_number=row_number,
        )
        immutable = _immutable_payload(
            item_id=item_id,
            category=row["category"].strip(),
            difficulty=row["difficulty"].strip(),
            question=row["question"],
            candidate_answer=row["candidate_answer"],
            reference_points=reference_points,
        )
        recomputed_hash = _immutable_hash(immutable)
        if (
            row["immutable_sha256"].strip() != recomputed_hash
            or manifest.immutable_row_sha256[item_id] != recomputed_hash
        ):
            raise ValueError(f"immutable content was changed at row {row_number}")
        score_payload = {
            field: _parse_bounded_integer(
                row[field],
                field=field,
                minimum=0,
                maximum=maximum,
                row_number=row_number,
            )
            for field, maximum in RUBRIC_LIMITS_V3.items()
        }
        confidence_raw = row["confidence"].strip()
        confidence = (
            _parse_bounded_integer(
                confidence_raw,
                field="confidence",
                minimum=1,
                maximum=5,
                row_number=row_number,
            )
            if confidence_raw
            else None
        )
        try:
            annotations.append(
                V3HumanAnnotation(
                    item_id=item_id,
                    category=row["category"].strip(),
                    **score_payload,
                    confidence=confidence,
                    notes=row["notes"].strip(),
                )
            )
        except ValidationError as exc:
            raise ValueError(f"invalid v3 rating at row {row_number}") from exc
    return annotations


def _band(score: int) -> int:
    if score < 60:
        return 0
    if score < 80:
        return 1
    return 2


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cohen_kappa(left: list[int], right: list[int]) -> float:
    observed = _rate([a == b for a, b in zip(left, right, strict=True)])
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[category] / len(left))
        * (right_counts[category] / len(right))
        for category in (0, 1, 2)
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def _quadratic_weighted_kappa(left: list[int], right: list[int]) -> float:
    scale = 100
    observed = mean(
        ((a - b) / scale) ** 2
        for a, b in zip(left, right, strict=True)
    )
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
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if math.isclose(left_scale * right_scale, 0.0):
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def calculate_v3_agreement(
    rater_annotations: list[list[V3HumanAnnotation]],
) -> V3HumanAgreementReport:
    if len(rater_annotations) < 2:
        raise ValueError("at least two completed human raters are required")
    expected_ids = {annotation.item_id for annotation in rater_annotations[0]}
    if len(expected_ids) != 30 or len(rater_annotations[0]) != 30:
        raise ValueError("each rater must contain the same unique set of 30 items")

    normalized: list[dict[str, V3HumanAnnotation]] = []
    for ratings in rater_annotations:
        by_id = {annotation.item_id: annotation for annotation in ratings}
        if len(by_id) != len(ratings) or set(by_id) != expected_ids:
            raise ValueError("all raters must contain the same unique set of 30 items")
        normalized.append(by_id)

    ordered_ids = sorted(expected_ids)
    categories = {
        item_id: normalized[0][item_id].category for item_id in ordered_ids
    }
    for rater in normalized[1:]:
        if any(
            rater[item_id].category != categories[item_id]
            for item_id in ordered_ids
        ):
            raise ValueError("item categories must match across raters")

    dimension_differences: dict[str, list[int]] = {
        field: [] for field in RUBRIC_LIMITS_V3
    }
    total_differences: list[int] = []
    within_10: list[bool] = []
    band_matches: list[bool] = []
    kappas: list[float] = []
    weighted_kappas: list[float] = []
    correlations: list[float] = []
    category_differences: dict[str, list[int]] = {
        category: [] for category in CATEGORY_QUOTAS_V3
    }

    rater_pairs = list(itertools.combinations(normalized, 2))
    for left, right in rater_pairs:
        left_totals = [left[item_id].total for item_id in ordered_ids]
        right_totals = [right[item_id].total for item_id in ordered_ids]
        left_bands = [_band(total) for total in left_totals]
        right_bands = [_band(total) for total in right_totals]
        differences = [
            abs(a - b)
            for a, b in zip(left_totals, right_totals, strict=True)
        ]
        total_differences.extend(differences)
        within_10.extend(difference <= 10 for difference in differences)
        band_matches.extend(
            a == b for a, b in zip(left_bands, right_bands, strict=True)
        )
        kappas.append(_cohen_kappa(left_bands, right_bands))
        weighted_kappas.append(
            _quadratic_weighted_kappa(left_totals, right_totals)
        )
        correlations.append(_pearson(left_totals, right_totals))
        for item_id, difference in zip(
            ordered_ids,
            differences,
            strict=True,
        ):
            category_differences[categories[item_id]].append(difference)
        for field in RUBRIC_LIMITS_V3:
            dimension_differences[field].extend(
                abs(
                    getattr(left[item_id], field)
                    - getattr(right[item_id], field)
                )
                for item_id in ordered_ids
            )

    return V3HumanAgreementReport(
        report_version="anonymized-baoyan-human-grading-agreement-v3",
        status="completed",
        rater_count=len(normalized),
        item_count=30,
        pair_count=len(rater_pairs),
        dimension_mae={
            field: mean(values)
            for field, values in dimension_differences.items()
        },
        total_mae=mean(total_differences),
        within_10_proportion=_rate(within_10),
        band_agreement=_rate(band_matches),
        cohen_kappa=mean(kappas),
        quadratic_weighted_kappa=mean(weighted_kappas),
        pearson_correlation=mean(correlations),
        category_total_mae={
            category: V3CategoryAgreement(
                item_count=CATEGORY_QUOTAS_V3[category],
                pair_count=len(rater_pairs),
                total_mae=mean(values),
            )
            for category, values in category_differences.items()
        },
        privacy={
            "aggregate_only": True,
            "item_level_scores_public": False,
            "rater_names_public": False,
            "private_source_text_public": False,
        },
    )


def build_awaiting_v3_status(
    *,
    items_sha256: str,
    manifest_sha256: str,
) -> V3AnnotationStatus:
    return V3AnnotationStatus(
        report_version="anonymized-baoyan-human-grading-status-v3",
        dataset_id=DATASET_ID_V3,
        status="awaiting_human_labels",
        item_count=30,
        rater_count=0,
        required_rater_count=2,
        items_sha256=items_sha256,
        manifest_sha256=manifest_sha256,
        metrics=None,
        privacy_scan_passed=True,
        note=(
            "Two independent real human rating exports are required. "
            "Models, agents, and rule scorers are not human raters."
        ),
    )


def build_completed_v3_status(
    report: V3HumanAgreementReport,
    *,
    items_sha256: str,
    manifest_sha256: str,
) -> V3AnnotationStatus:
    return V3AnnotationStatus(
        report_version="anonymized-baoyan-human-grading-status-v3",
        dataset_id=DATASET_ID_V3,
        status="completed",
        item_count=30,
        rater_count=report.rater_count,
        required_rater_count=2,
        items_sha256=items_sha256,
        manifest_sha256=manifest_sha256,
        metrics=report,
        privacy_scan_passed=True,
        note=(
            "Aggregate agreement from independent human ratings; "
            "item text and item-level scores remain private."
        ),
    )


def write_v3_annotation_status(
    status: V3AnnotationStatus,
    output_path: Path,
) -> None:
    payload = (
        json.dumps(
            status.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for forbidden in (
        "candidate_answer",
        "reference_points",
        '"item_id"',
        '"notes"',
        "rater-A",
        "rater-B",
    ):
        if forbidden in payload:
            raise ValueError("v3 annotation status contains item-level data")
    _atomic_text(output_path, payload)
