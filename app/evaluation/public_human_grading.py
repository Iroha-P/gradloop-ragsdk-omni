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
from io import StringIO
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

Category = Literal[
    "self_intro_motivation",
    "project_deep_dive",
    "technical_explanation",
    "pressure_followup",
    "research_reflection",
]
AnswerStrength = Literal["weak", "partial", "strong"]

DATASET_ID = "public-synthetic-human-grading-v2"
RUBRIC_VERSION = "human-grading-five-dimension-v2"
MANIFEST_VERSION = "public-human-grading-manifest-v2"
PACK_VERSION = "public-human-grading-pack-v2"

# These identities are deliberately pinned independently from the manifest payload.
# A replacement dataset cannot authorize itself merely by regenerating its manifest.
CANONICAL_ITEMS_SHA256 = (
    "2d0ee13298389de52eaf2feb02c1be2b9889490449b4ff6dd93a13645f3c528c"
)
CANONICAL_MANIFEST_SHA256 = (
    "5680c6fa99143c5896b3175e9444955d3f6c8585e5ca4566ce344ab1a94ec8a2"
)

RUBRIC_LIMITS = {
    "clarity": 15,
    "coverage": 30,
    "reasoning": 20,
    "evidence": 20,
    "reflection": 15,
}
CATEGORY_QUOTAS: dict[str, int] = {
    "self_intro_motivation": 6,
    "project_deep_dive": 6,
    "technical_explanation": 6,
    "pressure_followup": 6,
    "research_reflection": 6,
}
STRENGTH_QUOTAS: dict[str, int] = {
    "weak": 10,
    "partial": 10,
    "strong": 10,
}
IMMUTABLE_FIELDS = [
    "item_id",
    "category",
    "question",
    "candidate_answer",
    "reference_points",
]
RATER_FIELDS = [
    *IMMUTABLE_FIELDS,
    "clarity",
    "coverage",
    "reasoning",
    "evidence",
    "reflection",
    "confidence",
    "notes",
]
PACK_FILE_NAMES = frozenset(
    {"README.txt", "SHA256SUMS.txt", "评分说明.md", "评分者A.csv", "评分者B.csv"}
)

_PUBLIC_CASES: tuple[dict[str, object], ...] = (
    {
        "category": "self_intro_motivation",
        "question": "请用一分钟介绍自己，并说明你希望继续学习的原因。",
        "candidate_answer": "我做事比较认真，也愿意学习新东西，所以希望继续学习。",
        "reference_points": ["形成清晰的个人主线", "说明继续学习的具体动机", "给出能力或经历证据"],
        "answer_strength": "weak",
    },
    {
        "category": "self_intro_motivation",
        "question": "你为什么选择数据建模方向，而不是只做应用开发？",
        "candidate_answer": "因为数据建模更有前景，我也比较感兴趣。",
        "reference_points": ["解释方向选择依据", "比较建模与应用开发的差异", "连接已有准备与未来目标"],
        "answer_strength": "weak",
    },
    {
        "category": "self_intro_motivation",
        "question": "请概括你最突出的两项能力，并说明它们如何支持后续学习。",
        "candidate_answer": (
            "我的两项能力是快速拆解问题和持续复盘。面对开放任务时，我会先列出目标、约束和验收条件；"
            "任务结束后再记录偏差。不过我目前还缺少对复盘效果的量化验证。"
        ),
        "reference_points": ["明确两项能力", "说明能力与后续学习的关系", "提供具体做法", "承认证据边界"],
        "answer_strength": "partial",
    },
    {
        "category": "self_intro_motivation",
        "question": "如果学习任务同时要求理论理解和工程实现，你会怎样安排投入？",
        "candidate_answer": (
            "我会先用小规模例子确认核心概念，再实现可运行的最小版本，最后回到理论检查假设。"
            "这样可以避免只背结论，也能尽早暴露实现问题，但我还没有说明各阶段的时间比例。"
        ),
        "reference_points": ["说明理论与实践的先后关系", "给出阶段性安排", "解释安排理由", "提出校验方式"],
        "answer_strength": "partial",
    },
    {
        "category": "self_intro_motivation",
        "question": "请用一个公开合成场景说明你的学习方式如何产生可验证的进步。",
        "candidate_answer": (
            "在一个虚构的分类练习中，我先用三十道公开生成题做诊断，发现错误主要来自概念混淆。"
            "随后我把错因分成定义、边界和计算三类，每类补做十道变式题。一周后用未见过的三十道题复测，"
            "正确数从十八提高到二十六。这个结果只说明短期迁移有所改善，下一步还要间隔复测以排除记忆效应。"
        ),
        "reference_points": ["说明初始诊断", "描述针对性练习", "提供前后对照证据", "解释证据局限与后续验证"],
        "answer_strength": "strong",
    },
    {
        "category": "self_intro_motivation",
        "question": "你希望在下一阶段形成什么研究能力，为什么把它作为优先目标？",
        "candidate_answer": (
            "我优先希望形成可证伪的实验设计能力，因为提出想法并不等于证明想法有效。"
            "我会练习把每个主张写成假设、对照、指标和停止条件，并在实验前登记预期。"
            "在一个合成排序任务中，这种做法曾让我发现提升来自数据划分而不是新方法。"
            "它的局限是前置设计可能降低探索速度，所以我会为探索性实验单独保留预算并明确标记。"
        ),
        "reference_points": [
            "明确优先能力",
            "解释优先级依据",
            "给出形成能力的方法",
            "提供合成例证",
            "讨论权衡",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "project_deep_dive",
        "question": "在一个虚构的文本分类任务中，你负责的核心部分是什么？",
        "candidate_answer": "我主要负责模型部分，也做了一些数据处理，最后效果还可以。",
        "reference_points": ["界定个人职责", "说明关键技术动作", "给出结果证据", "区分个人与协作贡献"],
        "answer_strength": "weak",
    },
    {
        "category": "project_deep_dive",
        "question": "一个合成传感器任务的准确率突然下降，你会如何定位原因？",
        "candidate_answer": "我会检查代码和数据，找到问题后重新训练模型。",
        "reference_points": [
            "区分数据、代码与分布变化",
            "设计逐层排查顺序",
            "使用可观测证据",
            "验证修复是否有效",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "project_deep_dive",
        "question": "请解释你会怎样为一个虚构检索任务选择基线。",
        "candidate_answer": (
            "我会选一个词项匹配方法和一个稠密向量方法，前者代表低成本可解释方案，后者代表语义匹配方案。"
            "两者使用相同语料划分和命中指标，但还需要补充随机基线与计算成本统计。"
        ),
        "reference_points": ["覆盖简单与强基线", "说明选择理由", "保持评测条件一致", "比较效果与成本"],
        "answer_strength": "partial",
    },
    {
        "category": "project_deep_dive",
        "question": "虚构回归实验中训练误差降低而验证误差升高，你会采取哪些措施？",
        "candidate_answer": (
            "这通常提示过拟合。我会先确认数据划分没有重复，再比较正则化、早停和减少模型容量。"
            "每次只改变一个因素并记录验证误差。不过我还需要说明如何选择最终模型以及是否重复多次实验。"
        ),
        "reference_points": [
            "识别过拟合信号",
            "排除数据泄漏",
            "提出可比较措施",
            "说明控制变量",
            "定义选择规则",
        ],
        "answer_strength": "partial",
    },
    {
        "category": "project_deep_dive",
        "question": "请完整复盘一个虚构排序系统从问题定义到验证结论的过程。",
        "candidate_answer": (
            "目标是在一千条合成查询上提高前五位命中率，同时把单次处理时延控制在五十毫秒以内。"
            "我先固定八百条训练、两百条测试，再比较词项匹配、向量匹配和两阶段重排。"
            "五次不同随机初始化后，两阶段方案的平均命中率比词项基线高九个百分点，时延中位数为四十二毫秒。"
            "消融显示大部分提升来自重排器，但长查询上的收益不稳定，因此结论只适用于当前合成分布，"
            "下一步要增加长度分层和分布漂移测试。"
        ),
        "reference_points": [
            "定义目标与约束",
            "说明数据划分和基线",
            "报告重复实验结果",
            "提供消融证据",
            "限定结论边界",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "project_deep_dive",
        "question": "如果虚构实验的提升只有两个百分点，你如何判断它是否值得保留？",
        "candidate_answer": (
            "我不会只看平均提升。先用配对重复实验估计置信区间，再检查提升是否集中在少数样本，"
            "并同时比较计算、标注和维护成本。若五次重复的区间跨过零，或新增时延超过预设预算，"
            "我会把结论写成尚不稳定而不是有效。若整体提升小但关键困难子集稳定改善，则可以保留为条件策略。"
            "最后还要在新的合成批次上做一次冻结复验，避免因为反复调参高估收益。"
        ),
        "reference_points": [
            "评估统计不确定性",
            "检查样本分布",
            "比较收益与成本",
            "给出保留条件",
            "安排冻结复验",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "technical_explanation",
        "question": "请解释什么是过拟合，以及它为什么会发生。",
        "candidate_answer": "过拟合就是模型学得太好了，所以换一批数据效果会变差。",
        "reference_points": [
            "区分训练表现与泛化表现",
            "解释模型拟合噪声或偶然模式",
            "说明常见诊断与缓解方法",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "technical_explanation",
        "question": "准确率和召回率有什么区别，何时更关注召回率？",
        "candidate_answer": "准确率表示预测得准不准，召回率表示找到多少，漏掉结果时要看召回率。",
        "reference_points": [
            "给出精确率或准确概念的明确口径",
            "定义召回率",
            "说明漏检成本高的场景",
            "讨论指标权衡",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "technical_explanation",
        "question": "请解释向量检索的基本流程，并指出一个主要风险。",
        "candidate_answer": (
            "向量检索先把查询和候选内容编码成向量，再按相似度寻找邻近项。"
            "它能匹配表述不同但语义接近的内容。主要风险是相似度高不代表事实相关，"
            "所以仍需过滤或重排，但我没有展开索引构建和评测方法。"
        ),
        "reference_points": [
            "说明编码过程",
            "说明相似度检索",
            "解释语义优势",
            "指出相关性风险",
            "提出校验方法",
        ],
        "answer_strength": "partial",
    },
    {
        "category": "technical_explanation",
        "question": "什么是数据泄漏，为什么它会让离线指标失真？",
        "candidate_answer": (
            "数据泄漏是训练过程获得了部署时不可用的信息，例如同一来源的近重复样本同时进入训练和测试。"
            "模型因此可以利用捷径，离线分数被高估。应在划分前按来源分组并检查重复，但还需补充时间泄漏的处理。"
        ),
        "reference_points": [
            "定义不可用信息进入训练",
            "给出泄漏例子",
            "解释指标被高估的机制",
            "提出预防检查",
        ],
        "answer_strength": "partial",
    },
    {
        "category": "technical_explanation",
        "question": "请用一个数值例子解释模型校准，并说明校准不等于准确。",
        "candidate_answer": (
            "如果模型对一百个合成样本都给出约百分之七十的正类概率，而其中约七十个确为正类，"
            "这一组预测可视为校准良好。准确率只看最终类别是否判断正确，校准则比较置信度与实际频率。"
            "一个总是给出百分之五十一概率且类别大多正确的模型可能准确但不够有区分度；"
            "反过来，分组频率吻合也不保证排序能力强。评估时应同时看可靠性曲线、分箱误差和任务指标，"
            "并检查小样本分箱带来的不稳定。"
        ),
        "reference_points": [
            "给出概率与频率的数值例子",
            "区分校准与准确率",
            "说明反例",
            "提出联合评估",
            "讨论分箱局限",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "technical_explanation",
        "question": "请解释消融实验如何支持因果归因，以及它不能证明什么。",
        "candidate_answer": (
            "消融实验在其余条件固定时移除一个组件，比较指标变化，从而估计该组件在当前系统中的边际贡献。"
            "例如完整合成系统得分八十二，移除重排后为七十四，而仅移除缓存仍为八十二，"
            "这支持重排与收益相关、缓存主要影响速度。为了减少偶然性，应重复实验并报告波动。"
            "但消融不能自动证明普遍因果：组件间可能有交互，替代实现也可能改变结论，"
            "所以还要做组合消融和新分布复验。"
        ),
        "reference_points": [
            "说明控制变量思想",
            "用数值变化解释边际贡献",
            "要求重复实验",
            "指出交互混杂",
            "限定外推范围",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "pressure_followup",
        "question": "如果你的实验没有按期完成，你会怎样向协作者说明？",
        "candidate_answer": "我会说明遇到了一些困难，希望大家理解，然后尽快补上。",
        "reference_points": [
            "及时说明事实",
            "量化影响范围",
            "给出补救计划",
            "明确需要的协助",
            "复盘预警机制",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "pressure_followup",
        "question": "有人质疑你的结果只是运气，你会如何回应？",
        "candidate_answer": "我会告诉对方实验是认真做的，结果应该没有问题。",
        "reference_points": [
            "正面回应可重复性质疑",
            "提供重复实验或不确定性",
            "检查数据与代码",
            "允许结论被推翻",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "pressure_followup",
        "question": "当速度和准确性无法同时达到目标时，你会如何取舍？",
        "candidate_answer": (
            "我会先确认哪一项是硬约束，再画出不同方案的速度与准确性曲线。"
            "如果时延必须低于固定预算，就在满足预算的方案中选准确性最高者。"
            "我也会尝试分层处理，但还没有说明如何监控取舍在新数据上是否变化。"
        ),
        "reference_points": ["识别硬约束", "量化帕累托取舍", "给出选择规则", "提出折中方案", "持续监控"],
        "answer_strength": "partial",
    },
    {
        "category": "pressure_followup",
        "question": "如果协作者对任务边界理解不同，你会怎样快速对齐？",
        "candidate_answer": (
            "我会把分歧改写成可选择的验收条件，例如输入范围、输出格式和截止时间，"
            "再用一个最小样例共同确认。确认后记录负责人和接口，避免口头理解再次漂移。"
            "不过对于仍有争议的优先级，我还需要补充决策负责人和升级时点。"
        ),
        "reference_points": ["把抽象分歧变成验收条件", "使用最小样例", "记录接口和责任", "设置争议升级机制"],
        "answer_strength": "partial",
    },
    {
        "category": "pressure_followup",
        "question": "复现实验的人得到相反结论时，你会如何处理而不回避责任？",
        "candidate_answer": (
            "我会先暂停对原结论的扩张表述，并请双方分别冻结环境摘要、数据版本和运行日志。"
            "随后用同一组三个合成样例逐层比较预处理、中间输出和随机种子，定位最早分叉点。"
            "若差异来自我未记录的默认参数，我会公开更正复现实验说明；若来自分布差异，"
            "则把结论改为只在原分布成立。无论原因如何，都保留两次原始结果并增加自动化复现检查，"
            "而不是挑选支持自己的那一次。"
        ),
        "reference_points": [
            "收缩未经验证的结论",
            "冻结并比较复现条件",
            "定位最早分叉点",
            "按原因修正结论",
            "保留原始证据",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "pressure_followup",
        "question": "当评审指出你的核心假设可能错误时，请给出一套可执行回应。",
        "candidate_answer": (
            "我会先复述质疑，确认争点是“合成样本独立”这一假设，而不是实现细节。"
            "接着设计两项反证测试：按来源分组重新划分，以及提高重复样本比例观察指标变化。"
            "如果分组后提升从八个百分点降到一个百分点，我会撤回普遍有效的主张，"
            "只保留对当前随机划分的描述。若结果稳定，也只说明这两项测试未推翻假设。"
            "下一步再加入时间顺序划分，因为单次反证测试不能穷尽所有泄漏途径。"
        ),
        "reference_points": [
            "准确界定被质疑的假设",
            "设计可证伪测试",
            "预先说明结果解释",
            "愿意收缩或撤回结论",
            "提出后续反证",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "research_reflection",
        "question": "你会如何判断一个实验结果是否可信？",
        "candidate_answer": "只要结果比基线高，而且多跑几次差不多，我觉得就可信。",
        "reference_points": [
            "检查实验设计",
            "量化不确定性",
            "排除泄漏与选择偏差",
            "验证可复现性",
            "限定结论范围",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "research_reflection",
        "question": "实验得到负结果时，你认为还有必要记录吗？为什么？",
        "candidate_answer": "可以简单记一下，因为以后可能有用，但主要还是看成功结果。",
        "reference_points": [
            "说明负结果的信息价值",
            "记录条件与失败模式",
            "避免重复成本",
            "区分无效与证据不足",
        ],
        "answer_strength": "weak",
    },
    {
        "category": "research_reflection",
        "question": "请说明如何设计消融实验来比较三个模块的贡献。",
        "candidate_answer": (
            "我会先固定完整系统和统一测试集，然后分别移除三个模块并比较总指标。"
            "如果计算预算允许，再补充两两组合以观察交互。所有设置使用相同随机种子集合，"
            "但我还需要预先定义统计判断规则，避免看到结果后挑选解释。"
        ),
        "reference_points": [
            "设置完整系统对照",
            "逐项移除模块",
            "检查模块交互",
            "控制随机性",
            "预先定义判断规则",
        ],
        "answer_strength": "partial",
    },
    {
        "category": "research_reflection",
        "question": "如何区分方法无效和实验尚不足以检验方法？",
        "candidate_answer": (
            "方法无效意味着在足够敏感且条件匹配的检验中未达到预设效应，"
            "而证据不足可能来自样本太少、指标不敏感或实现未达到方法前提。"
            "我会检查统计功效、实现校验和操纵是否生效，再决定收缩假设或补实验，"
            "不过还需要明确补实验的停止条件。"
        ),
        "reference_points": [
            "区分零效应与低检验能力",
            "检查样本和指标敏感性",
            "验证实现与前提",
            "给出决策路径",
            "设置停止条件",
        ],
        "answer_strength": "partial",
    },
    {
        "category": "research_reflection",
        "question": "请为一个虚构预测方法设计从基线到外推验证的完整证据链。",
        "candidate_answer": (
            "我先把主张限定为“在受控合成噪声下减少预测误差”，并预先固定均值基线、线性基线和强非线性基线。"
            "训练与测试按生成批次分组，主指标是平均绝对误差，附带运行成本。"
            "在五个随机批次上报告均值与区间后，再做去除正则项和去除特征筛选的消融。"
            "若方法只在低噪声有效，就把噪声水平写成适用边界。最后冻结参数，"
            "在新的生成机制上复验；失败时记录分布差异，而不把调参后的结果混入首次验证。"
        ),
        "reference_points": [
            "限定可检验主张",
            "选择多层基线",
            "采用分组划分与预设指标",
            "报告不确定性和消融",
            "进行冻结外推验证",
        ],
        "answer_strength": "strong",
    },
    {
        "category": "research_reflection",
        "question": "当两个指标给出相反结论时，你如何形成谨慎且可复验的判断？",
        "candidate_answer": (
            "我会先回到任务损失，判断哪个指标更接近真实决策成本，而不是简单平均。"
            "在一个合成告警任务中，方案甲把总体正确率提高四个百分点，却让关键事件漏检率从百分之八升到百分之十三。"
            "如果关键漏检是硬约束，我会拒绝方案甲，并报告它只改善了多数样本。"
            "随后按事件难度分层，检查冲突是否由类别比例造成，并用预先固定的成本权重复算。"
            "权重本身带有价值判断，因此会同时发布敏感性区间，并在新批次上复验结论。"
        ),
        "reference_points": [
            "依据任务成本确定指标优先级",
            "提供指标冲突的数值证据",
            "进行分层诊断",
            "预设成本权重",
            "报告敏感性与复验",
        ],
        "answer_strength": "strong",
    },
)

_PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("id-card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    (
        "student-id",
        re.compile(
            r"(?:学号|student[ _-]?id)\s*[:：=]?\s*[A-Za-z0-9_-]{5,32}",
            re.IGNORECASE,
        ),
    ),
    (
        "wechat",
        re.compile(
            r"(?:微信号?|wechat)\s*[:：=]\s*[A-Za-z][-_A-Za-z0-9]{5,19}",
            re.IGNORECASE,
        ),
    ),
    ("qq", re.compile(r"\bqq\s*[:：=]\s*[1-9]\d{4,11}\b", re.IGNORECASE)),
    ("url-scheme", re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://")),
    (
        "bare-domain",
        re.compile(
            r"\b(?:[A-Za-z0-9-]+\.)+(?:com|org|net|cn|edu|io|ai|dev|co)\b",
            re.IGNORECASE,
        ),
    ),
    ("windows-path", re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")),
    ("unc-path", re.compile(r"(?<!\\)\\\\[^\\\s]+\\[^\\\s]+")),
    (
        "posix-path",
        re.compile(
            r"(?<![\w.])/(?:home|users|var|tmp|etc|opt|private|mnt|srv|usr)"
            r"(?:/[A-Za-z0-9._-]+)+",
            re.IGNORECASE,
        ),
    ),
    (
        "credential",
        re.compile(
            r"(?:api[_ -]?key|password|passwd|secret|token)\s*[:=：]\s*[A-Za-z0-9_-]{4,}",
            re.IGNORECASE,
        ),
    ),
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    (
        "feishu-identifier",
        re.compile(
            r"(?:飞书|lark|wiki\s+token|"
            r"\b(?:doccn|wikcn|bascn|shtcn|fldcn|boxcn)[A-Za-z0-9_-]{6,}\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "named-person",
        re.compile(
            r"(?:真实姓名|姓名|申请人|推荐人|联系人|作者|name)\s*[:：]\s*"
            r"(?:[\u4e00-\u9fff·]{2,12}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            re.IGNORECASE,
        ),
    ),
    ("placeholder-person", re.compile(r"(?:张三|李四|王五|赵六|John Doe|Jane Doe)", re.IGNORECASE)),
    (
        "unlabeled-latin-name",
        re.compile(
            r"\b(?!(?:Public|Human|Agent|Language|Rating|Synthetic)\s)"
            r"[A-Z][a-z]{2,}\s+"
            r"(?!(?:Grading|Agreement|Model|Labels|Workflow)\b)[A-Z][a-z]{2,}\b"
        ),
    ),
    (
        "institution",
        re.compile(
            r"(?:[\u4e00-\u9fff]{2,20}(?:大学|学院|研究院|公司|集团)|"
            r"\b[A-Z][A-Za-z& ]{1,40}(?:University|College|Institute|Corporation|Corp\.|Inc\.)\b)"
        ),
    ),
    (
        "private-context",
        re.compile(r"(?:私人题库|真实简历|个人故事|本地学习记录|真实项目经历|私有语料)"),
    ),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PublicHumanGradingItem(StrictModel):
    item_id: str = Field(pattern=r"^phg_[a-f0-9]{20}$")
    category: Category
    question: str = Field(min_length=8, max_length=2000)
    candidate_answer: str = Field(min_length=1, max_length=6000)
    reference_points: list[str] = Field(min_length=2, max_length=6)
    answer_strength: AnswerStrength
    rubric_version: Literal["human-grading-five-dimension-v2"]
    rubric_limits: dict[str, int]

    @field_validator("rubric_limits")
    @classmethod
    def _fixed_rubric_limits(cls, value: dict[str, int]) -> dict[str, int]:
        if value != RUBRIC_LIMITS:
            raise ValueError("rubric_limits must match the fixed v2 rubric")
        return value


class PublicHumanGradingManifest(StrictModel):
    manifest_version: Literal["public-human-grading-manifest-v2"]
    dataset_id: Literal["public-synthetic-human-grading-v2"]
    rubric_version: Literal["human-grading-five-dimension-v2"]
    item_count: Literal[30]
    category_counts: dict[str, int]
    strength_counts: dict[str, int]
    items_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    immutable_row_sha256: dict[str, str]
    privacy_scan_passed: Literal[True]


class PublicPackManifest(StrictModel):
    pack_version: Literal["public-human-grading-pack-v2"]
    dataset_id: Literal["public-synthetic-human-grading-v2"]
    file_sha256: dict[str, str]


class PublicHumanAnnotation(StrictModel):
    item_id: str = Field(pattern=r"^phg_[a-f0-9]{20}$")
    category: Category
    clarity: int = Field(ge=0, le=15)
    coverage: int = Field(ge=0, le=30)
    reasoning: int = Field(ge=0, le=20)
    evidence: int = Field(ge=0, le=20)
    reflection: int = Field(ge=0, le=15)
    confidence: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)

    @property
    def total(self) -> int:
        return sum(getattr(self, field) for field in RUBRIC_LIMITS)


class PublicCategoryAgreement(StrictModel):
    item_count: int = Field(ge=1)
    pair_count: int = Field(ge=1)
    total_mae: float = Field(ge=0)


class PublicHumanAgreementReport(StrictModel):
    report_version: Literal["public-human-grading-agreement-v2"]
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
    category_total_mae: dict[str, PublicCategoryAgreement]
    privacy: dict[str, bool]


class PublicHumanAnnotationStatus(StrictModel):
    report_version: Literal["public-human-grading-annotation-status-v2"]
    dataset_id: Literal["public-synthetic-human-grading-v2"]
    status: Literal["awaiting_human_labels", "completed"]
    item_count: Literal[30]
    rater_count: int = Field(ge=0)
    required_rater_count: Literal[2]
    items_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    metrics: PublicHumanAgreementReport | None
    privacy_scan_passed: Literal[True]
    note: str

    @model_validator(mode="after")
    def _completed_status_binds_dataset_identity(self) -> PublicHumanAnnotationStatus:
        if self.status == "completed" and (
            self.items_sha256 is None or self.manifest_sha256 is None
        ):
            raise ValueError("completed status requires pinned dataset hashes")
        return self


def assert_public_shareable_text(text: str) -> None:
    violations = [label for label, pattern in _PRIVACY_PATTERNS if pattern.search(text)]
    if violations:
        raise ValueError("public-shareable privacy scan failed: " + ", ".join(sorted(violations)))


def _normalized_question(question: str) -> str:
    return re.sub(r"[\W_]+", "", question, flags=re.UNICODE).casefold()


def _numbered_reference_points(points: list[str]) -> str:
    return "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))


def _immutable_payload(item: PublicHumanGradingItem) -> dict[str, str]:
    return {
        "item_id": item.item_id,
        "category": item.category,
        "question": item.question,
        "candidate_answer": item.candidate_answer,
        "reference_points": _numbered_reference_points(item.reference_points),
    }


def _immutable_hash(payload: dict[str, str]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_public_items(items: list[PublicHumanGradingItem]) -> None:
    if len(items) != 30:
        raise ValueError("public dataset requires exactly 30 items")
    item_ids = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate item_id values are not allowed")
    if Counter(item.category for item in items) != Counter(CATEGORY_QUOTAS):
        raise ValueError("public dataset category quotas do not match")
    if Counter(item.answer_strength for item in items) != Counter(STRENGTH_QUOTAS):
        raise ValueError("public dataset answer-strength quotas do not match")
    joint_counts = Counter((item.category, item.answer_strength) for item in items)
    if any(
        joint_counts[(category, strength)] != 2
        for category in CATEGORY_QUOTAS
        for strength in STRENGTH_QUOTAS
    ):
        raise ValueError("public dataset category-strength joint quota does not match")
    for item in items:
        assert_public_shareable_text(
            "\n".join([item.question, item.candidate_answer, *item.reference_points])
        )
        if item.rubric_limits != RUBRIC_LIMITS:
            raise ValueError("public dataset rubric limits do not match")
    normalized = [_normalized_question(item.question) for item in items]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate questions are not allowed")
    for left, right in itertools.combinations(normalized, 2):
        if SequenceMatcher(a=left, b=right).ratio() >= 0.92:
            raise ValueError("near-duplicate questions are not allowed")


def build_public_items() -> list[PublicHumanGradingItem]:
    items: list[PublicHumanGradingItem] = []
    for case in _PUBLIC_CASES:
        category = str(case["category"])
        question = str(case["question"])
        digest = hashlib.sha256(f"{category}|{question}".encode()).hexdigest()
        items.append(
            PublicHumanGradingItem(
                item_id=f"phg_{digest[:20]}",
                category=category,
                question=question,
                candidate_answer=str(case["candidate_answer"]),
                reference_points=list(case["reference_points"]),
                answer_strength=str(case["answer_strength"]),
                rubric_version=RUBRIC_VERSION,
                rubric_limits=dict(RUBRIC_LIMITS),
            )
        )
    validate_public_items(items)
    return items


def serialize_public_items(items: list[PublicHumanGradingItem]) -> str:
    validate_public_items(items)
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for item in items
    )


def build_public_manifest(
    items: list[PublicHumanGradingItem],
) -> PublicHumanGradingManifest:
    serialized = serialize_public_items(items)
    return PublicHumanGradingManifest(
        manifest_version=MANIFEST_VERSION,
        dataset_id=DATASET_ID,
        rubric_version=RUBRIC_VERSION,
        item_count=30,
        category_counts=dict(Counter(item.category for item in items)),
        strength_counts=dict(Counter(item.answer_strength for item in items)),
        items_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        immutable_row_sha256={item.item_id: _immutable_hash(_immutable_payload(item)) for item in items},
        privacy_scan_passed=True,
    )


def _atomic_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def write_public_dataset(
    items: list[PublicHumanGradingItem],
    output_dir: Path,
) -> PublicHumanGradingManifest:
    manifest = build_public_manifest(items)
    _atomic_text(output_dir / "items.jsonl", serialize_public_items(items))
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


def load_public_items(path: Path) -> list[PublicHumanGradingItem]:
    try:
        items = [
            PublicHumanGradingItem.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValidationError) as exc:
        raise ValueError("invalid public human-grading dataset") from exc
    validate_public_items(items)
    return items


def load_public_manifest(path: Path) -> PublicHumanGradingManifest:
    try:
        manifest = PublicHumanGradingManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError("invalid public human-grading manifest") from exc
    items_path = path.with_name("items.jsonl")
    try:
        raw_items = items_path.read_bytes()
    except OSError as exc:
        raise ValueError("public items.jsonl is required beside the manifest") from exc
    if hashlib.sha256(raw_items).hexdigest() != manifest.items_sha256:
        raise ValueError("public dataset hash does not match the manifest")
    items = load_public_items(items_path)
    if build_public_manifest(items) != manifest:
        raise ValueError("public manifest content does not match the dataset")
    return manifest


def _blank_rater_csv(items: list[PublicHumanGradingItem]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=RATER_FIELDS,
        lineterminator="\r\n",
    )
    writer.writeheader()
    for item in items:
        row = {field: "" for field in RATER_FIELDS}
        row.update(_immutable_payload(item))
        writer.writerow(row)
    return stream.getvalue()


def read_rater_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RATER_FIELDS:
            raise ValueError("rater columns do not match the public v2 template")
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if (
                None in row
                or set(row) != set(RATER_FIELDS)
                or any(not isinstance(row.get(field), str) for field in RATER_FIELDS)
            ):
                raise ValueError(f"malformed CSV cell count at row {row_number}")
            rows.append({field: row[field] for field in RATER_FIELDS})
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


def load_completed_public_ratings(
    path: Path,
    manifest: PublicHumanGradingManifest,
) -> list[PublicHumanAnnotation]:
    rows = read_rater_csv(path)
    if len(rows) != manifest.item_count:
        raise ValueError("rating file must contain exactly 30 rows")
    expected_ids = set(manifest.immutable_row_sha256)
    actual_ids = [row["item_id"].strip() for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("rating file item_id values must be unique")
    if set(actual_ids) != expected_ids:
        raise ValueError("rating file contains missing or unexpected item_id values")

    annotations: list[PublicHumanAnnotation] = []
    for row_number, row in enumerate(rows, start=2):
        immutable = {field: row[field] for field in IMMUTABLE_FIELDS}
        item_id = immutable["item_id"].strip()
        if _immutable_hash(immutable) != manifest.immutable_row_sha256[item_id]:
            raise ValueError(f"immutable content was changed at row {row_number}")
        score_payload = {
            field: _parse_bounded_integer(
                row[field],
                field=field,
                minimum=0,
                maximum=maximum,
                row_number=row_number,
            )
            for field, maximum in RUBRIC_LIMITS.items()
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
                PublicHumanAnnotation(
                    item_id=item_id,
                    category=immutable["category"],
                    **score_payload,
                    confidence=confidence,
                    notes=row["notes"].strip(),
                )
            )
        except ValidationError as exc:
            raise ValueError(f"invalid rating at row {row_number}: {exc}") from exc
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
        (left_counts[category] / len(left)) * (right_counts[category] / len(right)) for category in (0, 1, 2)
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
        left_count * right_count * (((left_score - right_score) / scale) ** 2)
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


def calculate_public_agreement(
    rater_annotations: list[list[PublicHumanAnnotation]],
) -> PublicHumanAgreementReport:
    if len(rater_annotations) < 2:
        raise ValueError("at least two completed human raters are required")
    expected_ids = {annotation.item_id for annotation in rater_annotations[0]}
    if len(expected_ids) != 30 or len(rater_annotations[0]) != 30:
        raise ValueError("each rater must contain the same unique set of 30 items")

    normalized: list[dict[str, PublicHumanAnnotation]] = []
    for ratings in rater_annotations:
        by_id = {annotation.item_id: annotation for annotation in ratings}
        if len(by_id) != len(ratings) or set(by_id) != expected_ids:
            raise ValueError("all raters must contain the same unique set of 30 items")
        normalized.append(by_id)

    ordered_ids = sorted(expected_ids)
    categories = {item_id: normalized[0][item_id].category for item_id in ordered_ids}
    for rater in normalized[1:]:
        if any(rater[item_id].category != categories[item_id] for item_id in ordered_ids):
            raise ValueError("item categories must match across raters")

    dimension_differences: dict[str, list[int]] = {field: [] for field in RUBRIC_LIMITS}
    total_differences: list[int] = []
    within_10: list[bool] = []
    band_matches: list[bool] = []
    kappas: list[float] = []
    weighted_kappas: list[float] = []
    correlations: list[float] = []
    category_differences: dict[str, list[int]] = {category: [] for category in CATEGORY_QUOTAS}

    rater_pairs = list(itertools.combinations(normalized, 2))
    for left, right in rater_pairs:
        left_totals = [left[item_id].total for item_id in ordered_ids]
        right_totals = [right[item_id].total for item_id in ordered_ids]
        left_bands = [_band(total) for total in left_totals]
        right_bands = [_band(total) for total in right_totals]
        differences = [abs(a - b) for a, b in zip(left_totals, right_totals, strict=True)]
        total_differences.extend(differences)
        within_10.extend(difference <= 10 for difference in differences)
        band_matches.extend(a == b for a, b in zip(left_bands, right_bands, strict=True))
        kappas.append(_cohen_kappa(left_bands, right_bands))
        weighted_kappas.append(_quadratic_weighted_kappa(left_totals, right_totals))
        correlations.append(_pearson(left_totals, right_totals))
        for item_id, difference in zip(ordered_ids, differences, strict=True):
            category_differences[categories[item_id]].append(difference)
        for field in RUBRIC_LIMITS:
            dimension_differences[field].extend(
                abs(getattr(left[item_id], field) - getattr(right[item_id], field)) for item_id in ordered_ids
            )

    return PublicHumanAgreementReport(
        report_version="public-human-grading-agreement-v2",
        status="completed",
        rater_count=len(normalized),
        item_count=30,
        pair_count=len(rater_pairs),
        dimension_mae={
            field: mean(values) for field, values in dimension_differences.items()
        },
        total_mae=mean(total_differences),
        within_10_proportion=_rate(within_10),
        band_agreement=_rate(band_matches),
        cohen_kappa=mean(kappas),
        quadratic_weighted_kappa=mean(weighted_kappas),
        pearson_correlation=mean(correlations),
        category_total_mae={
            category: PublicCategoryAgreement(
                item_count=CATEGORY_QUOTAS[category],
                pair_count=len(rater_pairs),
                total_mae=mean(values),
            )
            for category, values in category_differences.items()
        },
        privacy={
            "aggregate_only": True,
            "item_level_scores_public": False,
            "rater_names_public": False,
        },
    )


_RATER_GUIDE = """# 公开合成双盲人工评分说明

预计用时：30–45 分钟。请对全部 30 条回答独立完成评分，不得查看另一名评分者的文件、
系统评分、模型建议或作者侧强弱标签。首次评分完成后保留原文件，不为提高一致性而回改。

五个维度必须全部填写；`confidence` 可填 1–5，不计入总分；`notes` 只记录歧义或评分依据。

## 清晰度 clarity（0–15）

- 0–3：没有直接回答，难以理解。
- 4–7：能辨认主要意思，但结构混乱或重复明显。
- 8–11：有明确结论和基本结构，表达基本自然。
- 12–15：结论先行、层次清楚、简洁自然，便于继续追问。

## 覆盖度 coverage（0–30）

- 0–6：几乎未覆盖参考要点。
- 7–14：只覆盖少数要点，存在明显缺漏。
- 15–22：覆盖主要要点，但边界、细节或关联不足。
- 23–30：核心要点完整，并包含必要条件、细节和边界。

## 推理 reasoning（0–20）

- 0–4：结论没有理由或存在明显逻辑冲突。
- 5–9：给出部分原因，但存在跳跃或因果不清。
- 10–15：结论、方法与原因基本连贯。
- 16–20：推理链完整，能够解释权衡、条件和反例。

## 证据 evidence（0–20）

- 0–4：没有证据，只有态度、口号或空泛判断。
- 5–9：有例子但不具体，不能有效支撑结论。
- 10–15：有具体做法、技术细节、数据或事实，基本支持结论。
- 16–20：证据具体可核验，并明确解释证据与结论的关系。

## 反思 reflection（0–15）

- 0–3：没有局限、适用条件或改进方向。
- 4–7：提到反思，但内容笼统。
- 8–11：能说明主要局限和可执行改进。
- 12–15：能说明适用边界、失败原因、权衡和下一步验证。

## 双盲与归档

1. 只填写分配给自己的 CSV，并独立完成全部项目。
2. 不查看对方文件或任何系统、规则、模型评分。
3. 原始评分只读归档；先计算一致性，再另建争议讨论记录。
4. 争议讨论不得覆盖首次独立评分。
"""

_PACK_README = """已建立公开合成双盲标注与一致性计算流程。
两名真实评分者需分别完成各自文件；Agent、模型或规则评分器不计作人工评分者。
在两份真实评分完成前，不存在可报告的人工一致性指标。
"""


def _render_blind_pack(
    items: list[PublicHumanGradingItem],
    output_dir: Path,
) -> PublicPackManifest:
    template = _blank_rater_csv(items)
    _atomic_text(output_dir / "评分说明.md", _RATER_GUIDE)
    _atomic_text(output_dir / "评分者A.csv", template, encoding="utf-8-sig")
    _atomic_text(output_dir / "评分者B.csv", template, encoding="utf-8-sig")
    _atomic_text(output_dir / "README.txt", _PACK_README)

    shareable_names = ["README.txt", "评分说明.md", "评分者A.csv", "评分者B.csv"]
    file_sha256 = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in shareable_names
    }
    checksums = "".join(f"{file_sha256[name]}  {name}\n" for name in sorted(file_sha256))
    _atomic_text(output_dir / "SHA256SUMS.txt", checksums)
    return PublicPackManifest(
        pack_version=PACK_VERSION,
        dataset_id=DATASET_ID,
        file_sha256=file_sha256,
    )


def _assert_pack_target_inventory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError("pack target contains an unexpected non-directory entry")
    unexpected = [
        entry
        for entry in output_dir.iterdir()
        if entry.name not in PACK_FILE_NAMES or not entry.is_file() or entry.is_symlink()
    ]
    if unexpected:
        raise ValueError("pack target contains unexpected entries")


def write_blind_pack(
    items: list[PublicHumanGradingItem],
    output_dir: Path,
) -> PublicPackManifest:
    validate_public_items(items)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _assert_pack_target_inventory(output_dir)
    with tempfile.TemporaryDirectory(
        prefix=".public-human-grading-v2-",
        dir=output_dir.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        manifest = _render_blind_pack(items, staging_dir)
        staged_inventory = {
            entry.name for entry in staging_dir.iterdir() if entry.is_file()
        }
        if staged_inventory != PACK_FILE_NAMES:
            raise RuntimeError("staged pack inventory does not match the contract")
        _assert_pack_target_inventory(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(PACK_FILE_NAMES):
            os.replace(staging_dir / name, output_dir / name)
    final_inventory = {
        entry.name for entry in output_dir.iterdir() if entry.is_file()
    }
    if final_inventory != PACK_FILE_NAMES:
        raise RuntimeError("published pack inventory does not match the contract")
    return manifest


def build_awaiting_public_status(
    *,
    items_sha256: str,
    manifest_sha256: str,
) -> PublicHumanAnnotationStatus:
    return PublicHumanAnnotationStatus(
        report_version="public-human-grading-annotation-status-v2",
        dataset_id=DATASET_ID,
        status="awaiting_human_labels",
        item_count=30,
        rater_count=0,
        required_rater_count=2,
        items_sha256=items_sha256,
        manifest_sha256=manifest_sha256,
        metrics=None,
        privacy_scan_passed=True,
        note=(
            "The public synthetic double-blind workflow is ready. Agreement metrics "
            "remain unavailable until two real people independently complete all items; "
            "Agents and language models do not count as human raters."
        ),
    )


def build_completed_public_status(
    report: PublicHumanAgreementReport,
    *,
    items_sha256: str,
    manifest_sha256: str,
) -> PublicHumanAnnotationStatus:
    return PublicHumanAnnotationStatus(
        report_version="public-human-grading-annotation-status-v2",
        dataset_id=DATASET_ID,
        status="completed",
        item_count=30,
        rater_count=report.rater_count,
        required_rater_count=2,
        items_sha256=items_sha256,
        manifest_sha256=manifest_sha256,
        metrics=report,
        privacy_scan_passed=True,
        note=(
            "Metrics are aggregate-only results from complete independent human rating "
            "files. Original rating files remain local and unchanged."
        ),
    )


def write_public_annotation_status(
    status: PublicHumanAnnotationStatus,
    path: Path,
) -> None:
    _atomic_text(
        path,
        json.dumps(
            status.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
