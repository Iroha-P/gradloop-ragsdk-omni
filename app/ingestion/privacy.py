from __future__ import annotations

import re
from dataclasses import dataclass

from .models import PrivacyFindingSummary, Sensitivity

_PATTERNS = {
    "email": re.compile(r"(?i)(?<![\w.-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])"),
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "student_id": re.compile(r"(?i)(?:学号|student[ _-]?id)\s*[:：=]?\s*[a-z0-9_-]{5,32}"),
    "wechat": re.compile(r"(?i)(?:微信号?|wechat)\s*[:：=]\s*[a-z][-_a-z0-9]{5,19}"),
    "qq": re.compile(r"(?i)\bqq\s*[:：=]\s*[1-9]\d{4,11}\b"),
    "name_label": re.compile(r"(?:姓名|申请人|推荐人)\s*[:：]\s*[\u4e00-\u9fff·]{2,12}"),
    "secret": re.compile(
        r"(?i)(?:"
        r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\b"
        r"\s*[:=]\s*[^\s,;]{4,}"
        r"|(?:密码|口令|解压码)\s*[:：=]\s*[a-z0-9_!@#$%^&*.+-]{4,128}"
        r")"
    ),
}

_PATH_RISK_TERMS = {
    "身份证",
    "成绩单",
    "排名",
    "推荐信",
    "个人陈述",
    "申请表",
    "简历",
    "聊天记录",
    "导师私聊",
}

_PROMPT_INJECTION = [
    re.compile(r"(?i)ignore\s+(?:all\s+)?previous\s+instructions"),
    re.compile(r"(?i)reveal\s+(?:the\s+)?system\s+prompt"),
    re.compile(r"忽略(?:以上|之前|所有)指令"),
    re.compile(r"泄露(?:系统提示|密钥|其他用户)"),
    re.compile(r"执行(?:命令|删除工具|任意工具)"),
]


@dataclass(frozen=True)
class PrivacyDecision:
    sensitivity: Sensitivity
    requires_confirmation: bool
    reject: bool
    reason: str


def scan_privacy(text: str, *, path_hint: str = "") -> PrivacyFindingSummary:
    counts = {name: len(pattern.findall(text)) for name, pattern in _PATTERNS.items()}
    counts = {name: count for name, count in counts.items() if count}
    path_terms = sorted(term for term in _PATH_RISK_TERMS if term in path_hint)
    prompt_injection = any(pattern.search(text) for pattern in _PROMPT_INJECTION)

    if counts.get("id_card") or counts.get("secret") or path_terms:
        recommended = Sensitivity.HIGH
    elif counts:
        recommended = Sensitivity.MEDIUM
    else:
        recommended = Sensitivity.PUBLIC

    warnings: list[str] = []
    if prompt_injection:
        warnings.append("document contains a possible prompt-injection instruction")
    if path_terms:
        warnings.append("file name or relative path contains high-risk material terms")
    return PrivacyFindingSummary(
        finding_counts=counts,
        prompt_injection_detected=prompt_injection,
        path_risk_terms=path_terms,
        recommended_sensitivity=recommended,
        warnings=warnings,
    )


def decide_ingestion(
    declared: Sensitivity,
    scan: PrivacyFindingSummary,
    *,
    public_mode: bool,
) -> PrivacyDecision:
    rank = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.LOW: 1,
        Sensitivity.MEDIUM: 2,
        Sensitivity.HIGH: 3,
    }
    effective = (
        declared
        if rank[declared] >= rank[scan.recommended_sensitivity]
        else scan.recommended_sensitivity
    )

    if public_mode and (effective != Sensitivity.PUBLIC or scan.prompt_injection_detected):
        return PrivacyDecision(
            sensitivity=effective,
            requires_confirmation=False,
            reject=True,
            reason="public ingestion accepts only public data without PII or prompt injection",
        )
    if effective == Sensitivity.HIGH:
        return PrivacyDecision(
            sensitivity=effective,
            requires_confirmation=False,
            reject=True,
            reason="high-sensitivity sources are rejected by default",
        )
    if effective == Sensitivity.MEDIUM or scan.prompt_injection_detected:
        return PrivacyDecision(
            sensitivity=effective,
            requires_confirmation=True,
            reject=False,
            reason=(
                "medium-sensitivity or prompt-injection content requires "
                "a redacted preview and confirmation"
            ),
        )
    return PrivacyDecision(
        sensitivity=effective,
        requires_confirmation=False,
        reject=False,
        reason="source passed the privacy gate",
    )


def redact_text(text: str) -> str:
    redacted = text
    labels = {
        "email": "<redacted:email>",
        "phone": "<redacted:phone>",
        "id_card": "<redacted:id_card>",
        "student_id": "<redacted:student_id>",
        "wechat": "<redacted:wechat>",
        "qq": "<redacted:qq>",
        "name_label": "<redacted:name>",
        "secret": "<redacted:secret>",
    }
    for name, pattern in _PATTERNS.items():
        redacted = pattern.sub(labels[name], redacted)
    return redacted
