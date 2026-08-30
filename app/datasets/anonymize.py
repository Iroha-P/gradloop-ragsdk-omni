from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "name": (
        re.compile(
            r"(?i)(?:学生姓名|申请人|姓名|名字|candidate\s*name|name)"
            r"(?:\s*[:：]\s*|\s+)[^\s,，;；]{2,40}"
        ),
    ),
    "school": (
        re.compile(
            r"(?i)(?:本科院校|毕业院校|就读学校|学校|院校|university|college)"
            r"\s*[:：]\s*[^\n,，;；]{2,80}"
        ),
        re.compile(r"[\u4e00-\u9fff]{2,30}(?:大学|学院)"),
        re.compile(r"(?i)\b(?:[A-Z][A-Z&.'-]*\s+){0,5}(?:University|College)\b"),
    ),
    "student_id": (re.compile(r"(?i)(?:学号|student\s*id)\s*[:：]?\s*[A-Z0-9-]{6,32}"),),
    "email": (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b"),),
    "phone": (
        re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9](?:[-\s]?\d){9}(?!\d)"),
        re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)"),
    ),
    "address": (
        re.compile(
            r"(?i)(?:家庭住址|通讯地址|住址|地址|address)"
            r"(?:\s*[:：]\s*|\s+)[^\n,，;；]{3,120}"
        ),
    ),
    "supervisor": (
        re.compile(
            r"(?i)(?:导师|指导教师|supervisor|advisor)"
            r"(?:\s*[:：]\s*|\s+)[^\n,，;；]{2,60}"
        ),
    ),
    "absolute_path": (
        re.compile(r"(?i)(?:[A-Z]:[\\/]+|\\\\+)[^\s\"'<>|]+"),
        re.compile(r"(?<![:/])/(?:[^/\s\"'<>|]+/)+[^/\s\"'<>|]*"),
    ),
    "password_hint": (
        re.compile(
            r"(?i)(?:解压密码|访问密码|文件密码|密码提示|提取码|密码|口令|password|passcode)"
            r"\s*(?:is|为|[:：=])\s*[^\s,，;；]{1,120}"
        ),
    ),
    "qr_text": (re.compile(r"(?i)(?:二维码(?:内容|文本)?|QR\s*(?:code|text)?)\s*[:：]?\s*[^\n]{0,200}"),),
    "organization_label": (
        re.compile(
            r"(?i)(?:单位|机构|组织|工作单位|organization)"
            r"(?:\s*[:：]\s*|\s+)[^\n,，;；]{2,120}"
        ),
    ),
    "copyright_label": (re.compile(r"(?i)(?:©|\bcopyright\b|版权所有)[^\n]{0,160}"),),
    "watermark_label": (re.compile(r"(?i)(?:水印|watermark)\s*[:：]?\s*[^\n]{0,160}"),),
    "prompt_injection": (
        re.compile(
            r"(?i)(?:ignore|disregard|override).{0,40}(?:previous|prior|system)"
            r".{0,30}(?:instructions?|prompts?)"
        ),
        re.compile(r"(?:忽略|无视|覆盖|绕过).{0,16}(?:之前|先前|系统).{0,12}(?:指令|提示词)"),
        re.compile(r"(?:输出|泄露|展示).{0,12}(?:原始材料|系统提示词|隐私数据|完整原文)"),
    ),
}

_REIDENTIFICATION_SIGNALS = (
    re.compile(
        r"(?i)(?:(?:来自|籍贯|户籍|生源地|常住).{0,12}(?:省|市|县|区)"
        r"|\bfrom\b.{0,24}\b(?:province|city|county|district)\b)"
    ),
    re.compile(r"(?i)(?:(?:专业|年级|班级|综合)?排名\s*(?:第)?\s*\d+|\branked?\s*\d+)"),
    re.compile(r"(?i)(?:唯一|仅有|独一|\b(?:only|unique|sole)\b)"),
    re.compile(r"(?i)(?:第?\d{2,4}届|该届|年级|班级|\bcohort\b|\bclass\s+of\s+\d{4}\b)"),
    re.compile(r"(?i)(?:获奖|竞赛|奖项|保送|录取|\b(?:award|winner|admitted|enrolled)\w*\b)"),
)
_DIRECT_IDENTITY_CATEGORIES = frozenset(
    {"name", "school", "student_id", "email", "phone", "address", "supervisor", "absolute_path"}
)
_FAIL_CLOSED_CATEGORIES = frozenset(
    {
        "password_hint",
        "qr_text",
        "organization_label",
        "copyright_label",
        "watermark_label",
        "prompt_injection",
        "reidentification_combo",
    }
)


class SensitiveTextScan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[str]
    unsafe: bool


def scan_sensitive_text(text: str) -> SensitiveTextScan:
    hits = [category for category, patterns in _PATTERNS.items() if any(p.search(text) for p in patterns)]
    direct_identity_count = len(_DIRECT_IDENTITY_CATEGORIES.intersection(hits))
    if (
        sum(bool(pattern.search(text)) for pattern in _REIDENTIFICATION_SIGNALS) >= 3
        or direct_identity_count >= 3
    ):
        hits.append("reidentification_combo")
    unique_hits = list(dict.fromkeys(hits))
    return SensitiveTextScan(
        hits=unique_hits,
        unsafe=bool(_FAIL_CLOSED_CATEGORIES.intersection(unique_hits)),
    )


def anonymize_text(text: str) -> str:
    scan = scan_sensitive_text(text)
    if scan.unsafe:
        return "[REDACTED_UNSAFE_BLOCK]"

    anonymized = text
    for category in scan.hits:
        for pattern in _PATTERNS.get(category, ()):
            anonymized = pattern.sub(f"[REDACTED_{category.upper()}]", anonymized)

    if scan_sensitive_text(anonymized).hits:
        return "[REDACTED_UNSAFE_BLOCK]"
    return anonymized
