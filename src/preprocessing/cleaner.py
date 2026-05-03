"""
cleaner.py
==========
ParsedSection content 를 RAG 검색에 적합하게 정제한다.

원칙:
- raw 정보를 너무 많이 잃지 않도록 가벼운 정제만 한다.
- 표(content_type=table / excel_raw_table)는 행/탭 구조를 보존한다.
- 대화(content_type=conversation)는 라인 구분을 유지한다.
"""
from __future__ import annotations

import re
import unicodedata

# 비표시 문자 제거
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

# Slack 멘션/이모지 텍스트 정리
_SLACK_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
_SLACK_CHANNEL_RE = re.compile(r"<#[A-Z0-9]+\|([^>]+)>")
_SLACK_LINK_RE = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_INLINE_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def clean_text(content: str, content_type: str = "text") -> str:
    """모든 content_type 에 공통 적용되는 부드러운 정제."""
    if not content:
        return ""

    s = unicodedata.normalize("NFC", content)
    s = _INVISIBLE_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Slack 마크업 정리
    s = _SLACK_MENTION_RE.sub("[멘션]", s)
    s = _SLACK_CHANNEL_RE.sub(lambda m: f"#{m.group(1)}", s)
    s = _SLACK_LINK_RE.sub(lambda m: f"{m.group(2)} ({m.group(1)})", s)

    if content_type in {"table", "excel_raw_table"}:
        # 표는 줄/탭 보존, 줄 끝 공백만 정리
        s = _TRAILING_SPACE_RE.sub("\n", s)
        s = _MULTI_NEWLINE_RE.sub("\n\n", s)
        return s.strip()

    if content_type == "conversation":
        s = _TRAILING_SPACE_RE.sub("\n", s)
        s = _MULTI_NEWLINE_RE.sub("\n\n", s)
        return s.strip()

    # 일반 텍스트
    s = _TRAILING_SPACE_RE.sub("\n", s)
    s = _INLINE_MULTI_SPACE_RE.sub(" ", s)
    s = _MULTI_NEWLINE_RE.sub("\n\n", s)
    return s.strip()
