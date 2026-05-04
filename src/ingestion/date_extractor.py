"""
date_extractor.py
=================
파일명 / 본문 첫 줄에서 문서의 "업무일 (document_date)" 을 추출한다.

우선순위
--------
1. 파일명에서 추출
   - "[2026년 4월 29일 TODO].txt"
   - "2026-04-29 TODO.md"
   - "2026_04_29_slack.txt"
2. 본문 첫 줄/제목에서 추출
   - "[2026년 4월 29일 TODO]"
   - "4/29 동제 TODO"  (현재 연도 기준)
3. 추출 실패 시 (None, "unknown")

return: ``{"document_date": "YYYY-MM-DD" | None,
           "date_text": str | None,
           "date_source": "file_name" | "content_title" | "unknown"}``

UI 표시용 ``display_date`` 는 anonymizer.anonymize_date 가 담당한다.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

# 2026년 4월 29일
_FULL_KO_RE = re.compile(r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일")
# 2026-04-29 / 2026/4/29 / 2026.4.29 / 2026_04_29
_DASH_RE = re.compile(r"\b(?P<y>20\d{2})[-/_.\s](?P<m>\d{1,2})[-/_.\s](?P<d>\d{1,2})\b")
# "4월 29일" (연도 없음, 한국어)
_KO_MD_RE = re.compile(r"(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일")
# 4/29 (연도 없음)
_SHORT_RE = re.compile(r"(?<!\d)(?P<m>\d{1,2})/(?P<d>\d{1,2})(?!\d)")


def _to_iso(year: int, month: int, day: int) -> Optional[str]:
    try:
        dt = datetime(year=year, month=month, day=day)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d")


def _try_match(text: str, default_year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """문자열 1개에서 날짜 1개를 추출."""
    if not text:
        return None
    m = _FULL_KO_RE.search(text)
    if m:
        iso = _to_iso(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        if iso:
            return {"document_date": iso, "date_text": m.group(0)}
    m = _DASH_RE.search(text)
    if m:
        iso = _to_iso(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        if iso:
            return {"document_date": iso, "date_text": m.group(0)}
    m = _KO_MD_RE.search(text)
    if m and default_year:
        iso = _to_iso(int(default_year), int(m.group("m")), int(m.group("d")))
        if iso:
            return {"document_date": iso, "date_text": m.group(0)}
    m = _SHORT_RE.search(text)
    if m and default_year:
        iso = _to_iso(int(default_year), int(m.group("m")), int(m.group("d")))
        if iso:
            return {"document_date": iso, "date_text": m.group(0)}
    return None


def extract_document_date(
    *,
    file_name: Optional[str] = None,
    content: Optional[str] = None,
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns
    -------
    dict with keys: document_date, date_text, date_source
    """
    default_year = int(default_year or datetime.now().year)

    if file_name:
        hit = _try_match(file_name, default_year=default_year)
        if hit:
            hit["date_source"] = "file_name"
            return hit

    if content:
        # 본문 처음 ~20줄까지만 살핀다 (첫 헤더/제목 위주)
        head = "\n".join(content.splitlines()[:20])
        hit = _try_match(head, default_year=default_year)
        if hit:
            hit["date_source"] = "content_title"
            return hit

    return {"document_date": None, "date_text": None, "date_source": "unknown"}
