"""
time_utils.py
=============
시간 관련 헬퍼.
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """ISO 8601 (UTC) 문자열."""
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def now_compact() -> str:
    """파일명에 쓸 수 있는 형태. 예: 20260503_173012"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
