"""
normalizer.py
=============
임베딩 직전에 추가로 적용하는 가벼운 정규화.

- URL을 [link:domain] 으로 단순화 (검색 잡음 줄이기 / 토큰 절약)
- 같은 라인의 반복되는 안내성 문구 압축
- 너무 짧은 라인 / 의미 없는 라인 제거 (옵션)
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://[^\s)]+")


def _shorten_url(match: re.Match) -> str:
    url = match.group(0)
    try:
        domain = urlparse(url).netloc or "link"
    except Exception:
        domain = "link"
    return f"[link:{domain}]"


def normalize_for_embedding(text: str) -> str:
    if not text:
        return ""
    s = _URL_RE.sub(_shorten_url, text)
    return s.strip()


def drop_noise_lines(text: str, min_line_chars: int = 1) -> str:
    """min_line_chars 미만인 라인을 제거. 표/대화에는 사용하지 않는다."""
    if not text:
        return text
    out = []
    for line in text.splitlines():
        if len(line.strip()) >= min_line_chars:
            out.append(line)
    return "\n".join(out)
