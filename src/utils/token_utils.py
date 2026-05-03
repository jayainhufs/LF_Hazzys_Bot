"""
token_utils.py
==============
대략적인 토큰/문자 수 추정과 컨텍스트 절단 유틸.

- 정확한 Gemini 토크나이저는 사용하지 않는다 (외부 호출 없이 가볍게 처리).
- 한국어 기준으로 보수적인 rough estimate 만 제공.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def rough_token_count(text: str) -> int:
    """
    매우 거친 토큰 수 추정.

    - 한국어는 평균적으로 1 token ~ 1.3~1.6자 정도이고
      Gemini 의 경우도 단어/형태소에 따라 다르다.
    - 본 함수는 단순히 max(공백구분 단어수, len/2.5) 정도의 보수적 추정.
    """
    if not text:
        return 0
    by_words = len(text.split())
    by_chars = max(1, int(len(text) / 2.5))
    return max(by_words, by_chars)


def truncate_to_chars(text: str, max_chars: int) -> str:
    """문자열을 max_chars 까지 자르고 끝에 ... 표시."""
    if not text or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def cap_chunks_by_total_chars(
    items: List[T],
    text_getter,
    max_total_chars: int,
) -> List[T]:
    """
    item 리스트를 순회하며 누적 글자 수가 max_total_chars 를 넘기 직전까지만 유지.
    """
    total = 0
    out: List[T] = []
    for it in items:
        t = text_getter(it) or ""
        if total + len(t) > max_total_chars and out:
            break
        out.append(it)
        total += len(t)
    return out


def cap_chunks_per_file(items: List[T], file_getter, max_per_file: int) -> List[T]:
    """파일별로 max_per_file 개까지만 유지 (입력 순서 보존)."""
    if max_per_file <= 0:
        return list(items)
    counts: dict = defaultdict(int)
    out: List[T] = []
    for it in items:
        f = file_getter(it) or ""
        if counts[f] >= max_per_file:
            continue
        counts[f] += 1
        out.append(it)
    return out


def join_with_budget(parts: Iterable[str], max_chars: int, sep: str = "\n\n") -> str:
    """문자열 조각들을 max_chars 예산 내에서 이어붙인다."""
    out: List[str] = []
    total = 0
    for p in parts:
        if not p:
            continue
        if total + len(p) + len(sep) > max_chars and out:
            break
        out.append(p)
        total += len(p) + len(sep)
    return sep.join(out)
