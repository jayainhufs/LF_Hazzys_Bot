"""
hash_utils.py
=============
파일 / 텍스트 hash 생성 유틸.
중복 색인 / 중복 임베딩 / Excel summary 캐시 키로 사용된다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """파일 내용을 SHA-256으로 해싱. 큰 파일도 안전하게 처리."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """문자열 SHA-256 hash."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def short_hash(text: str, length: int = 12) -> str:
    """짧은 hash (id 생성용)."""
    return text_hash(text)[:length]
