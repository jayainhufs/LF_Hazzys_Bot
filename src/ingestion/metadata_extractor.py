"""
metadata_extractor.py
=====================
파일에서 가져올 수 있는 공통 metadata (파일 크기, 수정시각, 확장자, 카테고리)
를 추출한다. 파서별 metadata 와 합쳐 ParsedSection.metadata 로 들어간다.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def extract_basic_metadata(path: Path, uploaded_category: str) -> Dict[str, Any]:
    """파일 기본 metadata."""
    try:
        stat = path.stat()
        size = int(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except Exception:
        size = 0
        mtime = ""
    return {
        "file_name": path.name,
        "file_ext": path.suffix.lower().lstrip("."),
        "file_size_bytes": size,
        "file_mtime": mtime,
        "uploaded_category": uploaded_category,
    }
