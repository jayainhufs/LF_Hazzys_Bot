"""
upload_box.py
=============
Streamlit drag-and-drop 업로드 박스 + 카테고리 선택 헬퍼.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.config import settings
from src.utils.path_utils import safe_filename

CATEGORY_LABELS: Dict[str, str] = {
    "Slack 대화": "slack",
    "가이드": "guide",
    "카톡 대화": "kakao",
    "Excel": "excel",
    "기타": "misc",
}

EXT_PER_CATEGORY: Dict[str, List[str]] = {
    "slack": ["txt", "md", "docx"],
    "guide": ["docx", "md", "txt"],
    "kakao": ["txt"],
    "excel": ["xlsx", "xlsm"],
    "misc": ["txt", "md", "docx", "xlsx", "xlsm"],
}


def render_category_selector(label: str = "업로드할 자료의 카테고리") -> str:
    pretty = st.radio(
        label,
        list(CATEGORY_LABELS.keys()),
        horizontal=True,
        help="카테고리에 따라 저장 폴더와 파서 / source_weight 가 달라집니다.",
    )
    return CATEGORY_LABELS[pretty]


def render_uploader(uploaded_category: str) -> List[Any]:
    accepted = EXT_PER_CATEGORY.get(uploaded_category, EXT_PER_CATEGORY["misc"])
    files = st.file_uploader(
        f"파일 업로드 (지원: {', '.join(accepted)})",
        type=accepted,
        accept_multiple_files=True,
    )
    return files or []


def save_uploaded_files(uploaded_files: List[Any], uploaded_category: str) -> List[Tuple[Path, str]]:
    """업로드된 streamlit UploadedFile 들을 카테고리 폴더에 저장."""
    target_dir = settings.category_to_dir(uploaded_category)
    saved: List[Tuple[Path, str]] = []
    for uf in uploaded_files:
        try:
            file_name = safe_filename(uf.name)
            target = target_dir / file_name
            with target.open("wb") as f:
                f.write(uf.getbuffer())
            saved.append((target, "saved"))
        except Exception as e:  # noqa: BLE001
            st.error(f"저장 실패: {uf.name} ({e})")
    return saved
