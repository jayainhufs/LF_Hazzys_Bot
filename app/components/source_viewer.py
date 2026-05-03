"""
source_viewer.py
================
RetrievedChunk 들을 사용자가 보기 좋은 형태로 expander 로 보여 준다.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

from src.schemas import RetrievedChunk


def render_retrieved_chunks(chunks: Iterable[RetrievedChunk], show_metadata: bool = True) -> None:
    chunks = list(chunks)
    if not chunks:
        st.info("검색된 근거가 없습니다.")
        return
    for i, c in enumerate(chunks, start=1):
        title = (
            f"#{i} [{c.uploaded_category}/{c.source_type}/{c.content_type}] "
            f"{c.file_name} · {c.section_title or '-'} "
            f"(score={c.score:.4f}, final={c.final_score:.4f})"
        )
        with st.expander(title, expanded=(i == 1)):
            st.code(c.content[:4000], language="markdown")
            if show_metadata:
                st.caption("metadata")
                st.json({k: v for k, v in c.metadata.items() if v is not None}, expanded=False)
