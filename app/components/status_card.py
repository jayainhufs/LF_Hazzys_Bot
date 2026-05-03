"""
status_card.py
==============
홈 화면 / API 상태확인 페이지에서 쓰이는 상태 카드 모음.
"""
from __future__ import annotations

import streamlit as st

from src.config import settings
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore
from src.summarization.summary_store import SummaryStore


def render_status_grid() -> None:
    col1, col2, col3, col4 = st.columns(4)

    try:
        d_stats = DocumentStore().stats()
    except Exception as e:  # noqa: BLE001
        d_stats = {"documents": 0, "chunks": 0, "error": str(e)}

    try:
        v_stats = VectorStore().stats()
    except Exception as e:  # noqa: BLE001
        v_stats = {"count": -1, "collection_name": settings.chroma_collection, "error": str(e)}

    try:
        s_stats = len(SummaryStore().list_all())
    except Exception:  # noqa: BLE001
        s_stats = 0

    col1.metric("적재된 문서 수", d_stats.get("documents", 0))
    col2.metric("Chunk 수 (디스크)", d_stats.get("chunks", 0))
    col3.metric("Vector DB chunk 수", max(0, v_stats.get("count", 0)))
    col4.metric("Excel 요약 수", s_stats)

    with st.expander("Vector DB 정보"):
        st.write(v_stats)
    with st.expander("Document Store 정보"):
        st.write(d_stats)


def render_api_status() -> None:
    if settings.has_api_key():
        st.success("Google API Key: 설정됨")
    else:
        st.error("Google API Key: 미설정 (.env 확인)")
    st.write({
        "Generation Model": settings.generation_model,
        "Fallback Model": settings.fallback_generation_model,
        "Excel Summary Model": settings.excel_summary_model,
        "Embedding Provider": settings.embedding_provider,
        "Embedding Model": (
            settings.gemini_embedding_model
            if settings.embedding_provider == "gemini"
            else settings.local_embedding_model
        ),
    })
