"""
sidebar.py
==========
모든 페이지에서 공통으로 보여줄 사이드바.
"""
from __future__ import annotations

import streamlit as st

from src.config import settings


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {settings.app_name}")
        st.caption("로컬 RAG · Google Gemini · ChromaDB")
        st.divider()

        st.markdown("**현재 설정**")
        st.write({
            "ENV": settings.env,
            "Generation Model": settings.generation_model,
            "Fallback": settings.fallback_generation_model,
            "Embedding Provider": settings.embedding_provider,
            "Embedding Model": (
                settings.gemini_embedding_model
                if settings.embedding_provider == "gemini"
                else settings.local_embedding_model
            ),
            "TOP_K": settings.top_k,
            "Excel Summary 기본": settings.enable_excel_summary,
            "Query Rewrite": settings.enable_query_rewrite,
        })

        st.divider()
        if not settings.has_api_key():
            st.error("GOOGLE_API_KEY 가 설정되지 않았습니다. `.env` 를 확인하세요.")
        else:
            st.success("GOOGLE_API_KEY 감지됨")

        st.caption("Mac / Windows 양쪽에서 동작하도록 설계됨")
