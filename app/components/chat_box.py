"""
chat_box.py
===========
업무 QA 페이지의 답변 출력 컴포넌트.
"""
from __future__ import annotations

import streamlit as st


def render_answer(answer: str, model_name: str | None = None) -> None:
    st.markdown("### 답변")
    if model_name:
        st.caption(f"model: `{model_name}`")
    st.markdown(answer or "(답변 없음)")
