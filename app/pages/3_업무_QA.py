"""
3_업무_QA.py
=============
업무 질문 → Gemini 답변.
- 카테고리 필터 / top_k / query rewrite 옵션 제공
- 근거 chunk 를 expander 로 표시
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.chat_box import render_answer
from app.components.sidebar import render_sidebar
from app.components.source_viewer import render_retrieved_chunks
from src.config import settings
from src.rag.qa_pipeline import QAPipeline

st.set_page_config(page_title="업무 QA", layout="wide")
render_sidebar()

st.title("업무 QA")
st.caption("내부 자료 기반으로 업무 처리 순서, 단계별 설명, 체크리스트, 근거를 답변합니다.")

# ----------------------------- 옵션 ----------------------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    cat = st.selectbox(
        "카테고리 필터",
        ["전체", "Slack 대화", "가이드", "카톡 대화", "Excel"],
        index=0,
    )
with col2:
    k = st.slider("top_k", 3, 16, value=int(settings.top_k))
with col3:
    use_rewrite = st.toggle(
        "Query Rewrite",
        value=settings.enable_query_rewrite,
        help="질문을 검색 친화적 질의로 재작성합니다 (Gemini 호출 1회 추가).",
    )

question = st.text_area(
    "질문",
    placeholder="예) Meta 캠페인 세팅할 때 검수 체크리스트가 어떻게 돼?",
    height=120,
)

run = st.button("질문하기", type="primary", disabled=not question.strip())

if run:
    if not settings.has_api_key():
        st.error(
            "GOOGLE_API_KEY 가 설정되어 있지 않습니다. `.env` 를 확인하세요. "
            "(검색만 확인하려면 '4_검색_테스트' 페이지를 사용하세요.)"
        )
        st.stop()

    with st.spinner("답변 생성 중..."):
        try:
            pipeline = QAPipeline()
            result = pipeline.ask(
                question=question,
                top_k=k,
                uploaded_category=cat if cat != "전체" else None,
                enable_query_rewrite=use_rewrite,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"QA 파이프라인 실행 실패: {e}")
            st.stop()

    if result.get("rewritten_query"):
        st.caption(f"검색용 재작성 질의: `{result['rewritten_query']}`")

    render_answer(result["answer"], model_name=result.get("model_name") or settings.generation_model)

    st.divider()
    st.markdown("### 참고 근거 (검색된 Chunk)")
    chunks = result.get("retrieved_chunks") or []
    st.caption(
        f"embedding: `{result.get('embedding_provider')}` / `{result.get('embedding_model')}` · "
        f"chunks: {len(chunks)}"
    )
    render_retrieved_chunks(chunks, show_metadata=True)
