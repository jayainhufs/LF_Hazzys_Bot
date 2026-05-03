"""
4_검색_테스트.py
================
LLM generation 호출 없이 retrieval 결과만 확인.
임베딩 모델/카테고리 필터/top_k 를 조합해 검색 품질을 가볍게 비교한다.

비용 노트:
- Gemini Embedding 사용 시: 질의 임베딩 1회 호출 (검색 단계).
- 로컬 임베딩 사용 시: 외부 호출 0회.
- Generation API 는 호출하지 않는다.
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.source_viewer import render_retrieved_chunks
from src.config import settings
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever

st.set_page_config(page_title="검색 테스트", layout="wide")
render_sidebar()

st.title("검색 테스트")
st.caption(
    "비용 절감을 위해 Gemini Generation API 는 호출하지 않습니다. "
    "검색된 chunk 의 score / final_score / 메타데이터만 확인할 수 있습니다."
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    cat = st.selectbox(
        "카테고리 필터",
        ["전체", "Slack 대화", "가이드", "카톡 대화", "Excel"],
        index=0,
    )
with col2:
    k = st.slider("top_k", 3, 20, value=int(settings.top_k))
with col3:
    with_children = st.toggle(
        "Excel parent-child 보강",
        value=True,
        help="excel_summary 가 검색되면 같은 sheet 의 raw_table chunk 일부를 추가로 가져옵니다.",
    )

st.markdown(
    f"**현재 임베딩**: `{settings.embedding_provider}` / "
    f"`{settings.gemini_embedding_model if settings.embedding_provider == 'gemini' else settings.local_embedding_model}`"
)

# Gemini provider 인 경우 API Key 가 필요
if settings.embedding_provider == "gemini" and not settings.has_api_key():
    st.warning(
        "EMBEDDING_PROVIDER=gemini 이지만 GOOGLE_API_KEY 가 비어 있습니다. "
        ".env 의 키를 입력하거나 EMBEDDING_PROVIDER=local 로 변경하세요."
    )

query = st.text_input("검색 질문", placeholder="예) 캠페인 세팅 시 ROAS 기준이 어떻게 돼?")

if st.button("검색 실행", type="primary", disabled=not query.strip()):
    try:
        retriever = Retriever(embedder=Embedder())
        chunks = retriever.retrieve(
            query,
            top_k=k,
            uploaded_category=cat if cat != "전체" else None,
            with_parent_children=with_children,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"검색 실패: {e}")
        st.stop()

    st.write(f"검색 결과: {len(chunks)}개")
    if chunks:
        st.subheader("랭킹 요약")
        rows = []
        for i, c in enumerate(chunks, start=1):
            rows.append({
                "rank": i,
                "score": round(c.score, 4),
                "final_score": round(c.final_score, 4),
                "source_type": c.source_type,
                "category": c.uploaded_category,
                "content_type": c.content_type,
                "file_name": c.file_name,
                "section": c.section_title,
                "preview": (c.content or "")[:80].replace("\n", " "),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.divider()
        render_retrieved_chunks(chunks, show_metadata=True)
