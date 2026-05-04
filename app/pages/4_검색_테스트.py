"""
4_검색_테스트.py
================
LLM generation 호출 없이 retrieval 결과만 확인.
임베딩 모델/카테고리 필터/top_k/threshold 조합으로 검색 품질을 검증한다.

비용 노트:
- Gemini Embedding 사용 시: 질의 임베딩 1회 호출 (검색 단계).
- 로컬 임베딩 사용 시: 외부 호출 0회.
- Generation API 는 호출하지 않는다.

이번 개선:
- candidate / passed / dropped 카운트 표시
- score / final_score / source_weight / category_boost / content_type_boost 노출
- passed_threshold / filter_reason 노출
- MIN_SIMILARITY / MIN_FINAL / MAX_PER_FILE / USE_MMR 슬라이더 / 토글
- score 해석 방식 안내
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from src.config import settings
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever

st.set_page_config(page_title="검색 테스트", layout="wide")
render_sidebar()

st.title("검색 테스트")
st.caption(
    "비용 절감을 위해 Gemini Generation API 는 호출하지 않습니다. "
    "검색된 chunk 의 score / final_score / 메타데이터 / 탈락 사유까지 확인할 수 있습니다."
)

st.info(
    "**Score 해석:** ChromaDB cosine distance 를 `1 - distance` 로 변환한 값입니다. "
    "**값이 높을수록 더 유사**합니다 (정규화된 임베딩에서는 보통 0.0 ~ 1.0 범위)."
)

# ---------------------------- 옵션 라인 1 ------------------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    cat = st.selectbox(
        "카테고리 필터",
        ["전체", "Slack 대화", "가이드", "카톡 대화", "Excel"],
        index=0,
    )
with col2:
    k = st.slider("top_k", 1, 20, value=int(settings.top_k))
with col3:
    with_children = st.toggle(
        "Excel parent-child 보강",
        value=True,
        help="excel_summary 가 검색되면 같은 sheet 의 raw_table chunk 일부를 추가로 가져옵니다.",
    )

# ---------------------------- 옵션 라인 2 ------------------------------------
col4, col5, col6, col7 = st.columns([1, 1, 1, 1])
with col4:
    min_sim = st.slider(
        "MIN_SIMILARITY_SCORE",
        min_value=0.0, max_value=0.95, step=0.01,
        value=float(settings.min_similarity_score),
        help="raw similarity (1 - cosine_distance) 가 이 값보다 낮으면 제거",
    )
with col5:
    min_final = st.slider(
        "MIN_FINAL_SCORE",
        min_value=0.0, max_value=1.5, step=0.01,
        value=float(settings.min_final_score),
        help="source_weight/category_boost/content_type_boost 까지 적용된 final_score 임계값",
    )
with col6:
    max_per_file = st.selectbox(
        "MAX_CHUNKS_PER_FILE",
        options=[1, 2, 3, 4, 5, 10],
        index=[1, 2, 3, 4, 5, 10].index(int(settings.max_chunks_per_file))
        if int(settings.max_chunks_per_file) in [1, 2, 3, 4, 5, 10] else 1,
        help="같은 파일에서 최대 몇 개 chunk 까지 통과시킬지",
    )
with col7:
    use_mmr = st.checkbox(
        "USE_MMR",
        value=bool(settings.use_mmr),
        help="비슷한 chunk 반복을 줄이는 다양성 보정 적용",
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

query = st.text_input("검색 질문", placeholder="예) 정산 프로세스는 어떤 순서로 진행해야 해?")

if st.button("검색 실행", type="primary", disabled=not query.strip()):
    try:
        retriever = Retriever(embedder=Embedder())
        details = retriever.retrieve_with_details(
            query,
            top_k=k,
            uploaded_category=cat if cat != "전체" else None,
            with_parent_children=with_children,
            min_similarity=min_sim,
            min_final=min_final,
            max_per_file=max_per_file,
            use_mmr=use_mmr,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"검색 실패: {e}")
        st.stop()

    summary = details.summary
    passed = details.passed
    candidates = details.candidates

    # -------------------- 요약 메트릭 ---------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("후보 (candidate)", summary.get("candidate_count", 0))
    m2.metric("통과 (passed)", summary.get("passed_count", 0))
    m3.metric("탈락 (dropped)", summary.get("dropped_count", 0))
    m4.metric("top_k", summary.get("top_k", k))

    qc = summary.get("query_class") or {}
    qc_tags = [name.replace("is_", "") for name, v in qc.items() if v]
    st.caption(
        f"keyword class: {', '.join(qc_tags) if qc_tags else '(none)'} · "
        f"min_sim={summary.get('min_similarity'):.2f} · "
        f"min_final={summary.get('min_final'):.2f} · "
        f"max_per_file={summary.get('max_per_file')} · "
        f"use_mmr={summary.get('use_mmr')} · "
        f"mmr_lambda={summary.get('mmr_lambda'):.2f}"
    )

    # -------------------- 통과 결과 없음 ------------------------------------
    if not passed:
        st.warning(
            "기준을 통과한 검색 결과가 없습니다.\n\n"
            "- MIN_SIMILARITY_SCORE / MIN_FINAL_SCORE 값을 낮춰서 다시 시도해보세요.\n"
            "- 또는 관련 가이드/Slack 스레드/Excel 파일을 추가로 업로드한 뒤 색인하세요.\n"
            "- 검색어를 더 구체적으로 작성해보세요.\n\n"
            "Gemini Generation 은 호출되지 않았습니다."
        )

    # -------------------- 통과 결과 ------------------------------------------
    if passed:
        st.subheader("통과 결과 (Passed)")
        rows = []
        for i, c in enumerate(passed, start=1):
            md = c.metadata or {}
            rows.append({
                "rank": i,
                "score": round(c.score, 4),
                "final_score": round(c.final_score, 4),
                "source_weight": round(float(md.get("source_weight") or 0.0), 4),
                "category_boost": round(float(md.get("category_boost") or 0.0), 4),
                "content_type_boost": round(float(md.get("content_type_boost") or 0.0), 4),
                "passed": c.passed_threshold,
                "category": c.uploaded_category,
                "source_type": c.source_type,
                "content_type": c.content_type,
                "file_name": c.file_name,
                "section": c.section_title,
                "chunk_index": md.get("chunk_index"),
                "preview": (c.content or "")[:80].replace("\n", " "),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

        st.subheader("상세 보기 (Passed)")
        for i, c in enumerate(passed, start=1):
            md = c.metadata or {}
            title = (
                f"#{i} [PASS] {c.file_name} · {c.section_title or '-'} "
                f"(score={c.score:.4f}, final={c.final_score:.4f})"
            )
            with st.expander(title, expanded=(i == 1)):
                st.markdown(
                    f"- **file_name**: `{c.file_name}`\n"
                    f"- **uploaded_category**: `{c.uploaded_category}`\n"
                    f"- **source_type**: `{c.source_type}`\n"
                    f"- **content_type**: `{c.content_type}`\n"
                    f"- **section_title**: `{c.section_title or '-'}`\n"
                    f"- **chunk_index**: `{md.get('chunk_index')}`\n"
                    f"- **score (similarity)**: `{c.score:.4f}`\n"
                    f"- **final_score**: `{c.final_score:.4f}`\n"
                    f"- source_weight=`{md.get('source_weight')}` · "
                    f"category_boost=`{md.get('category_boost')}` · "
                    f"content_type_boost=`{md.get('content_type_boost')}` · "
                    f"recency_score=`{md.get('recency_score')}`"
                )
                st.code(c.content[:4000], language="markdown")
                st.caption("metadata")
                st.json({k_: v for k_, v in md.items() if v is not None}, expanded=False)

    # -------------------- 탈락 결과 ------------------------------------------
    dropped = [c for c in candidates if not c.passed_threshold]
    if dropped:
        with st.expander(f"탈락 후보 보기 ({len(dropped)}개) — filter_reason 별 사유 확인", expanded=False):
            rows_d = []
            for i, c in enumerate(dropped, start=1):
                md = c.metadata or {}
                rows_d.append({
                    "rank": i,
                    "score": round(c.score, 4),
                    "final_score": round(c.final_score, 4),
                    "filter_reason": c.filter_reason,
                    "category": c.uploaded_category,
                    "source_type": c.source_type,
                    "content_type": c.content_type,
                    "file_name": c.file_name,
                    "section": c.section_title,
                    "chunk_index": md.get("chunk_index"),
                    "preview": (c.content or "")[:80].replace("\n", " "),
                })
            st.dataframe(rows_d, hide_index=True, use_container_width=True)
