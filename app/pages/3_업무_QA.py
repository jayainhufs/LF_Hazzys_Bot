"""
3_업무_QA.py
=============
업무 질문 → Gemini 답변.
- 카테고리 필터 / top_k / query rewrite 옵션 제공
- 근거 chunk 를 expander 로 표시 (final_score / passed_threshold 포함)
- 근거 부족 시 generation 호출을 skip 하고 안내 메시지 표시
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.chat_box import render_answer
from app.components.sidebar import render_sidebar
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
    k = st.slider("top_k", 1, 16, value=int(settings.top_k))
with col3:
    use_rewrite = st.toggle(
        "Query Rewrite",
        value=settings.enable_query_rewrite,
        help="질문을 검색 친화적 질의로 재작성합니다 (Gemini 호출 1회 추가).",
    )

st.caption(
    f"검색 임계값 (.env): MIN_SIMILARITY={settings.min_similarity_score:.2f} · "
    f"MIN_FINAL={settings.min_final_score:.2f} · "
    f"MIN_RETRIEVED={settings.min_retrieved_chunks} · "
    f"MAX_PER_FILE={settings.max_chunks_per_file} · USE_MMR={settings.use_mmr}"
)

question = st.text_area(
    "질문",
    placeholder="예) 정산 프로세스는 어떤 순서로 진행해야 해?",
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

    with st.spinner("검색 중..."):
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

    summary = result.get("retrieval_summary") or {}
    skipped = bool(result.get("generation_skipped"))
    skip_reason = result.get("skip_reason")
    used_chunks = result.get("used_chunks") or []
    chunks = result.get("retrieved_chunks") or []
    candidates = result.get("all_candidates") or []

    # ---------------- 검색/생성 진단 메트릭 ----------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("후보 (candidate)", summary.get("candidate_count", len(candidates)))
    m2.metric("통과 (passed)", summary.get("passed_count", 0))
    m3.metric("탈락 (dropped)", summary.get("dropped_count", 0))
    m4.metric("top_k", summary.get("top_k", k))

    qc = summary.get("query_class") or {}
    qc_tags = [name.replace("is_", "") for name, v in qc.items() if v]
    st.caption(
        f"keyword class: {', '.join(qc_tags) if qc_tags else '(none)'} · "
        f"min_sim={summary.get('min_similarity', settings.min_similarity_score):.2f} · "
        f"min_final={summary.get('min_final', settings.min_final_score):.2f} · "
        f"max_per_file={summary.get('max_per_file', settings.max_chunks_per_file)} · "
        f"use_mmr={summary.get('use_mmr', settings.use_mmr)}"
    )

    if skipped:
        st.warning(
            "**근거 부족으로 Gemini Generation 을 호출하지 않았습니다.**\n\n"
            f"사유: `{skip_reason or '-'}`\n\n"
            f"통과 chunk 수가 MIN_RETRIEVED_CHUNKS"
            f"({settings.min_retrieved_chunks}) 미만이어서 답변을 생성하지 않았습니다.\n\n"
            "- MIN_SIMILARITY_SCORE / MIN_FINAL_SCORE 임계값을 낮추거나\n"
            "- 관련 가이드 / Slack 스레드 / Excel 파일을 추가로 업로드한 뒤\n"
            "- 검색어를 더 구체화하여 다시 시도해 주세요."
        )
        # 답변 슬롯에는 안내문만 출력 (generation 호출 없음)
        st.info(result.get("answer", ""))
    else:
        if result.get("rewritten_query"):
            st.caption(f"검색용 재작성 질의: `{result['rewritten_query']}`")

        st.caption(
            f"used_chunks={len(used_chunks)} · "
            f"prompt_chars={result.get('prompt_chars', 0)} · "
            f"max_context_chars={settings.max_context_chars} · "
            f"generation_called=True"
        )
        render_answer(
            result["answer"],
            model_name=result.get("model_name") or settings.generation_model,
        )

    # ---------------- 참고 근거 ----------------------------------------------
    st.divider()
    st.markdown("### 참고 근거 (검색된 Chunk)")
    st.caption(
        f"embedding: `{result.get('embedding_provider')}` / `{result.get('embedding_model')}` · "
        f"표시 chunks: {len(chunks)}"
    )
    if not chunks:
        st.info("표시할 근거 chunk 가 없습니다.")
    else:
        for i, c in enumerate(chunks, start=1):
            md = c.metadata or {}
            flag = "PASS" if c.passed_threshold else "DROP"
            title = (
                f"#{i} [{flag}] [{c.uploaded_category}/{c.source_type}/{c.content_type}] "
                f"{c.file_name} · {c.section_title or '-'} "
                f"(score={c.score:.4f}, final={c.final_score:.4f})"
            )
            with st.expander(title, expanded=(i == 1)):
                st.markdown(
                    f"- **passed_threshold**: `{c.passed_threshold}`"
                    + (f" · filter_reason=`{c.filter_reason}`" if c.filter_reason else "")
                )
                st.markdown(
                    f"- source_weight=`{md.get('source_weight')}` · "
                    f"category_boost=`{md.get('category_boost')}` · "
                    f"content_type_boost=`{md.get('content_type_boost')}`"
                )
                st.code(c.content[:4000], language="markdown")
                st.caption("metadata")
                st.json({k_: v for k_, v in md.items() if v is not None}, expanded=False)
