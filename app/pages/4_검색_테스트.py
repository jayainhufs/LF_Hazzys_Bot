"""
4_검색_테스트.py
================
LLM generation 호출 없이 retrieval 결과만 확인.
임베딩 모델/카테고리 필터/top_k/threshold 조합으로 검색 품질을 검증한다.

비용 노트:
- Gemini Embedding 사용 시: 질의 임베딩 1회 호출 (검색 단계).
- 로컬 임베딩 사용 시: 외부 호출 0회.
- Generation API 는 호출하지 않는다.

이번 개선 (v2):
- candidate / passed / dropped 카운트 표시
- score / final_score / source_weight / category_boost / content_type_boost 노출
- date_boost / topic_boost / date_match / topic_match 노출
- query_date / query_topics / query_intent / ENABLE_DATE_FILTER 노출
- ANONYMIZE_OUTPUT / SHOW_RAW_CONTENT 노출
- sanitized preview 기본 사용, raw expander 는 SHOW_RAW_CONTENT=true 일 때만
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from src.config import settings
from src.preprocessing.anonymizer import anonymize_text
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
        help="source_weight/category_boost/content_type_boost/date_boost/topic_boost 까지 적용된 final_score 임계값",
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

# ---------------------------- 옵션 라인 3 (date / 비식별화) -------------------
col8, col9, col10 = st.columns([1, 1, 1])
with col8:
    enable_date_filter = st.checkbox(
        "ENABLE_DATE_FILTER",
        value=bool(settings.enable_date_filter),
        help="질문에 날짜가 있고 chunk.document_date 와 다르면 제거. boost/penalty 로 부족할 때 사용.",
    )
with col9:
    show_raw = st.checkbox(
        "SHOW_RAW_CONTENT (디버깅용)",
        value=bool(settings.show_raw_content),
        help="원문 chunk.content 도 함께 표시. 사람 이름/시간/멘션이 그대로 노출되니 주의.",
    )
with col10:
    st.markdown(
        f"**ANONYMIZE_OUTPUT**: `{settings.anonymize_output}`<br>"
        f"**MASK_MENTIONS**: `{settings.mask_mentions}` · "
        f"**MASK_LINKS**: `{settings.mask_links}`",
        unsafe_allow_html=True,
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

query = st.text_input("검색 질문", placeholder="예) 4월 29일 메타 캠페인 세팅에서 놓치면 안 되는 건 뭐였어?")


def _content_preview(chunk, max_chars: int = 4000) -> str:
    """
    sanitized_content 가 있으면 그것을 우선 사용.
    ANONYMIZE_OUTPUT=false 인 경우만 원본을 보여준다.
    """
    md = chunk.metadata or {}
    if settings.anonymize_output:
        body = md.get("sanitized_content") or anonymize_text(chunk.content or "")
    else:
        body = chunk.content or ""
    return (body or "")[:max_chars]


def _display_date(chunk) -> str:
    md = chunk.metadata or {}
    doc_date = md.get("document_date")
    if not doc_date:
        return "-"
    if settings.show_exact_dates:
        return str(doc_date)
    if settings.anonymize_output:
        return f"해당 {settings.anonymized_date_label}"
    return str(doc_date)


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
            enable_date_filter=enable_date_filter,
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
    q_topics = summary.get("query_topics") or []
    q_intent = summary.get("query_intent") or []
    q_date = summary.get("query_date")
    q_date_text = summary.get("query_date_text")

    st.caption(
        f"keyword class: {', '.join(qc_tags) if qc_tags else '(none)'} · "
        f"query_topics: {', '.join(q_topics) if q_topics else '(none)'} · "
        f"query_intent: {', '.join(q_intent) if q_intent else '(none)'} · "
        f"query_date: {q_date or '(none)'}"
        + (f" ({q_date_text})" if q_date_text else "")
    )
    st.caption(
        f"min_sim={summary.get('min_similarity'):.2f} · "
        f"min_final={summary.get('min_final'):.2f} · "
        f"max_per_file={summary.get('max_per_file')} · "
        f"use_mmr={summary.get('use_mmr')} · "
        f"mmr_lambda={summary.get('mmr_lambda'):.2f} · "
        f"enable_date_filter={summary.get('enable_date_filter')} · "
        f"anonymize_output={summary.get('anonymize_output')} · "
        f"show_raw={show_raw}"
    )

    # -------------------- 통과 결과 없음 ------------------------------------
    if not passed:
        st.warning(
            "기준을 통과한 검색 결과가 없습니다.\n\n"
            "- MIN_SIMILARITY_SCORE / MIN_FINAL_SCORE 값을 낮춰서 다시 시도해보세요.\n"
            "- ENABLE_DATE_FILTER 를 끄면 날짜가 다른 문서도 결과에 포함됩니다.\n"
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
            note = ""
            if md.get("date_match") == "mismatch":
                note = "date mismatch but passed"
            rows.append({
                "rank": i,
                "score": round(c.score, 4),
                "final_score": round(c.final_score, 4),
                "source_weight": round(float(md.get("source_weight") or 0.0), 4),
                "category_boost": round(float(md.get("category_boost") or 0.0), 4),
                "content_type_boost": round(float(md.get("content_type_boost") or 0.0), 4),
                "date_boost": round(float(md.get("date_boost") or 1.0), 4),
                "topic_boost": round(float(md.get("topic_boost") or 1.0), 4),
                "date_match": md.get("date_match"),
                "topic_match": md.get("topic_match"),
                "primary_topic": md.get("primary_topic"),
                "todo_phase": md.get("todo_phase"),
                "parser_format": md.get("parser_format"),
                "category": c.uploaded_category,
                "content_type": c.content_type,
                "file_name": c.file_name,
                "section": c.section_title,
                "display_date": _display_date(c),
                "note": note,
                "preview": _content_preview(c, 80).replace("\n", " "),
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
                    f"- **display_date**: `{_display_date(c)}`\n"
                    f"- **primary_topic**: `{md.get('primary_topic') or '-'}`\n"
                    f"- **topic_tags**: `{md.get('topic_tags') or []}`\n"
                    f"- **todo_phase**: `{md.get('todo_phase') or '-'}`\n"
                    f"- **parser_format**: `{md.get('parser_format') or '-'}`\n"
                    f"- **time_range_display**: `{md.get('time_range_display') or '-'}`\n"
                    f"- **chunk_index**: `{md.get('chunk_index')}`\n"
                    f"- **score**: `{c.score:.4f}` · **final**: `{c.final_score:.4f}`\n"
                    f"- source_weight=`{md.get('source_weight')}` · "
                    f"category_boost=`{md.get('category_boost')}` · "
                    f"content_type_boost=`{md.get('content_type_boost')}` · "
                    f"recency=`{md.get('recency_score')}` · "
                    f"date_boost=`{md.get('date_boost')}` (`{md.get('date_match')}`) · "
                    f"topic_boost=`{md.get('topic_boost')}` (`{md.get('topic_match')}`)"
                )

                st.markdown("**Preview (sanitized)**")
                st.code(_content_preview(c, 4000), language="markdown")

                if show_raw:
                    st.warning(
                        "원문에는 사람 이름/멘션/정확한 시간이 포함될 수 있습니다."
                    )
                    st.code((c.content or "")[:4000], language="markdown")

                st.caption("metadata")
                # sanitized_content 는 Preview 와 중복이라 metadata json 에서 숨김
                md_view = {k_: v for k_, v in md.items() if v is not None and k_ != "sanitized_content"}
                st.json(md_view, expanded=False)

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
                    "date_match": md.get("date_match"),
                    "topic_match": md.get("topic_match"),
                    "primary_topic": md.get("primary_topic"),
                    "category": c.uploaded_category,
                    "content_type": c.content_type,
                    "file_name": c.file_name,
                    "section": c.section_title,
                    "display_date": _display_date(c),
                    "preview": _content_preview(c, 80).replace("\n", " "),
                })
            st.dataframe(rows_d, hide_index=True, use_container_width=True)
