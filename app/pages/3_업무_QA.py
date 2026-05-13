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
from src.preprocessing.anonymizer import anonymize_text
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
st.caption(
    f"비식별화: ANONYMIZE_OUTPUT={settings.anonymize_output} · "
    f"SHOW_RAW_CONTENT={settings.show_raw_content} · "
    f"SHOW_SPEAKER_NAMES={settings.show_speaker_names} · "
    f"SHOW_EXACT_TIMESTAMPS={settings.show_exact_timestamps} · "
    f"SHOW_EXACT_DATES={settings.show_exact_dates}"
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

    # Task 7: Normalized Document 중심 답변 진단 메트릭
    answer_mode = result.get("answer_mode") or "default"
    answer_format_label = result.get("answer_format_label") or "default"
    primary_card_count = int(
        result.get("primary_normalized_document_count", result.get("primary_card_count", 0))
    )
    raw_evidence_count = int(result.get("raw_evidence_count", 0))
    raw_fallback_count = int(result.get("raw_fallback_count", 0))
    primary_documents = result.get("primary_normalized_documents") or result.get("primary_cards") or []

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("answer_mode", answer_mode)
    m6.metric("primary_normalized_document", primary_card_count)
    m7.metric("raw_evidence", raw_evidence_count)
    m8.metric("raw_fallback", raw_fallback_count)
    st.caption(
        f"answer_format_label=`{answer_format_label}` · "
        f"answer_with_normalized_documents=`{settings.answer_with_knowledge_cards}` · "
        f"max_primary_normalized_documents=`{settings.max_primary_cards}` · "
        f"max_raw_evidence_chunks=`{settings.max_raw_evidence_chunks}` · "
        f"include_raw_evidence_appendix=`{settings.include_raw_evidence_appendix}` · "
        f"template=`{result.get('normalized_document_answer_template_version') or result.get('knowledge_card_answer_template_version')}`"
    )

    if primary_documents:
        with st.expander(
            f"사용된 Normalized Document ({len(primary_documents)}개)", expanded=False
        ):
            rows = []
            for i, c in enumerate(primary_documents, start=1):
                md = c.metadata or {}
                rows.append({
                    "rank": i,
                    "normalized_document_id": (
                        md.get("normalized_document_id") or md.get("card_id") or "-"
                    ),
                    "normalized_document_type": (
                        md.get("normalized_document_type") or md.get("card_type") or "-"
                    ),
                    "primary_topic": md.get("primary_topic") or "-",
                    "task_type": md.get("task_type") or "-",
                    "source_file_name": c.file_name,
                    "final_score": round(c.final_score, 4),
                    "title": md.get("title") or c.section_title or "-",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)

    qc = summary.get("query_class") or {}
    qc_tags = [name.replace("is_", "") for name, v in qc.items() if v]
    q_topics = summary.get("query_topics") or []
    q_intent = summary.get("query_intent") or []
    q_date = summary.get("query_date")
    q_topic_single = summary.get("query_topic") or (q_topics[0] if q_topics else None)
    st.caption(
        f"keyword class: {', '.join(qc_tags) if qc_tags else '(none)'} · "
        f"query_topic: {q_topic_single or '(none)'} · "
        f"query_topics: {', '.join(q_topics) if q_topics else '(none)'} · "
        f"query_intent: {', '.join(q_intent) if q_intent else '(none)'} · "
        f"query_date: {q_date or '(none)'}"
    )
    # MVP 2차 Step 1: Retrieval Diagnostics 강화 — candidate 구성 / topic mismatch
    # 카운트를 한 줄로 추가 (UI 대규모 변경 없이 caption 만 보강).
    st.caption(
        f"retrieved_count: {summary.get('retrieved_count', summary.get('candidate_count', 0))} · "
        f"passed_count: {summary.get('passed_count', 0)} · "
        f"normalized_document_candidate: {summary.get('normalized_document_candidate_count', 0)} · "
        f"raw_candidate: {summary.get('raw_candidate_count', 0)} · "
        f"topic_mismatch: {summary.get('topic_mismatch_count', 0)}"
    )
    st.caption(
        f"min_sim={summary.get('min_similarity', settings.min_similarity_score):.2f} · "
        f"min_final={summary.get('min_final', settings.min_final_score):.2f} · "
        f"max_per_file={summary.get('max_per_file', settings.max_chunks_per_file)} · "
        f"use_mmr={summary.get('use_mmr', settings.use_mmr)} · "
        f"enable_date_filter={summary.get('enable_date_filter', settings.enable_date_filter)}"
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

            # display_date 라벨
            if md.get("document_date"):
                if settings.show_exact_dates:
                    disp_date = str(md.get("document_date"))
                elif settings.anonymize_output:
                    disp_date = f"해당 {settings.anonymized_date_label}"
                else:
                    disp_date = str(md.get("document_date"))
            else:
                disp_date = "-"

            role = md.get("retrieval_role") or "-"
            role_badge = ""
            if role == "primary_card":
                role_badge = " · 🟣 PRIMARY NORMALIZED DOC"
            elif role == "raw_evidence":
                role_badge = " · 🔵 RAW EVIDENCE"
            title = (
                f"#{i} [{flag}] [{c.uploaded_category}/{c.source_type}/{c.content_type}] "
                f"{c.file_name} · {c.section_title or '-'} "
                f"(score={c.score:.4f}, final={c.final_score:.4f})"
                f"{role_badge}"
            )
            with st.expander(title, expanded=(i == 1)):
                st.markdown(
                    f"- **passed_threshold**: `{c.passed_threshold}`"
                    + (f" · filter_reason=`{c.filter_reason}`" if c.filter_reason else "")
                )
                nd_id = md.get("normalized_document_id") or md.get("card_id") or "-"
                nd_type = md.get("normalized_document_type") or md.get("card_type") or "-"
                nd_match = bool(
                    md.get("normalized_document_type_match")
                    or md.get("card_type_match", False)
                )
                nd_boost = md.get("normalized_document_boost") or md.get("knowledge_card_boost")
                nd_type_boost = md.get("normalized_document_type_boost") or md.get("card_type_boost")
                st.markdown(
                    f"- **retrieval_role**: `{role}` · "
                    f"**content_type**: `{c.content_type}` · "
                    f"**source_type**: `{c.source_type}`\n"
                    f"- **normalized_document_id**: `{nd_id}` · "
                    f"**normalized_document_type**: `{nd_type}` · "
                    f"**type_match**: `{nd_match}`\n"
                    f"- **normalized_document_boost**: `{nd_boost}` · "
                    f"**normalized_document_type_boost**: `{nd_type_boost}`\n"
                    f"- **parent_raw_chunk_ids**: `{md.get('parent_raw_chunk_ids') or []}`"
                )
                st.markdown(
                    f"- file_name=`{c.file_name}` · "
                    f"uploaded_category=`{c.uploaded_category}`\n"
                    f"- section_title=`{c.section_title or '-'}` · "
                    f"display_date=`{disp_date}` · "
                    f"primary_topic=`{md.get('primary_topic') or '-'}` · "
                    f"task_type=`{md.get('task_type') or '-'}` · "
                    f"todo_phase=`{md.get('todo_phase') or '-'}`\n"
                    f"- topic_tags=`{md.get('topic_tags') or []}` · "
                    f"parser_format=`{md.get('parser_format') or '-'}`\n"
                    f"- date_match=`{md.get('date_match')}` · "
                    f"topic_match=`{md.get('topic_match')}` · "
                    f"date_boost=`{md.get('date_boost')}` · "
                    f"topic_boost=`{md.get('topic_boost')}`\n"
                    f"- source_weight=`{md.get('source_weight')}` · "
                    f"category_boost=`{md.get('category_boost')}` · "
                    f"content_type_boost=`{md.get('content_type_boost')}`"
                )

                # Preview: sanitized 우선
                if settings.anonymize_output:
                    body_src = md.get("sanitized_content") or anonymize_text(c.content or "")
                else:
                    body_src = c.content or ""
                st.markdown("**Preview (sanitized)**")
                st.code((body_src or "")[:4000], language="markdown")

                if settings.show_raw_content:
                    st.warning(
                        "원문에는 사람 이름/멘션/정확한 시간이 포함될 수 있습니다."
                    )
                    st.code(c.content[:4000], language="markdown")

                st.caption("metadata")
                md_view = {
                    k_: v for k_, v in md.items()
                    if v is not None and k_ != "sanitized_content"
                }
                st.json(md_view, expanded=False)
