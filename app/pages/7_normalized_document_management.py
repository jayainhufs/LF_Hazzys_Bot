"""
7_normalized_document_management.py
===================================
LLM-based Document Normalization 으로 생성된 Normalized Document 를 확인하는
read-only 관리 페이지.

(UI 표시명은 "정규화 문서 관리" 로 유지한다.)

주의:
- raw 원문은 표시하지 않는다.
- 삭제 / 강제 재정규화는 Task 5 범위에서 제공하지 않는다.
- retrieval 우선순위 / QA prompt 변경은 Task 6~7 에서 처리한다.
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import json
from pathlib import Path

import streamlit as st

from app.components.sidebar import render_sidebar
from src.config import settings
from src.normalization import (
    NormalizationStore,
    filter_normalized_documents,
    list_normalized_json_files,
    list_normalized_markdown_files,
    load_all_normalized_documents_from_store,
    markdown_for_normalized_document,
    normalized_document_to_display_dict,
    summarize_normalized_documents,
)


NORMALIZED_DOCUMENT_TYPE_OPTIONS = [
    "전체",
    "workflow",
    "issue",
    "checklist",
    "faq",
    "decision",
    "glossary",
    "communication_template",
]

TOPIC_OPTIONS = [
    "전체",
    "meta",
    "kakao",
    "settlement",
    "outdoor",
    "report",
    "nbt",
    "greenp",
    "youtube",
    "common",
    "unknown",
]


def _counter_to_rows(counter: dict, name_key: str) -> list[dict]:
    if not counter:
        return [{name_key: "-", "count": 0}]
    return [
        {name_key: key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _render_list(title: str, values: list[str], *, numbered: bool = False) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.caption("(없음)")
        return
    for idx, value in enumerate(values, start=1):
        prefix = f"{idx}. " if numbered else "- "
        st.write(f"{prefix}{value}")


st.set_page_config(page_title="정규화 문서 관리", layout="wide")
render_sidebar()

st.title("정규화 문서 관리")
st.caption(
    "LLM-based Document Normalization 으로 생성된 Normalized Document JSON/Markdown 을 "
    "확인합니다. raw 원문은 표시하지 않고 sanitized markdown 중심으로 보여줍니다."
)

store = NormalizationStore(
    output_dir=settings.normalization_output_dir,
    cache_dir=settings.normalization_cache_dir,
)

if st.button("새로고침"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(show_spinner=False)
def _load_documents_and_files(json_dir: str, markdown_dir: str):
    cached_store = NormalizationStore(
        output_dir=Path(json_dir).parent,
        cache_dir=settings.normalization_cache_dir,
    )
    # cache_data 함수 안에서 넘겨받은 경로를 명시적으로 맞춘다.
    cached_store.json_dir = Path(json_dir)
    cached_store.markdown_dir = Path(markdown_dir)
    docs = load_all_normalized_documents_from_store(cached_store)
    json_files = list_normalized_json_files(cached_store)
    md_files = list_normalized_markdown_files(cached_store)
    return docs, json_files, md_files


if st.button("정규화 문서 JSON 다시 로드"):
    _load_documents_and_files.clear()
    st.rerun()


cards, json_files, md_files = _load_documents_and_files(
    str(store.json_dir),
    str(store.markdown_dir),
)
summary = summarize_normalized_documents(cards)

st.subheader("요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("JSON 파일", len(json_files))
col2.metric("Markdown 파일", len(md_files))
col3.metric("Normalized Document", summary["total_cards"])
col4.metric("Cache index", "있음" if store.cache_index_path.exists() else "없음")

with st.expander("저장 위치", expanded=False):
    st.write({
        "normalization_output_dir": str(store.output_dir),
        "json_dir": str(store.json_dir),
        "markdown_dir": str(store.markdown_dir),
        "cache_index": str(store.cache_index_path),
    })

dist_col1, dist_col2, dist_col3 = st.columns(3)
with dist_col1:
    st.markdown("**normalized_document_type 분포**")
    st.dataframe(
        _counter_to_rows(summary["card_type_counts"], "normalized_document_type"),
        use_container_width=True,
        hide_index=True,
    )
with dist_col2:
    st.markdown("**primary_topic 분포**")
    st.dataframe(
        _counter_to_rows(summary["primary_topic_counts"], "primary_topic"),
        use_container_width=True,
        hide_index=True,
    )
with dist_col3:
    st.markdown("**source_file 분포**")
    st.dataframe(
        _counter_to_rows(summary["source_file_counts"], "source_file_name"),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("필터")

source_files = ["전체"] + sorted({c.source_file_name for c in cards if c.source_file_name})
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1.4, 1.8])
with filter_col1:
    selected_card_type = st.selectbox(
        "normalized_document_type", NORMALIZED_DOCUMENT_TYPE_OPTIONS
    )
with filter_col2:
    selected_topic = st.selectbox("primary_topic", TOPIC_OPTIONS)
with filter_col3:
    selected_source_file = st.selectbox("source_file_name", source_files)
with filter_col4:
    query = st.text_input(
        "텍스트 검색",
        placeholder="title, summary, when_to_use, steps, checkpoints, cautions, related_terms",
    )

filtered_cards = filter_normalized_documents(
    cards,
    card_type=selected_card_type,
    primary_topic=selected_topic,
    source_file_name=selected_source_file,
    query=query,
)

st.caption(f"필터 결과: {len(filtered_cards)} / {len(cards)} 개")

if not cards:
    st.info(
        "아직 생성된 Normalized Document 가 없습니다. "
        "`ENABLE_LLM_NORMALIZATION=true` 로 Guide/Slack 파일을 색인하면 결과가 생성됩니다."
    )
    st.stop()

table_rows = [
    normalized_document_to_display_dict(c) | {"_idx": i}
    for i, c in enumerate(filtered_cards)
]
if not table_rows:
    st.warning("필터 조건에 맞는 정규화 문서가 없습니다.")
    st.stop()

st.subheader("정규화 문서 목록")
st.dataframe(
    [{k: v for k, v in row.items() if k != "_idx"} for row in table_rows],
    use_container_width=True,
    hide_index=True,
)

label_to_idx = {
    f"{row['title']} · {row['card_type']} · {row['source_file_name']}": row["_idx"]
    for row in table_rows
}
selected_label = st.selectbox("상세 확인할 정규화 문서 선택", list(label_to_idx.keys()))
selected_card = filtered_cards[label_to_idx[selected_label]]

st.divider()
st.subheader("정규화 문서 상세")

detail_col1, detail_col2 = st.columns([1, 1])
with detail_col1:
    st.markdown(f"### {selected_card.title}")
    st.write(selected_card.summary)
    st.write({
        "normalized_document_type": selected_card.normalized_document_type,
        "primary_topic": selected_card.primary_topic or "unknown",
        "topic_tags": selected_card.topic_tags,
        "task_type": selected_card.task_type,
        "source_file_name": selected_card.source_file_name,
        "display_date": selected_card.display_date,
    })
    st.markdown("**언제 사용하는가**")
    st.write(selected_card.when_to_use or "-")

with detail_col2:
    _render_list("선행 조건", selected_card.prerequisites)
    _render_list("업무 절차", selected_card.steps, numbered=True)
    _render_list("체크포인트", selected_card.checkpoints)
    _render_list("주의사항", selected_card.cautions)

detail_col3, detail_col4 = st.columns(2)
with detail_col3:
    _render_list("예시", selected_card.examples)
    _render_list("관련 용어", selected_card.related_terms)
with detail_col4:
    _render_list("미확인 사항", selected_card.open_questions)
    st.markdown("**근거**")
    if selected_card.evidence_spans:
        for span in selected_card.evidence_spans:
            st.write({
                "section_title": span.get("section_title") or span.get("section") or "",
                "chunk_index": span.get("chunk_index", ""),
                "quote_or_summary": span.get("quote_or_summary") or span.get("summary") or "",
            })
    else:
        st.caption("(근거 없음)")

st.markdown("### Sanitized Markdown Preview")
markdown_text = markdown_for_normalized_document(selected_card)
st.markdown(markdown_text or "(표시할 sanitized markdown 이 없습니다.)")

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "정규화 문서 Markdown 다운로드",
        data=(markdown_text or "").encode("utf-8"),
        file_name=f"{selected_card.normalized_document_id or 'normalized_document'}.md",
        mime="text/markdown",
    )
with download_col2:
    st.download_button(
        "정규화 문서 JSON 다운로드",
        data=json.dumps(selected_card.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{selected_card.normalized_document_id or 'normalized_document'}.json",
        mime="application/json",
    )

with st.expander("JSON metadata 확인", expanded=False):
    st.json({
        "normalized_document_id": selected_card.normalized_document_id,
        "source_file_hash": selected_card.source_file_hash,
        "parent_raw_chunk_ids": selected_card.parent_raw_chunk_ids,
        "metadata": selected_card.metadata,
    })

st.caption(
    "삭제 / 강제 재정규화 버튼은 Task 5 범위에서 제공하지 않습니다. "
    "후속 작업에서 안전장치를 포함해 추가할 예정입니다."
)
