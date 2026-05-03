"""
6_Excel_요약관리.py
===================
Excel 한국어 요약본을 미리 보고, 필요 시 재생성한다.

비용 노트:
- 요약 생성은 사용자가 버튼을 눌렀을 때만 Gemini API 를 호출한다.
- raw_table_hash 가 같으면 재생성하지 않는다 (force 옵션이 켜져 있지 않은 한).
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

from collections import defaultdict
from typing import Dict, List

import streamlit as st

from app.components.sidebar import render_sidebar
from src.config import settings
from src.ingestion.excel_parser import parse_excel
from src.rag.gemini_client import GeminiError
from src.schemas import ExcelSummary
from src.summarization.excel_summarizer import ExcelSummarizer
from src.summarization.summary_store import SummaryStore
from src.utils.hash_utils import file_hash, short_hash

st.set_page_config(page_title="Excel 요약 관리", layout="wide")
render_sidebar()

st.title("Excel 요약 관리")
st.caption(
    "Excel 한국어 업무 요약본을 미리 보고, 필요 시 재생성합니다. "
    "Gemini API 호출 비용이 발생하므로 신중하게 사용하세요. "
    "동일한 raw_table_hash 면 자동으로 캐시됩니다."
)

store = SummaryStore()

# ------------------------- 1) 저장된 요약본 목록 ----------------------------
st.subheader("저장된 Excel 요약 목록")
all_summaries: List[ExcelSummary] = store.list_all()
by_file: Dict[str, List[ExcelSummary]] = defaultdict(list)
for s in all_summaries:
    by_file[s.file_name].append(s)

if not all_summaries:
    st.info(
        "저장된 요약본이 없습니다. '2_문서_색인' 페이지에서 'Excel 상세 요약 생성' 옵션을 켜고 색인하세요."
    )
else:
    for file_name, items in sorted(by_file.items()):
        with st.expander(f"[FILE] {file_name}  ·  {len(items)}개 시트 요약", expanded=False):
            for s in items:
                st.markdown(
                    f"**[SHEET] {s.sheet_name}**  ·  raw_table_hash=`{s.raw_table_hash[:10]}...`  ·  model=`{s.model_name}`"
                )
                st.caption(f"원본: {s.source_raw_path}")
                st.caption(f"파일: {s.summary_markdown_path}")
                with st.popover("미리보기"):
                    st.markdown(s.summary_text or "(빈 요약)")

st.divider()

# ------------------------- 2) Excel 파일 → 요약 (재)생성 -------------------
st.subheader("Excel 파일 단위 요약 (재)생성")
excel_dir = settings.raw_data_dir / settings.category_dirs["excel"]
excel_files = sorted([p for p in excel_dir.glob("*") if p.suffix.lower() in {".xlsx", ".xlsm"}])

if not excel_files:
    st.info(f"`{excel_dir}` 에 Excel 파일이 없습니다.")
else:
    file_choice = st.selectbox(
        "Excel 파일 선택",
        excel_files,
        format_func=lambda p: p.name,
    )
    force = st.toggle("기존 요약을 무시하고 강제 재생성", value=False)

    if st.button("선택한 파일의 모든 시트 요약 (재)생성", type="primary"):
        if not settings.has_api_key():
            st.error("GOOGLE_API_KEY 가 비어 있습니다. `.env` 를 확인하세요.")
            st.stop()

        try:
            fhash = file_hash(file_choice)
            document_id = f"doc_{short_hash(fhash, length=10)}"
            sections = parse_excel(file_choice, document_id)
        except Exception as e:  # noqa: BLE001
            st.error(f"Excel 파싱 실패: {e}")
            st.stop()

        summarizer = ExcelSummarizer()
        n_total = len(sections)
        progress = st.progress(0.0, text="시작 중...")
        ok = 0
        fail = 0
        for i, sec in enumerate(sections):
            sheet_name = sec["metadata"].get("sheet_name") or sec.get("section_title") or "Sheet"
            progress.progress(
                (i + 0.001) / max(1, n_total), text=f"[{i + 1}/{n_total}] {sheet_name} 요약 중..."
            )
            try:
                summarizer.summarize_section(
                    document_id=document_id,
                    file_name=file_choice.name,
                    sheet_name=sheet_name,
                    raw_table_text=sec.get("content", ""),
                    table_range=sec["metadata"].get("table_range"),
                    source_raw_path=str(file_choice),
                    force=force,
                )
                ok += 1
                st.success(f"[OK] {sheet_name}")
            except GeminiError as e:
                fail += 1
                st.error(f"[FAIL] {sheet_name} : {e}")
            except Exception as e:  # noqa: BLE001
                fail += 1
                st.error(f"[FAIL] {sheet_name} : {e}")
            progress.progress((i + 1) / max(1, n_total))

        progress.empty()
        st.success(f"완료: 성공 {ok} · 실패 {fail}")

        st.warning(
            "요약본을 다시 ChromaDB 에 반영하려면 '2_문서_색인' 페이지에서 "
            "해당 파일의 색인을 다시 실행해 주세요. (file_hash 가 같으면 skip 되므로 "
            "강제 재색인이 필요한 경우 `python scripts/reset_vector_db.py` 로 초기화하세요.)"
        )
