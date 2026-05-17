"""
2_document_indexing.py
======================
data/raw/<카테고리>/ 폴더를 스캔하여 새 파일만 색인한다.
- 이미 색인된 파일은 file_hash 기준으로 skip
- Excel 상세 요약(LLM) 생성 옵션 제공
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from src.config import settings
from src.pipeline import IngestResult, discover_files, ingest_file
from src.rag.embedder import get_default_embedder
from src.storage.document_store import DocumentStore
from src.storage.file_registry import FileRegistry
from src.storage.vector_store import VectorStore
from src.summarization.excel_summarizer import ExcelSummarizer
from src.utils.cost_utils import tracker

st.set_page_config(page_title="문서 색인", layout="wide")
render_sidebar()

st.title("문서 색인")
st.caption(
    "data/raw 하위 폴더를 스캔하여 새 파일만 색인합니다. "
    "(file_hash 기반 중복 방지. 파일 내용이 바뀌면 자동으로 재색인 됩니다.)"
)

# ----------------------------- 옵션 ----------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    enable_summary = st.toggle(
        "Excel 상세 요약 생성",
        value=settings.enable_excel_summary,
        help=(
            "Excel 시트별로 Gemini 한국어 업무 요약을 생성합니다. "
            "비용이 발생하므로 필요한 경우에만 켜세요. "
            "(raw_table_hash 기반 캐시되어 동일한 표는 재호출하지 않습니다.)"
        ),
    )
with col2:
    skip_indexed = st.toggle("이미 색인된 파일 skip", value=True)
with col3:
    advanced_mode = st.toggle(
        "고급 색인 모드",
        value=False,
        help="기본 색인 모드는 비용 절약형. 고급 모드에서는 Excel 상세 요약을 자동으로 켭니다.",
    )
with col4:
    enable_normalization = st.toggle(
        "LLM 기반 문서 정규화",
        value=settings.enable_llm_normalization,
        help=(
            "Guide/Slack 파일 색인 시 LLM-based Document Normalization 을 함께 수행합니다. "
            "Gemini API 호출 비용이 발생할 수 있습니다."
        ),
    )

if advanced_mode:
    enable_summary = True

if enable_normalization:
    st.info(
        "ENABLE_LLM_NORMALIZATION 이 켜져 있습니다. "
        "Guide/Slack 파일은 raw chunk 색인 후 LLM-based Document Normalization 을 함께 수행합니다."
    )
else:
    st.caption("LLM 기반 문서 정규화: OFF (기존 raw ingest 흐름만 실행)")

# ----------------------------- 대상 파일 미리보기 --------------------------
files = discover_files()
st.write(f"스캔 결과: 총 **{len(files)}** 개 파일")
with st.expander("대상 파일 목록", expanded=False):
    if not files:
        st.caption("data/raw/* 에 파일이 없습니다. '문서 업로드' 페이지에서 먼저 업로드하세요.")
    else:
        for item in files:
            p = item["path"]
            st.write(f"- [{item['uploaded_category']}] {p.relative_to(settings.project_root)}")

# ----------------------------- 실행 ----------------------------------------
need_api = settings.embedding_provider == "gemini" or enable_summary or enable_normalization
if st.button("새 파일만 색인 실행", type="primary", disabled=not files):
    if need_api and not settings.has_api_key():
        st.error(
            "GOOGLE_API_KEY 가 설정되어 있지 않습니다. `.env` 의 키를 입력한 뒤 다시 시도하세요. "
            "(EMBEDDING_PROVIDER=local 로 설정하면 API Key 없이 색인할 수 있습니다.)"
        )
        st.stop()

    embedder = get_default_embedder()
    vstore = VectorStore()
    dstore = DocumentStore()
    registry = FileRegistry()
    summarizer = ExcelSummarizer() if enable_summary else None

    progress = st.progress(0.0, text="시작 중...")
    log_box = st.container()
    indexed = skipped = failed = chunks_total = normalized_total = 0
    results: list[IngestResult] = []

    n = len(files)
    for i, item in enumerate(files):
        path = item["path"]
        uploaded_category = item["uploaded_category"]
        progress.progress(
            (i + 0.001) / n,
            text=f"[{i + 1}/{n}] {path.name} ({uploaded_category}) 처리 중...",
        )
        res = ingest_file(
            path,
            uploaded_category,
            embedder=embedder,
            vector_store=vstore,
            document_store=dstore,
            file_registry=registry,
            excel_summarizer=summarizer,
            enable_excel_summary=enable_summary,
            enable_llm_normalization=enable_normalization,
            skip_if_indexed=skip_indexed,
        )
        results.append(res)
        if res.status == "indexed":
            indexed += 1
            chunks_total += res.chunks_added
            normalized_total += res.normalized_chunks_added
            norm_bits = ""
            if enable_normalization:
                norm_bits = (
                    f" normalized_cards={res.normalized_card_count} "
                    f"normalized_chunks={res.normalized_chunks_added}"
                )
                if res.normalized_skipped_reason:
                    norm_bits += f" ({res.normalized_skipped_reason})"
            log_box.success(
                f"[OK] [{uploaded_category}] {res.file_name} | chunks={res.chunks_added} "
                f"sections={res.sections_count} summary={res.summary_count}{norm_bits}"
            )
        elif res.status == "skipped":
            skipped += 1
            log_box.info(f"[SKIP] {res.file_name} : {res.error}")
        else:
            failed += 1
            log_box.error(f"[FAIL] {res.file_name} : {res.error}")
        progress.progress((i + 1) / n)

    progress.empty()
    st.success(
        f"완료: indexed={indexed} · skipped={skipped} · failed={failed} · "
        f"신규 chunk={chunks_total} · normalized chunk={normalized_total}"
    )

    snap = tracker.snapshot()
    st.caption("이번 세션 누적 호출 통계 (대략):")
    st.json(snap)

# ----------------------------- 현재 상태 -----------------------------------
st.divider()
st.subheader("현재 색인 상태")
try:
    d_stats = DocumentStore().stats()
except Exception as e:  # noqa: BLE001
    d_stats = {"error": str(e)}
try:
    v_stats = VectorStore().stats()
except Exception as e:  # noqa: BLE001
    v_stats = {"error": str(e)}
col1, col2, col3 = st.columns(3)
col1.metric("Documents (디스크)", d_stats.get("documents", 0))
col2.metric("Chunks (디스크)", d_stats.get("chunks", 0))
col3.metric("Vector DB chunks", max(0, v_stats.get("count", 0)))
with st.expander("자세히"):
    st.write({"documents": d_stats, "vector_store": v_stats})

st.caption(
    "Vector DB / 처리 결과를 초기화하고 싶다면 터미널에서 "
    "`python scripts/reset_vector_db.py` 를 실행하세요."
)
