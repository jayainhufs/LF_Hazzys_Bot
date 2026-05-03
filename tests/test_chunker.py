"""chunker 동작 테스트."""
from __future__ import annotations

from src.preprocessing.chunker import (
    build_context_header,
    chunk_sections,
    get_source_weight,
    link_excel_parent_child,
)
from src.schemas import Document, ParsedSection


def _doc(uploaded_category: str = "guide", source_type: str = "guide") -> Document:
    return Document(
        document_id="doc_abc12345",
        source_type=source_type,
        uploaded_category=uploaded_category,
        file_name="가이드_샘플.docx",
        file_path="data/raw/guide/가이드_샘플.docx",
        file_hash="0" * 64,
        title="가이드_샘플",
        created_at="2026-05-03T17:00:00+09:00",
        ingested_at="2026-05-03T17:00:00+09:00",
        metadata={},
    )


def test_chunk_section_basic():
    doc = _doc()
    sec = ParsedSection(
        section_id="sec_1",
        document_id=doc.document_id,
        section_title="ROAS 기준",
        content_type="text",
        content=("ROAS 는 광고 수익률 지표이다. " * 200).strip(),
        metadata={},
    )
    chunks = chunk_sections(document=doc, sections=[sec], chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.content.startswith("[파일명]")
        assert c.metadata["source_weight"] > 0
        assert c.metadata["chunk_hash"]


def test_source_weight_excel_summary_higher_than_raw():
    sw_summary = get_source_weight("excel", "excel_summary", "excel")
    sw_raw = get_source_weight("excel", "excel_raw_table", "excel")
    assert sw_summary > sw_raw


def test_link_excel_parent_child():
    doc = _doc(uploaded_category="excel", source_type="excel")
    summary = ParsedSection(
        section_id="sec_sum",
        document_id=doc.document_id,
        section_title="캠페인세팅",
        content_type="excel_summary",
        content="# 1. 시트/표 개요\n캠페인 세팅 시 ...",
        metadata={"sheet_name": "캠페인세팅"},
    )
    raw = ParsedSection(
        section_id="sec_raw",
        document_id=doc.document_id,
        section_title="캠페인세팅",
        content_type="excel_raw_table",
        content="컬럼A\t컬럼B\n1\t2\n3\t4",
        metadata={"sheet_name": "캠페인세팅"},
    )
    chunks = chunk_sections(document=doc, sections=[summary, raw])
    chunks = link_excel_parent_child(chunks)
    raw_chunks = [c for c in chunks if c.content_type == "excel_raw_table"]
    summary_chunks = [c for c in chunks if c.content_type == "excel_summary"]
    assert summary_chunks
    for c in raw_chunks:
        assert c.parent_chunk_id == summary_chunks[0].chunk_id


def test_build_context_header_contains_keys():
    h = build_context_header(
        file_name="x.xlsx",
        uploaded_category="excel",
        source_type="excel",
        section_title="캠페인",
        content_type="excel_summary",
    )
    assert "[파일명]" in h
    assert "[카테고리]" in h
    assert "[시트/섹션]" in h
    assert "[콘텐츠유형]" in h
