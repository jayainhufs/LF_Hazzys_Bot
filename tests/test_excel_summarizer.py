"""
excel_summarizer 캐시/스토어 동작 단위 테스트.
실제 Gemini 호출을 하지 않기 위해 GeminiClient 를 monkeypatch 한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.summarization.excel_summarizer import ExcelSummarizer
from src.summarization.summary_store import SummaryStore


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def generate_text(self, prompt, model=None, temperature=0.2, max_output_tokens=None):  # noqa: D401
        self.calls += 1
        return "# 1. 시트/표 개요\nFAKE SUMMARY"


def test_summary_cached_by_raw_table_hash(tmp_path: Path, monkeypatch):
    store = SummaryStore(base_dir=tmp_path)
    fake = _FakeClient()
    summarizer = ExcelSummarizer(client=fake, store=store, model_name="fake-model")

    raw = "컬럼A\t컬럼B\n1\t2\n3\t4"
    s1 = summarizer.summarize_section(
        document_id="doc1",
        file_name="x.xlsx",
        sheet_name="시트1",
        raw_table_text=raw,
        table_range="시트1!1:3",
        source_raw_path="data/raw/excel/x.xlsx",
    )
    s2 = summarizer.summarize_section(
        document_id="doc1",
        file_name="x.xlsx",
        sheet_name="시트1",
        raw_table_text=raw,  # 같은 raw → 캐시 hit 기대
        table_range="시트1!1:3",
        source_raw_path="data/raw/excel/x.xlsx",
    )
    assert s1.raw_table_hash == s2.raw_table_hash
    assert fake.calls == 1, "같은 raw_table_hash 면 한 번만 호출되어야 한다"
    md_files = list(tmp_path.glob("*.md"))
    assert md_files, "summary md 파일이 저장되어야 한다"
