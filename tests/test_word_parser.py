"""word_parser 최소 동작 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.word_parser import parse_word


def _make_docx(tmp_path: Path) -> Path:
    docx_mod = pytest.importorskip("docx")
    doc = docx_mod.Document()
    h = doc.add_paragraph("운영 가이드")
    h.style = doc.styles["Heading 1"]
    doc.add_paragraph("이 문서는 캠페인 세팅 절차를 정리한 가이드입니다.")
    doc.add_paragraph("ROAS 기준은 2.0 이상을 권장합니다.")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "항목"
    table.cell(0, 1).text = "기준"
    table.cell(1, 0).text = "ROAS"
    table.cell(1, 1).text = "2.0+"

    p = tmp_path / "guide_샘플.docx"
    doc.save(p)
    return p


def test_parse_word_basic(tmp_path: Path):
    p = _make_docx(tmp_path)
    sections = parse_word(p, document_id="doc_test")
    assert sections, "최소 1개 이상 섹션이 추출되어야 한다"
    types = {s["content_type"] for s in sections}
    assert "text" in types or "table" in types
