"""excel_parser 최소 동작 테스트.

실제 .xlsx 파일은 openpyxl 로 즉석 생성한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.excel_parser import parse_excel


def _make_xlsx(tmp_path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "캠페인세팅"
    ws.append(["캠페인명", "예산", "ROAS"])
    ws.append(["Brand A", 1000000, 3.2])
    ws.append(["Brand B", 500000, 2.5])

    ws2 = wb.create_sheet("소재검수")
    ws2.append(["소재", "검수자", "통과여부"])
    ws2.append(["banner_v1", "민지", "Y"])
    ws2.append(["video_v1", "철수", "N"])

    p = tmp_path / "운영가이드_샘플.xlsx"
    wb.save(p)
    wb.close()
    return p


def test_parse_excel_returns_sections(tmp_path: Path):
    xlsx = _make_xlsx(tmp_path)
    sections = parse_excel(xlsx, document_id="doc_test")
    assert isinstance(sections, list)
    assert len(sections) >= 2
    sheets = {s["metadata"].get("sheet_name") for s in sections}
    assert "캠페인세팅" in sheets
    assert "소재검수" in sheets
    for s in sections:
        assert s["content"]
        assert s["content_type"] == "excel_raw_table"
        assert "[행]" not in s["content"]  # raw text 에는 마커가 들어가지 않음
