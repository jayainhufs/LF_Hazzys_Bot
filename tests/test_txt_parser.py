"""txt_parser 최소 동작 + 인코딩 fallback 테스트."""
from __future__ import annotations

from pathlib import Path

from src.ingestion.txt_parser import parse_txt


def test_parse_txt_utf8(tmp_path: Path):
    p = tmp_path / "memo.txt"
    p.write_text("오늘 캠페인 세팅 끝났음.\n다음에는 ROAS 점검 필요.", encoding="utf-8")
    sections = parse_txt(p, document_id="doc_test", uploaded_category="misc")
    assert len(sections) == 1
    assert "ROAS" in sections[0]["content"]
    assert sections[0]["metadata"]["encoding"] == "utf-8"


def test_parse_txt_cp949(tmp_path: Path):
    p = tmp_path / "memo_cp949.txt"
    p.write_bytes("한글 텍스트 테스트입니다.".encode("cp949"))
    sections = parse_txt(p, document_id="doc_test", uploaded_category="misc")
    assert len(sections) == 1
    assert "한글 텍스트" in sections[0]["content"]
