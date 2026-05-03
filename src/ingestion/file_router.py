"""
file_router.py
==============
업로드된 파일을 적절한 파서로 라우팅한다.

규칙:
- .xlsx / .xlsm                 -> excel_parser
- .docx                         -> word_parser   (단, uploaded_category=slack 이면 slack_manual_parser)
- .txt                          -> txt_parser    (kakao/slack 이면 위임)
- .md                           -> markdown_parser (slack 이면 위임)
- 그 외                         -> skip + 경고 로그

source_type 도 함께 결정한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.ingestion.excel_parser import parse_excel
from src.ingestion.word_parser import parse_word
from src.ingestion.txt_parser import parse_txt
from src.ingestion.markdown_parser import parse_markdown
from src.ingestion.kakao_parser import parse_kakao
from src.ingestion.slack_manual_parser import parse_slack_manual
from src.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTS = {".xlsx", ".xlsm", ".docx", ".txt", ".md"}


def determine_source_type(path: Path, uploaded_category: str) -> str:
    """
    source_type 결정 규칙.

    uploaded_category 가 명시적이면(slack/kakao/excel) 우선 적용.
    그 외에는 확장자에 따른 기본 source_type 사용.
    """
    ext = path.suffix.lower()
    if uploaded_category == "excel" or ext in {".xlsx", ".xlsm"}:
        return "excel"
    if uploaded_category == "kakao":
        return "kakao"
    if uploaded_category == "slack":
        return "slack_manual"
    if uploaded_category == "guide":
        if ext == ".docx":
            return "guide"
        if ext == ".md":
            return "guide"
        if ext == ".txt":
            return "guide"
    if ext == ".docx":
        return "word"
    if ext == ".md":
        return "markdown"
    if ext == ".txt":
        return "txt"
    return "misc"


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def parse_file(
    path: Path, document_id: str, uploaded_category: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    파일 1개를 파싱하여 (source_type, sections) 를 반환.

    상위 호출자가 파일 단위로 try/except 하므로 여기서는 raise 한다.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not is_supported(path):
        raise ValueError(f"지원하지 않는 확장자: {path.suffix} ({path.name})")

    source_type = determine_source_type(path, uploaded_category)
    ext = path.suffix.lower()

    log.info("[router] %s | category=%s | source_type=%s", path.name, uploaded_category, source_type)

    sections: List[Dict[str, Any]]
    if ext in {".xlsx", ".xlsm"}:
        sections = parse_excel(path, document_id)
    elif ext == ".docx":
        if uploaded_category == "slack":
            sections = parse_slack_manual(path, document_id)
        else:
            sections = parse_word(path, document_id)
    elif ext == ".txt":
        if uploaded_category == "kakao":
            sections = parse_kakao(path, document_id)
        elif uploaded_category == "slack":
            sections = parse_slack_manual(path, document_id)
        else:
            sections = parse_txt(path, document_id, uploaded_category=uploaded_category)
    elif ext == ".md":
        if uploaded_category == "slack":
            sections = parse_slack_manual(path, document_id)
        else:
            sections = parse_markdown(path, document_id, uploaded_category=uploaded_category)
    else:  # 사실상 도달 안 함
        raise ValueError(f"지원하지 않는 확장자: {ext}")

    return source_type, sections
