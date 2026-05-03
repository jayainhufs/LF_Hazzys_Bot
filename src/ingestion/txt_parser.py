"""
txt_parser.py
=============
.txt 파일 파서.

- UTF-8, UTF-8-SIG, CP949, EUC-KR 순으로 인코딩 시도.
- uploaded_category 가 "kakao" 면 kakao_parser 위임.
- uploaded_category 가 "slack" 이면 slack_manual_parser 위임.
- 그 외에는 plain text 로 단일 ParsedSection 반환.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.logger import get_logger
from src.utils.encoding_utils import read_text_safely

log = get_logger(__name__)


def parse_txt(path: Path, document_id: str, uploaded_category: str = "misc") -> List[Dict[str, Any]]:
    # 위임 라우팅
    if uploaded_category == "kakao":
        from src.ingestion.kakao_parser import parse_kakao
        return parse_kakao(path, document_id)
    if uploaded_category == "slack":
        from src.ingestion.slack_manual_parser import parse_slack_manual
        return parse_slack_manual(path, document_id)

    try:
        text, used_enc = read_text_safely(path)
    except UnicodeDecodeError as e:
        log.error(
            "TXT 인코딩 디코딩 실패: %s. UTF-8 또는 CP949 로 다시 저장 후 시도해 주세요. (%s)",
            path.name,
            e,
        )
        raise

    text = text.strip()
    if not text:
        log.info("빈 TXT 파일 skip: %s", path.name)
        return []

    return [{
        "section_title": path.stem,
        "content_type": "text",
        "content": text,
        "metadata": {
            "encoding": used_enc,
            "char_count": len(text),
        },
    }]
