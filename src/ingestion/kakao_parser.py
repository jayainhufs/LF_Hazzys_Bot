"""
kakao_parser.py
===============
카카오톡 PC '대화 내보내기' TXT 형식 파서.

대표적인 형식 (한국어 PC 카카오톡):
- "2025년 5월 1일 오후 2:13, 김민지 : 오늘 캠페인 세팅 끝났어요"
- "[김민지] [오후 2:13] 안녕하세요"
- "--------------- 2025년 5월 1일 ---------------" (날짜 구분선)

전략:
- 위 두 패턴을 정규식으로 감지해 (datetime, speaker, message) 추출.
- 매칭 실패 줄은 직전 메시지에 이어 붙인다.
- 너무 작은 단위로 chunk 가 만들어지지 않게 일정 갯수(MESSAGES_PER_BLOCK)
  씩 묶어 ParsedSection 으로 반환한다.
- source_weight 는 chunker 단계에서 0.5 로 부여된다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.logger import get_logger
from src.utils.encoding_utils import read_text_safely

log = get_logger(__name__)

MESSAGES_PER_BLOCK = 50  # 대화 50개씩 묶어 한 chunk 의 후보로 만든다.

# 패턴 1: "YYYY년 M월 D일 오전/오후 H:MM, 이름 : 메시지"
_KAKAO_LINE_KOR = re.compile(
    r"^(?P<dt>\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s+오[전후]\s*\d{1,2}:\d{2}),\s*"
    r"(?P<speaker>[^:]+?)\s*:\s*(?P<message>.*)$"
)
# 패턴 2: "[이름] [오전/오후 H:MM] 메시지"
_KAKAO_LINE_BRACKET = re.compile(
    r"^\[(?P<speaker>[^\]]+)\]\s*\[(?P<dt>오[전후]\s*\d{1,2}:\d{2})\]\s*(?P<message>.*)$"
)
# 날짜 구분선: "--------------- YYYY년 M월 D일 ---------------"
_DATE_DIVIDER = re.compile(r"-{3,}\s*(?P<date>\d{4}년\s*\d{1,2}월\s*\d{1,2}일.*?)\s*-{3,}")


def _try_parse_line(line: str) -> Optional[Tuple[str, str, str]]:
    m = _KAKAO_LINE_KOR.match(line)
    if m:
        return m.group("dt"), m.group("speaker").strip(), m.group("message").strip()
    m = _KAKAO_LINE_BRACKET.match(line)
    if m:
        return m.group("dt"), m.group("speaker").strip(), m.group("message").strip()
    return None


def parse_kakao(path: Path, document_id: str) -> List[Dict[str, Any]]:
    try:
        text, used_enc = read_text_safely(path)
    except UnicodeDecodeError:
        log.error("Kakao TXT 디코딩 실패: %s", path.name)
        raise

    if not text.strip():
        return []

    messages: List[Dict[str, str]] = []
    current_date: str = ""
    last_msg_idx: Optional[int] = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _DATE_DIVIDER.match(line.strip())
        if m:
            current_date = m.group("date").strip()
            continue
        parsed = _try_parse_line(line.strip())
        if parsed:
            dt, speaker, message = parsed
            messages.append({
                "datetime": dt,
                "date_section": current_date,
                "speaker": speaker,
                "message": message,
            })
            last_msg_idx = len(messages) - 1
        else:
            # 형식 불명: 직전 메시지에 이어 붙임
            if last_msg_idx is not None:
                messages[last_msg_idx]["message"] += "\n" + line.strip()
            else:
                # 아직 첫 메시지도 없는 상태이면 미파싱 잡담으로 추가
                messages.append({
                    "datetime": "",
                    "date_section": current_date,
                    "speaker": "(unknown)",
                    "message": line.strip(),
                })
                last_msg_idx = len(messages) - 1

    if not messages:
        log.info("카카오톡 파싱 결과 0건: %s", path.name)
        return []

    sections: List[Dict[str, Any]] = []
    for block_idx, start in enumerate(range(0, len(messages), MESSAGES_PER_BLOCK)):
        block = messages[start : start + MESSAGES_PER_BLOCK]
        body_lines = []
        for msg in block:
            stamp = msg.get("datetime") or ""
            speaker = msg.get("speaker") or ""
            content = msg.get("message") or ""
            head = " ".join(x for x in [stamp, speaker] if x)
            if head:
                body_lines.append(f"[{head}] {content}")
            else:
                body_lines.append(content)
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        sections.append({
            "section_title": f"카톡대화 블록 {block_idx + 1}",
            "content_type": "conversation",
            "content": body,
            "metadata": {
                "encoding": used_enc,
                "block_index": block_idx,
                "message_count": len(block),
                "first_dt": block[0].get("datetime", ""),
                "last_dt": block[-1].get("datetime", ""),
            },
        })

    log.info("카카오톡 파싱 완료: %s -> %d section", path.name, len(sections))
    return sections
