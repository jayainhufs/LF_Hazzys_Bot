"""
slack_manual_parser.py
======================
Slack 채널/스레드 내용을 직접 복사해 만든 TXT/MD/DOCX 파일 파서.

회사 보안상 Slack API 를 쓰지 않으므로, 사용자가 직접 복사한 텍스트를 적재한다.

가정하는 형식 (느슨하게):
- "[09:12] 김선배: 메시지"
- "김선배 9:12 AM\n메시지"
- 위 형식이 아니면 전체를 단일 conversation section 으로 처리.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.logger import get_logger
from src.utils.encoding_utils import read_text_safely

log = get_logger(__name__)

MESSAGES_PER_BLOCK = 40

# "[09:12] 이름: 메시지"
_SLACK_BRACKET = re.compile(
    r"^\[(?P<time>\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\]\s+"
    r"(?P<speaker>[^:]+?):\s*(?P<message>.*)$"
)
# "이름 9:12 AM" (다음 줄이 메시지)
_SLACK_HEADER = re.compile(
    r"^(?P<speaker>[^\s].{0,40}?)\s+(?P<time>\d{1,2}:\d{2}\s*[APap][Mm])\s*$"
)


def _read_text(path: Path) -> Tuple[str, str]:
    """Slack 자료는 .txt/.md 또는 .docx 모두 가능."""
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document as DocxDocument  # python-docx
            doc = DocxDocument(str(path))
            text = "\n".join((p.text or "") for p in doc.paragraphs)
            return text, "docx"
        except Exception as e:
            log.error("Slack DOCX 로드 실패: %s (%s)", path.name, e)
            raise
    return read_text_safely(path)


def _parse_lines(text: str) -> List[Dict[str, str]]:
    """라인 단위로 (time, speaker, message) 를 추출. 실패 시 직전 메시지에 이어붙임."""
    messages: List[Dict[str, str]] = []
    last_idx: Optional[int] = None
    pending_header: Optional[Tuple[str, str]] = None  # (speaker, time)

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            pending_header = None
            continue

        # 1) [HH:MM] 이름: 메시지
        m = _SLACK_BRACKET.match(line.strip())
        if m:
            messages.append({
                "time": m.group("time"),
                "speaker": m.group("speaker").strip(),
                "message": m.group("message").strip(),
            })
            last_idx = len(messages) - 1
            pending_header = None
            continue

        # 2) "이름 9:12 AM" 헤더 → 다음 줄을 메시지로
        m2 = _SLACK_HEADER.match(line.strip())
        if m2:
            pending_header = (m2.group("speaker").strip(), m2.group("time").strip())
            continue

        if pending_header:
            speaker, t = pending_header
            messages.append({"time": t, "speaker": speaker, "message": line.strip()})
            last_idx = len(messages) - 1
            pending_header = None
            continue

        # 3) 형식 불명 → 직전 메시지에 이어붙이거나 새 메시지
        if last_idx is not None:
            messages[last_idx]["message"] += "\n" + line.strip()
        else:
            messages.append({"time": "", "speaker": "(unknown)", "message": line.strip()})
            last_idx = len(messages) - 1

    return messages


def parse_slack_manual(path: Path, document_id: str) -> List[Dict[str, Any]]:
    try:
        text, used_enc = _read_text(path)
    except Exception as e:
        log.error("Slack 수동 자료 읽기 실패: %s (%s)", path.name, e)
        raise

    text = text.strip()
    if not text:
        return []

    messages = _parse_lines(text)

    # 패턴 매칭이 거의 안 됐다면 단일 conversation section 으로 fallback
    matched = sum(1 for m in messages if m.get("speaker") and m["speaker"] != "(unknown)")
    if not messages or matched == 0:
        return [{
            "section_title": path.stem,
            "content_type": "conversation",
            "content": text,
            "metadata": {"encoding": used_enc, "format": "fallback_block"},
        }]

    sections: List[Dict[str, Any]] = []
    for block_idx, start in enumerate(range(0, len(messages), MESSAGES_PER_BLOCK)):
        block = messages[start : start + MESSAGES_PER_BLOCK]
        body_lines = []
        for msg in block:
            head = " ".join(x for x in [msg.get("time"), msg.get("speaker")] if x)
            content = msg.get("message", "")
            if head:
                body_lines.append(f"[{head}] {content}")
            else:
                body_lines.append(content)
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        sections.append({
            "section_title": f"Slack 대화 블록 {block_idx + 1}",
            "content_type": "conversation",
            "content": body,
            "metadata": {
                "encoding": used_enc,
                "block_index": block_idx,
                "message_count": len(block),
            },
        })

    log.info("Slack 수동 자료 파싱 완료: %s -> %d section", path.name, len(sections))
    return sections
