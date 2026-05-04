"""
slack_manual_parser.py  (v2)
============================
Slack 채널/스레드 내용을 직접 복사해 만든 TXT/MD/DOCX 파일 파서.

회사 보안상 Slack API 를 쓰지 않으므로, 사용자가 직접 복사한 텍스트를 적재한다.

지원 패턴
---------
- "이동제[마케팅4팀]  [오전 10:02]"
- "김보미[마케팅4팀]  [오후 1:50]"
- "[09:12] 이름: 메시지"
- "이름 9:12 AM" + 다음 줄 메시지
- "[2026년 4월 29일 TODO]"  (제목)
- "4/29 동제 TODO"  / "4/29 동제 오전 TODO" / "4/29 동제 중간 TODO"
- "4/29 동제 퇴근! TODO"  / "동제 명일 TODO" / "내일 동제쓰 TODO"

산출물 (sections)
-----------------
- ``content_type`` : "conversation"
- ``section_title`` : TODO 섹션 헤더 또는 메시지 그룹 라벨 (사람 이름 비포함)
- ``content`` : 원본 라인 묶음 (raw, 익명화 전)
- ``metadata`` :
  - ``document_date`` / ``date_text`` / ``date_source``
  - ``todo_phase`` : initial / morning / mid / end_of_day / tomorrow / unknown
  - ``topic_tags`` : list[str]
  - ``primary_topic`` : str | None
  - ``original_speakers`` : list[str]
  - ``speaker_roles`` : list[str]
  - ``display_speakers`` : list[str]
  - ``time_buckets`` : list[str]
  - ``time_range_display`` : str
  - ``parser_format`` : "structured_slack_messages" / "slack_todo_sections" / "fallback_block"
  - ``sanitized_content`` : 비식별화된 본문 (UI / prompt context 우선 사용)
  - ``message_count`` : int
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings as _default_settings
from src.ingestion.date_extractor import extract_document_date
from src.logger import get_logger
from src.preprocessing.anonymizer import (
    anonymize_text,
    anonymize_timestamp,
    role_label_for,
    time_bucket_for,
    time_range_display_for,
)
from src.utils.encoding_utils import read_text_safely

log = get_logger(__name__)

MESSAGES_PER_BLOCK = 40

# ---------------------------------------------------------------------------
# Slack 라인 패턴
# ---------------------------------------------------------------------------
# "[09:12] 이름: 메시지"
_SLACK_BRACKET = re.compile(
    r"^\[(?P<time>\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\]\s+"
    r"(?P<speaker>[^:]+?):\s*(?P<message>.*)$"
)
# "이름 9:12 AM" (다음 줄이 메시지)
_SLACK_HEADER = re.compile(
    r"^(?P<speaker>[^\s].{0,40}?)\s+(?P<time>\d{1,2}:\d{2}\s*[APap][Mm])\s*$"
)

# v2 추가: "이동제[마케팅4팀]  [오전 10:02]" 같은 헤더
_NAME_TEAM_TIME_RE = re.compile(
    r"^(?P<speaker>[가-힣A-Za-z0-9_ ]{1,30})\[(?P<team>[^\[\]]{1,30})\]\s*"
    r"\[(?P<ampm>오전|오후|AM|PM|am|pm)?\s*(?P<hour>\d{1,2}):(?P<min>\d{2})\s*\]\s*$"
)

# v2 추가: "[오전 10:57]@김보미[마케팅4팀] ..." 라인 (시간 + 멘션 + 본문이 한 줄)
_TIME_MENTION_INLINE_RE = re.compile(
    r"^\[(?P<ampm>오전|오후|AM|PM|am|pm)?\s*(?P<hour>\d{1,2}):(?P<min>\d{2})\s*\]\s*"
    r"(?P<rest>.*)$"
)

# TODO 섹션 헤더 패턴
_TODO_SECTION_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\[?\s*\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\s*TODO\s*\]?\s*$"), "initial"),
    (re.compile(r"^\d{1,2}/\d{1,2}\s+[^\n]*?오전\s*TODO\s*$"), "morning"),
    (re.compile(r"^\d{1,2}/\d{1,2}\s+[^\n]*?중간\s*TODO\s*$"), "mid"),
    (re.compile(r"^\d{1,2}/\d{1,2}\s+[^\n]*?퇴근[!]?\s*TODO\s*$"), "end_of_day"),
    (re.compile(r"^[^\n]*?(명일|내일)[^\n]*?TODO\s*$"), "tomorrow"),
    (re.compile(r"^\d{1,2}/\d{1,2}\s+[^\n]*?TODO\s*$"), "initial"),
    # 본문 안의 sub-section
    (re.compile(r"^\s*셋팅\s*내용\s*$"), "mid"),
    (re.compile(r"^\s*크첵\s*해주실\s*내용\s*$"), "mid"),
    (re.compile(r"^\s*\d+\.\s+.{0,80}원인\s*분석\s*$"), "mid"),
    (re.compile(r"^\s*\d+\.\s+.{0,80}운영\s*방향\s*제언\s*$"), "mid"),
]

# Topic keyword 매핑
_TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "meta": (
        "메타", "asc", "bau", "캠페인", "광고세트", "카탈로그", "컨첵", "크첵",
        "t&d", "td", "랜딩", "네이밍", "매핑", "토글",
    ),
    "kakao": ("카카오", "카카오톡", "카카오메시지", "메시지", "발송", "잔액", "충전"),
    "settlement": (
        "정산", "세금계산서", "인보이스", "모비사인", "sf", "입금", "거래명세서",
    ),
    "outdoor": ("옥외", "파르나스", "편성표", "구좌", "선입금"),
    "report": ("dr", "rd", "리포트", "월간보고", "월간 보고"),
    "nbt": ("nbt", "토스"),
    "greenp": ("그린피", "greenp"),
    "youtube": ("유튜브", "youtube", "구글 유튜브"),
    "common": ("광고주 공유", "노티", "메일 전달", "크첵"),
}


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


# ---------------------------------------------------------------------------
# 메시지 파싱 (v2)
# ---------------------------------------------------------------------------
def _parse_lines(text: str) -> List[Dict[str, Any]]:
    """라인 단위 메시지 추출. 형식 미일치 라인은 직전 메시지에 이어붙임."""
    messages: List[Dict[str, Any]] = []
    last_idx: Optional[int] = None
    pending_header: Optional[Dict[str, str]] = None  # {"speaker", "time", "team"}

    def _push(speaker: str, time_text: str, message: str, team: str = "") -> None:
        nonlocal last_idx
        messages.append({
            "speaker": speaker.strip(),
            "team": team.strip(),
            "time": time_text.strip(),
            "message": message.strip(),
        })
        last_idx = len(messages) - 1

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            pending_header = None
            continue

        s = line.strip()

        # 0) TODO 섹션 헤더 라인은 직전 메시지 본문에 합치지 않고
        #    별도 "(section_header)" 메시지로 push 한다.
        phase_hit = _detect_todo_phase(line)
        if phase_hit is not None:
            messages.append({
                "speaker": "(section_header)",
                "team": "",
                "time": "",
                "message": s,
                "phase": phase_hit,
                "is_section_header": True,
            })
            last_idx = len(messages) - 1
            pending_header = None
            continue

        # 1) v2: "이름[팀]  [오전 10:02]" 단독 헤더 → 다음 줄을 메시지로
        m_nt = _NAME_TEAM_TIME_RE.match(s)
        if m_nt:
            ampm = m_nt.group("ampm") or ""
            time_text = f"{ampm} {m_nt.group('hour')}:{m_nt.group('min')}".strip()
            pending_header = {
                "speaker": m_nt.group("speaker").strip(),
                "team": m_nt.group("team").strip(),
                "time": time_text,
            }
            continue

        # 2) "[09:12] 이름: 메시지"
        m = _SLACK_BRACKET.match(s)
        if m:
            _push(
                speaker=m.group("speaker"),
                time_text=m.group("time"),
                message=m.group("message"),
            )
            pending_header = None
            continue

        # 3) "이름 9:12 AM" 헤더 (legacy)
        m2 = _SLACK_HEADER.match(s)
        if m2:
            pending_header = {
                "speaker": m2.group("speaker").strip(),
                "team": "",
                "time": m2.group("time").strip(),
            }
            continue

        # 4) "[오전 10:57] @김보미[마케팅4팀] ..." 같은 시간+멘션+본문 한 줄
        m_tm = _TIME_MENTION_INLINE_RE.match(s)
        if m_tm:
            ampm = m_tm.group("ampm") or ""
            time_text = f"{ampm} {m_tm.group('hour')}:{m_tm.group('min')}".strip()
            rest = m_tm.group("rest").strip()
            if pending_header:
                _push(
                    speaker=pending_header["speaker"],
                    time_text=time_text or pending_header.get("time", ""),
                    message=rest,
                    team=pending_header.get("team", ""),
                )
                pending_header = None
            else:
                _push(speaker="(unknown)", time_text=time_text, message=rest)
            continue

        # 5) pending 헤더가 있으면 본문으로 합침
        if pending_header:
            _push(
                speaker=pending_header["speaker"],
                time_text=pending_header.get("time", ""),
                message=s,
                team=pending_header.get("team", ""),
            )
            pending_header = None
            continue

        # 6) 형식 불명 → 직전 메시지에 이어붙이거나 새 메시지
        if last_idx is not None:
            messages[last_idx]["message"] += "\n" + s
        else:
            _push(speaker="(unknown)", time_text="", message=s)

    return messages


# ---------------------------------------------------------------------------
# Topic / TODO 섹션 분류
# ---------------------------------------------------------------------------
def detect_topic_tags(text: str) -> List[str]:
    """텍스트에서 업무 topic_tags 를 추출."""
    if not text:
        return []
    low = text.lower()
    tags: List[str] = []
    for tag, kws in _TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in low:
                tags.append(tag)
                break
    return tags


def _detect_todo_phase(line: str) -> Optional[str]:
    """라인이 TODO 섹션 헤더인지 판단하고 phase 를 반환. 헤더가 아니면 None."""
    if not line:
        return None
    s = line.strip().strip("[]").strip()
    for pat, phase in _TODO_SECTION_PATTERNS:
        if pat.match(line.strip()) or pat.match(s):
            return phase
    return None


# ---------------------------------------------------------------------------
# 메시지 → 섹션 그룹핑
# ---------------------------------------------------------------------------
def _build_sections_from_messages(
    messages: List[Dict[str, Any]],
    *,
    encoding: str,
    file_text: str,
    document_date: Optional[str],
    date_text: Optional[str],
    date_source: str,
    settings_obj=None,
) -> List[Dict[str, Any]]:
    """
    메시지 리스트와 원본 텍스트의 라인을 보고 TODO 섹션 헤더 또는
    speaker/time 그룹 단위로 ParsedSection-like dict 들을 만든다.

    - TODO 섹션 헤더가 보이면 새 섹션을 연다.
    - 한 섹션 안에서 메시지 수가 너무 많으면 MESSAGES_PER_BLOCK 단위로 sub-block 으로 분리.
    - 각 섹션에 topic_tags, todo_phase, time_buckets, sanitized_content 를 채운다.
    """
    cfg = settings_obj or _default_settings

    if not messages:
        return []

    # _parse_lines 가 TODO 헤더 라인을 별도 (section_header) 메시지로 push 했기 때문에,
    # 메시지 리스트 안에서 직접 boundary 를 찾을 수 있다.
    unique_breaks: List[Tuple[int, str, str]] = []
    for i, msg in enumerate(messages):
        if msg.get("is_section_header"):
            unique_breaks.append((i, msg.get("message", "").strip(), msg.get("phase", "unknown")))

    # 섹션 분리: 각 section = [start_msg_idx, end_msg_idx)
    boundaries: List[Tuple[int, int, str, str]] = []  # (start, end, title, phase)
    if unique_breaks:
        # 첫 헤더 이전 메시지가 있으면 "인트로" 섹션
        first_break_idx = unique_breaks[0][0]
        if first_break_idx > 0:
            boundaries.append((0, first_break_idx, "Slack 인트로", "unknown"))
        for j, (idx, header, phase) in enumerate(unique_breaks):
            end = unique_breaks[j + 1][0] if j + 1 < len(unique_breaks) else len(messages)
            boundaries.append((idx, end, header, phase))
    else:
        boundaries.append((0, len(messages), "Slack 대화", "unknown"))
    # 사용하지 않는 file_text 매개변수 (지난 구현 호환을 위해 유지)
    _ = file_text

    sections: List[Dict[str, Any]] = []
    for sec_idx, (start, end, title, phase) in enumerate(boundaries):
        block = messages[start:end]
        if not block:
            continue

        # 메시지 수가 많으면 MESSAGES_PER_BLOCK 단위로 추가 분할
        for sub_idx, sub_start in enumerate(range(0, len(block), MESSAGES_PER_BLOCK)):
            sub = block[sub_start : sub_start + MESSAGES_PER_BLOCK]
            if not sub:
                continue
            section_dict = _build_one_section(
                messages=sub,
                title=title,
                phase=phase,
                section_idx=sec_idx,
                sub_idx=sub_idx,
                encoding=encoding,
                document_date=document_date,
                date_text=date_text,
                date_source=date_source,
                settings_obj=cfg,
            )
            sections.append(section_dict)

    return sections


def _build_one_section(
    *,
    messages: List[Dict[str, Any]],
    title: str,
    phase: str,
    section_idx: int,
    sub_idx: int,
    encoding: str,
    document_date: Optional[str],
    date_text: Optional[str],
    date_source: str,
    settings_obj,
) -> Dict[str, Any]:
    cfg = settings_obj
    body_lines: List[str] = []
    sanitized_lines: List[str] = []
    original_speakers: List[str] = []
    speaker_roles: List[str] = []
    display_speakers: List[str] = []
    time_buckets: List[str] = []

    speaker_to_role_idx: Dict[str, int] = {}

    for msg in messages:
        speaker = (msg.get("speaker") or "").strip()
        team = (msg.get("team") or "").strip()
        time_text = (msg.get("time") or "").strip()
        body = msg.get("message", "") or ""

        # 섹션 헤더 메시지는 prefix 없이 그대로 본문에 둔다.
        if msg.get("is_section_header"):
            body_lines.append(body)
            sanitized_lines.append(anonymize_text(body, settings_obj=cfg))
            continue

        # speaker role 추정
        from src.preprocessing.anonymizer import _heuristic_role  # late import to avoid cycle
        role = _heuristic_role(speaker)
        # display speaker (occurrence based)
        if speaker not in speaker_to_role_idx:
            speaker_to_role_idx[speaker] = len(speaker_to_role_idx)
        occ = speaker_to_role_idx[speaker]
        display_name = role_label_for(role, occurrence_idx=occ if role == "participant" else None)

        bucket = time_bucket_for(time_text)

        # raw line (원본): 이름/시간 그대로 유지
        head_raw = " ".join(x for x in [time_text, f"{speaker}[{team}]" if team else speaker] if x)
        if head_raw:
            body_lines.append(f"[{head_raw}] {body}")
        else:
            body_lines.append(body)

        # sanitized line
        time_label = anonymize_timestamp(time_text, settings_obj=cfg)
        # body 자체도 익명화
        body_san = anonymize_text(body, settings_obj=cfg)
        head_san = " ".join(x for x in [time_label, display_name] if x)
        if head_san:
            sanitized_lines.append(f"[{head_san}] {body_san}")
        else:
            sanitized_lines.append(body_san)

        if speaker:
            original_speakers.append(speaker)
        speaker_roles.append(role)
        display_speakers.append(display_name)
        time_buckets.append(bucket)

    raw_body = "\n".join(body_lines).strip()
    sanitized_body = "\n".join(sanitized_lines).strip()

    section_text_for_topic = "\n".join([title, raw_body])
    topic_tags = detect_topic_tags(section_text_for_topic)
    primary_topic = topic_tags[0] if topic_tags else None

    section_title = title
    if sub_idx > 0:
        section_title = f"{title} (계속 {sub_idx + 1})"

    return {
        "section_title": section_title,
        "content_type": "conversation",
        "content": raw_body,
        "metadata": {
            "encoding": encoding,
            "block_index": section_idx,
            "sub_block_index": sub_idx,
            "message_count": len(messages),
            "document_date": document_date,
            "date_text": date_text,
            "date_source": date_source,
            "todo_phase": phase or "unknown",
            "topic_tags": topic_tags,
            "primary_topic": primary_topic,
            "original_speakers": list(dict.fromkeys(original_speakers)),
            "speaker_roles": speaker_roles,
            "display_speakers": list(dict.fromkeys(display_speakers)),
            "time_buckets": time_buckets,
            "time_range_display": time_range_display_for(time_buckets),
            "parser_format": "structured_slack_messages",
            "sanitized_content": sanitized_body,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_slack_manual(path: Path, document_id: str) -> List[Dict[str, Any]]:  # noqa: ARG001
    try:
        text, used_enc = _read_text(path)
    except Exception as e:
        log.error("Slack 수동 자료 읽기 실패: %s (%s)", path.name, e)
        raise

    text = (text or "").strip()
    if not text:
        return []

    # 0) 문서 날짜 추출 (파일명 우선)
    date_info = extract_document_date(file_name=path.name, content=text)
    document_date = date_info.get("document_date")
    date_text = date_info.get("date_text")
    date_source = date_info.get("date_source", "unknown")

    # 1) 메시지 파싱
    messages = _parse_lines(text)

    matched = sum(
        1 for m in messages
        if m.get("speaker") and m["speaker"] != "(unknown)"
    )

    # 2) 매칭이 거의 안 됐다면 fallback_block 으로 처리
    #    (단, TODO 섹션 헤더가 본문에 보이면 slack_todo_sections 로 살린다)
    if not messages or matched == 0:
        topic_tags = detect_topic_tags(text)
        sanitized = anonymize_text(text)
        return [{
            "section_title": path.stem,
            "content_type": "conversation",
            "content": text,
            "metadata": {
                "encoding": used_enc,
                "format": "fallback_block",
                "parser_format": "fallback_block",
                "document_date": document_date,
                "date_text": date_text,
                "date_source": date_source,
                "todo_phase": "unknown",
                "topic_tags": topic_tags,
                "primary_topic": topic_tags[0] if topic_tags else None,
                "original_speakers": [],
                "speaker_roles": [],
                "display_speakers": [],
                "time_buckets": [],
                "time_range_display": "업무 시간대",
                "sanitized_content": sanitized,
                "message_count": 0,
            },
        }]

    # 3) 정상 파싱 → TODO 섹션 + speaker/time block 단위로 분리
    sections = _build_sections_from_messages(
        messages,
        encoding=used_enc,
        file_text=text,
        document_date=document_date,
        date_text=date_text,
        date_source=date_source,
    )

    # 어떤 섹션에 TODO 헤더가 잡혔는지에 따라 parser_format 라벨 결정
    has_todo = any(s["metadata"].get("todo_phase") not in (None, "unknown") for s in sections)
    parser_label = "slack_todo_sections" if has_todo else "structured_slack_messages"
    for s in sections:
        s["metadata"]["parser_format"] = parser_label

    log.info(
        "Slack v2 파싱 완료: %s -> %d section (parser=%s, document_date=%s)",
        path.name,
        len(sections),
        parser_label,
        document_date,
    )
    return sections
