"""
anonymizer.py
=============
출력/표시/QA 컨텍스트용 비식별화 유틸.

원칙
----
- 원본 raw 문서/파일은 절대 변경하지 않는다. 이 모듈은 텍스트 변환만 수행한다.
- 검색 품질을 위해 발화자 "역할", "시간대", "TODO 단계", "업무 주제" 같은
  메타 정보는 metadata 에 보존한다.
- 실명, 멘션, 정확한 시간, 정확한 날짜는 표시(UI / QA 답변 / prompt context)에서
  기본적으로 비식별화한다.
- 설정(`settings.show_*`, `settings.mask_*`)에 따라 동작이 달라진다.

주요 API
--------
- ``anonymize_text(text, metadata=None, *, settings_obj=None) -> str``
- ``anonymize_speaker(name, role_hint=None, *, settings_obj=None, occurrence_idx=None) -> str``
- ``anonymize_timestamp(time_text, *, settings_obj=None) -> str``
- ``anonymize_date(date_text, *, settings_obj=None) -> str``
- ``mask_mentions(text) -> str``
- ``mask_links_and_files(text) -> str``
- ``time_bucket_for(time_text) -> str``
- ``time_range_display_for(buckets) -> str``
- ``role_label_for(role, occurrence_idx) -> str``
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

# 지연 import: settings 가 로드 시점에 .env 를 읽기 때문에 모듈 최상단 import 가능
from src.config import settings as _default_settings


# ---------------------------------------------------------------------------
# 정규식
# ---------------------------------------------------------------------------
# "이름[부서/팀]" 패턴 (한국어 이름 2~4자 + 대괄호 부서명)
_NAME_TEAM_RE = re.compile(r"([가-힣]{2,4})\[([^\[\]]{1,30})\]")

# @멘션 (Slack/카톡 복붙 형태). "@김보미[마케팅4팀]" / "@김보미"
_AT_MENTION_TEAM_RE = re.compile(r"@\s*([가-힣A-Za-z0-9_]{2,30})(?:\[([^\[\]]{1,30})\])?")

# Slack 내부 mention id 형태 <@U12345>
_SLACK_INTERNAL_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")

# 정확한 시간: "[오전 10:02]", "[오후 1:50]", "[10:02]", "(10:02)", "10:02 AM"
_EXACT_TIME_BRACKET_RE = re.compile(
    r"\[\s*(?P<ampm>오전|오후|AM|PM|am|pm)?\s*(?P<hour>\d{1,2}):(?P<min>\d{2})\s*\]"
)
_EXACT_TIME_INLINE_RE = re.compile(
    r"(?<![\d:])(?P<hour>\d{1,2}):(?P<min>\d{2})\s*(?P<ampm>오전|오후|AM|PM|am|pm)?"
)

# 정확한 날짜: "2026년 4월 29일", "2026-04-29", "2026/4/29", "4/29"
_DATE_FULL_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_DATE_DASH_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DATE_SHORT_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b(?!\d)")

# URL / 드라이브 / 파일명
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_DRIVE_RE = re.compile(r"(drive\.google\.com|docs\.google\.com)[^\s)>\]]*")
_IMAGE_FILE_RE = re.compile(r"\b\S+\.(png|jpg|jpeg|gif|bmp|webp)\b", re.IGNORECASE)
_OFFICE_FILE_RE = re.compile(r"\b\S+\.(xlsx|xlsm|docx|pdf|csv)\b", re.IGNORECASE)

# Slack 자동 안내 문구
_SLACK_FILE_COUNT_RE = re.compile(r"\b\d+\s*개\s*파일\b")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def time_bucket_for(time_text: Optional[str]) -> str:
    """
    "오전 10:02" / "오후 6:45" / "13:30" 같은 시각 텍스트에서
    morning / afternoon / evening / unknown 버킷을 추출한다.

    규칙:
    - 0~11시: morning
    - 12~17시: afternoon
    - 18~23시: evening
    - 추출 불가: unknown
    """
    if not time_text:
        return "unknown"
    s = time_text.strip()

    ampm = None
    if "오전" in s or s.lower().endswith("am"):
        ampm = "am"
    elif "오후" in s or s.lower().endswith("pm"):
        ampm = "pm"

    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return "unknown"
    hh = int(m.group(1))
    if ampm == "pm" and hh < 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    if 0 <= hh <= 11:
        return "morning"
    if 12 <= hh <= 17:
        return "afternoon"
    if 18 <= hh <= 23:
        return "evening"
    return "unknown"


_BUCKET_LABEL = {
    "morning": "오전",
    "afternoon": "오후",
    "evening": "퇴근 전",
    "unknown": "업무 시간대",
}


def time_range_display_for(buckets: Iterable[str]) -> str:
    """여러 메시지 시간대를 1~2개 라벨로 압축. UI 표시용."""
    seen: List[str] = []
    for b in buckets:
        if b and b not in seen:
            seen.append(b)
    if not seen:
        return _BUCKET_LABEL["unknown"]
    labels = [_BUCKET_LABEL.get(b, _BUCKET_LABEL["unknown"]) for b in seen[:2]]
    return " · ".join(labels)


def anonymize_timestamp(time_text: Optional[str], *, settings_obj=None) -> str:
    """정확한 시간을 시간대 라벨로 치환. 빈 입력은 그대로."""
    cfg = settings_obj or _default_settings
    if not time_text:
        return ""
    if cfg.show_exact_timestamps:
        return time_text
    bucket = time_bucket_for(time_text)
    return _BUCKET_LABEL.get(bucket, _BUCKET_LABEL["unknown"])


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def anonymize_date(date_text: Optional[str], *, settings_obj=None) -> str:
    """
    "2026년 4월 29일" 류 텍스트를 "해당 업무일" 같은 라벨로 치환.
    빈 입력은 빈 문자열.
    """
    cfg = settings_obj or _default_settings
    if not date_text:
        return ""
    if cfg.show_exact_dates:
        return date_text
    return f"해당 {cfg.anonymized_date_label}"


# ---------------------------------------------------------------------------
# Speaker helpers
# ---------------------------------------------------------------------------
_ROLE_LABEL_MAP = {
    "author": "작성자",
    "reviewer": "검토자",
    "bot": "봇/채널",
    "participant": "담당자",
}

# 작성자 본인을 가리키는 흔한 닉네임 키워드 (상대적으로 약한 신호)
_AUTHOR_HINT_TOKENS = ("동제", "동제쓰", "본인", "이동제")
# 명백한 봇/채널 (이름이 한국어 이름이 아닌 경우)
_BOT_HINT_TOKENS = ("LF 데일리", "TODO 봇", "Channel", "channel")


def _heuristic_role(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "participant"
    for tok in _BOT_HINT_TOKENS:
        if tok in s:
            return "bot"
    for tok in _AUTHOR_HINT_TOKENS:
        if tok in s:
            return "author"
    return "participant"


def role_label_for(role: str, occurrence_idx: Optional[int] = None) -> str:
    """
    역할 라벨 생성.
    - author      → "작성자"
    - reviewer    → "검토자"
    - bot         → "봇/채널"
    - participant → "담당자 A" / "담당자 B" ... (occurrence_idx 가 있으면 알파벳 부여)
    """
    role = (role or "participant").lower()
    if role == "participant" and occurrence_idx is not None and occurrence_idx >= 0:
        return f"담당자 {chr(ord('A') + min(occurrence_idx, 25))}"
    return _ROLE_LABEL_MAP.get(role, "담당자")


def anonymize_speaker(
    name: str,
    role_hint: Optional[str] = None,
    *,
    settings_obj=None,
    occurrence_idx: Optional[int] = None,
) -> str:
    """발화자 이름을 역할 기반 라벨로 치환. show_speaker_names=true 면 그대로 반환."""
    cfg = settings_obj or _default_settings
    if cfg.show_speaker_names:
        return name or ""
    role = (role_hint or _heuristic_role(name)).lower()
    return role_label_for(role, occurrence_idx=occurrence_idx)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def mask_mentions(text: str) -> str:
    """@이름[팀] 또는 <@U123> 멘션을 @담당자 로 치환."""
    if not text:
        return text
    text = _SLACK_INTERNAL_MENTION_RE.sub("@담당자", text)
    text = _AT_MENTION_TEAM_RE.sub("@담당자", text)
    return text


def mask_links_and_files(text: str) -> str:
    """링크/이미지/오피스 파일명을 라벨로 치환."""
    if not text:
        return text
    text = _DRIVE_RE.sub("[링크]", text)
    text = _URL_RE.sub("[링크]", text)
    text = _IMAGE_FILE_RE.sub("[이미지]", text)
    text = _OFFICE_FILE_RE.sub("[파일]", text)
    text = _SLACK_FILE_COUNT_RE.sub("[이미지/파일 첨부]", text)
    return text


def _replace_name_team(text: str) -> str:
    """
    "이동제[마케팅4팀]" 류 패턴을 발화자 라벨로 일괄 치환.
    동일 텍스트 안에서 동일 이름이 반복되면 같은 라벨을 부여한다.
    """
    if not text:
        return text
    seen: dict[str, str] = {}
    counter = {"idx": 0}

    def _sub(m: re.Match) -> str:
        name = m.group(1).strip()
        if name not in seen:
            role = _heuristic_role(name)
            if role == "participant":
                seen[name] = role_label_for("participant", occurrence_idx=counter["idx"])
                counter["idx"] += 1
            else:
                seen[name] = role_label_for(role)
        return seen[name]

    return _NAME_TEAM_RE.sub(_sub, text)


def _replace_dates(text: str, *, show: bool, label: str) -> str:
    if show:
        return text
    text = _DATE_FULL_RE.sub(f"해당 {label}", text)
    text = _DATE_DASH_RE.sub(f"해당 {label}", text)
    text = _DATE_SHORT_RE.sub(f"해당 {label}", text)
    return text


def _replace_times(text: str, *, show: bool, label: str) -> str:
    if show:
        return text

    def _bracket(m: re.Match) -> str:
        return f"[{label}]"

    def _inline(m: re.Match) -> str:
        # "오전 10:02" 같은 표현이면 시간대 라벨, 아니면 [시간대]
        full = m.group(0)
        bucket = time_bucket_for(full)
        return f"[{_BUCKET_LABEL.get(bucket, label)}]"

    text = _EXACT_TIME_BRACKET_RE.sub(_inline, text)
    text = _EXACT_TIME_INLINE_RE.sub(_inline, text)
    return text


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def anonymize_text(
    text: str,
    metadata: Optional[dict] = None,  # noqa: ARG001 - 향후 확장 여지
    *,
    settings_obj=None,
) -> str:
    """
    UI/QA 출력용 비식별화 변환.

    설정(`ANONYMIZE_OUTPUT=false`) 이면 원문을 그대로 반환한다.
    """
    cfg = settings_obj or _default_settings
    if not text:
        return text or ""
    if not cfg.anonymize_output:
        return text

    s = text

    if cfg.mask_links:
        s = mask_links_and_files(s)
    if cfg.mask_mentions:
        s = mask_mentions(s)

    if not cfg.show_speaker_names:
        s = _replace_name_team(s)

    s = _replace_times(s, show=cfg.show_exact_timestamps, label=cfg.anonymized_time_label)
    s = _replace_dates(s, show=cfg.show_exact_dates, label=cfg.anonymized_date_label)

    return s
