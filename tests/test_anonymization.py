"""
test_anonymization.py
=====================
src.preprocessing.anonymizer 단위 테스트.

핵심 검증:
1. "이동제[마케팅4팀]" 같은 이름+팀 패턴이 역할 라벨로 치환된다.
2. "@김보미[마케팅4팀]" 멘션이 "@담당자" 로 치환된다.
3. "[오전 10:02]" 같은 정확한 시간이 시간대 라벨로 치환된다.
4. "2026년 4월 29일" / "4/29" 같은 정확한 날짜가 "해당 업무일" 로 치환된다.
5. URL / 드라이브 / 이미지 / 오피스 파일 링크가 [링크] / [이미지] / [파일] 로 치환된다.
6. ANONYMIZE_OUTPUT=False / SHOW_SPEAKER_NAMES=True 처럼 토글이 비활성이면 원문 유지.
7. time_bucket_for 가 morning/afternoon/evening/unknown 을 올바르게 반환한다.
"""
from __future__ import annotations

from copy import copy

from src.config import settings as real_settings
from src.preprocessing.anonymizer import (
    anonymize_date,
    anonymize_speaker,
    anonymize_text,
    anonymize_timestamp,
    mask_links_and_files,
    mask_mentions,
    role_label_for,
    time_bucket_for,
)


def _make_settings(**overrides):
    """real settings 를 복사해 일부만 덮어쓴 임시 settings."""
    s = copy(real_settings)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# 1. 이름+팀 → 역할 라벨
# ---------------------------------------------------------------------------
def test_name_team_pattern_replaced_with_role_label():
    text = "이동제[마케팅4팀]: 카탈로그 매핑 다 끝냈습니다.\n김보미[마케팅4팀]: 컨첵해드릴게요"
    s = _make_settings(
        anonymize_output=True,
        show_speaker_names=False,
        mask_mentions=True,
        mask_links=True,
    )
    out = anonymize_text(text, settings_obj=s)
    assert "이동제" not in out
    assert "김보미" not in out
    assert "마케팅4팀" not in out
    # 작성자 힌트 토큰('동제') 으로 author 라벨이 들어가야 한다
    assert ("작성자" in out) or ("담당자" in out)


def test_show_speaker_names_keeps_original():
    text = "이동제[마케팅4팀]: 안녕하세요"
    s = _make_settings(anonymize_output=True, show_speaker_names=True)
    out = anonymize_text(text, settings_obj=s)
    # show_speaker_names=True 면 이름은 보존
    assert "이동제" in out


def test_anonymize_off_keeps_original():
    text = "이동제[마케팅4팀]: 안녕하세요 https://example.com [오전 10:02]"
    s = _make_settings(anonymize_output=False)
    out = anonymize_text(text, settings_obj=s)
    assert out == text


# ---------------------------------------------------------------------------
# 2. 멘션
# ---------------------------------------------------------------------------
def test_mention_with_team_is_masked():
    text = "@김보미[마케팅4팀] 컨첵 부탁드립니다"
    s = _make_settings(anonymize_output=True, show_speaker_names=False, mask_mentions=True)
    out = anonymize_text(text, settings_obj=s)
    assert "김보미" not in out
    assert "@담당자" in out


def test_mask_mentions_helper():
    assert "@담당자" in mask_mentions("@김보미[마케팅4팀] 확인 부탁")
    assert "@담당자" in mask_mentions("<@U12345> 확인 요청")


# ---------------------------------------------------------------------------
# 3. 시간
# ---------------------------------------------------------------------------
def test_exact_timestamp_replaced_with_bucket_label():
    s = _make_settings(anonymize_output=True, show_exact_timestamps=False)
    out = anonymize_text("[오전 10:02] 회의 시작", settings_obj=s)
    assert "10:02" not in out
    assert "오전" in out  # bucket label "오전"

    out2 = anonymize_text("[오후 6:45] 퇴근 보고", settings_obj=s)
    assert "6:45" not in out2
    assert "퇴근 전" in out2 or "오후" in out2


def test_anonymize_timestamp_helper():
    s = _make_settings(anonymize_output=True, show_exact_timestamps=False)
    assert anonymize_timestamp("오전 10:02", settings_obj=s) == "오전"
    assert anonymize_timestamp("오후 1:50", settings_obj=s) == "오후"
    assert anonymize_timestamp("오후 6:45", settings_obj=s) == "퇴근 전"
    assert anonymize_timestamp("", settings_obj=s) == ""


def test_show_exact_timestamps_keeps_original():
    s = _make_settings(anonymize_output=True, show_exact_timestamps=True)
    assert anonymize_timestamp("오전 10:02", settings_obj=s) == "오전 10:02"


def test_time_bucket_for():
    assert time_bucket_for("오전 10:02") == "morning"
    assert time_bucket_for("오후 1:50") == "afternoon"
    assert time_bucket_for("오후 6:45") == "evening"
    assert time_bucket_for("") == "unknown"
    assert time_bucket_for("hello") == "unknown"


# ---------------------------------------------------------------------------
# 4. 날짜
# ---------------------------------------------------------------------------
def test_exact_date_replaced_with_label():
    s = _make_settings(anonymize_output=True, show_exact_dates=False)
    out = anonymize_text("2026년 4월 29일에 캠페인 셋팅 시작", settings_obj=s)
    assert "2026년 4월 29일" not in out
    assert "해당 업무일" in out

    out2 = anonymize_text("4/29 동제 TODO", settings_obj=s)
    assert "4/29" not in out2
    assert "해당 업무일" in out2


def test_anonymize_date_helper():
    s = _make_settings(anonymize_output=True, show_exact_dates=False)
    assert anonymize_date("2026년 4월 29일", settings_obj=s) == "해당 업무일"
    s2 = _make_settings(anonymize_output=True, show_exact_dates=True)
    assert anonymize_date("2026년 4월 29일", settings_obj=s2) == "2026년 4월 29일"


# ---------------------------------------------------------------------------
# 5. 링크 / 파일
# ---------------------------------------------------------------------------
def test_links_and_files_masked():
    text = (
        "참고: https://example.com/path\n"
        "드라이브: https://drive.google.com/file/d/abc\n"
        "image.png 첨부\n"
        "report.xlsx 확인 부탁"
    )
    out = mask_links_and_files(text)
    assert "https://" not in out
    assert "[링크]" in out
    assert "[이미지]" in out
    assert "[파일]" in out


def test_anonymize_text_masks_links_when_enabled():
    s = _make_settings(anonymize_output=True, mask_links=True)
    out = anonymize_text("자료 https://example.com 참고", settings_obj=s)
    assert "https://" not in out
    assert "[링크]" in out


def test_anonymize_text_skips_links_when_disabled():
    s = _make_settings(anonymize_output=True, mask_links=False)
    out = anonymize_text("자료 https://example.com 참고", settings_obj=s)
    assert "https://example.com" in out


# ---------------------------------------------------------------------------
# 6. role_label_for / anonymize_speaker
# ---------------------------------------------------------------------------
def test_role_label_for():
    assert role_label_for("author") == "작성자"
    assert role_label_for("reviewer") == "검토자"
    assert role_label_for("bot") == "봇/채널"
    assert role_label_for("participant", occurrence_idx=0) == "담당자 A"
    assert role_label_for("participant", occurrence_idx=1) == "담당자 B"
    assert role_label_for("participant", occurrence_idx=2) == "담당자 C"


def test_anonymize_speaker_uses_role():
    s = _make_settings(anonymize_output=True, show_speaker_names=False)
    # "동제" 가 들어간 이름은 author 로 추정
    out = anonymize_speaker("이동제", settings_obj=s, occurrence_idx=0)
    assert out == "작성자"
    out2 = anonymize_speaker("김보미", role_hint="reviewer", settings_obj=s)
    assert out2 == "검토자"
    out3 = anonymize_speaker("강민지", settings_obj=s, occurrence_idx=2)
    assert out3 == "담당자 C"


def test_anonymize_speaker_keeps_original_when_show():
    s = _make_settings(anonymize_output=True, show_speaker_names=True)
    assert anonymize_speaker("이동제", settings_obj=s) == "이동제"
