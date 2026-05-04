"""
test_slack_parser_v2.py
=======================
Slack 복붙 자료 v2 파서 단위 테스트.

검증:
1. 파일명 "[2026년 4월 29일 TODO].txt" 에서 document_date=2026-04-29 추출
2. "이동제[마케팅4팀] [오전 10:02]" 헤더 + 다음 줄 본문 → message 1개로 묶이고
   original_speaker 는 metadata 에 저장, sanitized_content 에는 실명 미포함
3. "@김보미[마케팅4팀]" 멘션이 sanitized_content 에서 제거/치환됨
4. "4/29 동제 중간 TODO" 섹션이 todo_phase=mid 로 감지됨
5. "셋팅 내용" 영역에서 topic_tags 에 meta 가 포함됨
6. 카카오 발송 이슈 영역에서 topic_tags 에 kakao 가 포함됨
7. parser_format 이 fallback_block 이 아닌 slack_todo_sections 또는
   structured_slack_messages 가 됨
8. date_extractor 자체 단위 동작 확인
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.date_extractor import extract_document_date
from src.ingestion.slack_manual_parser import detect_topic_tags, parse_slack_manual


SLACK_TODO_TEXT = """[2026년 4월 29일 TODO]

LF 데일리 TODO  [오전 8:00]
오늘도 화이팅하세요!

이동제[마케팅4팀]  [오전 10:02]
@김보미[마케팅4팀] 안녕하세요. 4/29 동제 오전 TODO 공유드려요.
- BAU/ASC 캠페인 컨첵 시트 작성
- 카카오 메시지 발송 준비 (잔액 충전 필요)

김보미[마케팅4팀]  [오후 1:50]
@이동제[마케팅4팀] 컨첵 결과 확인했습니다. ASC vs BAU 성과 차이는 소재 학습 차이로 설명드리면 좋을 것 같아요.

4/29 동제 중간 TODO

셋팅 내용
- 메타 캠페인 및 광고세트 셋팅
- 네이밍 및 매핑
- 랜딩페이지 변경
- T&D 변경
- 게재지면 업데이트
- 캠페인 토글 ON/OFF

크첵 해주실 내용
- 컨첵시트 항목별 확인
- 광고주 공유 노티 메일

이동제[마케팅4팀]  [오후 4:20]
카카오 메시지 발송 관련해서 잔액 부족으로 즉시 발송 불가했습니다.
미진행 건 셋팅해두고 충전 후 시간 조율해서 발송하겠습니다.

4/29 동제 퇴근! TODO
- 옥외 파르나스 편성표 정리는 내일

내일 동제쓰 TODO
- 그린피, 유튜브 리포트 정리
"""


# ---------------------------------------------------------------------------
# date_extractor
# ---------------------------------------------------------------------------
def test_date_extract_from_filename():
    info = extract_document_date(file_name="[2026년 4월 29일 TODO].txt", content=None)
    assert info["document_date"] == "2026-04-29"
    assert info["date_source"] == "file_name"


def test_date_extract_from_content_when_no_filename_match():
    info = extract_document_date(file_name="random.txt", content="[2026년 4월 29일 TODO]\n본문")
    assert info["document_date"] == "2026-04-29"
    assert info["date_source"] == "content_title"


def test_date_extract_short_form_with_default_year():
    # 4/29 만 있고 연도 없음 → default_year 적용
    info = extract_document_date(file_name=None, content="4/29 동제 TODO", default_year=2026)
    assert info["document_date"] == "2026-04-29"


def test_date_extract_unknown_when_missing():
    info = extract_document_date(file_name="random.txt", content="안녕하세요")
    assert info["document_date"] is None
    assert info["date_source"] == "unknown"


# ---------------------------------------------------------------------------
# detect_topic_tags
# ---------------------------------------------------------------------------
def test_detect_topic_tags_meta():
    tags = detect_topic_tags("ASC vs BAU 성과 차이 분석. 메타 캠페인 셋팅, 컨첵 시트 작성")
    assert "meta" in tags


def test_detect_topic_tags_kakao():
    tags = detect_topic_tags("카카오 메시지 발송 잔액 충전 필요")
    assert "kakao" in tags


def test_detect_topic_tags_settlement():
    tags = detect_topic_tags("정산 / 세금계산서 / 모비사인 발행 요청")
    assert "settlement" in tags


def test_detect_topic_tags_outdoor():
    tags = detect_topic_tags("옥외 파르나스 편성표 구좌")
    assert "outdoor" in tags


# ---------------------------------------------------------------------------
# parse_slack_manual: 파일 단위 통합 시나리오
# ---------------------------------------------------------------------------
@pytest.fixture
def slack_file(tmp_path: Path) -> Path:
    p = tmp_path / "[2026년 4월 29일 TODO].txt"
    p.write_text(SLACK_TODO_TEXT, encoding="utf-8")
    return p


def test_parser_extracts_document_date_from_filename(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    assert sections, "섹션이 적어도 1개는 나와야 한다"
    for s in sections:
        meta = s["metadata"]
        assert meta.get("document_date") == "2026-04-29"
        assert meta.get("date_source") == "file_name"


def test_parser_avoids_fallback_block_for_structured_input(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    formats = {s["metadata"].get("parser_format") for s in sections}
    assert "fallback_block" not in formats, (
        f"v2 파서는 구조화된 입력에서 fallback_block 을 만들면 안 된다. "
        f"실제: {formats}"
    )
    # slack_todo_sections 또는 structured_slack_messages 라벨 중 하나
    assert formats <= {"slack_todo_sections", "structured_slack_messages"}


def test_parser_detects_todo_phase_mid(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    phases = [s["metadata"].get("todo_phase") for s in sections]
    assert "mid" in phases, f"중간 TODO 섹션이 mid phase 로 잡혀야 한다. 실제: {phases}"


def test_parser_detects_todo_phase_end_of_day_and_tomorrow(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    phases = [s["metadata"].get("todo_phase") for s in sections]
    assert "end_of_day" in phases
    assert "tomorrow" in phases


def test_parser_detects_topic_meta_for_setting_section(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    # "셋팅 내용" / "크첵 해주실 내용" 가 들어있는 섹션 중 하나에 meta topic 이 있어야 함
    found = False
    for s in sections:
        body = s["content"]
        if "셋팅 내용" in body or "크첵" in body:
            tags = s["metadata"].get("topic_tags") or []
            if "meta" in tags:
                found = True
                break
    assert found, "셋팅 내용 섹션에서 meta topic 이 감지되어야 한다"


def test_parser_detects_topic_kakao_for_kakao_section(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    # 카카오 발송 관련 텍스트가 들어 있는 섹션을 찾는다
    kakao_section_tags = []
    for s in sections:
        if "카카오" in s["content"]:
            kakao_section_tags.append(s["metadata"].get("topic_tags") or [])
    assert any("kakao" in tags for tags in kakao_section_tags), (
        "카카오 메시지 발송 섹션에서 kakao topic 이 감지되어야 한다"
    )


def test_parser_keeps_original_speaker_in_metadata_but_hides_in_sanitized(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    # 어떤 섹션에는 "이동제" / "김보미" 가 original_speakers 에 있어야 한다.
    has_original = False
    for s in sections:
        meta = s["metadata"]
        if "이동제" in (meta.get("original_speakers") or []):
            has_original = True
        sanitized = meta.get("sanitized_content") or ""
        # sanitized 본문에는 실명이 없어야 한다
        assert "이동제" not in sanitized, "sanitized_content 에 실명이 남아있으면 안 된다"
        assert "김보미" not in sanitized, "sanitized_content 에 실명이 남아있으면 안 된다"
        assert "마케팅4팀" not in sanitized, "팀명도 sanitized 에 남으면 안 된다"
    assert has_original, "최소 한 섹션에는 original_speakers 가 보존되어야 한다"


def test_parser_masks_at_mention_in_sanitized(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    # "@김보미[마케팅4팀]" 멘션은 sanitized 에서 제거되어야 함
    for s in sections:
        sanitized = s["metadata"].get("sanitized_content") or ""
        assert "@김보미" not in sanitized
        assert "@이동제" not in sanitized


def test_parser_records_time_buckets(slack_file: Path):
    sections = parse_slack_manual(slack_file, document_id="doc1")
    # 어떤 섹션에는 morning / afternoon / evening 이 섞여 있어야 함
    all_buckets = []
    for s in sections:
        all_buckets.extend(s["metadata"].get("time_buckets") or [])
    assert "morning" in all_buckets
    assert "afternoon" in all_buckets
    # time_range_display 라벨이 채워져 있어야 함
    for s in sections:
        rd = s["metadata"].get("time_range_display")
        assert rd, "time_range_display 가 채워져 있어야 함"
