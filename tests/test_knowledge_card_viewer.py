"""Normalized Document viewer helper 단위 테스트.

명칭 변경 노트
----------------
- 함수 이름 (``load_all_cards_from_store`` 등) 은 기존 import 호환을 위해 유지된다.
- ``KnowledgeCard`` 는 ``NormalizedDocument`` 의 alias 다.
"""
from __future__ import annotations

from src.normalization import NormalizationStore
from src.normalization.card_viewer import (
    card_to_display_dict,
    filter_cards,
    list_normalized_json_files,
    list_normalized_markdown_files,
    load_all_cards_from_store,
    markdown_for_card,
    summarize_cards,
)
from src.schemas import KnowledgeCard


def _card(
    *,
    card_id: str = "kc_001",
    card_type: str = "workflow",
    title: str = "메타 캠페인 셋업 절차",
    summary: str = "메타 신규 캠페인을 설정하는 절차입니다.",
    primary_topic: str = "meta",
    source_file_name: str = "meta_guide.txt",
) -> KnowledgeCard:
    card = KnowledgeCard(
        card_id=card_id,
        card_type=card_type,
        title=title,
        summary=summary,
        source_file_name=source_file_name,
        source_file_hash="a" * 64,
        source_category="guide",
        source_type="guide",
        display_date="해당 업무일",
        primary_topic=primary_topic,
        topic_tags=[primary_topic, "setup"],
        task_type="setup",
        when_to_use="신규 캠페인 세팅 시 사용합니다.",
        prerequisites=["광고 계정 권한"],
        steps=["광고 계정을 선택합니다.", "타겟을 설정합니다."],
        checkpoints=["전환 이벤트 수신 여부 확인"],
        cautions=["임의 설정 금지"],
        examples=["신규 브랜드 런칭 캠페인"],
        related_terms=["광고 세트", "전환 이벤트"],
        open_questions=[],
        evidence_spans=[
            {
                "section_title": "메타 셋업",
                "chunk_index": 0,
                "quote_or_summary": "메타 셋업 절차 근거",
            }
        ],
        metadata={"prompt_version": "guide_v1"},
    )
    card.sanitized_markdown = card.to_markdown()
    return card


def test_load_all_cards_from_store_reads_saved_json(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    cards = [
        _card(card_id="kc_001"),
        _card(
            card_id="kc_002",
            card_type="checklist",
            title="정산 체크리스트",
            summary="정산 전 확인 항목입니다.",
            primary_topic="settlement",
            source_file_name="settlement_guide.txt",
        ),
    ]
    store.save_cards_json("abcdef12", cards)
    store.save_cards_markdown("abcdef12", cards)

    loaded = load_all_cards_from_store(store)
    json_files = list_normalized_json_files(store)
    markdown_files = list_normalized_markdown_files(store)

    assert len(loaded) == 2
    assert loaded[0].title == "메타 캠페인 셋업 절차"
    assert len(json_files) == 1
    assert len(markdown_files) == 1


def test_load_all_cards_from_empty_store_returns_empty_list(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")

    assert load_all_cards_from_store(store) == []
    assert list_normalized_json_files(store) == []
    assert list_normalized_markdown_files(store) == []


def test_summarize_cards_counts_distributions():
    cards = [
        _card(card_id="kc_001"),
        _card(card_id="kc_002", card_type="issue", primary_topic="settlement", source_file_name="slack.txt"),
        _card(card_id="kc_003", card_type="issue", primary_topic="settlement", source_file_name="slack.txt"),
    ]

    summary = summarize_cards(cards)

    assert summary["total_cards"] == 3
    assert summary["card_type_counts"]["workflow"] == 1
    assert summary["card_type_counts"]["issue"] == 2
    assert summary["primary_topic_counts"]["settlement"] == 2
    assert summary["source_file_counts"]["slack.txt"] == 2


def test_filter_cards_by_card_type():
    cards = [_card(), _card(card_id="kc_002", card_type="issue")]

    filtered = filter_cards(cards, card_type="issue")

    assert len(filtered) == 1
    assert filtered[0].card_type == "issue"


def test_filter_cards_by_primary_topic():
    cards = [_card(), _card(card_id="kc_002", primary_topic="settlement")]

    filtered = filter_cards(cards, primary_topic="settlement")

    assert len(filtered) == 1
    assert filtered[0].primary_topic == "settlement"


def test_filter_cards_by_source_file_name():
    cards = [_card(), _card(card_id="kc_002", source_file_name="slack.txt")]

    filtered = filter_cards(cards, source_file_name="slack.txt")

    assert len(filtered) == 1
    assert filtered[0].source_file_name == "slack.txt"


def test_filter_cards_by_text_query_title_summary_steps_checkpoints():
    cards = [
        _card(card_id="kc_title", title="유튜브 캠페인 운영"),
        _card(card_id="kc_summary", summary="카카오 리포트 확인 절차입니다."),
        _card(card_id="kc_step"),
        _card(card_id="kc_checkpoint"),
    ]
    cards[2].steps = ["옥외 매체 결과를 정리합니다."]
    cards[3].checkpoints = ["정산 금액 단위 확인"]

    assert [c.card_id for c in filter_cards(cards, query="유튜브")] == ["kc_title"]
    assert [c.card_id for c in filter_cards(cards, query="카카오")] == ["kc_summary"]
    assert [c.card_id for c in filter_cards(cards, query="옥외")] == ["kc_step"]
    assert [c.card_id for c in filter_cards(cards, query="정산 금액")] == ["kc_checkpoint"]


def test_filter_cards_query_checks_cautions_related_terms():
    card = _card()
    card.cautions = ["월말 마감 전 임의 수정 금지"]
    card.related_terms = ["그린피", "매체비"]

    assert filter_cards([card], query="월말 마감") == [card]
    assert filter_cards([card], query="그린피") == [card]


def test_filter_cards_all_options_do_not_filter():
    cards = [_card(), _card(card_id="kc_002", card_type="issue", primary_topic="settlement")]

    filtered = filter_cards(
        cards,
        card_type="전체",
        primary_topic="전체",
        source_file_name="전체",
        query="",
    )

    assert filtered == cards


def test_card_to_display_dict_for_table():
    card = _card()
    card.open_questions = ["추가 확인 필요"]

    row = card_to_display_dict(card)

    assert row["title"] == "메타 캠페인 셋업 절차"
    assert row["card_type"] == "workflow"
    assert row["primary_topic"] == "meta"
    assert row["task_type"] == "setup"
    assert row["source_file_name"] == "meta_guide.txt"
    assert row["display_date"] == "해당 업무일"
    assert row["checkpoints_count"] == 1
    assert row["steps_count"] == 2
    assert row["open_questions_count"] == 1


def test_markdown_for_card_prefers_sanitized_markdown():
    card = _card()
    card.sanitized_markdown = "# Sanitized\n\n정리된 지식카드"

    assert markdown_for_card(card) == "# Sanitized\n\n정리된 지식카드"


def test_korean_title_summary_preserved(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    card = _card(title="한글 제목 카드", summary="한글 요약이 깨지면 안 됩니다.")
    store.save_cards_json("korean01", [card])

    loaded = load_all_cards_from_store(store)

    assert loaded[0].title == "한글 제목 카드"
    assert loaded[0].summary == "한글 요약이 깨지면 안 됩니다."


def test_invalid_json_file_is_skipped(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    (store.json_dir / "broken.json").write_text("{not-json", encoding="utf-8")

    assert load_all_cards_from_store(store) == []
