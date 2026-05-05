"""NormalizationStore 단위 테스트."""
from __future__ import annotations

import json

from src.normalization import NormalizationStore
from src.schemas import KnowledgeCard


def _card(card_id: str = "kc_001", title: str = "정산 프로세스 카드") -> KnowledgeCard:
    return KnowledgeCard(
        card_id=card_id,
        card_type="workflow",
        title=title,
        summary="정산 업무 절차를 정리한 지식카드.",
        source_file_name="정산 가이드_한글.docx",
        source_file_hash="b" * 64,
        source_category="guide",
        source_type="guide",
        document_date=None,
        display_date="해당 업무일",
        primary_topic="settlement",
        topic_tags=["settlement"],
        task_type="settlement",
        when_to_use="정산 업무를 처음 진행할 때 사용한다.",
        prerequisites=["거래명세서 확인"],
        steps=["정산 대상 기간을 확인한다.", "세금계산서 발행 요청을 준비한다."],
        checkpoints=["금액 단위 확인"],
        cautions=["원본 파일은 수정하지 않는다."],
        examples=["광고주 공유용 정산 시트 확인"],
        related_terms=["SF", "모비사인"],
        open_questions=[],
        evidence_spans=[
            {"section": "정산 프로세스", "chunk_index": 1, "summary": "정산 단계 근거"}
        ],
        parent_raw_chunk_ids=["chunk_001"],
        sanitized_markdown="",
        metadata={"prompt_version": "mock-v1"},
    )


def test_make_cache_key_is_stable(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    key1 = store.make_cache_key("abc", "prompt-v1", "model-a")
    key2 = store.make_cache_key("abc", "prompt-v1", "model-a")
    key3 = store.make_cache_key("abc", "prompt-v2", "model-a")
    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64


def test_set_cache_has_cache_get_cache(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    key = store.make_cache_key("abc", "prompt-v1", "model-a")
    assert store.has_cache(key) is False
    assert store.get_cache(key) is None

    store.set_cache(key, {"json_path": "json/abc.json", "card_count": 1, "한글": "정상"})

    assert store.has_cache(key) is True
    cached = store.get_cache(key)
    assert cached is not None
    assert cached["card_count"] == 1
    assert cached["한글"] == "정상"


def test_cache_index_uses_utf8_json(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    key = store.make_cache_key("hash", "프롬프트", "모델")
    store.set_cache(key, {"title": "한글 제목"})

    raw = store.cache_index_path.read_text(encoding="utf-8")
    assert "한글 제목" in raw
    parsed = json.loads(raw)
    assert parsed[key]["title"] == "한글 제목"


def test_save_cards_json_and_load_roundtrip(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    cards = [_card(), _card(card_id="kc_002", title="메타 캠페인 카드")]

    path = store.save_cards_json("abcdef12", cards)
    loaded = store.load_cards_json(path)

    assert path.exists()
    assert loaded == cards
    assert loaded[0].source_file_name == "정산 가이드_한글.docx"
    assert loaded[1].title == "메타 캠페인 카드"


def test_save_cards_json_preserves_korean(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    path = store.save_cards_json("abcdef12", [_card(title="한글 제목 카드")])

    text = path.read_text(encoding="utf-8")
    assert "한글 제목 카드" in text
    assert "정산 가이드_한글.docx" in text


def test_save_cards_markdown_creates_file(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    path = store.save_cards_markdown("abcdef12", [_card()])

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# 정산 프로세스 카드" in text
    assert "## 업무 절차" in text
    assert "정산 대상 기간을 확인한다." in text


def test_save_cards_markdown_uses_sanitized_markdown_when_present(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    card = _card()
    card.sanitized_markdown = "# Sanitized\n\n정규화된 본문"

    path = store.save_cards_markdown("abcdef12", [card])
    text = path.read_text(encoding="utf-8")

    assert "# Sanitized" in text
    assert "정규화된 본문" in text
    assert "정산 프로세스 카드" not in text


def test_store_creates_expected_directories(tmp_path):
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    assert store.json_dir.exists()
    assert store.markdown_dir.exists()
    assert store.cache_dir.exists()
    assert store.cache_index_path.name == "index.json"
