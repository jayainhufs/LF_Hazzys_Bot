"""GuideKnowledgeNormalizer 단위 테스트.

외부 Gemini API 호출 없이 ``FakeGeminiClient`` 로 대체해서 검증한다.
- cache miss → LLM 1회 호출
- cache hit → LLM 호출 없음
- JSON 파싱 실패 / cards 키 없음 / 코드블록 응답
- card 수 제한, 한글 보존
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src.normalization import (
    GUIDE_NORMALIZER_PROMPT_VERSION,
    GuideKnowledgeNormalizer,
    NormalizationStore,
)
from src.schemas import KnowledgeCard


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGeminiClient:
    """generate_text 시그니처만 GeminiClient 와 동일하게 맞춘 가짜 클라이언트.

    - response: 항상 동일하게 반환할 문자열
    - responses: 호출 순서대로 반환할 문자열 리스트 (response 보다 우선)
    """

    def __init__(self, response: Optional[str] = None, responses=None) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None
        self.last_model: Optional[str] = None
        self.last_temperature: Optional[float] = None

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_model = model
        self.last_temperature = temperature
        self.last_system_instruction = system_instruction
        if self.responses:
            return self.responses.pop(0)
        return self.response or ""


def _make_settings(
    *,
    max_cards_per_file: int = 30,
    save_json: bool = True,
    save_markdown: bool = True,
    max_chars_per_call: int = 18000,
    temperature: float = 0.1,
    model: str = "fake-gemini-model",
) -> SimpleNamespace:
    return SimpleNamespace(
        llm_normalization_model=model,
        normalization_temperature=temperature,
        normalization_max_chars_per_call=max_chars_per_call,
        normalization_max_cards_per_file=max_cards_per_file,
        normalization_save_json=save_json,
        normalization_save_markdown=save_markdown,
        normalization_use_anonymized_input=True,
    )


def _make_normalizer(
    *,
    tmp_path,
    fake_client: FakeGeminiClient,
    settings: Optional[SimpleNamespace] = None,
) -> GuideKnowledgeNormalizer:
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    return GuideKnowledgeNormalizer(
        gemini_client=fake_client,
        store=store,
        settings=settings or _make_settings(),
    )


# ---------------------------------------------------------------------------
# Sample LLM responses
# ---------------------------------------------------------------------------
WORKFLOW_AND_CHECKLIST_JSON = json.dumps(
    {
        "cards": [
            {
                "card_type": "workflow",
                "title": "메타 캠페인 셋업 절차",
                "summary": "메타 광고 캠페인을 처음 셋업할 때의 표준 절차.",
                "primary_topic": "meta",
                "topic_tags": ["meta", "setup"],
                "task_type": "setup",
                "when_to_use": "메타 신규 캠페인 세팅이 필요할 때 사용한다.",
                "prerequisites": ["광고 계정 권한 확인", "예산 승인"],
                "steps": [
                    "광고 계정에서 캠페인 목표를 선택한다.",
                    "타겟과 게재 위치를 설정한다.",
                    "광고 세트와 소재를 등록한다.",
                ],
                "checkpoints": ["전환 이벤트가 정상 수신되는지 확인"],
                "cautions": ["원본 가이드와 다른 임의 설정 금지"],
                "examples": ["베스트 케이스: 신규 브랜드 런칭"],
                "related_terms": ["광고 세트", "전환 이벤트"],
                "open_questions": ["일부 카탈로그 광고 정책 확인 필요"],
                "evidence_spans": [
                    {
                        "section_title": "메타 셋업",
                        "chunk_index": 0,
                        "quote_or_summary": "메타 캠페인 표준 셋업 절차 근거",
                    }
                ],
            },
            {
                "card_type": "checklist",
                "title": "정산 사전 점검 체크리스트",
                "summary": "정산 전에 반드시 확인해야 하는 항목 모음.",
                "primary_topic": "settlement",
                "topic_tags": ["settlement", "check"],
                "task_type": "check",
                "when_to_use": "월말 정산 직전",
                "prerequisites": ["거래명세서 확보"],
                "steps": [],
                "checkpoints": [
                    "광고비 단위(원/USD) 확인",
                    "수수료율 적용 여부 확인",
                ],
                "cautions": ["환율 적용 시점에 주의"],
                "examples": [],
                "related_terms": ["SF", "정산서"],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "정산 체크리스트",
                        "chunk_index": 0,
                        "quote_or_summary": "정산 점검 항목 근거",
                    }
                ],
            },
        ]
    },
    ensure_ascii=False,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_guide_normalizer_converts_response_into_knowledge_cards(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="메타 가이드 본문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    assert isinstance(cards, list)
    assert len(cards) == 2
    assert all(isinstance(c, KnowledgeCard) for c in cards)
    assert fake_client.call_count == 1


def test_guide_normalizer_creates_workflow_and_checklist_cards(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    workflow = cards[0]
    checklist = cards[1]

    assert workflow.card_type == "workflow"
    assert workflow.primary_topic == "meta"
    assert "광고 계정에서 캠페인 목표를 선택한다." in workflow.steps

    assert checklist.card_type == "checklist"
    assert checklist.primary_topic == "settlement"
    assert "광고비 단위(원/USD) 확인" in checklist.checkpoints


def test_guide_normalizer_fills_source_metadata(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
        document_date="2026-04-29",
        display_date="해당 업무일",
        metadata={"section_title": "메타 셋업"},
    )

    for card in cards:
        assert card.source_file_name == "meta_guide.txt"
        assert card.source_file_hash == "a" * 64
        assert card.source_category == "guide"
        assert card.source_type == "guide"
        assert card.document_date == "2026-04-29"
        assert card.display_date == "해당 업무일"
        assert card.metadata.get("prompt_version") == GUIDE_NORMALIZER_PROMPT_VERSION
        assert card.metadata.get("model_name") == "fake-gemini-model"
        assert card.metadata.get("section_title") == "메타 셋업"
        assert card.card_id.startswith("kc_a1b2c3d4_")


def test_guide_normalizer_sanitized_markdown_includes_workflow_sections(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    workflow_md = cards[0].sanitized_markdown
    assert "# 메타 캠페인 셋업 절차" in workflow_md
    assert "## 업무 절차" in workflow_md
    assert "## 체크포인트" in workflow_md
    assert "## 주의사항" in workflow_md
    assert "## 근거" in workflow_md
    assert "광고 계정에서 캠페인 목표를 선택한다." in workflow_md


def test_guide_normalizer_calls_llm_once_on_cache_miss(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    assert fake_client.call_count == 1


def test_guide_normalizer_skips_llm_on_cache_hit(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    first = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )
    assert fake_client.call_count == 1

    second = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )
    assert fake_client.call_count == 1, "cache hit 시 LLM 이 다시 호출되면 안 됨"
    assert len(second) == len(first)
    assert [c.card_id for c in second] == [c.card_id for c in first]
    assert second[0].title == first[0].title


def test_guide_normalizer_invalid_json_raises_value_error(tmp_path):
    fake_client = FakeGeminiClient(response="this is not json")
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    with pytest.raises(ValueError):
        normalizer.normalize_guide_text(
            text="원문",
            file_name="meta_guide.txt",
            file_hash="a" * 64,
            file_hash_short="a1b2c3d4",
        )


def test_guide_normalizer_missing_cards_key_raises_value_error(tmp_path):
    fake_client = FakeGeminiClient(response='{"results": []}')
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    with pytest.raises(ValueError):
        normalizer.normalize_guide_text(
            text="원문",
            file_name="meta_guide.txt",
            file_hash="a" * 64,
            file_hash_short="a1b2c3d4",
        )


def test_guide_normalizer_handles_code_fenced_json(tmp_path):
    fenced = "```json\n" + WORKFLOW_AND_CHECKLIST_JSON + "\n```"
    fake_client = FakeGeminiClient(response=fenced)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    assert len(cards) == 2
    assert cards[0].card_type == "workflow"


def test_guide_normalizer_respects_max_cards_per_file(tmp_path):
    many_cards = {
        "cards": [
            {
                "card_type": "workflow",
                "title": f"카드 {i}",
                "summary": f"요약 {i}",
                "primary_topic": "common",
                "topic_tags": [],
                "task_type": "setup",
                "when_to_use": "",
                "prerequisites": [],
                "steps": ["단계"],
                "checkpoints": [],
                "cautions": [],
                "examples": [],
                "related_terms": [],
                "open_questions": [],
                "evidence_spans": [
                    {"section_title": "", "chunk_index": 0, "quote_or_summary": "근거"}
                ],
            }
            for i in range(10)
        ]
    }
    fake_client = FakeGeminiClient(response=json.dumps(many_cards, ensure_ascii=False))
    normalizer = _make_normalizer(
        tmp_path=tmp_path,
        fake_client=fake_client,
        settings=_make_settings(max_cards_per_file=3),
    )

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="b" * 64,
        file_hash_short="bbbb1111",
    )

    assert len(cards) == 3
    titles = [c.title for c in cards]
    assert titles == ["카드 0", "카드 1", "카드 2"]


def test_guide_normalizer_preserves_korean_in_titles_and_summaries(tmp_path):
    payload = {
        "cards": [
            {
                "card_type": "faq",
                "title": "정산 처리 방법은 어떻게 됩니까?",
                "summary": "월말 정산 처리 방법을 정리한 카드입니다.",
                "primary_topic": "settlement",
                "topic_tags": ["정산"],
                "task_type": "settlement",
                "when_to_use": "월말 정산 시",
                "prerequisites": [],
                "steps": ["거래명세서를 확인합니다.", "수수료율을 적용합니다."],
                "checkpoints": [],
                "cautions": [],
                "examples": [],
                "related_terms": ["광고 정산"],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "정산 가이드",
                        "chunk_index": 0,
                        "quote_or_summary": "월말 정산 처리 절차 근거",
                    }
                ],
            }
        ]
    }
    fake_client = FakeGeminiClient(response=json.dumps(payload, ensure_ascii=False))
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="settle_guide.txt",
        file_hash="c" * 64,
        file_hash_short="cccc2222",
    )

    assert len(cards) == 1
    card = cards[0]
    assert card.title == "정산 처리 방법은 어떻게 됩니까?"
    assert "월말 정산" in card.summary
    assert "거래명세서를 확인합니다." in card.steps

    json_path = tmp_path / "normalized" / "json" / "cccc2222.json"
    assert json_path.exists()
    raw_json = json_path.read_text(encoding="utf-8")
    assert "정산 처리 방법은 어떻게 됩니까?" in raw_json

    md_path = tmp_path / "normalized" / "markdown" / "cccc2222.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "정산 처리 방법은 어떻게 됩니까?" in md_text


def test_guide_normalizer_cache_value_records_paths_and_metadata(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="a" * 64,
        file_hash_short="a1b2c3d4",
    )

    cache_key = normalizer.store.make_cache_key(
        file_hash="a" * 64,
        prompt_version=GUIDE_NORMALIZER_PROMPT_VERSION,
        model_name="fake-gemini-model",
    )
    cached = normalizer.store.get_cache(cache_key)

    assert cached is not None
    assert cached["card_count"] == 2
    assert cached["prompt_version"] == GUIDE_NORMALIZER_PROMPT_VERSION
    assert cached["model_name"] == "fake-gemini-model"
    assert cached["json_path"]
    assert cached["markdown_path"]


def test_guide_normalizer_invalid_card_type_falls_back_to_workflow(tmp_path):
    payload = {
        "cards": [
            {
                "card_type": "unknown_type_xyz",
                "title": "잘못된 타입 카드",
                "summary": "fallback 검증",
                "primary_topic": "common",
                "evidence_spans": [
                    {"section_title": "", "chunk_index": 0, "quote_or_summary": "근거"}
                ],
            }
        ]
    }
    fake_client = FakeGeminiClient(response=json.dumps(payload, ensure_ascii=False))
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="d" * 64,
        file_hash_short="dddd3333",
    )

    assert len(cards) == 1
    assert cards[0].card_type == "workflow"


def test_guide_normalizer_passes_system_instruction_to_client(tmp_path):
    fake_client = FakeGeminiClient(response=WORKFLOW_AND_CHECKLIST_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_guide_text(
        text="원문",
        file_name="meta_guide.txt",
        file_hash="e" * 64,
        file_hash_short="eeee4444",
    )

    assert fake_client.last_system_instruction is not None
    assert "KnowledgeCard" in fake_client.last_system_instruction
    assert "JSON" in fake_client.last_system_instruction
    assert fake_client.last_model == "fake-gemini-model"
    assert fake_client.last_temperature == pytest.approx(0.1)


def test_guide_normalizer_llm_failure_raises_runtime_error(tmp_path):
    class FailingClient:
        def generate_text(self, *args, **kwargs):
            raise RuntimeError("network down")

    store = NormalizationStore(output_dir=tmp_path / "normalized")
    normalizer = GuideKnowledgeNormalizer(
        gemini_client=FailingClient(),
        store=store,
        settings=_make_settings(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        normalizer.normalize_guide_text(
            text="원문",
            file_name="meta_guide.txt",
            file_hash="f" * 64,
            file_hash_short="ffff5555",
        )

    assert "LLM 호출 실패" in str(exc_info.value)
