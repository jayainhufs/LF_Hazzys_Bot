"""SlackThreadKnowledgeNormalizer 단위 테스트.

외부 Gemini API 호출 없이 ``FakeGeminiClient`` 로 대체해서 검증한다.
- cache miss → LLM 1회 호출
- cache hit → LLM 호출 없음
- JSON 파싱 실패 / cards 키 없음 / 코드블록 응답
- card 수 제한, 한글 보존, Slack 특화 metadata 반영
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import pytest

from src.normalization import (
    SLACK_NORMALIZER_PROMPT_VERSION,
    NormalizationStore,
    SlackThreadKnowledgeNormalizer,
    build_slack_normalization_prompt,
)
from src.schemas import KnowledgeCard


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGeminiClient:
    """generate_text 시그니처만 GeminiClient 와 동일하게 맞춘 가짜 클라이언트."""

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
) -> SlackThreadKnowledgeNormalizer:
    store = NormalizationStore(output_dir=tmp_path / "normalized")
    return SlackThreadKnowledgeNormalizer(
        gemini_client=fake_client,
        store=store,
        settings=settings or _make_settings(),
    )


# ---------------------------------------------------------------------------
# Sample LLM responses
# ---------------------------------------------------------------------------
ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON = json.dumps(
    {
        "cards": [
            {
                "card_type": "issue",
                "title": "정산서 단위 불일치 이슈 처리",
                "summary": "원/USD 단위 표기 누락으로 정산 금액이 어긋났던 사례와 처리 방향.",
                "primary_topic": "settlement",
                "topic_tags": ["settlement", "issue"],
                "task_type": "settlement",
                "when_to_use": "정산서에서 단위 누락이 의심될 때",
                "prerequisites": ["광고주 정산서 사본 확보"],
                "steps": [
                    "정산서에서 단위(원/USD) 표기 여부를 확인한다.",
                    "단위가 누락된 항목은 재무팀에 단위 명시 요청을 전달한다.",
                ],
                "checkpoints": ["수정된 정산서가 다시 공유되었는지 확인"],
                "cautions": ["단위 가정으로 임의 환산 금지"],
                "examples": [],
                "related_terms": ["정산서", "환율"],
                "open_questions": ["환율 적용 시점은 추가 확인 필요"],
                "evidence_spans": [
                    {
                        "section_title": "정산 이슈 공유",
                        "chunk_index": 0,
                        "quote_or_summary": "정산서 단위 누락 사례 근거",
                    }
                ],
            },
            {
                "card_type": "decision",
                "title": "이번 달 옥외 매체 우선순위 결정",
                "summary": "옥외 매체 중 그린피 우선 진행으로 합의된 결정.",
                "primary_topic": "outdoor",
                "topic_tags": ["outdoor", "greenp"],
                "task_type": "communication",
                "when_to_use": "이번 달 옥외 캠페인 운영 방향이 필요할 때",
                "prerequisites": [],
                "steps": [],
                "checkpoints": ["담당자 간 우선순위 합의 여부 확인"],
                "cautions": [],
                "examples": [],
                "related_terms": ["그린피", "옥외 매체"],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "옥외 운영 방향",
                        "chunk_index": 0,
                        "quote_or_summary": "그린피 우선 진행 합의 근거",
                    }
                ],
            },
            {
                "card_type": "checklist",
                "title": "퇴근 전 TODO 체크리스트",
                "summary": "퇴근 전 마감 점검 항목 모음.",
                "primary_topic": "common",
                "topic_tags": ["todo", "end_of_day"],
                "task_type": "check",
                "when_to_use": "퇴근 전 마감 점검 시",
                "prerequisites": [],
                "steps": [],
                "checkpoints": [
                    "광고비 단위(원/USD) 재확인",
                    "다음 업무일 일정 정리",
                ],
                "cautions": ["미처리 이슈는 다음 업무일로 이관 메모"],
                "examples": [],
                "related_terms": [],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "퇴근 전 정리",
                        "chunk_index": 0,
                        "quote_or_summary": "퇴근 전 점검 항목 근거",
                    }
                ],
            },
            {
                "card_type": "communication_template",
                "title": "광고주 정산서 재공유 안내 문안",
                "summary": "단위 누락 수정 후 광고주에게 재공유할 때 사용할 문안.",
                "primary_topic": "settlement",
                "topic_tags": ["settlement", "communication"],
                "task_type": "communication",
                "when_to_use": "정산서 단위 누락 수정 후 광고주에게 재공유할 때",
                "prerequisites": ["수정된 정산서 확정"],
                "steps": [],
                "checkpoints": [],
                "cautions": ["사람 실명 / 정확한 시간 노출 금지"],
                "examples": [
                    "안녕하세요, 정산서 단위 표기를 보완하여 재공유드립니다."
                ],
                "related_terms": [],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "재공유 문안",
                        "chunk_index": 0,
                        "quote_or_summary": "광고주 재공유 메시지 초안 근거",
                    }
                ],
            },
        ]
    },
    ensure_ascii=False,
)


# ---------------------------------------------------------------------------
# Prompt structure tests
# ---------------------------------------------------------------------------
def test_slack_prompt_includes_anonymization_principles():
    sys_instr, user_prompt = build_slack_normalization_prompt(
        file_name="slack_thread.txt",
        document_date="2026-04-29",
        display_date="해당 업무일",
        content="원문 일부",
        topic_tags=["settlement", "outdoor"],
        todo_phase="end_of_day",
        parser_format="slack_todo_sections",
    )

    assert "사람 실명" in sys_instr
    assert "@멘션" in sys_instr
    assert "정확한 시간" in sys_instr
    assert "[링크]" in sys_instr
    assert "[이미지]" in sys_instr
    assert "[파일]" in sys_instr
    assert "JSON" in sys_instr
    assert "Slack Thread" in sys_instr

    assert "topic_tags: settlement, outdoor" in user_prompt
    assert "todo_phase: end_of_day" in user_prompt
    assert "parser_format: slack_todo_sections" in user_prompt


def test_slack_prompt_handles_missing_optional_metadata():
    _, user_prompt = build_slack_normalization_prompt(
        file_name="slack_thread.txt",
        content="원문",
    )

    assert "topic_tags: -" in user_prompt
    assert "todo_phase: -" in user_prompt
    assert "parser_format: -" in user_prompt


# ---------------------------------------------------------------------------
# Normalizer behavior tests
# ---------------------------------------------------------------------------
def test_slack_normalizer_converts_response_into_knowledge_cards(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="Slack 본문",
        file_name="2026-04-29_정산_TODO.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    assert isinstance(cards, list)
    assert len(cards) == 4
    assert all(isinstance(c, KnowledgeCard) for c in cards)
    assert fake_client.call_count == 1


def test_slack_normalizer_creates_issue_card(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    issue_cards = [c for c in cards if c.card_type == "issue"]
    assert len(issue_cards) == 1
    issue = issue_cards[0]
    assert issue.primary_topic == "settlement"
    assert "정산서에서 단위(원/USD) 표기 여부를 확인한다." in issue.steps


def test_slack_normalizer_creates_decision_card(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    decision_cards = [c for c in cards if c.card_type == "decision"]
    assert len(decision_cards) == 1
    decision = decision_cards[0]
    assert decision.primary_topic == "outdoor"
    assert "그린피 우선" in decision.summary


def test_slack_normalizer_creates_checklist_card(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    checklist_cards = [c for c in cards if c.card_type == "checklist"]
    assert len(checklist_cards) == 1
    checklist = checklist_cards[0]
    assert "광고비 단위(원/USD) 재확인" in checklist.checkpoints


def test_slack_normalizer_creates_communication_template_card(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    template_cards = [c for c in cards if c.card_type == "communication_template"]
    assert len(template_cards) == 1
    template = template_cards[0]
    assert "재공유" in template.title
    assert any("정산서 단위 표기를 보완" in ex for ex in template.examples)


def test_slack_normalizer_fills_source_metadata(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="2026-04-29_정산_TODO.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
        document_date="2026-04-29",
        display_date="해당 업무일",
        topic_tags=["settlement"],
        todo_phase="end_of_day",
        parser_format="slack_todo_sections",
        metadata={"section_title": "퇴근 전 정리"},
    )

    for card in cards:
        assert card.source_file_name == "2026-04-29_정산_TODO.txt"
        assert card.source_file_hash == "a" * 64
        assert card.source_category == "slack"
        assert card.source_type == "slack_manual"
        assert card.document_date == "2026-04-29"
        assert card.display_date == "해당 업무일"
        assert card.metadata.get("prompt_version") == SLACK_NORMALIZER_PROMPT_VERSION
        assert card.metadata.get("model_name") == "fake-gemini-model"
        assert card.metadata.get("section_title") == "퇴근 전 정리"
        assert card.metadata.get("todo_phase") == "end_of_day"
        assert card.metadata.get("parser_format") == "slack_todo_sections"
        assert card.metadata.get("input_topic_tags") == ["settlement"]
        assert card.metadata.get("document_date") == "2026-04-29"
        assert card.metadata.get("display_date") == "해당 업무일"
        assert card.card_id.startswith("kc_aa11bb22_")


def test_slack_normalizer_merges_input_topic_tags_into_card(tmp_path):
    """LLM 이 반환한 topic_tags 와 입력 topic_tags 가 dedup 되어 합쳐지는지 검증."""
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
        topic_tags=["settlement", "month_end"],
    )

    issue_card = next(c for c in cards if c.card_type == "issue")
    assert "settlement" in issue_card.topic_tags
    assert "issue" in issue_card.topic_tags
    assert "month_end" in issue_card.topic_tags
    assert issue_card.topic_tags.count("settlement") == 1


def test_slack_normalizer_sanitized_markdown_includes_required_sections(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    issue_md = next(c for c in cards if c.card_type == "issue").sanitized_markdown
    assert "# 정산서 단위 불일치 이슈 처리" in issue_md
    assert "## 업무 절차" in issue_md
    assert "## 체크포인트" in issue_md
    assert "## 주의사항" in issue_md
    assert "## 근거" in issue_md
    assert "정산서에서 단위(원/USD) 표기 여부를 확인한다." in issue_md


def test_slack_normalizer_calls_llm_once_on_cache_miss(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    assert fake_client.call_count == 1


def test_slack_normalizer_skips_llm_on_cache_hit(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    first = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )
    assert fake_client.call_count == 1

    second = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )
    assert fake_client.call_count == 1, "cache hit 시 LLM 이 다시 호출되면 안 됨"
    assert len(second) == len(first)
    assert [c.card_id for c in second] == [c.card_id for c in first]
    assert [c.card_type for c in second] == [c.card_type for c in first]


def test_slack_normalizer_invalid_json_raises_value_error(tmp_path):
    fake_client = FakeGeminiClient(response="this is not json")
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    with pytest.raises(ValueError):
        normalizer.normalize_slack_thread_text(
            text="원문",
            file_name="thread.txt",
            file_hash="a" * 64,
            file_hash_short="aa11bb22",
        )


def test_slack_normalizer_missing_cards_key_raises_value_error(tmp_path):
    fake_client = FakeGeminiClient(response='{"results": []}')
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    with pytest.raises(ValueError):
        normalizer.normalize_slack_thread_text(
            text="원문",
            file_name="thread.txt",
            file_hash="a" * 64,
            file_hash_short="aa11bb22",
        )


def test_slack_normalizer_handles_code_fenced_json(tmp_path):
    fenced = "```json\n" + ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON + "\n```"
    fake_client = FakeGeminiClient(response=fenced)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    assert len(cards) == 4


def test_slack_normalizer_respects_max_cards_per_file(tmp_path):
    many_cards = {
        "cards": [
            {
                "card_type": "issue",
                "title": f"이슈 {i}",
                "summary": f"요약 {i}",
                "primary_topic": "common",
                "topic_tags": [],
                "task_type": "check",
                "when_to_use": "",
                "prerequisites": [],
                "steps": [],
                "checkpoints": ["점검"],
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

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="b" * 64,
        file_hash_short="bbbb1111",
    )

    assert len(cards) == 3
    assert [c.title for c in cards] == ["이슈 0", "이슈 1", "이슈 2"]


def test_slack_normalizer_preserves_korean_in_titles_and_summaries(tmp_path):
    payload = {
        "cards": [
            {
                "card_type": "issue",
                "title": "광고주 정산서 단위 누락 이슈",
                "summary": "원/USD 단위 누락으로 정산이 어긋난 사례를 정리한 카드.",
                "primary_topic": "settlement",
                "topic_tags": ["정산"],
                "task_type": "settlement",
                "when_to_use": "정산서 단위 누락 발생 시",
                "prerequisites": [],
                "steps": ["정산서 단위 표기를 다시 확인합니다."],
                "checkpoints": [],
                "cautions": [],
                "examples": [],
                "related_terms": ["정산", "환율"],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "정산 이슈 공유",
                        "chunk_index": 0,
                        "quote_or_summary": "정산서 단위 누락 사례 근거",
                    }
                ],
            }
        ]
    }
    fake_client = FakeGeminiClient(response=json.dumps(payload, ensure_ascii=False))
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="settle_thread.txt",
        file_hash="c" * 64,
        file_hash_short="cccc2222",
    )

    assert len(cards) == 1
    card = cards[0]
    assert card.title == "광고주 정산서 단위 누락 이슈"
    assert "원/USD" in card.summary
    assert "정산서 단위 표기를 다시 확인합니다." in card.steps

    json_path = tmp_path / "normalized" / "json" / "cccc2222.json"
    md_path = tmp_path / "normalized" / "markdown" / "cccc2222.md"
    assert json_path.exists()
    assert md_path.exists()
    assert "광고주 정산서 단위 누락 이슈" in json_path.read_text(encoding="utf-8")
    assert "광고주 정산서 단위 누락 이슈" in md_path.read_text(encoding="utf-8")


def test_slack_normalizer_passes_system_instruction_to_client(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="e" * 64,
        file_hash_short="eeee4444",
        topic_tags=["settlement"],
        todo_phase="end_of_day",
        parser_format="slack_todo_sections",
    )

    assert fake_client.last_system_instruction is not None
    assert "Slack" in fake_client.last_system_instruction
    assert "사람 실명" in fake_client.last_system_instruction
    assert "@멘션" in fake_client.last_system_instruction
    assert "정확한 시간" in fake_client.last_system_instruction
    assert "JSON" in fake_client.last_system_instruction
    assert fake_client.last_model == "fake-gemini-model"
    assert fake_client.last_temperature == pytest.approx(0.1)
    assert "topic_tags: settlement" in fake_client.last_prompt
    assert "todo_phase: end_of_day" in fake_client.last_prompt
    assert "parser_format: slack_todo_sections" in fake_client.last_prompt


def test_slack_normalizer_cache_value_records_paths_and_metadata(tmp_path):
    fake_client = FakeGeminiClient(response=ISSUE_DECISION_CHECKLIST_TEMPLATE_JSON)
    normalizer = _make_normalizer(tmp_path=tmp_path, fake_client=fake_client)

    normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="a" * 64,
        file_hash_short="aa11bb22",
    )

    cache_key = normalizer.store.make_cache_key(
        file_hash="a" * 64,
        prompt_version=SLACK_NORMALIZER_PROMPT_VERSION,
        model_name="fake-gemini-model",
    )
    cached = normalizer.store.get_cache(cache_key)

    assert cached is not None
    assert cached["card_count"] == 4
    assert cached["prompt_version"] == SLACK_NORMALIZER_PROMPT_VERSION
    assert cached["model_name"] == "fake-gemini-model"
    assert cached["json_path"]
    assert cached["markdown_path"]
    assert cached["source_kind"] == "slack_thread"


def test_slack_normalizer_invalid_card_type_falls_back_to_issue(tmp_path):
    payload = {
        "cards": [
            {
                "card_type": "totally_unknown",
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

    cards = normalizer.normalize_slack_thread_text(
        text="원문",
        file_name="thread.txt",
        file_hash="d" * 64,
        file_hash_short="dddd3333",
    )

    assert len(cards) == 1
    assert cards[0].card_type == "issue"


def test_slack_normalizer_llm_failure_raises_runtime_error(tmp_path):
    class FailingClient:
        def generate_text(self, *args, **kwargs):
            raise RuntimeError("network down")

    store = NormalizationStore(output_dir=tmp_path / "normalized")
    normalizer = SlackThreadKnowledgeNormalizer(
        gemini_client=FailingClient(),
        store=store,
        settings=_make_settings(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        normalizer.normalize_slack_thread_text(
            text="원문",
            file_name="thread.txt",
            file_hash="f" * 64,
            file_hash_short="ffff5555",
        )

    assert "LLM 호출 실패" in str(exc_info.value)
