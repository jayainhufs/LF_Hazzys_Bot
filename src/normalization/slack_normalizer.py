"""
slack_normalizer.py
===================
Slack Thread txt 문서를 LLM 으로 KnowledgeCard 로 정규화하는 normalizer.

Task 3 범위
-----------
- Slack Thread 본문(또는 Slack parser v2 가 만든 sanitized_content) 을 받아
  KnowledgeCard 리스트로 변환한다.
- file_hash + prompt_version + model_name 으로 cache 를 사용한다.
- LLM 호출은 ``GeminiClient.generate_text(system_instruction=..., temperature=...)``
  에 의존한다. 테스트에서는 동일 시그니처의 fake client 를 주입한다.
- pipeline 연결, Streamlit UI, retrieval / QA 변경은 다른 Task 에서 처리한다.

Slack 특화 동작
---------------
- Slack parser v2 가 추출한 ``topic_tags`` / ``todo_phase`` / ``parser_format``
  을 prompt 에 명시해 카드 분리·토픽 추론에 활용한다.
- 같은 metadata 를 결과 ``KnowledgeCard.metadata`` 에도 보존해, Task 6~7
  의 검색·답변 단계에서 활용할 수 있게 한다.

Guide normalizer 와 공통인 JSON 파싱 / 필드 정규화 / cache 흐름은
``guide_normalizer`` 모듈의 helper 를 그대로 재사용한다 (코드 중복 방지).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings as default_settings
from src.logger import get_logger
from src.normalization.guide_normalizer import (
    _as_str,
    _as_str_list,
    _coerce_card_type,
    _make_card_id,
    _normalize_evidence_spans,
    parse_cards_response,
)
from src.normalization.normalization_prompt import (
    SLACK_NORMALIZER_PROMPT_VERSION,
    build_slack_normalization_prompt,
)
from src.normalization.normalization_store import NormalizationStore
from src.schemas import KnowledgeCard

log = get_logger(__name__)


class SlackThreadKnowledgeNormalizer:
    """Slack Thread 문서를 KnowledgeCard 리스트로 정규화하는 LLM normalizer."""

    def __init__(
        self,
        gemini_client: Any,
        store: NormalizationStore,
        settings: Any = None,
        prompt_version: str = SLACK_NORMALIZER_PROMPT_VERSION,
    ) -> None:
        if gemini_client is None:
            raise ValueError("gemini_client 가 필요합니다.")
        if store is None:
            raise ValueError("NormalizationStore 가 필요합니다.")
        self.gemini_client = gemini_client
        self.store = store
        self.settings = settings or default_settings
        self.prompt_version = prompt_version or SLACK_NORMALIZER_PROMPT_VERSION

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def normalize_slack_thread_text(
        self,
        text: str,
        *,
        file_name: str,
        file_hash: str,
        file_hash_short: str,
        source_category: str = "slack",
        source_type: str = "slack_manual",
        document_date: Optional[str] = None,
        display_date: Optional[str] = None,
        topic_tags: Optional[List[str]] = None,
        todo_phase: Optional[str] = None,
        parser_format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
    ) -> List[KnowledgeCard]:
        """Slack Thread 본문을 LLM 으로 정규화해 KnowledgeCard 리스트를 반환한다.

        cache hit 이면 LLM 호출 없이 저장된 JSON 으로부터 KnowledgeCard 들을 복원한다.
        cache miss 이면 LLM 을 1회 호출하고, 결과를 JSON / Markdown 으로 저장한 뒤
        cache index 를 업데이트한다.
        """
        if not file_name:
            raise ValueError("file_name 이 필요합니다.")
        if not file_hash:
            raise ValueError("file_hash 가 필요합니다.")
        if not file_hash_short:
            raise ValueError("file_hash_short 가 필요합니다.")

        model_name = self.settings.llm_normalization_model
        cache_key = self.store.make_cache_key(
            file_hash=file_hash,
            prompt_version=self.prompt_version,
            model_name=model_name,
        )

        cached = self.store.get_cache(cache_key)
        if cached:
            cards = self._load_from_cache(cached)
            if cards is not None:
                log.info(
                    "Slack normalizer cache hit: file=%s cards=%d", file_name, len(cards)
                )
                return cards
            log.warning(
                "Slack normalizer cache 메타는 있으나 JSON 파일을 찾지 못해 LLM 재호출합니다. "
                "file=%s cache_key=%s",
                file_name,
                cache_key,
            )

        truncated_text = self._truncate_input(text or "")
        clean_topic_tags = [t for t in (topic_tags or []) if isinstance(t, str) and t.strip()]

        system_instruction, user_prompt = build_slack_normalization_prompt(
            file_name=file_name,
            source_category=source_category,
            source_type=source_type,
            document_date=document_date,
            display_date=display_date,
            content=truncated_text,
            chunk_index=chunk_index,
            topic_tags=clean_topic_tags or None,
            todo_phase=todo_phase,
            parser_format=parser_format,
            extra_metadata=metadata or None,
        )

        raw_response = self._call_llm(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            model=model_name,
        )

        parsed = parse_cards_response(raw_response)
        raw_cards = parsed.get("cards") or []

        max_cards = max(int(self.settings.normalization_max_cards_per_file), 0)
        if max_cards and len(raw_cards) > max_cards:
            log.info(
                "Slack normalizer card 수 %d → %d 로 제한 (file=%s)",
                len(raw_cards),
                max_cards,
                file_name,
            )
            raw_cards = raw_cards[:max_cards]

        cards = self._build_cards(
            raw_cards=raw_cards,
            file_name=file_name,
            file_hash=file_hash,
            file_hash_short=file_hash_short,
            source_category=source_category,
            source_type=source_type,
            document_date=document_date,
            display_date=display_date,
            metadata=metadata,
            model_name=model_name,
            input_topic_tags=clean_topic_tags,
            todo_phase=todo_phase,
            parser_format=parser_format,
        )

        json_path: Optional[Path] = None
        markdown_path: Optional[Path] = None
        if self.settings.normalization_save_json:
            json_path = self.store.save_cards_json(file_hash_short, cards)
        if self.settings.normalization_save_markdown:
            markdown_path = self.store.save_cards_markdown(file_hash_short, cards)

        self.store.set_cache(
            cache_key,
            {
                "json_path": str(json_path) if json_path else None,
                "markdown_path": str(markdown_path) if markdown_path else None,
                "card_count": len(cards),
                "prompt_version": self.prompt_version,
                "model_name": model_name,
                "file_name": file_name,
                "file_hash_short": file_hash_short,
                "source_kind": "slack_thread",
            },
        )

        log.info(
            "Slack normalizer cache miss → LLM 호출 완료: file=%s cards=%d",
            file_name,
            len(cards),
        )
        return cards

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _truncate_input(self, text: str) -> str:
        cap = max(int(self.settings.normalization_max_chars_per_call), 0)
        if cap <= 0 or len(text) <= cap:
            return text
        return text[: cap - 3] + "..."

    def _call_llm(
        self,
        *,
        user_prompt: str,
        system_instruction: str,
        model: str,
    ) -> str:
        try:
            return self.gemini_client.generate_text(
                user_prompt,
                model=model,
                temperature=float(self.settings.normalization_temperature),
                system_instruction=system_instruction,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Slack normalizer LLM 호출 실패: {e}") from e

    def _load_from_cache(
        self, cached: Dict[str, Any]
    ) -> Optional[List[KnowledgeCard]]:
        json_path_str = cached.get("json_path")
        if not json_path_str:
            return None
        path = Path(json_path_str)
        if not path.exists():
            return None
        try:
            return self.store.load_cards_json(path)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.warning("Slack cache JSON 로드 실패 (재호출 fallback): %s", e)
            return None

    def _build_cards(
        self,
        *,
        raw_cards: List[Dict[str, Any]],
        file_name: str,
        file_hash: str,
        file_hash_short: str,
        source_category: str,
        source_type: str,
        document_date: Optional[str],
        display_date: Optional[str],
        metadata: Optional[Dict[str, Any]],
        model_name: str,
        input_topic_tags: List[str],
        todo_phase: Optional[str],
        parser_format: Optional[str],
    ) -> List[KnowledgeCard]:
        base_metadata: Dict[str, Any] = dict(metadata or {})
        base_metadata.setdefault("prompt_version", self.prompt_version)
        base_metadata.setdefault("model_name", model_name)
        if input_topic_tags:
            base_metadata.setdefault("input_topic_tags", list(input_topic_tags))
        if todo_phase:
            base_metadata.setdefault("todo_phase", todo_phase)
        if parser_format:
            base_metadata.setdefault("parser_format", parser_format)
        if document_date:
            base_metadata.setdefault("document_date", document_date)
        if display_date:
            base_metadata.setdefault("display_date", display_date)

        cards: List[KnowledgeCard] = []
        for idx, raw in enumerate(raw_cards):
            if not isinstance(raw, dict):
                continue
            card_type = _coerce_card_type(raw.get("card_type"), default="issue")

            llm_topic_tags = _as_str_list(raw.get("topic_tags"))
            merged_topic_tags = _merge_unique_strings(llm_topic_tags, input_topic_tags)

            card = KnowledgeCard(
                card_id=_make_card_id(file_hash_short, idx, card_type),
                card_type=card_type,
                title=_as_str(raw.get("title")).strip() or "(제목 없음)",
                summary=_as_str(raw.get("summary")).strip(),
                source_file_name=file_name,
                source_file_hash=file_hash,
                source_category=source_category,
                source_type=source_type,
                document_date=document_date,
                display_date=display_date,
                primary_topic=_as_str(raw.get("primary_topic")).strip() or None,
                topic_tags=merged_topic_tags,
                task_type=_as_str(raw.get("task_type")).strip() or None,
                when_to_use=_as_str(raw.get("when_to_use")).strip(),
                prerequisites=_as_str_list(raw.get("prerequisites")),
                steps=_as_str_list(raw.get("steps")),
                checkpoints=_as_str_list(raw.get("checkpoints")),
                cautions=_as_str_list(raw.get("cautions")),
                examples=_as_str_list(raw.get("examples")),
                related_terms=_as_str_list(raw.get("related_terms")),
                open_questions=_as_str_list(raw.get("open_questions")),
                evidence_spans=_normalize_evidence_spans(raw.get("evidence_spans")),
                parent_raw_chunk_ids=[],
                sanitized_markdown="",
                metadata=dict(base_metadata),
            )
            card.sanitized_markdown = card.to_markdown()
            cards.append(card)
        return cards


def _merge_unique_strings(primary: List[str], extra: List[str]) -> List[str]:
    """primary 우선 순서를 보존하면서 extra 의 신규 항목만 뒤에 덧붙인다."""
    seen: set = set()
    out: List[str] = []
    for src in (primary or []), (extra or []):
        for item in src:
            if not isinstance(item, str):
                continue
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out
