"""
guide_normalizer.py
===================
Guide txt 문서를 LLM-based Document Normalization 으로 Normalized Document
리스트로 변환하는 normalizer.

Task 2 범위
-----------
- Guide 본문(또는 guide chunk text) 를 받아 Normalized Document 리스트로 변환한다.
- file_hash + prompt_version + model_name 으로 cache 를 사용한다.
- LLM 호출은 ``GeminiClient.generate_text(system_instruction=..., temperature=...)``
  에 의존한다. 테스트에서는 동일 시그니처의 fake client 를 주입한다.
- pipeline 연결, Streamlit UI, retrieval / QA 변경은 다른 Task 에서 처리한다.

설계 의도
---------
- Slack / 다른 normalizer 가 추가될 때 동일한 구조 (cache → prompt → LLM →
  parse → Normalized Document 변환 → 저장 + cache set) 로 확장 가능하도록
  helper 들을 모듈 함수로 분리해 두었다.
- LLM 응답이 ```json 코드블록으로 감싸지는 모델 / 케이스가 흔하기 때문에
  파싱 단계에서 fence 를 robust 하게 제거한다.
- cache hit 이어도 저장 파일이 사라졌다면 graceful 하게 LLM 재호출로 fallback 한다.

명칭 변경 노트:
- 클래스 ``GuideKnowledgeNormalizer`` 는 ``GuideDocumentNormalizer`` 로 명칭이
  바뀌었다. 기존 import 호환을 위해 ``GuideKnowledgeNormalizer`` alias 가
  유지된다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings as default_settings
from src.logger import get_logger
from src.normalization.normalization_prompt import (
    GUIDE_NORMALIZER_PROMPT_VERSION,
    build_guide_normalization_prompt,
)
from src.normalization.normalization_store import NormalizationStore
from src.schemas import KnowledgeCard, NormalizedDocument
from src.schemas.normalized_document import VALID_NORMALIZED_DOCUMENT_TYPES

# legacy alias — 기존 코드와의 호환을 위해 노출
VALID_CARD_TYPES = VALID_NORMALIZED_DOCUMENT_TYPES

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """LLM 이 ```json ... ``` 로 응답을 감쌌을 경우 fence 만 제거한다.

    fence 가 없으면 strip 만 반환.
    """
    if not text:
        return ""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_cards_response(raw_text: str) -> Dict[str, Any]:
    """LLM 응답 문자열에서 cards JSON object 를 파싱한다.

    Raises
    ------
    ValueError
        - 응답이 비어 있을 때
        - JSON 파싱 자체가 실패했을 때
        - root 가 object 가 아닐 때
        - "cards" 키가 없거나 list 가 아닐 때
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM 응답이 비어 있습니다.")

    cleaned = _strip_code_fence(raw_text)
    if not cleaned:
        raise ValueError("LLM 응답이 코드블록만 포함되어 있고 본문이 없습니다.")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 응답을 JSON 으로 파싱할 수 없습니다: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("LLM 응답이 JSON object 가 아닙니다.")
    if "cards" not in parsed:
        raise ValueError("LLM 응답에 'cards' 키가 없습니다.")
    if not isinstance(parsed["cards"], list):
        raise ValueError("'cards' 값이 list 가 아닙니다.")

    return parsed


# ---------------------------------------------------------------------------
# Field normalization helpers
# ---------------------------------------------------------------------------
def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = item if isinstance(item, str) else str(item)
            if text.strip():
                out.append(text)
        return out
    return []


def _normalize_evidence_spans(spans: Any) -> List[Dict[str, Any]]:
    """LLM 이 반환한 evidence_spans 를 NormalizedDocument.to_markdown 이 인식 가능한 형태로 정규화."""
    if not isinstance(spans, list):
        return []
    out: List[Dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        section_title = span.get("section_title") or span.get("section") or ""
        chunk_index = span.get("chunk_index", -1)
        try:
            chunk_index_int = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index_int = -1
        quote_or_summary = (
            span.get("quote_or_summary")
            or span.get("summary")
            or span.get("text")
            or ""
        )
        out.append({
            "section_title": _as_str(section_title),
            "section": _as_str(section_title),
            "chunk_index": chunk_index_int,
            "quote_or_summary": _as_str(quote_or_summary),
            "summary": _as_str(quote_or_summary),
        })
    return out


def _coerce_card_type(value: Any, default: str = "workflow") -> str:
    candidate = _as_str(value, default).strip().lower()
    if candidate in VALID_NORMALIZED_DOCUMENT_TYPES:
        return candidate
    return default


def _make_card_id(file_hash_short: str, idx: int, card_type: str) -> str:
    """Normalized Document 의 식별자 (legacy 명: card_id) 를 만든다.

    기존 저장 JSON / metadata 호환을 위해 prefix 는 ``kc_`` 를 그대로 둔다.
    """
    short = (file_hash_short or "unknown").strip() or "unknown"
    safe_type = card_type or "card"
    return f"kc_{short}_{idx:03d}_{safe_type}"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class GuideDocumentNormalizer:
    """Guide 문서를 Normalized Document 리스트로 변환하는 LLM normalizer.

    legacy alias: ``GuideKnowledgeNormalizer`` (모듈 하단에서 노출).
    """

    def __init__(
        self,
        gemini_client: Any,
        store: NormalizationStore,
        settings: Any = None,
        prompt_version: str = GUIDE_NORMALIZER_PROMPT_VERSION,
    ) -> None:
        if gemini_client is None:
            raise ValueError("gemini_client 가 필요합니다.")
        if store is None:
            raise ValueError("NormalizationStore 가 필요합니다.")
        self.gemini_client = gemini_client
        self.store = store
        self.settings = settings or default_settings
        self.prompt_version = prompt_version or GUIDE_NORMALIZER_PROMPT_VERSION

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def normalize_guide_text(
        self,
        text: str,
        *,
        file_name: str,
        file_hash: str,
        file_hash_short: str,
        source_category: str = "guide",
        source_type: str = "guide",
        document_date: Optional[str] = None,
        display_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
    ) -> List[NormalizedDocument]:
        """Guide 본문을 LLM 으로 Normalized Document 리스트로 변환한다.

        cache hit 이면 LLM 호출 없이 저장된 JSON 으로부터 Normalized Document 들을 복원한다.
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
                    "Guide normalizer cache hit: file=%s cards=%d", file_name, len(cards)
                )
                return cards
            log.warning(
                "Guide normalizer cache 메타는 있으나 JSON 파일을 찾지 못해 LLM 재호출합니다. "
                "file=%s cache_key=%s",
                file_name,
                cache_key,
            )

        truncated_text = self._truncate_input(text or "")
        system_instruction, user_prompt = build_guide_normalization_prompt(
            file_name=file_name,
            source_category=source_category,
            source_type=source_type,
            document_date=document_date,
            display_date=display_date,
            content=truncated_text,
            chunk_index=chunk_index,
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
                "Guide normalizer card 수 %d → %d 로 제한 (file=%s)",
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
            },
        )

        log.info(
            "Guide normalizer cache miss → LLM 호출 완료: file=%s cards=%d",
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
            raise RuntimeError(f"Guide normalizer LLM 호출 실패: {e}") from e

    def _load_from_cache(
        self, cached: Dict[str, Any]
    ) -> Optional[List[NormalizedDocument]]:
        json_path_str = cached.get("json_path")
        if not json_path_str:
            return None
        path = Path(json_path_str)
        if not path.exists():
            return None
        try:
            return self.store.load_cards_json(path)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.warning("cache JSON 로드 실패 (재호출 fallback): %s", e)
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
    ) -> List[NormalizedDocument]:
        base_metadata: Dict[str, Any] = dict(metadata or {})
        base_metadata.setdefault("prompt_version", self.prompt_version)
        base_metadata.setdefault("model_name", model_name)

        cards: List[NormalizedDocument] = []
        for idx, raw in enumerate(raw_cards):
            if not isinstance(raw, dict):
                continue
            card_type = _coerce_card_type(raw.get("card_type"), default="workflow")
            card = NormalizedDocument(
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
                topic_tags=_as_str_list(raw.get("topic_tags")),
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
                answer_use_cases=_as_str_list(raw.get("answer_use_cases")),
                sanitized_markdown="",
                metadata=dict(base_metadata),
            )
            card.sanitized_markdown = card.to_markdown()
            cards.append(card)
        return cards


# ---------------------------------------------------------------------------
# legacy compatibility — 기존 import 유지
# ---------------------------------------------------------------------------
GuideKnowledgeNormalizer = GuideDocumentNormalizer
