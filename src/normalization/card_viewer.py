"""
card_viewer.py
==============
Streamlit 지식카드 관리 UI 에서 사용하는 read-only helper.

Task 5 범위
-----------
- NormalizationStore 가 저장한 KnowledgeCard JSON / Markdown 을 읽어 목록화한다.
- 요약 통계, 필터링, table 표시용 dict 변환을 제공한다.
- raw 원문 파일에는 접근하지 않는다. UI 는 sanitized_markdown 중심으로만 표시한다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from src.normalization.normalization_store import NormalizationStore
from src.schemas import KnowledgeCard


StoreOrPath = Union[NormalizationStore, Path, str]


def _as_path(value: StoreOrPath, *, subdir: str) -> Path:
    if isinstance(value, NormalizationStore):
        return value.json_dir if subdir == "json" else value.markdown_dir
    path = Path(value)
    if path.name == subdir:
        return path
    candidate = path / subdir
    return candidate if candidate.exists() else path


def list_normalized_json_files(store_or_path: StoreOrPath) -> List[Path]:
    """NormalizationStore 또는 json directory 에서 KnowledgeCard JSON 파일 목록을 반환."""
    json_dir = _as_path(store_or_path, subdir="json")
    if not json_dir.exists():
        return []
    return sorted(json_dir.glob("*.json"), key=lambda p: p.name.lower())


def list_normalized_markdown_files(store_or_path: StoreOrPath) -> List[Path]:
    """NormalizationStore 또는 markdown directory 에서 KnowledgeCard Markdown 파일 목록을 반환."""
    markdown_dir = _as_path(store_or_path, subdir="markdown")
    if not markdown_dir.exists():
        return []
    return sorted(markdown_dir.glob("*.md"), key=lambda p: p.name.lower())


def load_all_cards_from_store(store: NormalizationStore) -> List[KnowledgeCard]:
    """store.json_dir 아래 모든 KnowledgeCard JSON 을 읽어 평탄화한다.

    손상된 JSON 파일은 UI 전체를 깨지 않도록 건너뛴다.
    """
    cards: List[KnowledgeCard] = []
    for path in list_normalized_json_files(store):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                cards.append(KnowledgeCard.from_dict(item))
            except TypeError:
                continue
    return cards


def summarize_cards(cards: Iterable[KnowledgeCard]) -> Dict[str, Any]:
    """카드 수와 주요 분포를 계산한다."""
    card_list = list(cards or [])
    return {
        "total_cards": len(card_list),
        "card_type_counts": dict(Counter(c.card_type or "unknown" for c in card_list)),
        "primary_topic_counts": dict(Counter(c.primary_topic or "unknown" for c in card_list)),
        "source_file_counts": dict(Counter(c.source_file_name or "unknown" for c in card_list)),
    }


def filter_cards(
    cards: Iterable[KnowledgeCard],
    *,
    card_type: Optional[str] = None,
    primary_topic: Optional[str] = None,
    source_file_name: Optional[str] = None,
    query: Optional[str] = None,
) -> List[KnowledgeCard]:
    """card_type / primary_topic / source_file_name / text query 로 필터링한다."""
    q = (query or "").strip().lower()
    out: List[KnowledgeCard] = []
    for card in cards or []:
        if card_type and card_type != "전체" and card.card_type != card_type:
            continue
        if primary_topic and primary_topic != "전체" and (card.primary_topic or "unknown") != primary_topic:
            continue
        if source_file_name and source_file_name != "전체" and card.source_file_name != source_file_name:
            continue
        if q and q not in _search_blob(card):
            continue
        out.append(card)
    return out


def card_to_display_dict(card: KnowledgeCard) -> Dict[str, Any]:
    """Streamlit dataframe/table 표시용 납작한 dict."""
    return {
        "title": card.title,
        "card_type": card.card_type,
        "primary_topic": card.primary_topic or "unknown",
        "task_type": card.task_type or "",
        "source_file_name": card.source_file_name,
        "display_date": card.display_date or "",
        "checkpoints_count": len(card.checkpoints or []),
        "steps_count": len(card.steps or []),
        "open_questions_count": len(card.open_questions or []),
    }


def markdown_for_card(card: KnowledgeCard) -> str:
    """UI preview 에 사용할 sanitized markdown 을 반환한다."""
    return (card.sanitized_markdown or card.to_markdown() or "").strip()


def _search_blob(card: KnowledgeCard) -> str:
    parts: List[str] = [
        card.title or "",
        card.summary or "",
        card.when_to_use or "",
    ]
    for values in (
        card.steps,
        card.checkpoints,
        card.cautions,
        card.related_terms,
    ):
        parts.extend([v for v in (values or []) if isinstance(v, str)])
    return "\n".join(parts).lower()
