"""
knowledge_card.py
=================
Legacy compatibility shim.

이 모듈은 기존 ``from src.schemas.knowledge_card import KnowledgeCard``
형태의 import 가 그대로 동작하도록 유지된다. 실제 구현은
``src.schemas.normalized_document`` 로 이동했다 (LLM-based Document
Normalization 체계로 명칭 통일).

새 코드는 가능하면 다음을 사용한다::

    from src.schemas.normalized_document import NormalizedDocument
"""
from __future__ import annotations

from src.schemas.normalized_document import (  # noqa: F401  (re-export)
    VALID_ANSWER_USE_CASES,
    VALID_CARD_TYPES,
    VALID_NORMALIZED_DOCUMENT_TYPES,
    KnowledgeCard,
    NormalizedDocument,
)

__all__ = [
    "KnowledgeCard",
    "NormalizedDocument",
    "VALID_CARD_TYPES",
    "VALID_NORMALIZED_DOCUMENT_TYPES",
    "VALID_ANSWER_USE_CASES",
]
