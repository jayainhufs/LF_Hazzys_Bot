"""
query_rewriter.py
=================
사용자 질문을 검색 친화적인 형태로 재작성한다.

- 기본은 비활성화 (settings.enable_query_rewrite == False)
- True 일 때만 Gemini 호출 (비용 발생)
"""
from __future__ import annotations

from typing import Optional

from src.config import settings
from src.logger import get_logger
from src.rag.gemini_client import GeminiClient, GeminiError, get_default_client

log = get_logger(__name__)

_REWRITE_PROMPT = """너는 한국어 RAG 검색을 위한 질문 재작성 전문가다.
아래 사용자의 질문을 검색에 잘 걸리도록 핵심 키워드 중심으로 다시 써라.
- 한국어 1~2문장
- 중요한 업무 용어, 도메인 키워드 (예: 캠페인, ROAS, 소재, KPI 등) 를 보존
- 형용사/감탄사 등은 줄여라
- 답변 형태가 아니라 '검색용 질의' 형태로

원래 질문:
{question}

재작성:"""


def rewrite_query_if_enabled(
    question: str,
    enable: Optional[bool] = None,
    client: Optional[GeminiClient] = None,
) -> Optional[str]:
    """
    Returns
    -------
    재작성된 질의 문자열, 또는 비활성화/실패 시 None.
    """
    use = settings.enable_query_rewrite if enable is None else bool(enable)
    if not use:
        return None
    if not question or not question.strip():
        return None

    try:
        c = client or get_default_client()
        text = c.generate_text(
            _REWRITE_PROMPT.format(question=question),
            temperature=0.0,
            max_output_tokens=120,
        )
        text = (text or "").strip().splitlines()[0].strip()
        if text and text != question.strip():
            return text
        return None
    except GeminiError as e:
        log.warning("query rewrite 실패 (그대로 사용): %s", e)
        return None
