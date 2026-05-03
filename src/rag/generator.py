"""
generator.py
============
Gemini Generation API 래퍼.

- 1차 시도: settings.generation_model
- 실패 시: settings.fallback_generation_model 로 한 번만 재시도
- 그래도 실패하면 친절한 한국어 에러 메시지를 사용자 답변 위치에 반환
"""
from __future__ import annotations

from typing import Optional, Tuple

from src.config import settings
from src.logger import get_logger
from src.rag.gemini_client import GeminiClient, GeminiError, get_default_client

log = get_logger(__name__)


class Generator:
    def __init__(self, client: Optional[GeminiClient] = None) -> None:
        self.client = client or get_default_client()

    def generate(
        self,
        prompt: str,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Returns
        -------
        (answer_text, used_model_name)

        실패 시:
            answer_text 는 사용자에게 보일 한국어 에러 안내 메시지.
            used_model_name 은 "" 또는 시도한 모델명.
        """
        primary = primary_model or settings.generation_model
        fallback = fallback_model or settings.fallback_generation_model

        try:
            text = self.client.generate_text(
                prompt,
                model=primary,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if text:
                return text, primary
            log.warning("primary 모델 응답이 비었습니다 → fallback 시도")
        except GeminiError as e:
            log.warning("primary 모델 호출 실패: %s → fallback 시도", e)

        if fallback and fallback != primary:
            try:
                text = self.client.generate_text(
                    prompt,
                    model=fallback,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if text:
                    return text, fallback
            except GeminiError as e:
                log.error("fallback 모델 호출도 실패: %s", e)

        msg = (
            "## 답변 생성 실패\n\n"
            "Gemini API 호출에 실패했습니다. 다음을 확인해 주세요.\n"
            "- `.env` 의 GOOGLE_API_KEY 가 올바른지\n"
            "- 모델명(GENERATION_MODEL / FALLBACK_GENERATION_MODEL) 이 사용 가능한지\n"
            "- Gemini quota 한도가 남아있는지\n"
            "- 네트워크 연결 상태\n\n"
            "(검색된 근거는 아래 'Retrieved Chunks' 영역에서 그대로 확인하실 수 있습니다.)"
        )
        return msg, ""
