"""
gemini_client 동작 테스트.

API Key 가 없는 환경에서도 통과하도록, 실제 외부 호출을 하지 않고
GeminiError 가 친절한 한국어 메시지로 발생하는지만 검증한다.
"""
from __future__ import annotations

import pytest

from src.rag.gemini_client import GeminiClient, GeminiConfig, GeminiError


def test_missing_api_key_raises_friendly_error():
    cfg = GeminiConfig(api_key=None, timeout_seconds=10, max_retries=0)
    client = GeminiClient(cfg)
    with pytest.raises(GeminiError) as exc:
        client.generate_text("hello")
    assert "GOOGLE_API_KEY" in str(exc.value) or "키" in str(exc.value)


def test_check_connection_without_key():
    cfg = GeminiConfig(api_key=None, timeout_seconds=10, max_retries=0)
    client = GeminiClient(cfg)
    with pytest.raises(GeminiError):
        client.check_connection()
