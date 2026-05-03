"""
check_gemini_api.py
===================
Google Gemini API Key / 모델 / 임베딩 / 생성 동작을 콘솔에서 점검.

사용법:
    python scripts/check_gemini_api.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder
from src.rag.gemini_client import GeminiClient, GeminiError

log = get_logger(__name__)


def main() -> int:
    print(f"APP        : {settings.app_name} ({settings.env})")
    print(f"API Key    : {'set' if settings.has_api_key() else 'MISSING'}")
    print(f"Generation : {settings.generation_model}")
    print(f"Fallback   : {settings.fallback_generation_model}")
    print(f"Embedding  : {settings.embedding_provider} / "
          f"{settings.gemini_embedding_model if settings.embedding_provider == 'gemini' else settings.local_embedding_model}")
    print()

    if not settings.has_api_key() and settings.embedding_provider == "gemini":
        print("WARN: GOOGLE_API_KEY 가 비어 있어 Gemini 호출은 모두 실패합니다.")

    client = GeminiClient()

    # 1) Generation ping
    print("[1/3] Generation API ping...")
    try:
        res = client.check_connection()
        print(f"     -> ok={res.get('ok')}  preview={res.get('preview')}")
        if not res.get("ok"):
            print(f"     error: {res.get('error')}")
    except GeminiError as e:
        print(f"     GeminiError: {e}")

    # 2) Embedding (provider 에 따라)
    print("[2/3] Embedding 호출...")
    try:
        emb = Embedder()
        v = emb.embed_query("Meta 캠페인 세팅 가이드")
        print(f"     -> dim={len(v)}, first={v[:4] if v else []}")
    except Exception as e:  # noqa: BLE001
        print(f"     실패: {e}")

    # 3) models.list (가능한 경우)
    print("[3/3] models.list...")
    try:
        names = client.list_models()
        if names:
            print(f"     -> {len(names)} models")
            for n in names[:20]:
                print(f"       - {n}")
        else:
            print("     -> (목록을 가져오지 못했습니다)")
    except Exception as e:  # noqa: BLE001
        print(f"     실패: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
