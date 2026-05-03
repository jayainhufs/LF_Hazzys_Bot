"""
embedder.py
===========
EMBEDDING_PROVIDER 에 따라 Gemini / Local 임베딩을 동일 인터페이스로 제공.

- 기본 provider: gemini
- query / document 용도가 다르면 task_type 분리 (Gemini 만 해당)
- 같은 chunk 에 대한 중복 호출 방지를 위한 in-memory cache 제공
"""
from __future__ import annotations

import hashlib
from threading import Lock
from typing import Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.gemini_client import (
    EMBED_TASK_DOCUMENT,
    EMBED_TASK_QUERY,
    GeminiClient,
    GeminiError,
    get_default_client,
)
from src.rag.local_embedder import LocalEmbedder

log = get_logger(__name__)


def _key(provider: str, model: str, text: str, task: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(b"|")
    h.update(model.encode())
    h.update(b"|")
    h.update((task or "").encode())
    h.update(b"|")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


class Embedder:
    """provider 추상화."""

    def __init__(
        self,
        provider: Optional[str] = None,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.provider: str = (provider or settings.embedding_provider).lower()
        if self.provider not in {"gemini", "local"}:
            log.warning("알 수 없는 EMBEDDING_PROVIDER=%s -> gemini 로 fallback", self.provider)
            self.provider = "gemini"
        self._gemini = gemini_client
        self._local: Optional[LocalEmbedder] = None
        self._cache: Dict[str, List[float]] = {}
        self._lock = Lock()

    # --------------------------- 내부 client 핸들러 -----------------------
    def _get_gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = get_default_client()
        return self._gemini

    def _get_local(self) -> LocalEmbedder:
        if self._local is None:
            self._local = LocalEmbedder(settings.local_embedding_model)
        return self._local

    # --------------------------- model_name -------------------------------
    @property
    def model_name(self) -> str:
        if self.provider == "gemini":
            return settings.gemini_embedding_model
        return settings.local_embedding_model

    # --------------------------- query / document -------------------------
    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, task=EMBED_TASK_QUERY)

    def embed_document(self, text: str) -> List[float]:
        return self._embed_one(text, task=EMBED_TASK_DOCUMENT)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_many(texts, task=EMBED_TASK_DOCUMENT)

    # 호환용
    def embed_text(self, text: str) -> List[float]:
        return self.embed_document(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    # --------------------------- 내부 구현 --------------------------------
    def _embed_one(self, text: str, task: Optional[str]) -> List[float]:
        if not text:
            return []
        ck = _key(self.provider, self.model_name, text, task)
        with self._lock:
            if ck in self._cache:
                return self._cache[ck]

        if self.provider == "gemini":
            try:
                vec = self._get_gemini().embed_text(text, task_type=task)
            except GeminiError as e:
                log.error("Gemini embedding 실패: %s", e)
                raise
        else:
            vec = self._get_local().embed_text(text)

        with self._lock:
            self._cache[ck] = vec
        return vec

    def _embed_many(self, texts: List[str], task: Optional[str]) -> List[List[float]]:
        if not texts:
            return []
        # cache 적용
        cached: Dict[int, List[float]] = {}
        missing_idx: List[int] = []
        missing_texts: List[str] = []
        for i, t in enumerate(texts):
            if not t:
                cached[i] = []
                continue
            ck = _key(self.provider, self.model_name, t, task)
            with self._lock:
                if ck in self._cache:
                    cached[i] = self._cache[ck]
                else:
                    missing_idx.append(i)
                    missing_texts.append(t)

        new_vectors: List[List[float]] = []
        if missing_texts:
            if self.provider == "gemini":
                new_vectors = self._get_gemini().embed_texts(
                    missing_texts, task_type=task
                )
            else:
                new_vectors = self._get_local().embed_texts(missing_texts)

            with self._lock:
                for t, v in zip(missing_texts, new_vectors):
                    self._cache[_key(self.provider, self.model_name, t, task)] = v

        out: List[List[float]] = []
        new_iter = iter(new_vectors)
        for i in range(len(texts)):
            if i in cached:
                out.append(cached[i])
            else:
                out.append(next(new_iter, []))
        return out


# 글로벌 헬퍼
_default: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = Embedder()
    return _default


def reset_default_embedder() -> None:
    global _default
    _default = None
