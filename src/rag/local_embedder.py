"""
local_embedder.py
=================
sentence-transformers 기반 로컬 임베딩.

- 한국어/다국어 모델을 .env LOCAL_EMBEDDING_MODEL 으로 자유롭게 교체 가능.
- 기본 후보: BAAI/bge-m3 (다국어 retrieval 강세)
- 모델 로딩은 한 번만 (싱글턴 캐시).
- 회사 노트북 사양에서는 첫 모델 다운로드/로드가 길 수 있다.

NOTE: GPU 가 없는 환경에서는 cpu 로 자동 동작한다.
"""
from __future__ import annotations

from threading import Lock
from typing import List, Optional

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)


_model_cache: dict = {}
_lock = Lock()


def _load_model(model_name: str):
    with _lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers 가 설치되어 있지 않습니다. "
                "`pip install -r requirements.txt` 를 다시 실행하세요."
            ) from e
        log.info("[local_embedder] 모델 로드 시작: %s", model_name)
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        log.info("[local_embedder] 모델 로드 완료: %s", model_name)
        return model


class LocalEmbedder:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = model_name or settings.local_embedding_model
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        m = self._ensure()
        vec = m.encode([text], normalize_embeddings=True)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        m = self._ensure()
        arr = m.encode(
            list(texts),
            batch_size=int(batch_size),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in arr]
