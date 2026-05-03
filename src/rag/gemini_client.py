"""
gemini_client.py
================
Google Gemini (`google-genai`) SDK 의 얇은 wrapper.

- `client = genai.Client(api_key=...)` 한 번만 생성
- generate_text / embed_text / embed_texts / check_connection / list_models 제공
- task_type 분리 (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY) 지원 시 사용
- API Key 미설정 / quota 초과 / invalid model / network / timeout 친절한 에러 메시지
- retry 는 최대 2회 (지수 backoff)

`google-genai` 의 SDK 인터페이스가 시점에 따라 약간 다를 수 있으므로,
embedding 결과는 `embedding.values` 와 `embeddings[0].values` 둘 다 수용한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from src.config import settings
from src.logger import get_logger
from src.utils.cost_utils import tracker

log = get_logger(__name__)


class GeminiError(RuntimeError):
    """Gemini 호출 관련 모든 예외를 감싼다."""


@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    timeout_seconds: int = 60
    max_retries: int = 2

    @classmethod
    def from_settings(cls) -> "GeminiConfig":
        return cls(
            api_key=settings.google_api_key,
            timeout_seconds=int(settings.request_timeout_seconds),
            max_retries=2,
        )


# embedding 입력이 너무 길면 API 호출 자체가 실패할 수 있다. 보수적인 cap.
_EMBED_INPUT_CHAR_CAP = 8000

# task type
EMBED_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"


def _truncate_for_embedding(text: str, cap: int = _EMBED_INPUT_CHAR_CAP) -> str:
    if not text:
        return ""
    if len(text) <= cap:
        return text
    return text[: cap - 3] + "..."


def _is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("quota", "rate limit", "429", "exceeded"))


def _is_invalid_model(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("not found", "invalid model", "404", "permission"))


def _is_timeout_or_network(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("timeout", "timed out", "connection", "unavailable"))


class GeminiClient:
    def __init__(self, config: Optional[GeminiConfig] = None) -> None:
        self.config = config or GeminiConfig.from_settings()
        self._client = None
        self._types = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.config.api_key:
            log.warning(
                "GOOGLE_API_KEY 가 설정되어 있지 않습니다. "
                "Gemini 호출 시 친절한 에러 메시지를 반환합니다."
            )
            return
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise GeminiError(
                "google-genai SDK 가 설치되지 않았습니다. `pip install google-genai` 또는 "
                "`pip install -r requirements.txt` 를 실행하세요."
            ) from e

        try:
            self._client = genai.Client(api_key=self.config.api_key)
            self._types = genai_types
        except Exception as e:
            raise GeminiError(f"Gemini Client 초기화 실패: {e}") from e

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
    def _ensure_client(self) -> None:
        if not self.config.api_key:
            raise GeminiError(
                "GOOGLE_API_KEY 가 비어 있습니다. .env 파일에 키를 입력해 주세요."
            )
        if self._client is None:
            self._init_client()
        if self._client is None:
            raise GeminiError("Gemini Client 가 정상적으로 초기화되지 않았습니다.")

    def _retry(self, fn, *args, **kwargs):
        """지수 backoff 로 fn 을 재시도. quota / invalid model 은 즉시 실패."""
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_invalid_model(e):
                    raise GeminiError(
                        f"모델을 찾을 수 없습니다. .env 의 모델명을 확인하세요. ({e})"
                    ) from e
                if _is_quota_error(e):
                    raise GeminiError(
                        f"API quota 가 초과되었거나 rate limit 에 걸렸습니다. 잠시 후 다시 시도하세요. ({e})"
                    ) from e
                if attempt >= self.config.max_retries:
                    break
                wait = 1.5 * (2 ** attempt)
                log.warning(
                    "Gemini 호출 실패 (재시도 %d/%d, %.1fs 대기): %s",
                    attempt + 1,
                    self.config.max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
        raise GeminiError(f"Gemini 호출 실패: {last_err}") from last_err

    # ------------------------------------------------------------------
    # 연결 테스트 / 모델 목록
    # ------------------------------------------------------------------
    def check_connection(self) -> dict:
        """간단한 연결 확인. 실패 시 GeminiError 발생."""
        self._ensure_client()
        try:
            model = settings.generation_model
            resp = self._client.models.generate_content(
                model=model,
                contents="ping",
            )
            text = getattr(resp, "text", "") or ""
            return {"ok": True, "model": model, "preview": text[:50]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_models(self) -> List[str]:
        """현재 API Key 로 호출 가능한 모델 목록. 실패 시 빈 리스트."""
        self._ensure_client()
        try:
            it = self._client.models.list()
            names: List[str] = []
            for m in it:
                name = getattr(m, "name", None) or getattr(m, "model", None) or str(m)
                names.append(str(name))
            return names
        except Exception as e:  # noqa: BLE001
            log.warning("models.list 실패: %s", e)
            return []

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        self._ensure_client()
        model = model or settings.generation_model
        types = self._types

        config_kwargs: dict = {"temperature": float(temperature)}
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = int(max_output_tokens)
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        def _call():
            cfg = types.GenerateContentConfig(**config_kwargs) if types else None
            resp = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
            text = getattr(resp, "text", None)
            if not text:
                # candidates fallback
                try:
                    parts = []
                    for cand in getattr(resp, "candidates", []) or []:
                        for p in getattr(cand.content, "parts", []) or []:
                            if getattr(p, "text", None):
                                parts.append(p.text)
                    text = "".join(parts)
                except Exception:  # noqa: BLE001
                    text = ""
            return text or ""

        out = self._retry(_call)
        tracker.add_generation(prompt_chars=len(prompt or ""))
        return out

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------
    def _embed_call(
        self, contents: Any, model: str, task_type: Optional[str]
    ) -> List[List[float]]:
        types = self._types
        cfg = None
        if types and task_type:
            try:
                cfg = types.EmbedContentConfig(task_type=task_type)
            except Exception:  # noqa: BLE001
                # 일부 SDK 버전에서 task_type 미지원이면 무시
                cfg = None

        kwargs = {"model": model, "contents": contents}
        if cfg is not None:
            kwargs["config"] = cfg

        resp = self._client.models.embed_content(**kwargs)

        # 새 SDK: resp.embeddings (list) / 일부 버전: resp.embedding (single)
        if getattr(resp, "embeddings", None):
            vectors: List[List[float]] = []
            for e in resp.embeddings:
                v = getattr(e, "values", None) or getattr(e, "value", None) or []
                vectors.append(list(v))
            return vectors
        if getattr(resp, "embedding", None):
            v = getattr(resp.embedding, "values", None) or getattr(resp.embedding, "value", None)
            return [list(v or [])]
        raise GeminiError("Embedding 응답 형식을 인식할 수 없습니다.")

    def embed_text(
        self,
        text: str,
        model: Optional[str] = None,
        task_type: Optional[str] = EMBED_TASK_DOCUMENT,
    ) -> List[float]:
        self._ensure_client()
        model = model or settings.gemini_embedding_model
        text = _truncate_for_embedding(text or "")
        if not text:
            return []

        def _call():
            return self._embed_call(text, model, task_type)

        vectors = self._retry(_call)
        tracker.add_embedding(n_chunks=1)
        return vectors[0] if vectors else []

    def embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
        task_type: Optional[str] = EMBED_TASK_DOCUMENT,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """배치 임베딩. SDK 가 list 입력을 지원하지 않으면 단건씩 호출한다."""
        self._ensure_client()
        model = model or settings.gemini_embedding_model
        if not texts:
            return []
        clean = [_truncate_for_embedding(t or "") for t in texts]

        out: List[List[float]] = []
        for i in range(0, len(clean), batch_size):
            batch = clean[i : i + batch_size]
            try:
                def _call():
                    return self._embed_call(batch, model, task_type)
                vectors = self._retry(_call)
            except GeminiError:
                # 일부 환경에서 list 입력이 거부되면 단건 호출 fallback
                log.info("배치 embedding 실패 → 단건 호출 fallback")
                vectors = []
                for t in batch:
                    def _call_one(_t=t):
                        return self._embed_call(_t, model, task_type)
                    vs = self._retry(_call_one)
                    vectors.append(vs[0] if vs else [])
            tracker.add_embedding(n_chunks=len(batch))
            out.extend(vectors)
        return out


# ------------------------ 모듈 단위 싱글턴 ----------------------------------
_default_client: Optional[GeminiClient] = None


def get_default_client() -> GeminiClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client


def reset_default_client() -> None:
    """API key 변경 시 다시 만들기 위한 헬퍼."""
    global _default_client
    _default_client = None
