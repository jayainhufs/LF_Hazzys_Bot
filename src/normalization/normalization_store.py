"""
normalization_store.py
======================
LLM-based Document Normalization 결과 (Normalized Document) 를 저장/캐시하는
파일 저장소.

Task 1 범위
-----------
- Normalized Document JSON / Markdown 저장
- raw_file_hash + prompt_version + model_name 기반 cache key 생성
- cache index.json 읽기/쓰기
- 외부 Gemini API 호출 없음
- 원본 raw 파일 수정 없음

명칭 변경 노트:
- 메서드 ``save_cards_json`` / ``save_cards_markdown`` / ``load_cards_json`` 은
  legacy 명을 그대로 유지한다. 새 코드도 동일 메서드를 사용한다 (저장 파일
  포맷이 같음).
- 새 명칭 alias ``save_normalized_documents_json`` / ``save_normalized_documents_markdown``
  / ``load_normalized_documents_json`` 을 함께 노출한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.schemas import KnowledgeCard, NormalizedDocument


class NormalizationStore:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir or settings.normalization_output_dir)
        self.json_dir = self.output_dir / "json"
        self.markdown_dir = self.output_dir / "markdown"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.output_dir / "cache"
        self.cache_index_path = self.cache_dir / "index.json"

        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_cache_key(self, file_hash: str, prompt_version: str, model_name: str) -> str:
        """동일 raw 파일/프롬프트/모델 조합을 식별하는 안정적인 cache key."""
        raw = "|".join([
            (file_hash or "").strip(),
            (prompt_version or "").strip(),
            (model_name or "").strip(),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def has_cache(self, cache_key: str) -> bool:
        return cache_key in self._load_cache_index()

    def get_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        value = self._load_cache_index().get(cache_key)
        if isinstance(value, dict):
            return value
        return None

    def set_cache(self, cache_key: str, value: Dict[str, Any]) -> None:
        index = self._load_cache_index()
        index[cache_key] = dict(value or {})
        self._write_cache_index(index)

    def save_cards_json(
        self, file_hash_short: str, cards: List[NormalizedDocument]
    ) -> Path:
        path = self.json_dir / f"{file_hash_short}.json"
        payload = [card.to_dict() for card in cards]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save_cards_markdown(
        self, file_hash_short: str, cards: List[NormalizedDocument]
    ) -> Path:
        path = self.markdown_dir / f"{file_hash_short}.md"
        blocks = [
            (card.sanitized_markdown.strip() if card.sanitized_markdown else card.to_markdown())
            for card in cards
        ]
        path.write_text("\n\n---\n\n".join(blocks).strip() + "\n", encoding="utf-8")
        return path

    def load_cards_json(self, path: Path) -> List[NormalizedDocument]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Normalized Document JSON must be a list.")
        return [NormalizedDocument.from_dict(item) for item in raw]

    # ------------------------------------------------------------------
    # 신규 명칭 alias — 새 코드는 가능한 한 이 이름을 사용한다.
    # ------------------------------------------------------------------
    def save_normalized_documents_json(
        self, file_hash_short: str, documents: List[NormalizedDocument]
    ) -> Path:
        return self.save_cards_json(file_hash_short, documents)

    def save_normalized_documents_markdown(
        self, file_hash_short: str, documents: List[NormalizedDocument]
    ) -> Path:
        return self.save_cards_markdown(file_hash_short, documents)

    def load_normalized_documents_json(self, path: Path) -> List[NormalizedDocument]:
        return self.load_cards_json(path)

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_index_path.exists():
            return {}
        try:
            raw = json.loads(self.cache_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(raw, dict):
            return raw
        return {}

    def _write_cache_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        self.cache_index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
