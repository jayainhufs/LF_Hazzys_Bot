"""
NormalizedDocument
==================
LLM-based Document Normalization 결과를 담는 dataclass schema.

개념:
- LLM-based Document Normalization
    raw 업무 문서를 LLM 으로 절차 / 체크리스트 / 이슈 / FAQ / 공유 문안 등
    검색·QA 친화적 구조로 정규화하는 ingestion-time preprocessing 단계.
- Normalized Document
    위 정규화의 결과물이며, 검색과 답변의 1차 근거로 사용되는 구조화된 문서 단위.

호환성:
- 기존 코드가 ``KnowledgeCard`` 라는 이름으로 동일한 dataclass 를 사용해 왔다.
  ``KnowledgeCard`` 는 ``NormalizedDocument`` 의 backward-compat alias 로 유지된다.
- dataclass 필드명 자체 (``card_id`` / ``card_type``) 는 기존 저장 JSON / metadata
  호환을 위해 그대로 둔다. 신규 코드에서는
  ``normalized_document_id`` / ``normalized_document_type`` property 를 통해
  새 이름으로도 접근할 수 있다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VALID_NORMALIZED_DOCUMENT_TYPES = {
    "workflow",
    "issue",
    "checklist",
    "faq",
    "decision",
    "glossary",
    "communication_template",
    "context_note",
    "status_update",
    "action_item",
    "issue_log",
    "decision_log",
    "campaign_summary",
    "communication_history",
    "reference_note",
    "report_insight",
}

VALID_ANSWER_USE_CASES = {
    "procedure",
    "summary",
    "troubleshooting",
    "draft_message",
    "compare",
    "history_lookup",
    "checklist",
    "freeform_grounded",
}

# legacy compatibility — 기존 코드 / 테스트에서 import 가능
VALID_CARD_TYPES = VALID_NORMALIZED_DOCUMENT_TYPES


@dataclass
class NormalizedDocument:
    """LLM-based Document Normalization 결과 1건."""

    # --- 핵심 식별자 (legacy field 명 유지: card_id / card_type) ---
    card_id: str
    card_type: str

    # --- 본문 ---
    title: str
    summary: str

    # --- 출처 메타 ---
    source_file_name: str
    source_file_hash: str
    source_category: str
    source_type: str

    # --- 부가 메타 ---
    document_date: Optional[str] = None
    display_date: Optional[str] = None
    primary_topic: Optional[str] = None
    topic_tags: List[str] = field(default_factory=list)
    task_type: Optional[str] = None
    when_to_use: str = ""
    prerequisites: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    parent_raw_chunk_ids: List[str] = field(default_factory=list)
    answer_use_cases: List[str] = field(default_factory=list)
    sanitized_markdown: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 신규 명칭 alias property — legacy field (card_id/card_type) 와 1:1 대응
    # ------------------------------------------------------------------
    @property
    def normalized_document_id(self) -> str:
        return self.card_id

    @normalized_document_id.setter
    def normalized_document_id(self, value: str) -> None:
        self.card_id = value

    @property
    def normalized_document_type(self) -> str:
        return self.card_type

    @normalized_document_type.setter
    def normalized_document_type(self, value: str) -> None:
        self.card_type = value

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedDocument":
        """dict 에서 NormalizedDocument 를 복원한다.

        - 누락된 list/dict 필드는 안전하게 기본값 처리.
        - legacy compatibility: ``normalized_document_id`` / ``normalized_document_type``
          키도 입력으로 허용한다.
        """
        data = dict(data or {})

        if "card_id" not in data and "normalized_document_id" in data:
            data["card_id"] = data.pop("normalized_document_id")
        if "card_type" not in data and "normalized_document_type" in data:
            data["card_type"] = data.pop("normalized_document_type")

        list_fields = {
            "topic_tags",
            "prerequisites",
            "steps",
            "checkpoints",
            "cautions",
            "examples",
            "related_terms",
            "open_questions",
            "evidence_spans",
            "parent_raw_chunk_ids",
            "answer_use_cases",
        }
        for key in list_fields:
            if data.get(key) is None:
                data[key] = []
        if data.get("metadata") is None:
            data["metadata"] = {}

        # 알려지지 않은 키는 무시 (forward compatibility)
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    def to_markdown(self) -> str:
        """Normalized Document 를 사람이 읽기 쉬운 Markdown 형식으로 변환한다."""
        lines: List[str] = [
            f"# {self.title}",
            "",
            f"- card_type: {self.card_type}",
            f"- primary_topic: {self.primary_topic or '-'}",
            f"- task_type: {self.task_type or '-'}",
            f"- source_file: {self.source_file_name}",
            f"- source_category: {self.source_category}",
            f"- display_date: {self.display_date or '-'}",
            f"- answer_use_cases: {', '.join(self.answer_use_cases) if self.answer_use_cases else '-'}",
            "",
            "## 요약",
            self.summary or "",
            "",
            "## 언제 사용하는가",
            self.when_to_use or "",
            "",
            "## 선행 조건",
            *_bullet_lines(self.prerequisites),
            "",
            "## 업무 절차",
            *_numbered_lines(self.steps),
            "",
            "## 체크포인트",
            *_bullet_lines(self.checkpoints),
            "",
            "## 주의사항",
            *_bullet_lines(self.cautions),
            "",
            "## 예시",
            *_bullet_lines(self.examples),
            "",
            "## 관련 용어",
            *_bullet_lines(self.related_terms),
            "",
            "## 미확인 사항",
            *_bullet_lines(self.open_questions),
            "",
            "## 근거",
            *_evidence_lines(self.evidence_spans),
        ]
        return "\n".join(lines).strip()

    def validate_minimum(self) -> bool:
        """저장/검색에 필요한 최소 필드를 검증한다."""
        if not self.card_id or not self.card_id.strip():
            return False
        if not self.card_type or not self.card_type.strip():
            return False
        if self.card_type not in VALID_NORMALIZED_DOCUMENT_TYPES:
            return False
        if any(use_case not in VALID_ANSWER_USE_CASES for use_case in self.answer_use_cases):
            return False
        if not self.title or not self.title.strip():
            return False
        if not self.summary or not self.summary.strip():
            return False
        if not self.source_file_name or not self.source_file_name.strip():
            return False
        markdown = (self.sanitized_markdown or self.to_markdown() or "").strip()
        return bool(markdown)


# ---------------------------------------------------------------------------
# legacy compatibility — 기존 import / 테스트가 그대로 동작하도록 alias 제공
# ---------------------------------------------------------------------------
KnowledgeCard = NormalizedDocument


def _bullet_lines(items: List[str]) -> List[str]:
    if not items:
        return ["- (없음)"]
    return [f"- {item}" for item in items]


def _numbered_lines(items: List[str]) -> List[str]:
    if not items:
        return ["1. (없음)"]
    return [f"{idx}. {item}" for idx, item in enumerate(items, start=1)]


def _evidence_lines(spans: List[Dict[str, Any]]) -> List[str]:
    if not spans:
        return ["- (근거 없음)"]
    lines: List[str] = []
    for span in spans:
        section = span.get("section") or span.get("section_title") or "-"
        chunk_index = span.get("chunk_index", "-")
        summary = span.get("summary") or span.get("text") or "-"
        lines.append(f"- section: {section}, chunk_index: {chunk_index}, summary: {summary}")
    return lines
