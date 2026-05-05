"""
KnowledgeCard
=============
LLM 기반 업무 지식카드 정규화 결과를 담는 dataclass schema.

Task 1 에서는 schema / store / mock test 만 제공하고, 실제 LLM 호출이나
pipeline 연결은 하지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VALID_CARD_TYPES = {
    "workflow",
    "issue",
    "checklist",
    "faq",
    "decision",
    "glossary",
    "communication_template",
}


@dataclass
class KnowledgeCard:
    card_id: str
    card_type: str
    title: str
    summary: str
    source_file_name: str
    source_file_hash: str
    source_category: str
    source_type: str
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
    sanitized_markdown: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeCard":
        """dict 에서 KnowledgeCard 를 복원한다. 누락된 list/dict 필드는 안전하게 기본값 처리."""
        data = dict(data or {})
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
        }
        for key in list_fields:
            if data.get(key) is None:
                data[key] = []
        if data.get("metadata") is None:
            data["metadata"] = {}
        return cls(**data)

    def to_markdown(self) -> str:
        """지식카드를 사람이 읽기 쉬운 Markdown 형식으로 변환한다."""
        lines: List[str] = [
            f"# {self.title}",
            "",
            f"- card_type: {self.card_type}",
            f"- primary_topic: {self.primary_topic or '-'}",
            f"- task_type: {self.task_type or '-'}",
            f"- source_file: {self.source_file_name}",
            f"- source_category: {self.source_category}",
            f"- display_date: {self.display_date or '-'}",
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
        """Task 1 저장/테스트에 필요한 최소 필드만 검증한다."""
        if not self.card_id or not self.card_id.strip():
            return False
        if not self.card_type or not self.card_type.strip():
            return False
        if self.card_type not in VALID_CARD_TYPES:
            return False
        if not self.title or not self.title.strip():
            return False
        if not self.summary or not self.summary.strip():
            return False
        if not self.source_file_name or not self.source_file_name.strip():
            return False
        markdown = (self.sanitized_markdown or self.to_markdown() or "").strip()
        return bool(markdown)


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
