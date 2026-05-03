"""
prompt_builder.py
=================
RAG 답변용 프롬프트 빌더.

답변 형식:
1. 결론
2. 업무 처리 순서
3. 단계별 상세 설명
4. 실무 주의사항
5. 체크리스트
6. 참고 근거
7. 불확실한 부분
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from src.config import settings
from src.schemas import RetrievedChunk
from src.utils.token_utils import (
    cap_chunks_by_total_chars,
    cap_chunks_per_file,
    truncate_to_chars,
)

SYSTEM_INSTRUCTION = (
    "너는 광고대행사 퍼포먼스마케팅 업무를 도와주는 한국어 업무지원 챗봇이다. "
    "반드시 제공된 근거(컨텍스트)에 기반해 답변한다. "
    "근거에 없는 내용은 확정적으로 단정하지 않는다. "
    "업무 초보자가 바로 따라할 수 있도록 단계별로 자세히 설명한다. "
    "답변 마지막에는 참고 근거의 파일명/카테고리/섹션명/chunk index 를 표기한다. "
    "불확실한 부분이 있으면 솔직하게 모른다고 말한다. "
    "광고대행사 퍼포먼스마케팅 맥락(캠페인 세팅, 소재 검수, KPI/ROAS, 보고서 등) 을 우선 고려한다."
)

ANSWER_FORMAT = """반드시 아래 7가지 섹션 형식을 모두 갖춰서 한국어로 답변하라.

## 1. 결론
- 질문에 대한 핵심 답변 1~3줄

## 2. 업무 처리 순서
- 1) ...
- 2) ...
- 3) ...

## 3. 단계별 상세 설명
- 각 단계에서 해야 할 일, 보는 곳, 클릭/입력해야 할 항목 등을 자세히 설명

## 4. 실무 주의사항
- 흔히 하는 실수, 주의해야 할 정책/숫자 단위/네이밍 규칙 등

## 5. 체크리스트
- [ ] 항목1
- [ ] 항목2
- [ ] 항목3

## 6. 참고 근거
- (파일명 · 카테고리 · 섹션 · chunk_index) 형식으로 표시
- 가능한 한 모든 근거 chunk 를 언급

## 7. 불확실한 부분
- 컨텍스트에서 확실히 알 수 없는 내용은 여기에 정직하게 적는다."""


def _format_chunk_block(idx: int, chunk: RetrievedChunk, max_chars_per_chunk: int) -> str:
    title = chunk.section_title or "-"
    cat = chunk.uploaded_category or "-"
    src = chunk.source_type or "-"
    ctype = chunk.content_type or "-"
    file_name = chunk.file_name or "-"
    chunk_index = chunk.metadata.get("chunk_index", "?")
    body = truncate_to_chars(chunk.content or "", max_chars_per_chunk)
    return (
        f"[근거 #{idx}]\n"
        f"- 파일명: {file_name}\n"
        f"- 카테고리: {cat}\n"
        f"- 출처(source_type): {src}\n"
        f"- 콘텐츠유형: {ctype}\n"
        f"- 섹션/시트: {title}\n"
        f"- chunk_index: {chunk_index}\n"
        f"- 점수(final): {chunk.final_score:.4f}\n"
        f"---\n"
        f"{body}\n"
    )


def _trim_context(
    chunks: Iterable[RetrievedChunk],
    max_total_chars: int,
    max_per_file: int,
    max_chars_per_chunk: int,
) -> Tuple[List[RetrievedChunk], List[str]]:
    """파일별 cap → 누적 cap 의 순으로 자른다."""
    chunks = list(chunks)
    chunks = cap_chunks_per_file(chunks, file_getter=lambda c: c.file_name, max_per_file=max_per_file)
    blocks: List[str] = []
    selected: List[RetrievedChunk] = []
    total = 0
    for c in chunks:
        block = _format_chunk_block(len(selected) + 1, c, max_chars_per_chunk)
        if total + len(block) > max_total_chars and selected:
            break
        selected.append(c)
        blocks.append(block)
        total += len(block)
    return selected, blocks


def build_qa_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    rewritten_query: str | None = None,
    max_total_chars: int | None = None,
    max_per_file: int | None = None,
) -> Tuple[str, List[RetrievedChunk]]:
    """
    Returns
    -------
    (prompt_text, used_chunks)
    """
    max_total = int(max_total_chars or settings.max_context_chars)
    max_per = int(max_per_file or settings.max_chunks_per_file)
    # chunk 1개당 최대 길이 (전체 예산의 1/3 까지 허용)
    per_chunk_max = max(800, max_total // max(3, len(chunks) or 3))
    used, blocks = _trim_context(
        chunks, max_total_chars=max_total, max_per_file=max_per, max_chars_per_chunk=per_chunk_max
    )

    parts: List[str] = []
    parts.append(SYSTEM_INSTRUCTION)
    parts.append("")
    parts.append("## 사용자 질문")
    parts.append(question.strip())
    if rewritten_query:
        parts.append("")
        parts.append("## 검색용 재작성 질의")
        parts.append(rewritten_query.strip())

    parts.append("")
    parts.append("## 답변 형식 가이드")
    parts.append(ANSWER_FORMAT)

    parts.append("")
    parts.append("## 검색된 근거 (총 %d개)" % len(used))
    if used:
        parts.extend(blocks)
    else:
        parts.append("(검색된 근거가 없습니다.)")

    parts.append("")
    parts.append("## 출력")
    parts.append("위 형식을 그대로 따라 한국어로 답변하라. 근거가 부족하면 7번 섹션에 명시하라.")

    return "\n".join(parts), used


def build_no_context_answer_prompt(question: str) -> str:
    """검색 결과가 0개일 때 사용. 모르는 부분을 인정하도록 한다."""
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"## 사용자 질문\n{question}\n\n"
        f"## 답변 형식 가이드\n{ANSWER_FORMAT}\n\n"
        f"## 검색된 근거\n(없음)\n\n"
        f"## 출력\n근거 자료를 찾지 못했으므로 7번 '불확실한 부분' 에 그 사실을 분명히 적고, "
        f"가능한 일반적 가이드만 조심스럽게 제시하라."
    )
