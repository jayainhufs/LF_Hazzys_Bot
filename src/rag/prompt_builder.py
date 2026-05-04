"""
prompt_builder.py
=================
RAG 답변용 프롬프트 빌더.

답변 형식 (기존 유지):
1. 결론
2. 업무 처리 순서
3. 단계별 상세 설명
4. 실무 주의사항
5. 체크리스트
6. 참고 근거
7. 불확실한 부분

이번 개선 포인트:
- 근거에 없는 내용 추정 금지 원칙을 강화한다.
- Slack 대화 vs Guide 문서의 우선순위 규칙을 명시한다.
- 정산/세금계산서/SF/모비사인 → guide 우선, TODO/당일 진행 → slack 우선.
- 참고 근거 표기에 file_name / uploaded_category / source_type / section_title /
  chunk_index / final_score 를 모두 포함한다.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from src.config import settings
from src.preprocessing.anonymizer import anonymize_text
from src.schemas import RetrievedChunk
from src.utils.token_utils import (
    cap_chunks_per_file,
    truncate_to_chars,
)

SYSTEM_INSTRUCTION = (
    "너는 광고대행사 퍼포먼스마케팅 업무를 도와주는 한국어 업무지원 챗봇이다. "
    "답변은 반드시 아래 '검색된 근거(컨텍스트)' 에 적힌 사실에만 기반해라. "
    "근거에 없는 내용은 절대 단정하거나 일반 상식으로 채우지 말라. "
    "근거가 부족하면 7번 섹션에서 '현재 근거만으로는 확실하지 않다' 고 분명히 밝혀라. "
    "Slack 대화 근거는 실무 맥락 참고용이다. 공식 절차에 대한 설명은 guide 근거를 우선 반영하라. "
    "정산/세금계산서/SF/모비사인/인보이스/입금 관련 질문은 guide 근거를 우선 인용하고, "
    "TODO/당일 업무 흐름/스레드 피드백 관련 질문은 Slack thread 근거를 우선 인용하라. "
    "업무 초보자가 바로 따라할 수 있도록 단계별로 자세히 설명한다. "
    "광고대행사 퍼포먼스마케팅 맥락(캠페인 세팅, 소재 검수, KPI/ROAS, 보고서 등) 을 우선 고려한다. "
    "제공된 근거에는 비식별화된 업무 내용이 포함되어 있다. "
    "사람 실명/멘션/정확한 시간/정확한 날짜를 답변에 그대로 옮기지 말고, "
    "역할 기반 표현(작성자, 검토자, 담당자) 과 시간대/업무일 라벨로 표현하라. "
    "답변 마지막 참고근거에는 file_name, uploaded_category, source_type, section_title, "
    "display_date, primary_topic, chunk_index, final_score 를 표기하되 사람 실명과 정확한 시간은 제외한다."
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
- guide 근거가 있으면 절차 설명 시 guide 근거를 먼저 인용
- slack 근거는 보조 맥락(예: "이 단계에서 흔히 빠지는 실수") 으로 활용

## 4. 실무 주의사항
- 흔히 하는 실수, 주의해야 할 정책/숫자 단위/네이밍 규칙 등
- 근거에 명시된 주의사항만 적고, 일반 상식 추정은 적지 말 것

## 5. 체크리스트
- [ ] 항목1
- [ ] 항목2
- [ ] 항목3

## 6. 참고 근거
- 아래 형식으로 모든 인용 근거를 표기:
  - `file_name=... · uploaded_category=... · source_type=... · section_title=... · display_date=... · primary_topic=... · chunk_index=... · final_score=...`
- 사람 실명/정확한 시간/원본 날짜는 적지 말 것 (display_date 라벨만 표기)
- 가능한 한 답변에서 활용한 모든 근거 chunk 를 명시

## 7. 불확실한 부분
- 컨텍스트에서 확실히 알 수 없는 내용은 여기에 정직하게 적는다.
- "근거가 부족해 단정할 수 없습니다" 같은 표현을 적극적으로 사용하라."""

PRINCIPLES = """## 답변 원칙 (반드시 준수)
1. 제공된 근거 chunk 에 없는 내용을 추정하거나 일반 상식으로 채우지 말 것.
2. 근거가 불충분하면 단정하지 말고 "현재 근거만으로는 확실하지 않다" 고 말할 것.
3. Slack 대화 근거(uploaded_category=slack)는 실무 맥락 참고용이며,
   공식 절차/표준 프로세스에 대한 답변은 guide 근거(uploaded_category=guide)를 우선할 것.
4. 정산/세금계산서/SF/모비사인/인보이스/입금/광고주 공유용 정산 시트 관련 질문은 guide 근거 우선.
5. TODO/당일 업무 흐름/오늘의 스레드/피드백 관련 질문은 Slack thread 근거 우선.
6. 메타 캠페인 세팅/ASC/BAU/컨첵시트/토글 관련 질문은 slack 과 guide 근거 모두 활용 가능.
7. 답변 마지막에는 활용한 모든 근거를 정확히 표시할 것 (6번 섹션).
8. 제공된 근거 텍스트에는 비식별화된 업무 내용이 포함되어 있다. 답변에는 사람 실명,
   @멘션, "오전 10:02" 같은 정확한 시간, "2026년 4월 29일" 같은 정확한 날짜를
   그대로 옮기지 말 것. 대신 "작성자/검토자/담당자", "오전/오후/퇴근 전", "해당 업무일/
   다음 업무일" 같은 라벨로 표현할 것.
9. 사용자가 날짜를 명시적으로 묻더라도 답변에서는 가능한 "질문에서 언급한 업무일" 정도로
   표현하고, 정확한 날짜를 반복해 적지 말 것 (display_date 라벨 사용)."""


def _display_date_for(chunk: RetrievedChunk) -> str:
    """anonymize_output 이 켜져 있으면 'YYYY-MM-DD' 대신 라벨을 반환."""
    md = chunk.metadata or {}
    doc_date = md.get("document_date")
    if not doc_date:
        return "-"
    if settings.show_exact_dates:
        return str(doc_date)
    if settings.anonymize_output:
        return f"해당 {settings.anonymized_date_label}"
    return str(doc_date)


def _format_chunk_block(idx: int, chunk: RetrievedChunk, max_chars_per_chunk: int) -> str:
    title = chunk.section_title or "-"
    cat = chunk.uploaded_category or "-"
    src = chunk.source_type or "-"
    ctype = chunk.content_type or "-"
    file_name = chunk.file_name or "-"
    md = chunk.metadata or {}
    chunk_index = md.get("chunk_index", "?")
    primary_topic = md.get("primary_topic") or "-"
    todo_phase = md.get("todo_phase") or "-"
    parser_format = md.get("parser_format") or "-"
    display_date = _display_date_for(chunk)

    # body 는 sanitized_content 를 우선 사용 (anonymize_output=true).
    if settings.anonymize_output:
        body_src = md.get("sanitized_content") or anonymize_text(chunk.content or "")
    else:
        body_src = chunk.content or ""
    body = truncate_to_chars(body_src, max_chars_per_chunk)

    return (
        f"[근거 #{idx}]\n"
        f"- file_name: {file_name}\n"
        f"- uploaded_category: {cat}\n"
        f"- source_type: {src}\n"
        f"- content_type: {ctype}\n"
        f"- section_title: {title}\n"
        f"- display_date: {display_date}\n"
        f"- primary_topic: {primary_topic}\n"
        f"- todo_phase: {todo_phase}\n"
        f"- parser_format: {parser_format}\n"
        f"- chunk_index: {chunk_index}\n"
        f"- final_score: {chunk.final_score:.4f}\n"
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
    # chunk 1개당 최대 길이 (전체 예산의 1/2 까지 허용)
    per_chunk_max = max(800, max_total // max(2, len(chunks) or 2))
    used, blocks = _trim_context(
        chunks, max_total_chars=max_total, max_per_file=max_per, max_chars_per_chunk=per_chunk_max
    )

    parts: List[str] = []
    parts.append(SYSTEM_INSTRUCTION)
    parts.append("")
    parts.append(PRINCIPLES)
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
    parts.append(
        "위 형식을 그대로 따라 한국어로 답변하라. "
        "근거에 없는 내용은 추정하지 말고, 부족하면 7번 섹션에 명시하라. "
        "6번 섹션에 모든 인용 근거의 file_name, uploaded_category, source_type, "
        "section_title, display_date, primary_topic, chunk_index, final_score 를 빠짐없이 표기하라. "
        "사람 실명/멘션/정확한 시간/원본 날짜는 답변과 참고근거 모두에 적지 말 것."
    )

    return "\n".join(parts), used


def build_no_context_answer_prompt(question: str) -> str:
    """
    검색 결과가 0개일 때 사용 (호환용).

    NOTE: qa_pipeline.py 는 더 이상 이 prompt 를 호출하지 않는다 (근거 부족 시 generation skip).
    스크립트나 외부 코드에서 직접 호출할 가능성이 있어 함수는 남겨둔다.
    """
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{PRINCIPLES}\n\n"
        f"## 사용자 질문\n{question}\n\n"
        f"## 답변 형식 가이드\n{ANSWER_FORMAT}\n\n"
        f"## 검색된 근거\n(없음)\n\n"
        f"## 출력\n근거 자료를 찾지 못했으므로 7번 '불확실한 부분' 에 그 사실을 분명히 적고, "
        f"가능한 일반적 가이드만 조심스럽게 제시하라."
    )
