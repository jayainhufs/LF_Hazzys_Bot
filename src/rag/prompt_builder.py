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

Task 7 (knowledge_card 중심 답변):
- retrieval_role == "primary_card" 인 chunk 를 1차 근거로 사용한다.
- raw_evidence 는 보조 근거 섹션으로만 사용한다.
- raw_fallback 은 primary_card 가 없을 때 fallback 으로만 사용한다.
- card_type (workflow / checklist / issue / decision / faq / communication_template /
  glossary) 에 따라 답변 구조를 다르게 유도한다.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

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

    Task 7: ANSWER_WITH_KNOWLEDGE_CARDS=true 이고 primary_card 가 있으면
    knowledge_card 중심 prompt 로 라우팅한다.
    """
    if settings.answer_with_knowledge_cards:
        groups = split_chunks_by_retrieval_role(chunks)
        if groups["primary_cards"]:
            prompt, used, _mode = build_knowledge_card_answer_prompt(
                question=question,
                chunks=chunks,
                rewritten_query=rewritten_query,
                max_total_chars=max_total_chars,
                max_per_file=max_per_file,
            )
            return prompt, used

    return _build_legacy_raw_qa_prompt(
        question=question,
        chunks=chunks,
        rewritten_query=rewritten_query,
        max_total_chars=max_total_chars,
        max_per_file=max_per_file,
    )


def _build_legacy_raw_qa_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    rewritten_query: str | None = None,
    max_total_chars: int | None = None,
    max_per_file: int | None = None,
) -> Tuple[str, List[RetrievedChunk]]:
    """기존 raw chunk 기반 prompt (fallback / 호환용)."""
    max_total = int(max_total_chars or settings.max_context_chars)
    max_per = int(max_per_file or settings.max_chunks_per_file)
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


# ---------------------------------------------------------------------------
# Task 7: KnowledgeCard 중심 답변 helpers
# ---------------------------------------------------------------------------
KNOWLEDGE_CARD_SYSTEM_INSTRUCTION = (
    "너는 광고대행사 퍼포먼스마케팅 업무를 도와주는 한국어 업무지원 챗봇이다. "
    "아래 컨텍스트의 '주 근거 (KnowledgeCard)' 섹션에 적힌 정규화된 카드를 1차 근거로 답변하라. "
    "'보조 근거 (Raw Evidence)' 섹션은 카드 내용을 보강할 때만 참고하고, 카드와 충돌하면 "
    "'근거 간 차이가 있습니다' 라고 분명히 적어라. "
    "근거에 없는 절차/수치/결정사항을 만들거나 일반 상식으로 채우지 말라. "
    "근거가 부족하면 답변의 마지막 섹션에서 '현재 근거만으로는 확실하지 않다' 고 분명히 밝혀라. "
    "Guide 기반 workflow/checklist 카드는 공식 절차 근거로 우선 반영하고, "
    "Slack 기반 카드(예: source_category=slack)는 실무 히스토리/맥락으로 취급하라. "
    "사람 실명, @멘션, '오전 10:02' 같은 정확한 시간, 원본 날짜를 답변에 그대로 옮기지 말라. "
    "사람 이름은 작성자 / 검토자 / 담당자 / 광고주 / 매체 담당자 / 재무팀 같은 역할 표현으로 바꾸고, "
    "날짜는 '해당 업무일', '전일', '다음 업무일', '월초', '월말' 같은 업무 맥락 표현을 우선 사용하라."
)

KNOWLEDGE_CARD_PRINCIPLES = """## 답변 원칙 (KnowledgeCard 중심)
1. '주 근거 (KnowledgeCard)' 의 카드를 1차 근거로 사용한다.
2. '보조 근거 (Raw Evidence)' 는 카드 내용을 보강하거나 구체 수치/사례를 보여줄 때만 사용한다.
3. 보조 근거가 주 근거 카드와 충돌하면 단정하지 말고 "근거 간 차이가 있습니다" 라고 적는다.
4. 주 근거가 비어 있고 'Raw Fallback' 만 있을 때는 그 사실을 7번/마지막 섹션에 분명히 적고
   "정규화된 카드가 아직 없어 raw 근거에 의존했습니다" 라고 명시한다.
5. 근거에 없는 절차/수치/결정사항을 추정하거나 일반 상식으로 채우지 말 것.
6. 사람 실명, @멘션, "오전 10:02" 같은 정확한 시간, "2026년 4월 29일" 같은 원본 날짜를
   답변과 참고 근거 모두에 그대로 옮기지 말 것.
7. 이름은 "작성자/검토자/담당자/광고주/매체 담당자/재무팀" 같은 역할 표현,
   날짜는 "해당 업무일/전일/다음 업무일/월초/월말" 같은 업무 맥락 표현을 우선 사용한다.
8. Guide 기반 workflow/checklist 카드는 공식 절차 근거로 우선 인용한다.
9. Slack 기반 카드는 공식 가이드가 아니라 실무 히스토리/맥락으로 취급한다.
10. 답변 마지막 '참고 근거' 섹션은 KnowledgeCard 와 Raw Evidence 를 분리해 표기한다."""

# 기본 답변 형식 (workflow / checklist / issue / decision / faq)
DEFAULT_ANSWER_FORMAT = """반드시 아래 형식을 그대로 따라 한국어로 답변하라.

## 1. 결론
- 질문에 대한 핵심 답변 1~3줄. 주 근거 카드의 summary / when_to_use 를 우선 반영하라.

## 2. 업무 처리 순서 또는 핵심 체크포인트
- 카드의 steps 또는 checkpoints 를 1) 2) 3) 형식으로 정리한다.
- workflow 카드: 처리 순서 중심.
- checklist 카드: "놓치면 안 되는 항목" 중심.
- issue / decision / faq 카드: 핵심 판단 기준 / 결론 중심.

## 3. 단계별 상세 설명
- 각 단계에서 해야 할 일, 보는 곳, 클릭/입력해야 할 항목, 판단 기준을 자세히 설명.
- 보조 근거(Raw Evidence)는 구체 수치/사례 보강 용도로만 인용한다.

## 4. 실무 주의사항
- 카드의 cautions / 충돌하는 보조 근거 / 흔한 실수 중심.
- 일반 상식 추정은 적지 말 것. 카드/근거에 없는 내용은 7번 섹션으로 보낼 것.

## 5. 바로 사용할 수 있는 체크리스트
- [ ] 항목1
- [ ] 항목2
- [ ] 항목3

## 6. 참고 근거
- 아래 형식으로 주 근거(KnowledgeCard) 와 보조 근거(Raw Evidence) 를 분리해 표기한다.

[주 근거: KnowledgeCard]
- title:
- card_type:
- primary_topic:
- source_file_name:
- display_date:
- final_score:

[보조 근거: Raw Evidence]
- source_file_name:
- section_title:
- content_type:
- chunk_index 또는 parent_raw_chunk_ids:

## 7. 불확실한 부분
- 컨텍스트에서 확실히 알 수 없는 내용을 정직하게 적는다.
- 주 근거가 비어 있어 raw fallback 만 사용했다면 그 사실을 명시한다.
- "근거가 부족해 단정할 수 없습니다" 같은 표현을 적극적으로 사용한다."""

# 문안 / 광고주 공유 (communication_template) 형식
COMMUNICATION_ANSWER_FORMAT = """반드시 아래 형식을 그대로 따라 한국어로 답변하라.

## 1. 결론
- 어떤 상황에 어떤 톤으로 어떤 핵심 메시지를 전달해야 하는지 1~3줄.

## 2. 바로 사용할 수 있는 초안
- 카드의 steps / examples / template 본문을 정리해 사용 가능한 문안 초안을 제시한다.
- 사람 실명, @멘션, 정확한 시간, 원본 날짜는 빼고 역할 표현 / 업무일 라벨로 대체한다.

## 3. 문안 작성 포인트
- 어떤 정보를 반드시 포함해야 하는지, 어떤 표현을 피해야 하는지.
- 카드의 cautions / when_to_use / prerequisites 를 우선 반영.

## 4. 주의사항
- 광고주/매체/내부 공유 톤 차이, 금액/일정 표기 시 주의점, 익명화 원칙 등.

## 5. 참고 근거
[주 근거: KnowledgeCard]
- title:
- card_type:
- primary_topic:
- source_file_name:
- display_date:
- final_score:

[보조 근거: Raw Evidence]
- source_file_name:
- section_title:
- content_type:
- chunk_index 또는 parent_raw_chunk_ids:

## 6. 불확실한 부분
- 카드/근거에 명시되지 않은 항목, 추가 확인이 필요한 내용을 정직하게 적는다."""

# 용어 설명 (glossary) 형식
GLOSSARY_ANSWER_FORMAT = """반드시 아래 형식을 그대로 따라 한국어로 답변하라.

## 1. 용어 정의
- 카드의 title / summary / definition 을 1~3줄로 정리.

## 2. 실무에서의 의미
- 광고대행사 퍼포먼스마케팅 업무에서 이 용어가 어떻게 쓰이는지.
- 카드의 examples / when_to_use 를 우선 반영.

## 3. 헷갈리기 쉬운 점
- 비슷한 용어와의 차이, 흔히 잘못 쓰이는 사례 (카드 cautions 기반).

## 4. 관련 용어
- 카드의 related_terms 를 그대로 옮긴다.

## 5. 참고 근거
[주 근거: KnowledgeCard]
- title:
- card_type:
- primary_topic:
- source_file_name:
- display_date:
- final_score:

[보조 근거: Raw Evidence]
- source_file_name:
- section_title:
- content_type:
- chunk_index 또는 parent_raw_chunk_ids:

## 6. 불확실한 부분
- 정의가 다르게 쓰일 가능성이 있는 부분을 정직하게 적는다."""


def split_chunks_by_retrieval_role(
    chunks: Iterable[RetrievedChunk],
) -> Dict[str, List[RetrievedChunk]]:
    """
    retrieval_role 기준으로 chunk 를 3 그룹으로 나눈다.

    - primary_cards: retrieval_role=="primary_card" 또는
                     content_type=="knowledge_card" 또는
                     source_type=="llm_normalized"
    - raw_evidence:  retrieval_role=="raw_evidence"
    - raw_fallback:  retrieval_role=="raw_fallback" 또는 그 외 raw chunk

    Returns
    -------
    {"primary_cards": [...], "raw_evidence": [...], "raw_fallback": [...]}
    """
    primary: List[RetrievedChunk] = []
    raw_ev: List[RetrievedChunk] = []
    raw_fb: List[RetrievedChunk] = []
    for c in chunks or []:
        md = c.metadata or {}
        role = (md.get("retrieval_role") or "").lower()
        ctype = (c.content_type or "").lower()
        stype = (c.source_type or "").lower()
        is_card = (
            role == "primary_card"
            or ctype == "knowledge_card"
            or stype == "llm_normalized"
        )
        if is_card:
            primary.append(c)
        elif role == "raw_evidence":
            raw_ev.append(c)
        else:
            raw_fb.append(c)
    return {
        "primary_cards": primary,
        "raw_evidence": raw_ev,
        "raw_fallback": raw_fb,
    }


def _format_card_metadata_block(
    idx: int, card: RetrievedChunk, max_chars_per_chunk: int
) -> str:
    md = card.metadata or {}
    body_src: str
    if settings.anonymize_output:
        body_src = md.get("sanitized_content") or anonymize_text(card.content or "")
    else:
        body_src = card.content or ""
    body = truncate_to_chars(body_src or "", max_chars_per_chunk)
    return (
        f"[KC #{idx}]\n"
        f"- card_id: {md.get('card_id') or '-'}\n"
        f"- card_type: {md.get('card_type') or '-'}\n"
        f"- title: {md.get('title') or card.section_title or '-'}\n"
        f"- primary_topic: {md.get('primary_topic') or '-'}\n"
        f"- topic_tags: {md.get('topic_tags') or []}\n"
        f"- task_type: {md.get('task_type') or '-'}\n"
        f"- source_file_name: {card.file_name or '-'}\n"
        f"- source_category: {card.uploaded_category or '-'}\n"
        f"- display_date: {_display_date_for(card)}\n"
        f"- final_score: {card.final_score:.4f}\n"
        f"- parent_raw_chunk_ids: {md.get('parent_raw_chunk_ids') or []}\n"
        f"---\n"
        f"{body}\n"
    )


def _format_raw_evidence_block(
    idx: int, chunk: RetrievedChunk, max_chars_per_chunk: int
) -> str:
    md = chunk.metadata or {}
    if settings.anonymize_output:
        body_src = md.get("sanitized_content") or anonymize_text(chunk.content or "")
    else:
        body_src = chunk.content or ""
    body = truncate_to_chars(body_src or "", max_chars_per_chunk)
    return (
        f"[RAW #{idx}]\n"
        f"- file_name: {chunk.file_name or '-'}\n"
        f"- uploaded_category: {chunk.uploaded_category or '-'}\n"
        f"- source_type: {chunk.source_type or '-'}\n"
        f"- content_type: {chunk.content_type or '-'}\n"
        f"- section_title: {chunk.section_title or '-'}\n"
        f"- display_date: {_display_date_for(chunk)}\n"
        f"- chunk_index: {md.get('chunk_index')}\n"
        f"- final_score: {chunk.final_score:.4f}\n"
        f"---\n"
        f"{body}\n"
    )


def format_knowledge_card_context(
    cards: List[RetrievedChunk],
    *,
    max_total_chars: int,
    max_chars_per_chunk: int,
) -> Tuple[List[RetrievedChunk], List[str]]:
    """primary_cards 를 prompt block 형태로 직렬화한다 (총 char 예산 내)."""
    used: List[RetrievedChunk] = []
    blocks: List[str] = []
    total = 0
    for c in cards:
        block = _format_card_metadata_block(
            len(used) + 1, c, max_chars_per_chunk
        )
        if total + len(block) > max_total_chars and used:
            break
        used.append(c)
        blocks.append(block)
        total += len(block)
    return used, blocks


def format_raw_evidence_appendix(
    raw_chunks: List[RetrievedChunk],
    *,
    max_total_chars: int,
    max_chars_per_chunk: int,
) -> Tuple[List[RetrievedChunk], List[str]]:
    """보조 근거 raw chunk 를 prompt block 형태로 직렬화한다."""
    used: List[RetrievedChunk] = []
    blocks: List[str] = []
    total = 0
    for c in raw_chunks:
        block = _format_raw_evidence_block(len(used) + 1, c, max_chars_per_chunk)
        if total + len(block) > max_total_chars and used:
            break
        used.append(c)
        blocks.append(block)
        total += len(block)
    return used, blocks


def select_answer_format(
    primary_cards: List[RetrievedChunk],
    question: str,
) -> Tuple[str, str]:
    """
    primary_cards 의 card_type 와 질문 의도를 기반으로 적절한 답변 형식을 선택.

    Returns
    -------
    (answer_format_text, format_label)
    """
    q = (question or "").lower()
    types: List[str] = []
    for c in primary_cards or []:
        md = c.metadata or {}
        ct = str(md.get("card_type") or "").lower()
        if ct:
            types.append(ct)

    # 우선순위: glossary > communication_template > 기본
    if "glossary" in types or any(k in q for k in ("용어", "무슨 뜻", "정의")):
        return GLOSSARY_ANSWER_FORMAT, "glossary"
    if "communication_template" in types or any(
        k in q for k in ("문안", "메일", "공유", "전달", "회신")
    ):
        return COMMUNICATION_ANSWER_FORMAT, "communication_template"
    return DEFAULT_ANSWER_FORMAT, "default"


def build_knowledge_card_answer_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    rewritten_query: Optional[str] = None,
    max_total_chars: Optional[int] = None,
    max_per_file: Optional[int] = None,
    max_primary_cards: Optional[int] = None,
    max_raw_evidence_chunks: Optional[int] = None,
    include_raw_evidence_appendix: Optional[bool] = None,
) -> Tuple[str, List[RetrievedChunk], str]:
    """
    KnowledgeCard 중심 QA prompt 를 생성한다.

    Returns
    -------
    (prompt_text, used_chunks, format_label)
    """
    max_total = int(max_total_chars or settings.max_context_chars)
    _ = int(max_per_file or settings.max_chunks_per_file)  # 호환 (현재 미사용)
    n_primary = int(
        max_primary_cards
        if max_primary_cards is not None
        else settings.max_primary_cards
    )
    n_raw = int(
        max_raw_evidence_chunks
        if max_raw_evidence_chunks is not None
        else settings.max_raw_evidence_chunks
    )
    include_raw_app = bool(
        include_raw_evidence_appendix
        if include_raw_evidence_appendix is not None
        else settings.include_raw_evidence_appendix
    )

    groups = split_chunks_by_retrieval_role(chunks)
    primary_cards = list(groups["primary_cards"])[: max(0, n_primary)]
    raw_evidence = list(groups["raw_evidence"])[: max(0, n_raw)]
    raw_fallback = list(groups["raw_fallback"])

    # primary 가 비어 있으면 raw fallback 으로 폴백 (호출 측에서 더 일찍 분기해도 괜찮음)
    if not primary_cards:
        prompt, used = _build_legacy_raw_qa_prompt(
            question=question,
            chunks=chunks,
            rewritten_query=rewritten_query,
            max_total_chars=max_total_chars,
            max_per_file=max_per_file,
        )
        return prompt, used, "raw_fallback"

    answer_format, format_label = select_answer_format(primary_cards, question)

    # char 예산 분배: card 70%, raw evidence 25%, fallback 5%
    card_budget = int(max_total * 0.70)
    raw_budget = int(max_total * 0.25) if include_raw_app else 0
    fallback_budget = max_total - card_budget - raw_budget

    per_card_max = max(1000, card_budget // max(1, len(primary_cards)))
    per_raw_max = max(500, raw_budget // max(1, len(raw_evidence) or 1))
    per_fb_max = max(400, fallback_budget // max(1, len(raw_fallback) or 1))

    used_cards, card_blocks = format_knowledge_card_context(
        primary_cards,
        max_total_chars=card_budget,
        max_chars_per_chunk=per_card_max,
    )
    used_raw_ev: List[RetrievedChunk] = []
    raw_ev_blocks: List[str] = []
    if include_raw_app and raw_evidence:
        used_raw_ev, raw_ev_blocks = format_raw_evidence_appendix(
            raw_evidence,
            max_total_chars=raw_budget,
            max_chars_per_chunk=per_raw_max,
        )

    used_raw_fb: List[RetrievedChunk] = []
    raw_fb_blocks: List[str] = []
    if include_raw_app and raw_fallback and fallback_budget > 0:
        # fallback 은 짧게만 노출. 보통 primary 가 있으면 사용 안 함.
        used_raw_fb, raw_fb_blocks = format_raw_evidence_appendix(
            raw_fallback[: max(1, n_raw)],
            max_total_chars=fallback_budget,
            max_chars_per_chunk=per_fb_max,
        )

    parts: List[str] = []
    parts.append(KNOWLEDGE_CARD_SYSTEM_INSTRUCTION)
    parts.append("")
    parts.append(KNOWLEDGE_CARD_PRINCIPLES)
    parts.append("")
    parts.append("## 사용자 질문")
    parts.append((question or "").strip())
    if rewritten_query:
        parts.append("")
        parts.append("## 검색용 재작성 질의")
        parts.append(rewritten_query.strip())

    parts.append("")
    parts.append(
        f"## 답변 형식 가이드 (template={settings.knowledge_card_answer_template_version}, "
        f"label={format_label})"
    )
    parts.append(answer_format)

    parts.append("")
    parts.append("## 주 근거 (KnowledgeCard) — 1차 근거")
    parts.append(
        "이 섹션의 카드를 답변의 1차 근거로 사용하라. 카드에 명시된 steps / "
        "checkpoints / cautions / examples 만 인용하고, 카드에 없는 내용을 만들지 말 것."
    )
    parts.append(f"(총 {len(used_cards)}개)")
    parts.extend(card_blocks)

    if include_raw_app and used_raw_ev:
        parts.append("")
        parts.append("## 보조 근거 (Raw Evidence) — 카드 보강 용도")
        parts.append(
            "주 근거 카드를 보강하거나 구체 수치/사례를 보여줄 때만 참고하라. "
            "주 근거와 충돌하면 '근거 간 차이가 있습니다' 라고 명시하라."
        )
        parts.append(f"(총 {len(used_raw_ev)}개)")
        parts.extend(raw_ev_blocks)
    elif include_raw_app:
        parts.append("")
        parts.append("## 보조 근거 (Raw Evidence)")
        parts.append("(보조 근거가 없습니다.)")

    if include_raw_app and used_raw_fb:
        parts.append("")
        parts.append("## Raw Fallback (참고용)")
        parts.append(
            "카드와 직접 연결되지 않은 raw chunk 다. 본 답변에서는 보조 근거가 없을 때만 "
            "조심스럽게 참고하라. 본문 인용 시 '정규화된 카드가 아직 없는 영역입니다' 라고 적어라."
        )
        parts.append(f"(총 {len(used_raw_fb)}개)")
        parts.extend(raw_fb_blocks)

    parts.append("")
    parts.append("## 출력")
    parts.append(
        "위 답변 형식을 그대로 따라 한국어로 답변하라. "
        "주 근거 (KnowledgeCard) 의 카드를 1차 근거로 사용하고, 보조 근거는 보강 용도로만 인용하라. "
        "근거에 없는 내용은 추정하지 말고, 부족하면 마지막 섹션에 명시하라. "
        "참고 근거 섹션에서 KnowledgeCard 와 Raw Evidence 를 분리해 표기하고, "
        "사람 실명/멘션/정확한 시간/원본 날짜는 답변과 참고 근거 모두에 적지 말 것."
    )

    used: List[RetrievedChunk] = list(used_cards) + list(used_raw_ev) + list(used_raw_fb)
    return "\n".join(parts), used, format_label
