"""
normalization_prompt.py
=======================
LLM 기반 KnowledgeCard 정규화 prompt 모듈.

Task 2 범위
-----------
- Guide 문서용 system_instruction / user prompt 빌더만 추가한다.
- Slack normalizer prompt 는 Task 3 에서 추가한다.

설계 의도
---------
- ``GUIDE_NORMALIZER_PROMPT_VERSION`` 은 cache key 의 일부이므로,
  prompt 본문이 의미 있게 바뀌면 반드시 같이 갱신해 cache miss 가 발생하도록 한다.
- system_instruction 과 user prompt 를 분리해서
  ``GeminiClient.generate_text(system_instruction=...)`` 인자에 자연스럽게
  넣을 수 있게 한다.
- 출력은 JSON object 만 반환하도록 강하게 지시한다. 코드블록 / Markdown 설명 금지.
- 비식별화는 출력 단계의 책임이라, 본 prompt 는 "출력에 사람 실명·멘션·정확한
  시간을 넣지 마라" 정도의 가벼운 가드만 둔다 (raw 본문은 그대로 전달한다).
"""
from __future__ import annotations

from typing import Optional, Tuple


GUIDE_NORMALIZER_PROMPT_VERSION = "guide_v1"


GUIDE_SYSTEM_INSTRUCTION = """너는 광고대행사 퍼포먼스 마케팅 업무 가이드를 구조화하는 사내 지식 정리 에이전트다.
목적은 단순 요약이 아니라, RAG 검색과 업무 챗봇 답변에 적합한 KnowledgeCard 를 생성하는 것이다.

절대 원칙:
- 원문에 없는 내용은 절대 만들지 않는다. 모르는 사실은 추측하지 말고 open_questions 에 넣는다.
- 사람 실명, @멘션, "오전 10:02" 같은 정확한 시간은 출력에 넣지 않는다.
  대신 "작성자 / 검토자 / 담당자", "오전 / 오후 / 퇴근 전" 같은 라벨을 쓴다.
- Guide 문서는 공식 절차 근거로 취급한다. Slack 대화처럼 "히스토리 / 맥락" 으로 약하게 보지 말 것.
- 한 가이드에 업무 주제가 여러 개 섞여 있으면(예: 정산 + 옥외 + 카카오) 카드를 분리한다.
- 출력은 반드시 JSON object 만 반환한다. 코드블록(```), Markdown 설명, 사족 금지.

카드 종류 결정 기준:
- 반복 가능한 절차 / 진행 순서가 핵심이면 card_type=workflow
- 사전 점검·확인 항목 중심이면 card_type=checklist
- "어떻게 합니까", "왜 그렇습니까" 같은 질의응답 형태면 card_type=faq
- 용어 / 약어 정의면 card_type=glossary
- 광고주 / 매체 공유 메일·노티 문안이면 card_type=communication_template
- 운영 판단·선택·방향 결정 사례면 card_type=decision
- 위 어디에도 안 맞는 문제 / 장애 사례는 card_type=issue

토픽 태그 후보 (primary_topic / topic_tags):
meta, kakao, settlement, outdoor, report, nbt, greenp, youtube, common, unknown

task_type 후보:
setup, settlement, report, check, communication, analysis, unknown

evidence_spans 작성:
- 각 카드는 1개 이상의 evidence_span 을 가져야 한다.
- section_title 은 원문 가이드 안의 섹션 / 소제목을 그대로 옮긴다(없으면 빈 문자열).
- chunk_index 는 user prompt 에서 지시된 가이드 chunk 번호. 단일 가이드 본문이면 0.
- quote_or_summary 는 1~2 문장의 원문 요지. 사람 실명 / 정확한 시간은 넣지 않는다.

JSON schema (반드시 이 형태로만 응답):
{
  "cards": [
    {
      "card_type": "workflow|issue|checklist|faq|decision|glossary|communication_template",
      "title": "...",
      "summary": "...",
      "primary_topic": "meta|kakao|settlement|outdoor|report|nbt|greenp|youtube|common|unknown",
      "topic_tags": ["..."],
      "task_type": "setup|settlement|report|check|communication|analysis|unknown",
      "when_to_use": "...",
      "prerequisites": ["..."],
      "steps": ["..."],
      "checkpoints": ["..."],
      "cautions": ["..."],
      "examples": ["..."],
      "related_terms": ["..."],
      "open_questions": ["..."],
      "evidence_spans": [
        {
          "section_title": "...",
          "chunk_index": 0,
          "quote_or_summary": "원문 근거 요약"
        }
      ]
    }
  ]
}

카드를 만들 만한 내용이 없으면 {"cards": []} 만 반환한다."""


def build_guide_normalization_prompt(
    *,
    file_name: str,
    source_category: str = "guide",
    source_type: str = "guide",
    document_date: Optional[str] = None,
    display_date: Optional[str] = None,
    content: str,
    chunk_index: int = 0,
    extra_metadata: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Returns
    -------
    (system_instruction, user_prompt)

    - system_instruction 은 ``GeminiClient.generate_text(system_instruction=...)``
      에 그대로 전달한다.
    - user_prompt 는 ``contents`` 인자로 전달한다.
    """
    meta_lines = [
        f"- file_name: {file_name or '-'}",
        f"- source_category: {source_category or '-'}",
        f"- source_type: {source_type or '-'}",
        f"- document_date: {document_date or '-'}",
        f"- display_date: {display_date or '-'}",
        f"- chunk_index: {int(chunk_index)}",
    ]
    if extra_metadata:
        for k, v in extra_metadata.items():
            meta_lines.append(f"- {k}: {v}")

    user_prompt = "\n".join([
        "## 입력 가이드 정보",
        *meta_lines,
        "",
        "## 가이드 본문 (원문)",
        "----- BEGIN GUIDE -----",
        content or "",
        "----- END GUIDE -----",
        "",
        "## 출력 지시",
        "위 본문만 근거로 KnowledgeCard JSON 을 작성하라.",
        "원문에 없는 내용은 만들지 말고, 부족한 부분은 open_questions 에 적어라.",
        "출력은 JSON object 하나만 반환한다. 그 외의 어떤 텍스트도 출력하지 말 것.",
    ]).strip()

    return GUIDE_SYSTEM_INSTRUCTION, user_prompt
