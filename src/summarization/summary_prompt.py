"""
summary_prompt.py
=================
Excel 시트/표 → 한국어 업무 설명문 (Markdown) 변환용 프롬프트.

원본 raw_table_text 가 너무 길면 호출자 (excel_summarizer) 가 사전에 길이를 제한한다.
"""
from __future__ import annotations

EXCEL_SUMMARY_PROMPT_TEMPLATE = """너는 광고대행사 퍼포먼스마케팅 업무 가이드를 정리하는 분석가다.
아래 Excel 시트/표 내용을 한국어로 매우 자세하게 설명하라.

[원칙]
- 단순 값 나열이 아니라 이 표가 어떤 업무에 쓰이는지 설명한다.
- 컬럼의 의미를 설명한다.
- 숫자/지표가 있으면 무엇을 의미하는지 설명한다.
- 실무자가 이 표를 언제 참조해야 하는지 설명한다.
- 주의해야 할 점을 정리한다.
- 검색에 잘 걸리도록 핵심 키워드를 포함한다.
- 결과는 Markdown 으로 작성한다.
- 다음 7개 섹션을 반드시 포함한다.

[필수 섹션]
1. 시트/표 개요
2. 주요 컬럼 설명
3. 주요 업무 맥락
4. 실무 사용 방법
5. 주의사항
6. 검색 키워드
7. 원본 참조 정보

[입력 메타]
- 파일명: {file_name}
- 시트명: {sheet_name}
- 표 범위: {table_range}

[원본 표 텍스트]
```
{raw_table_text}
```

위 내용을 바탕으로 한국어 Markdown 으로만 답하라.
헤더 (# 1. 시트/표 개요 등) 가 누락되지 않도록 한다.
값을 너무 많이 그대로 옮겨 적지 말고, 의미와 사용 맥락 위주로 설명한다."""


def build_excel_summary_prompt(
    *,
    file_name: str,
    sheet_name: str,
    table_range: str | None,
    raw_table_text: str,
) -> str:
    return EXCEL_SUMMARY_PROMPT_TEMPLATE.format(
        file_name=file_name,
        sheet_name=sheet_name,
        table_range=table_range or "-",
        raw_table_text=raw_table_text,
    )
