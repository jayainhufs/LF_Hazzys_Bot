"""
src.slack_bot
=============
Slack QA Bot MVP.

Slack 채널에서 봇이 멘션되면 기존 ``src.rag.qa_pipeline`` 을 호출해 답변을
thread 로 포스트하는 얇은 adapter 패키지다.

설계 원칙
---------
- 기존 Streamlit UI 는 변경하지 않는다. Streamlit 은 관리자/운영 콘솔
  (문서 업로드/색인/정규화 문서 관리/검색 테스트/API 상태 확인) 역할을 그대로
  유지하고, Slack Bot 은 사용자 질문 인터페이스 역할만 담당한다.
- 자체 RAG 검색/답변 로직을 구현하지 않는다. 기존 retriever / reranker /
  prompt_builder / qa_pipeline 을 그대로 재사용한다.
- Slack token 등 비밀값은 항상 환경변수에서만 읽으며, 코드에 하드코딩하지
  않는다.
"""
