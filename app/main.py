"""
app/main.py
===========
Streamlit 진입점 (홈 화면).

실행:
    streamlit run app/main.py
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401  - 반드시 src import 보다 먼저
import platform

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.status_card import render_api_status, render_status_grid
from src.config import settings

st.set_page_config(
    page_title=settings.app_name,
    layout="wide",
)

render_sidebar()

st.title(settings.app_name)
st.caption(
    "광고대행사 퍼포먼스마케팅 인턴을 위한 로컬 RAG 챗봇. "
    "Slack/가이드/카톡/Excel 자료를 바탕으로 업무 처리 순서와 체크리스트를 답변합니다."
)

st.markdown(
    """
이 앱은 **로컬 PC** 에서 동작하며, 외부 호출은 **Google Gemini API** 만 사용합니다.

- 모든 원본 / chunk / Vector DB 는 로컬에 저장됩니다.
- 외부 Vector DB (Pinecone, Weaviate, Qdrant Cloud 등) 는 사용하지 않습니다.
- 자체 LLM 추론 (Ollama 등) 은 사용하지 않습니다.
- Slack API / Slack Bot 도 사용하지 않습니다 (대화는 직접 복사한 텍스트 파일로 적재).
"""
)

st.divider()
st.subheader("시스템 상태")
render_status_grid()

st.divider()
st.subheader("API / Embedding 설정")
render_api_status()

st.divider()
st.subheader("사용 흐름")
st.markdown(
    """
1. **1_문서_업로드** : Slack / 가이드 / 카톡 / Excel 자료 업로드
2. **2_문서_색인** : 새 파일만 색인 (옵션: Excel 한국어 요약 생성)
3. **3_업무_QA** : 자연어로 업무 질문 후 근거 기반 답변
4. **4_검색_테스트** : LLM 호출 없이 검색 결과만 확인 (비용 절약)
5. **5_API_상태확인** : Gemini API 연결 / 모델 목록 확인
6. **6_Excel_요약관리** : Excel 한국어 요약 미리보기 / 재생성
"""
)

st.divider()
with st.expander("실행 환경 정보"):
    st.write({
        "Python": platform.python_version(),
        "OS": f"{platform.system()} {platform.release()}",
        "Project Root": str(settings.project_root),
        "Chroma DB": str(settings.chroma_db_dir),
        "Raw Data": str(settings.raw_data_dir),
        "Processed": str(settings.processed_data_dir),
    })
