"""
5_api_status.py
===============
Gemini API 연결 / Embedding / Generation / 모델 목록 진단 페이지.

이 페이지는 사용자가 버튼을 눌렀을 때만 외부 API 를 호출한다.
페이지를 단순히 열기만 해서는 호출이 발생하지 않는다.
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.status_card import render_api_status
from src.config import settings
from src.rag.embedder import Embedder
from src.rag.gemini_client import GeminiClient, GeminiError

st.set_page_config(page_title="API 상태확인", layout="wide")
render_sidebar()

st.title("API 상태확인")
render_api_status()

st.divider()

# ----------------------------- 연결 테스트 ---------------------------------
st.subheader("1. Gemini Generation 연결 테스트")
if st.button("Generation API ping 보내기"):
    if not settings.has_api_key():
        st.error("GOOGLE_API_KEY 가 비어 있습니다.")
    else:
        try:
            client = GeminiClient()
            with st.spinner("호출 중..."):
                res = client.check_connection()
            if res.get("ok"):
                st.success(f"OK · model={res.get('model')}")
                st.caption(f"응답 미리보기: {res.get('preview')}")
            else:
                st.error(f"실패: {res.get('error')}")
        except GeminiError as e:
            st.error(f"Gemini 오류: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"예외: {e}")

st.divider()

st.subheader("2. Embedding 테스트")
sample = st.text_input("임베딩 테스트 문장", value="Meta 캠페인 세팅 시 검수 항목")
if st.button("임베딩 1회 호출"):
    try:
        emb = Embedder()
        with st.spinner("호출 중..."):
            vec = emb.embed_query(sample)
        st.success(
            f"OK · provider={emb.provider} · model={emb.model_name} · dim={len(vec)}"
        )
        st.caption(f"앞 8개 값: {vec[:8]}")
    except GeminiError as e:
        st.error(f"Gemini 오류: {e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"예외: {e}")

st.divider()

st.subheader("3. 사용 가능한 모델 목록")
if st.button("models.list 호출"):
    try:
        client = GeminiClient()
        names = client.list_models()
        if not names:
            st.warning(
                "모델 목록을 가져오지 못했거나 비어 있습니다. "
                "API Key/네트워크/SDK 버전을 확인하세요."
            )
        else:
            st.success(f"{len(names)} 개 모델 사용 가능")
            st.code("\n".join(names))
    except Exception as e:  # noqa: BLE001
        st.error(f"실패: {e}")

st.divider()

st.subheader("자주 발생하는 오류")
st.markdown(
    """
- `GOOGLE_API_KEY is missing` : `.env` 의 키 입력
- `model not found` : `.env` 의 `GENERATION_MODEL` / `EMBEDDING_MODEL` 을 사용 가능한 모델로 교체
- `quota exceeded` : 잠시 후 재시도, 또는 한도/요금제 확인
- `timeout` / `connection error` : 네트워크/방화벽 확인
"""
)
