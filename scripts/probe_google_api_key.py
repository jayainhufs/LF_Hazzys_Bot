# scripts/probe_google_api_key.py
"""
Google API Key 진단 스크립트

목적:
- .env의 GOOGLE_API_KEY가 어떤 Google API에서 동작하는지 최소 호출로 확인
- Gemini / Generative Language API 사용 가능 여부 확인
- Vertex AI API Key 방식 접근 가능 여부 확인
- Sheets / Drive 등 일반 Google API 접근 가능 여부 확인
- 에러 타입(API_KEY_INVALID, PERMISSION_DENIED, API_NOT_ENABLED 등) 구분

주의:
- API Key 전체를 출력하지 않는다.
- 일부 호출은 API 사용량에 기록될 수 있다.
- 민감한 회사 API Key를 외부에 공유하지 말 것.
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import Any

import requests
from dotenv import load_dotenv


TIMEOUT = 20


def mask_key(key: str | None) -> str:
    if not key:
        return "(not set)"
    key = key.strip()
    if len(key) <= 12:
        return key[:4] + "***"
    return f"{key[:6]}...{key[-4:]} (len={len(key)})"


def extract_error(resp: requests.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception:
        return {
            "status_code": resp.status_code,
            "raw_text": resp.text[:500],
        }

    err = data.get("error", data)
    return {
        "status_code": resp.status_code,
        "error_code": err.get("code"),
        "status": err.get("status"),
        "message": err.get("message"),
        "reason": _extract_reason(err),
        "raw": err,
    }


def _extract_reason(err: dict[str, Any]) -> str | None:
    details = err.get("details") or []
    for d in details:
        if isinstance(d, dict):
            reason = d.get("reason")
            if reason:
                return reason
            metadata = d.get("metadata") or {}
            if metadata.get("reason"):
                return metadata.get("reason")
    return None


def print_result(name: str, ok: bool, detail: dict[str, Any] | str) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("RESULT:", "OK" if ok else "FAIL")
    if isinstance(detail, str):
        print(detail)
    else:
        print(json.dumps(detail, ensure_ascii=False, indent=2)[:3000])


def get(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
    return requests.get(url, params=params, headers=headers, timeout=TIMEOUT)


def post(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, json_body: dict[str, Any] | None = None):
    return requests.post(url, params=params, headers=headers, json=json_body, timeout=TIMEOUT)


def test_gemini_models_list(api_key: str) -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = get(url, params={"key": api_key})
    if resp.ok:
        data = resp.json()
        models = data.get("models", [])
        preview = [m.get("name") for m in models[:10]]
        print_result(
            "1) Gemini Developer API - models.list",
            True,
            {
                "message": "Generative Language API 호출 성공",
                "model_count_preview": len(models),
                "first_models": preview,
            },
        )
    else:
        print_result("1) Gemini Developer API - models.list", False, extract_error(resp))


def test_gemini_generate(api_key: str) -> None:
    # 모델명을 너무 최신 고정값 하나만 쓰면 모델명 문제와 키 문제를 구분하기 어려워서
    # 비교적 흔한 flash 계열 이름을 순차 테스트한다.
    candidate_models = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": "Reply with exactly: pong"}
                    ]
                }
            ]
        }
        resp = post(
            url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json_body=body,
        )

        if resp.ok:
            data = resp.json()
            print_result(
                f"2) Gemini Developer API - generateContent ({model})",
                True,
                {
                    "message": "텍스트 생성 호출 성공",
                    "response_preview": str(data)[:1000],
                },
            )
            return

        err = extract_error(resp)
        # API_KEY_INVALID면 다른 모델도 다 실패할 가능성이 높으므로 바로 중단
        if err.get("reason") == "API_KEY_INVALID" or "API key not valid" in str(err.get("message")):
            print_result(f"2) Gemini Developer API - generateContent ({model})", False, err)
            return

        print_result(f"2) Gemini Developer API - generateContent ({model})", False, err)

    print("\n모든 candidate generation 모델 테스트 실패")


def test_gemini_embedding(api_key: str) -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    body = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [
                {"text": "테스트 문장입니다."}
            ]
        }
    }
    resp = post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json_body=body,
    )
    if resp.ok:
        data = resp.json()
        values = data.get("embedding", {}).get("values", [])
        print_result(
            "3) Gemini Embedding API - embedContent",
            True,
            {
                "message": "Embedding 호출 성공",
                "embedding_dim": len(values),
                "first_values_preview": values[:5],
            },
        )
    else:
        print_result("3) Gemini Embedding API - embedContent", False, extract_error(resp))


def test_vertex_ai_models(api_key: str, project_number: str | None) -> None:
    # Vertex AI API Key 방식 확인.
    # 주의: 실제 Vertex AI는 보통 OAuth/ADC/Service Account를 더 많이 사용한다.
    # API Key만으로 안 되는 환경이면 401/403이 정상적으로 나올 수 있다.
    project = project_number or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
    if not project:
        print_result(
            "4) Vertex AI API - publisher models list",
            False,
            "PROJECT_NUMBER 또는 GOOGLE_CLOUD_PROJECT가 없어 테스트를 건너뜁니다.",
        )
        return

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models"
    resp = get(url, params={"key": api_key})

    if resp.ok:
        data = resp.json()
        models = data.get("publisherModels", data.get("models", []))
        print_result(
            "4) Vertex AI API - publisher models list",
            True,
            {
                "message": "Vertex AI API Key 방식 호출 성공",
                "project": project,
                "location": location,
                "preview": str(models[:3])[:1000],
            },
        )
    else:
        print_result("4) Vertex AI API - publisher models list", False, extract_error(resp))


def test_sheets_discovery(api_key: str) -> None:
    # Sheets API가 키 restriction에 포함되어 있는지 간접 확인.
    # discovery 문서 접근은 실제 사용자 데이터 접근이 아니다.
    url = "https://sheets.googleapis.com/$discovery/rest"
    resp = get(url, params={"version": "v4", "key": api_key})
    if resp.ok:
        data = resp.json()
        print_result(
            "5) Google Sheets API - discovery",
            True,
            {
                "message": "Sheets API discovery 접근 성공",
                "title": data.get("title"),
                "version": data.get("version"),
            },
        )
    else:
        print_result("5) Google Sheets API - discovery", False, extract_error(resp))


def test_drive_discovery(api_key: str) -> None:
    url = "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"
    resp = get(url, params={"key": api_key})
    if resp.ok:
        data = resp.json()
        print_result(
            "6) Google Drive API - discovery",
            True,
            {
                "message": "Drive API discovery 접근 성공",
                "title": data.get("title"),
                "version": data.get("version"),
            },
        )
    else:
        print_result("6) Google Drive API - discovery", False, extract_error(resp))


def test_geocoding_simple(api_key: str) -> None:
    # Maps Geocoding API 테스트. 키가 Maps 계열인지 보는 용도.
    # 성공하더라도 과금/쿼터 대상일 수 있으므로 원치 않으면 아래 호출을 주석 처리.
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = get(url, params={"address": "Seoul", "key": api_key})
    if resp.ok:
        data = resp.json()
        status = data.get("status")
        ok = status in {"OK", "ZERO_RESULTS"}
        print_result(
            "7) Google Maps Geocoding API - simple call",
            ok,
            {
                "maps_status": status,
                "message": data.get("error_message"),
                "result_count": len(data.get("results", [])),
            },
        )
    else:
        print_result("7) Google Maps Geocoding API - simple call", False, extract_error(resp))


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    project_number = os.getenv("GOOGLE_PROJECT_NUMBER") or os.getenv("PROJECT_NUMBER")

    print("=" * 80)
    print("Google API Key Probe")
    print("=" * 80)
    print("GOOGLE_API_KEY:", mask_key(api_key))
    print("GOOGLE_PROJECT_NUMBER:", project_number or "(not set)")
    print("GOOGLE_CLOUD_LOCATION:", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))

    if not api_key:
        print("\n.env에 GOOGLE_API_KEY가 없습니다.")
        return

    print(
        textwrap.dedent(
            """
            해석 가이드:
            - Gemini에서 API_KEY_INVALID가 나오면 현재 키는 Generative Language API에 유효하지 않습니다.
            - PERMISSION_DENIED / API has not been used면 프로젝트에서 API 활성화 또는 제한 설정 문제일 수 있습니다.
            - Vertex AI가 실패해도 회사가 OAuth/Service Account 방식만 허용하는 경우일 수 있습니다.
            - Discovery API 성공은 해당 API의 실제 사용자 데이터 접근 권한을 의미하지 않습니다.
            """
        ).strip()
    )

    test_gemini_models_list(api_key)
    test_gemini_generate(api_key)
    test_gemini_embedding(api_key)
    test_vertex_ai_models(api_key, project_number)
    test_sheets_discovery(api_key)
    test_drive_discovery(api_key)

    # Maps 호출은 실제 과금 API일 수 있어 기본 주석 처리하고 싶으면 아래 줄을 주석 처리하세요.
    test_geocoding_simple(api_key)


if __name__ == "__main__":
    main()