#!/usr/bin/env bash
# Mac/Linux: streamlit run 헬퍼
# 사용:  bash scripts/run_app.sh

set -euo pipefail

# 스크립트 위치 기준 프로젝트 루트로 이동
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${ROOT_DIR}"

# .venv 있으면 안내
if [ -d ".venv" ]; then
  echo "[hint] 가상환경이 감지되었습니다. 활성화 권장:"
  echo "       source .venv/bin/activate"
fi

# .env 가 없으면 안내
if [ ! -f ".env" ]; then
  echo "[warn] .env 파일이 없습니다. cp .env.example .env 후 GOOGLE_API_KEY 입력하세요."
fi

exec python -m streamlit run app/main.py
