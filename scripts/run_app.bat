@echo off
REM Windows: streamlit run 헬퍼
REM 사용: scripts\run_app.bat

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
cd /d "%ROOT_DIR%"

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
) else (
  echo [hint] .venv 가 없습니다. python -m venv .venv 로 만들어 주세요.
)

if not exist .env (
  echo [warn] .env 파일이 없습니다. copy .env.example .env 후 GOOGLE_API_KEY 입력 필요.
)

python -m streamlit run app\main.py
endlocal
