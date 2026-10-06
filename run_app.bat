@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto havevenv

where py >nul 2>&1
if errorlevel 1 (
  python -m venv .venv
) else (
  py -3 -m venv .venv
)

:havevenv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e ".[api]"
echo Starting LLM Eval Suite at http://127.0.0.1:8765
python -m llm_eval_suite.app
