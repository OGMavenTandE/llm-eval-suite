@echo off
setlocal
cd /d "%~dp0"
rem Pins match llm_eval_suite\eval_env.py: torch 2.13.0, garak 0.17.0, transformers 5.18.0.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if exist ".venv-eval\Scripts\python.exe" goto havevenv

where py >nul 2>&1
if errorlevel 1 (
  python -m venv .venv-eval
) else (
  py -3 -m venv .venv-eval
)
if errorlevel 1 (
  echo Failed to create .venv-eval
  exit /b 1
)

:havevenv
call ".venv-eval\Scripts\activate.bat"
python -m pip install --upgrade pip
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo WARNING: No NVIDIA GPU was detected ^(nvidia-smi not found^). Installing the CPU build of torch. Garak and local Hugging Face models will be slow.
  python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
) else (
  echo NVIDIA GPU detected. Installing torch 2.13.0 from the CUDA 12.6 index.
  python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
)
if errorlevel 1 (
  echo torch install failed.
  exit /b 1
)
python -m pip install -e ".[api]"
if errorlevel 1 exit /b 1
rem Keep the torch build already installed. --extra-index-url stops pip from swapping in a PyPI CPU wheel.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  python -m pip install "garak==0.17.0" "transformers==5.18.0" "torch==2.13.0" --extra-index-url https://download.pytorch.org/whl/cpu
) else (
  python -m pip install "garak==0.17.0" "transformers==5.18.0" "torch==2.13.0" --extra-index-url https://download.pytorch.org/whl/cu126
)
if errorlevel 1 (
  echo garak install failed.
  exit /b 1
)
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
  if errorlevel 1 (
    echo WARNING: torch does not see CUDA. Generations will use the CPU.
  )
)
echo Starting LLM Eval Suite at http://127.0.0.1:8765
python -m llm_eval_suite.app
