@echo off
REM One command for the demo on Windows (same steps as run_demo.sh)
cd /d %~dp0
if not exist .venv (
  echo ==^> Creating virtual environment
  py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate.bat
echo ==^> Installing dependencies
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
if not exist .env (
  copy .env.example .env >nul
  echo ==^> Created .env - add GEMINI_API_KEY or GROQ_API_KEY for full LLM answers
)
if not "%SKIP_INDEX%"=="1" (
  echo ==^> Cloning the evaluation repositories
  python -m cqa setup
  echo ==^> Building indexes - first run downloads the embedding models
  python -m cqa index --all
  if not exist results\retrieval.json python -m cqa eval retrieval
)
python -m cqa doctor
start "" http://127.0.0.1:8000
python -m cqa serve --port 8000
