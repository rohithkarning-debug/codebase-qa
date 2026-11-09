#!/usr/bin/env bash
# One command for the demo:
#   virtualenv -> dependencies -> clone the 3 repos -> build indexes -> retrieval eval -> web app
#
#   ./run_demo.sh              full setup (first run downloads ~1 GB of embedding models)
#   SKIP_INDEX=1 ./run_demo.sh  just start the web app (indexes already built)
#   PORT=9000 ./run_demo.sh     use another port
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
PORT="${PORT:-8000}"

say() { printf "\n\033[1;32m==>\033[0m %s\n" "$*"; }

if [ ! -d .venv ]; then
  say "Creating virtual environment (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

say "Installing dependencies"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  say "Created .env - add GEMINI_API_KEY or GROQ_API_KEY to it for full LLM answers (optional)"
fi

if [ "${SKIP_INDEX:-0}" != "1" ]; then
  say "Cloning the evaluation repositories (requests, flask, starlette)"
  python -m cqa setup
  say "Building indexes: {function, fixed} chunks x {code, general} embeddings (first run takes a while)"
  python -m cqa index --all
  if [ ! -f results/retrieval.json ]; then
    say "Running the retrieval evaluation (Recall@k, no LLM needed)"
    python -m cqa eval retrieval
  fi
fi

python -m cqa doctor || true

say "Opening http://127.0.0.1:${PORT}"
( sleep 2
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:${PORT}"
  elif command -v open >/dev/null 2>&1; then open "http://127.0.0.1:${PORT}"; fi ) >/dev/null 2>&1 &
python -m cqa serve --port "$PORT"
