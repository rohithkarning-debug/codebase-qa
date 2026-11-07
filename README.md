# Codebase QA: ask questions about a GitHub repository

Point the assistant at a GitHub repository and ask things like *"where is authentication handled?"* or
*"explain this function"*. Every answer links to the exact file and line range, both inside the app
(a code viewer with the cited lines highlighted) and on GitHub (permalinks pinned to the indexed commit).

| Deliverable | Where |
|---|---|
| Code ingestion: **syntax-tree chunking** (function / class / method) vs **fixed-size chunking** | `cqa/chunking/` |
| Retrieval: **code-specific embedding model** vs **general model** (+ BM25 and hybrid baselines) | `cqa/embeddings.py`, `cqa/index.py` |
| **ReAct agent** that can search, `grep`, open files and look up symbols | `cqa/agent/` |
| **50-question test set** over 3 repositories (requests, flask, starlette) | `eval/questions.jsonl` |
| **Recall@k**, **LLM-as-Judge correctness**, **file-location accuracy** | `eval/` → `results/REPORT.md` |
| Web app for the demo (ask, live agent trace, compare retrieval, evaluation dashboard, add repos) | `server/`, `web/` |

## Quick start

```bash
git clone <this repo> && cd codebase-qa
./run_demo.sh            # Windows: run_demo.bat
```

That one script creates a virtualenv, installs the dependencies, clones the three evaluation repositories at
pinned versions, builds all indexes, runs the retrieval evaluation and opens **http://127.0.0.1:8000**.
The first run downloads the two embedding models (about 1 GB) and embeds ~6,000 chunks on the CPU, which takes a
while; later runs start in seconds (`SKIP_INDEX=1 ./run_demo.sh`).

### LLM key (free)

Copy `.env.example` to `.env` and set **one** key:

| Provider | Free key | Default model |
|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | `gemini-flash-latest` |
| Groq | https://console.groq.com/keys | `openai/gpt-oss-120b` |
| OpenAI / Anthropic / OpenRouter | paid | `gpt-4o-mini` / `claude-haiku-5-5` / Qwen coder |
| Ollama (local) | none - `CQA_LLM_PROVIDER=ollama` | `qwen2.5-coder:7b` |

Model names change often; override with `CQA_LLM_MODEL=...`. Without any key the app still works in
**offline mode**: the agent loop, tools, search and citations run, and the final answer is assembled from the
best search hit instead of written by an LLM. Run `python -m cqa doctor` to check everything before a demo.

## Using it

**Web app** (`python -m cqa serve`):

- **Ask**: type a question or click an example. In *Agent* mode you watch each step live (`search_code`,
  `grep`, `open_file`, `find_symbol`); the answer's citations open the code viewer with those lines highlighted.
  Switch to *Single retrieval* (RAG), *Fixed windows* or another embedding model to compare.
- **Compare retrieval**: one question against all 8 indexes side by side; for test-set questions the
  correct chunks are marked.
- **Evaluation**: Recall@k chart and tables, answer accuracy per mode, and every question with its answers and
  the judge's reasoning.
- **Repositories**: add any public GitHub repo by URL; it is cloned and indexed in the background.

**Command line**:

```bash
python -m cqa ask -r requests "Where is HTTP Basic authentication handled?"
python -m cqa ask -r flask "How does Flask dispatch a request?" --mode rag --chunker fixed
python -m cqa search -r starlette "path traversal static files" --retriever general
python -m cqa add https://github.com/encode/httpx --ref 0.27.0
python -m cqa eval all          # retrieval + answers + results/REPORT.md
```

## How it works

```
GitHub repo ──clone (pinned commit)──► source files
      │
      ├─► AST chunker  (Python ast / tree-sitter: one chunk per function, class, method)
      └─► Fixed chunker (40-line windows, 10-line overlap)
               │
               ├─► code embeddings    jinaai/jina-embeddings-v2-base-code
               ├─► general embeddings BAAI/bge-base-en-v1.5
               └─► BM25 / hybrid (reciprocal-rank fusion)
                         │
question ──► ReAct agent ─┤ Thought → Action → Observation … → Final Answer
                          │   tools: search_code · grep · open_file · find_symbol · list_dir
                          ▼
              answer with [path:Lstart-Lend] citations ──► validated ──► GitHub permalinks + code viewer
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Evaluation method: [docs/EVALUATION.md](docs/EVALUATION.md).
API: [docs/API.md](docs/API.md). Demo script for the presentation: [docs/DEMO.md](docs/DEMO.md).

## Evaluation

```bash
python -m cqa eval retrieval    # 2 chunkers x 4 retrievers, Recall@1/3/5/10, MRR, top-1 file accuracy (no LLM)
python -m cqa eval answers      # RAG vs agent: LLM-as-Judge + file/line-location accuracy (needs an LLM key)
python -m cqa eval report       # results/REPORT.md + results/charts/*.png
```

Answer evaluation makes roughly 400 LLM calls for 50 questions x 2 modes plus judging. Responses are cached
in `data/cache/llm`, and an interrupted run resumes where it stopped. Use `--limit 10` for a quick check or
`--workers 2` to go faster if your rate limit allows it.

## Project structure and team

| Owner | Area | Files |
|---|---|---|
| **Rohith** (lead) | Configuration, syntax-tree (AST) and fixed-size chunking, ReAct agent and its tools, RAG baseline, test fixtures, README, architecture | `cqa/config.py`, `cqa/chunking/`, `cqa/agent/`, `tests/conftest.py`, `tests/test_chunking.py`, `tests/test_agent.py`, `docs/ARCHITECTURE.md` |
| **Siddhardha** | Repository ingestion, multi-provider LLM client, FastAPI backend with streaming answers and indexing jobs, CLI and demo scripts | `cqa/repo.py`, `cqa/llm.py`, `cqa/cli.py`, `server/`, `run_demo.*`, `Makefile`, `tests/test_repo.py`, `tests/test_llm.py`, `tests/test_api.py`, `docs/API.md` |
| **Rohanth** | Web UI (live agent trace, code viewer, compare and evaluation dashboards), citation parsing and links, demo script | `web/`, `cqa/citations.py`, `tests/test_citations.py`, `tests/test_frontend.py`, `docs/DEMO.md` |
| **Nischkal** | Code vs general embedding models, search index (dense, BM25, hybrid), 50-question test set, metrics, LLM judge, evaluation runners and report | `cqa/embeddings.py`, `cqa/index.py`, `eval/`, `tests/test_retrieval.py`, `tests/test_metrics.py`, `tests/test_dataset.py`, `docs/EVALUATION.md` |

## Tests

```bash
python -m pytest        # ~70 tests, fully offline (hashing embedder + scripted LLM), runs in seconds
```

## Troubleshooting

- **Model download fails / no internet**: `CQA_EMBED_BACKEND=hash` uses an offline embedder so the app still
  runs (numbers are then not the real model comparison).
- **`429` from the LLM**: free tiers are rate limited; the client retries with back-off. For the answer
  evaluation use `--workers 1` and, if needed, `--limit`.
- **A model id was retired**: set `CQA_LLM_MODEL` in `.env` to a current model of your provider.
- **Port in use**: `PORT=9000 ./run_demo.sh` or `python -m cqa serve --port 9000`.
