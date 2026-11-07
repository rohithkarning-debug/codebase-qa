# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `cqa/config.py` | Paths, chunk sizes, embedding models, LLM provider presets, `.env` loading |
| `cqa/repo.py` | `RepoSpec`, registry of repos, shallow clone at a pinned tag/commit, file walking, GitHub permalinks, path-traversal guard |
| `cqa/chunking/ast_chunker.py` | Syntax-tree chunking (Python `ast`, other languages via tree-sitter) + symbol table |
| `cqa/chunking/fixed_chunker.py` | Fixed-size sliding windows (baseline) |
| `cqa/embeddings.py` | Code vs general embedding models (fastembed / ONNX), offline hashing embedder for tests |
| `cqa/index.py` | Per-repo index: chunks + embedding matrices on disk, BM25, hybrid (RRF), search |
| `cqa/llm.py` | One chat client for Gemini, Groq, OpenAI, OpenRouter, Ollama (OpenAI-compatible) and Anthropic; retries, disk cache, offline mode |
| `cqa/agent/tools.py` | Agent tools: `search_code`, `grep`, `open_file`, `find_symbol`, `list_dir` |
| `cqa/agent/react.py` | ReAct loop (text protocol), streaming events, step budget, offline policy |
| `cqa/agent/rag.py` | Single-shot RAG baseline |
| `cqa/citations.py` | Extracts `path:Lx-Ly` citations, resolves short paths, validates them, builds links |
| `cqa/cli.py` | `python -m cqa ...` commands |

## Ingestion

1. `ensure_cloned()` does `git clone --depth 1 --branch <tag>` (or fetches a commit SHA) into `data/repos/<name>`
   and records the exact commit. Every GitHub link uses that commit, so line numbers never drift.
2. `iter_source_files()` walks the repo, keeps known code extensions and skips `tests/`, `docs/`, `examples/`,
   vendored code, hidden folders and files over 400 KB.

## Chunking strategies

**Syntax-tree (AST) chunking** - `chunk_ast()`

- Python: the stdlib `ast` gives exact `lineno`/`end_lineno`; decorators are included in the span.
- JS/TS, Java, Go, Rust, Ruby, C/C++, C#, PHP, Kotlin: tree-sitter grammars from `tree-sitter-language-pack`,
  with per-language rules for which node types are functions or classes (`_TS_RULES`). Arrow functions assigned
  to `const` count as functions; `export` wrappers are unwrapped.
- A class up to 80 lines is one chunk. A bigger class becomes a `class_header` chunk (signature, docstring,
  attributes) plus one `method` chunk per method, named `Class.method`.
- Functions over 160 lines are split into overlapping windows that keep the symbol name.
- Code outside any definition (imports, constants) becomes `module` chunks, so nothing is lost.
- A file that fails to parse falls back to fixed windows.

**Fixed-size chunking** - `chunk_fixed()`: 40-line windows with a 10-line overlap, ignoring syntax.

Both chunkers store the exact source lines, so a chunk's text always equals `lines[start-1:end]` (checked by the
tests). The text sent to the embedding model gets a small header (`# file: path` and, for AST chunks,
`# method: Class.method`).

## Retrieval

| Retriever | What it is |
|---|---|
| `code` | `jinaai/jina-embeddings-v2-base-code`, trained on code and code/text pairs (768-d) |
| `general` | `BAAI/bge-base-en-v1.5`, a general English model of similar size (768-d), with its query instruction |
| `bm25` | Okapi BM25 over identifier-aware tokens (`HTTPDigestAuth` -> `http`, `digest`, `auth`) |
| `hybrid` | Reciprocal-rank fusion of `code` and `bm25` |

Embeddings are L2-normalised, so search is a single matrix-vector product (cosine similarity). Matrices are
cached as `.npy` files per (repo, chunker, model).

## ReAct agent

The agent follows the ReAct pattern (Yao et al., 2022) with a plain-text protocol that works on every chat model:

```
Thought: ...
Action: grep
Action Input: {"pattern": "def rebuild_auth"}
Observation: src/requests/sessions.py:282: def rebuild_auth(self, prepared_request, response):
...
Final Answer: ... [src/requests/sessions.py:L282-L300] ...
```

- The first step is always a `search_code` call with the question (a seeded search), which shows the model
  the format and gives it candidates.
- Stop sequences (`Observation:`) stop the model from inventing tool output; the client also enforces them.
- Malformed turns get a corrective observation; after `max_steps` (8) the model is asked for its final answer.
- `open_file` accepts short paths (`auth.py`) and resolves them uniquely; all paths go through `safe_join`.
- Citations in the final answer are validated against the repo's file list; invalid ones are marked, valid ones
  become GitHub permalinks and in-app links.

`run_stream()` yields events (`start`, `step`, `thought`, `answer`) that the web app streams as NDJSON.

## Offline mode

With no LLM key, `offline_react_policy` plays the model: search, open the best hit, answer from it. The full
loop, parser, tools and citation handling still run. Tests use the same mechanism with scripted replies.
