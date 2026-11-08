# Evaluation method

## Test set

`eval/questions.jsonl`: **50 questions over 3 repositories**, each pinned to a release so line numbers are stable.

| Repository | Version | Questions |
|---|---|---|
| [psf/requests](https://github.com/psf/requests) | v2.32.3 | 17 |
| [pallets/flask](https://github.com/pallets/flask) | 3.0.3 | 17 |
| [encode/starlette](https://github.com/encode/starlette) | 0.38.2 | 16 |

Question types: **locate** ("where is X handled?", 26), **how** ("how does X work?", 17) and **explain**
("explain this function", 7).

Each question has:

- `gold`: one or more `(file, symbol)` pairs, such as `("src/requests/auth.py", "HTTPDigestAuth.handle_401")`.
  They are resolved to exact line spans with the syntax tree of the pinned commit (`eval/dataset.py`), so the
  test set is readable and its line numbers cannot be wrong. `tests/test_dataset.py` checks that every symbol
  resolves and that the line really starts that definition.
- `answer`: a reference answer written after reading the code, used by the LLM judge.

## Retrieval metrics (`eval/run_retrieval_eval.py`)

For each of the 8 indexes (`{ast, fixed} x {code, general, bm25, hybrid}`) we retrieve the top 10 chunks.

**Hit rule.** A chunk is relevant if it is in a gold file and
`overlap(chunk, gold) / min(len(chunk), len(gold)) >= 0.5`. A 40-line window that contains a 10-line gold method
counts; a window that only touches its last 3 lines does not. The same rule applies to both chunkers.

- **Recall@k** (k = 1, 3, 5, 10): share of questions with at least one relevant chunk in the top k.
- **MRR@10**: mean of 1 / rank of the first relevant chunk.
- **File recall@k** and **top-1 file accuracy**: same, but only the file has to match.

Results are reported overall, per repository and per question type.

## Answer metrics (`eval/run_answer_eval.py`)

Both answer modes use the same index (default: AST chunks + code model):

- **RAG**: retrieve 6 chunks once, answer.
- **Agent**: the ReAct agent with search, grep, open_file, find_symbol and list_dir (max 8 actions).

For each answer:

- **LLM-as-Judge correctness**: the judge sees the question, the reference answer, the gold locations and the
  candidate, and returns `{"score": 1-5, "correct": score >= 4, "reason": ...}` at temperature 0
  (`eval/judge.py`). Set `CQA_JUDGE_PROVIDER` / `CQA_JUDGE_MODEL` to use a different model as judge.
- **File-location accuracy**: the answer cites at least one gold file.
- **Line accuracy**: a cited line range overlaps a gold span.
- **First-citation accuracy** and **citation precision**: is the first link right, and what share of cited
  files are gold files.

## Reproducing

```bash
python -m cqa setup && python -m cqa index --all
python -m cqa eval retrieval
python -m cqa eval answers [--limit 10] [--workers 2]
python -m cqa eval report        # -> results/REPORT.md, results/charts/
```

## Threats to validity

- 50 questions is small: a difference of one question is 2 percentage points.
- Three Python repositories; the tree-sitter path for other languages is tested but not part of the benchmark.
- The judge is itself an LLM. Its prompt includes a reference answer and gold locations to keep it grounded; the
  per-question table in the app shows each verdict and reason for spot checks.
- The AST chunker also prepends the symbol name to the embedded text, which is part of its advantage by design.
