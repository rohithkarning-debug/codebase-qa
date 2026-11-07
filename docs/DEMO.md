# Demo script (about 7 minutes)

**Author / Presenter**: Rohanth Ganesha C

## Before the presentation

1. `./run_demo.sh` once at home (downloads models, builds indexes, runs the retrieval eval).
2. Put an LLM key in `.env`, then run `python -m cqa eval answers` so the Evaluation tab has real judge scores.
   Every answer is cached, so the demo questions below answer instantly even on slow Wi-Fi.
3. Run `python -m cqa doctor`: every line should say `[ok]`.
4. On the day: `SKIP_INDEX=1 ./run_demo.sh` and keep the browser at 100% zoom.

## 1. The problem (30 s)

"New engineers spend days finding where things happen in an unfamiliar codebase. We built an assistant that
answers with links to the exact lines."

## 2. Ask (2 min) - Ask tab, repository `starlette`

1. Click the example **"Where is authentication handled?"** (or type it).
2. Point at the steps as they appear: the agent first runs a semantic search, then decides to `grep` or
   `open_file` - this is the ReAct loop (Thought, Action, Observation).
3. When the answer arrives, the code viewer opens on the right with the cited lines highlighted. Click a second
   citation to jump; click the GitHub icon to show the permalink pinned to the indexed commit.
4. Switch the repository to `requests`, ask **"Explain what Response.raise_for_status does."** - an
   "explain this function" question.

## 3. Chunking and embeddings (2 min) - Compare retrieval tab

1. Pick test-set question **req-03** (stripping the Authorization header on redirects).
2. Eight result lists appear: function chunks vs fixed windows, each with the code model, the general model,
   BM25 and hybrid. Green rows are chunks that contain the expected code.
3. Point out that function chunks are whole methods (`SessionRedirectMixin.should_strip_auth`) while fixed
   windows cut through functions.

## 4. Results (2 min) - Evaluation tab

1. The findings list states the headline: best index, AST vs fixed difference, code vs general model.
2. Toggle k = 1 / 5 / 10 on the chart.
3. Answers table: agent vs single retrieval on LLM-judged correctness and file-location accuracy.
4. Click a question row to show the answer, the judge's reason and the expected code.

## 5. Any repository (30 s) - Repositories tab

Paste a small repo (for example `https://github.com/pallets/itsdangerous`) and click **Add and index** - the
progress bar shows cloning, chunking and embedding. Then ask a question about it.

## Backup plan

- No internet: answers for the prepared questions come from the LLM cache. If the key or network fails entirely,
  set `CQA_LLM_PROVIDER=offline` - the agent loop still runs and cites code.
- Show `results/REPORT.md` and `results/charts/*.png` if the browser misbehaves.
