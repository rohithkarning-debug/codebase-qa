"""Answer evaluation: RAG baseline vs ReAct agent.

For each of the 50 questions and each mode it records the answer, its
citations, file-location accuracy and the LLM-as-Judge verdict.

    python -m eval.run_answer_eval --modes rag,agent [--limit 10] [--workers 2]

Results: results/answers.json (summary) and results/answers_detail.jsonl.
Interrupted runs resume where they stopped (finished rows are reused).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from cqa import config
from cqa.agent import RAGAnswerer, ReActAgent
from cqa.agent.rag import offline_rag_policy
from cqa.agent.react import offline_react_policy
from cqa.llm import make_llm
from cqa.repo import ensure_cloned, get_repo

from .dataset import load_questions
from .judge import Judge
from .metrics import avg, citation_scores

DETAIL_FILE = "answers_detail.jsonl"


def _summ(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "judge_correct": avg(r["judge"]["correct"] for r in rows),
        "judge_score": avg(r["judge"]["score"] for r in rows),
        "file_acc": avg(r["cite"]["file_hit"] for r in rows),
        "line_acc": avg(r["cite"]["line_hit"] for r in rows),
        "top_file_acc": avg(r["cite"]["top_file"] for r in rows),
        "citation_precision": avg(r["cite"]["precision"] for r in rows),
        "avg_steps": avg(r.get("steps", 0) for r in ok),
        "avg_seconds": avg(r.get("seconds", 0) for r in ok),
    }


def answer_one(q, mode: str, chunker: str, retriever: str, provider=None, model=None) -> dict:
    spec = get_repo(q.repo)
    t0 = time.time()
    try:
        if mode == "rag":
            res = RAGAnswerer(spec, make_llm(provider, model, offline_policy=offline_rag_policy),
                              chunker, retriever).run(q.question)
        else:
            res = ReActAgent(spec, make_llm(provider, model, offline_policy=offline_react_policy),
                             chunker, retriever).run(q.question)
        return {"answer": res["answer"], "citations": res["citations"], "steps": res.get("steps", 0),
                "seconds": res.get("seconds", round(time.time() - t0, 2)), "llm": res.get("llm"),
                "trace": [{k: s[k] for k in ("action", "input")} for s in res.get("trace", [])]}
    except Exception as e:
        traceback.print_exc()
        return {"answer": "", "citations": [], "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 2)}


def main(modes=("rag", "agent"), limit=None, chunker="ast", retriever="code", workers=1,
         provider=None, model=None, fresh=False) -> dict:
    questions = load_questions(limit=limit)
    for repo in {q.repo for q in questions}:
        ensure_cloned(get_repo(repo))
    js = config.judge_settings()
    judge = Judge(make_llm(js.provider, js.model))
    agent_llm = make_llm(provider, model).name
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = config.RESULTS_DIR / DETAIL_FILE

    done: dict[tuple, dict] = {}
    if detail_path.exists() and not fresh:
        for line in detail_path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                done[(r["mode"], r["id"], r["llm"], r["chunker"], r["retriever"])] = r

    rows: list[dict] = []
    for mode in modes:
        print(f"\n== mode={mode}  llm={agent_llm}  judge={judge.name}  index={chunker}+{retriever}")

        def work(q):
            key = (mode, q.id, agent_llm, chunker, retriever)
            if key in done:
                return done[key]
            res = answer_one(q, mode, chunker, retriever, provider, model)
            verdict = (judge.grade(q, res["answer"], res["citations"]) if not res.get("error")
                       else {"score": 1, "correct": False, "reason": "error"})
            row = {"mode": mode, "id": q.id, "repo": q.repo, "type": q.type, "question": q.question,
                   "llm": agent_llm, "judge_model": judge.name, "chunker": chunker, "retriever": retriever,
                   **res, "cite": citation_scores(res["citations"], q.gold), "judge": verdict,
                   "gold": [g.to_dict() for g in q.gold]}
            with open(detail_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            return row

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, row in enumerate(ex.map(work, questions), 1):
                rows.append(row)
                mark = "ok " if row["judge"]["correct"] else "-- "
                print(f"  [{i:>2}/{len(questions)}] {mark}{row['id']:<7} score={row['judge']['score']} "
                      f"file={'Y' if row['cite']['file_hit'] else 'n'} line={'Y' if row['cite']['line_hit'] else 'n'}"
                      f"  {row.get('error', '')[:80]}")

    summary = {}
    for mode in modes:
        mrows = [r for r in rows if r["mode"] == mode]
        s = _summ(mrows)
        s["by_repo"] = {r: _summ([x for x in mrows if x["repo"] == r]) for r in sorted({x["repo"] for x in mrows})}
        s["by_type"] = {t: _summ([x for x in mrows if x["type"] == t]) for t in sorted({x["type"] for x in mrows})}
        summary[mode] = s
        print(f"  {mode:<6} correct={s['judge_correct']:.2f} score={s['judge_score']:.2f} "
              f"file_acc={s['file_acc']:.2f} line_acc={s['line_acc']:.2f}")

    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "llm": agent_llm, "judge": judge.name,
              "index": f"{chunker}+{retriever}", "embed_backend": config.EMBED_BACKEND,
              "n_questions": len(questions), "modes": summary,
              "rows": [{k: r[k] for k in ("mode", "id", "repo", "type", "question", "answer", "citations",
                                          "cite", "judge", "gold") if k in r} for r in rows]}
    (config.RESULTS_DIR / "answers.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {config.RESULTS_DIR / 'answers.json'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="rag,agent")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunker", default="ast")
    ap.add_argument("--retriever", default="code")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore previously saved rows")
    a = ap.parse_args()
    main(a.modes.split(","), a.limit, a.chunker, a.retriever, a.workers, a.provider, a.model, a.fresh)
