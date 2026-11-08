"""Retrieval evaluation: {AST, fixed} chunking x {code, general, BM25, hybrid} retrieval.

Writes results/retrieval.json with overall, per-repo and per-question-type
Recall@k, file Recall@k, MRR@10 and top-1 file accuracy, plus per-question ranks.

    python -m eval.run_retrieval_eval [--retrievers code,general,bm25,hybrid]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from cqa import config
from cqa.embeddings import embedder_info
from cqa.index import get_index
from cqa.repo import ensure_cloned, get_repo

from .dataset import load_questions
from .metrics import first_file_rank, first_hit_rank, mrr, recall_at_k

TOP_N = 10


def summarize(rows: list[dict], ks) -> dict:
    ranks = [r["rank"] for r in rows]
    franks = [r["file_rank"] for r in rows]
    out = {f"recall@{k}": round(recall_at_k(ranks, k), 4) for k in ks}
    out.update({f"file_recall@{k}": round(recall_at_k(franks, k), 4) for k in ks})
    out["mrr@10"] = round(mrr(ranks, 10), 4)
    out["file_acc@1"] = out.get("file_recall@1", round(recall_at_k(franks, 1), 4))
    out["n"] = len(rows)
    return out


def main(ks=(1, 3, 5, 10), chunkers=None, retrievers=None, repos=None) -> dict:
    chunkers = chunkers or config.CHUNKERS
    retrievers = retrievers or config.RETRIEVERS
    questions = load_questions(repos)
    by_repo = defaultdict(list)
    for q in questions:
        by_repo[q.repo].append(q)

    configs, per_question, chunk_stats, skipped = {}, [], {}, {}
    t0 = time.time()
    for ch in chunkers:
        for ret in retrievers:
            name = f"{ch}+{ret}"
            rows = []
            try:
                for repo, qs in by_repo.items():
                    spec = ensure_cloned(get_repo(repo))
                    idx = get_index(spec, ch)
                    chunk_stats.setdefault(ch, {})[repo] = idx.stats()
                    for q in qs:
                        hits = idx.search(q.question, ret, k=TOP_N)
                        retrieved = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in hits]
                        rows.append({
                            "config": name, "chunker": ch, "retriever": ret, "id": q.id, "repo": repo,
                            "type": q.type, "rank": first_hit_rank(retrieved, q.gold),
                            "file_rank": first_file_rank(retrieved, q.gold_files),
                            "top3": [f"{p}:L{s}-L{e}" for p, s, e in retrieved[:3]],
                        })
            except Exception as e:  # e.g. embedding model cannot be downloaded
                skipped[name] = f"{type(e).__name__}: {e}"
                print(f"  [skip] {name}: {skipped[name]}")
                continue
            summary = summarize(rows, ks)
            summary["by_repo"] = {r: summarize([x for x in rows if x["repo"] == r], ks) for r in by_repo}
            summary["by_type"] = {t: summarize([x for x in rows if x["type"] == t], ks)
                                  for t in sorted({x["type"] for x in rows})}
            configs[name] = summary
            per_question.extend(rows)
            print(f"  {name:<16} R@1={summary['recall@1']:.2f}  R@5={summary['recall@5']:.2f}  "
                  f"R@10={summary['recall@10']:.2f}  MRR={summary['mrr@10']:.2f}  file@1={summary['file_acc@1']:.2f}")

    result = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
        "n_questions": len(questions), "ks": list(ks), "embed_backend": config.EMBED_BACKEND,
        "embedders": embedder_info(), "chunk_stats": chunk_stats, "configs": configs,
        "skipped": skipped, "per_question": per_question,
        "hit_rule": "same file and overlap >= 50% of the smaller span",
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "retrieval.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {config.RESULTS_DIR / 'retrieval.json'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunkers", default=",".join(config.CHUNKERS))
    ap.add_argument("--retrievers", default=",".join(config.RETRIEVERS))
    ap.add_argument("--repos", default="")
    a = ap.parse_args()
    main(chunkers=a.chunkers.split(","), retrievers=a.retrievers.split(","),
         repos=[r for r in a.repos.split(",") if r] or None)
