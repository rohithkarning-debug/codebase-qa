"""Turn results/*.json into results/REPORT.md (+ PNG charts when matplotlib is installed).

    python -m eval.report
"""
from __future__ import annotations

import json
from pathlib import Path

from cqa import config
from cqa.repo import get_repo


def _load(name: str) -> dict | None:
    p = config.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x) -> str:
    return f"{100 * float(x):.0f}%" if x is not None else "-"


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def retrieval_section(ret: dict) -> str:
    ks = ret["ks"]
    cfgs = ret["configs"]
    lines = ["## 1. Retrieval: chunking x embedding model", "",
             f"{ret['n_questions']} questions, top-10 retrieved chunks per question. Hit rule: {ret['hit_rule']}.", ""]
    if ret.get("embed_backend") == "hash":
        lines += ["> **Note:** run with the offline hashing embedder (CQA_EMBED_BACKEND=hash). "
                  "Re-run with real models for the final numbers.", ""]
    rows = []
    for name, s in sorted(cfgs.items(), key=lambda kv: -kv[1]["recall@5"]):
        rows.append([f"`{name}`"] + [_pct(s[f"recall@{k}"]) for k in ks] + [f"{s['mrr@10']:.3f}", _pct(s["file_acc@1"])])
    lines += [_table(["config"] + [f"Recall@{k}" for k in ks] + ["MRR@10", "File acc@1"], rows), ""]

    # chunking comparison
    rets = sorted({n.split("+")[1] for n in cfgs})
    rows = []
    for r in rets:
        a, f = cfgs.get(f"ast+{r}"), cfgs.get(f"fixed+{r}")
        if a and f:
            rows.append([r, _pct(a["recall@5"]), _pct(f["recall@5"]),
                         f"{100 * (a['recall@5'] - f['recall@5']):+.0f} pts", f"{a['mrr@10']:.3f} / {f['mrr@10']:.3f}"])
    if rows:
        lines += ["### AST (function/class) chunking vs fixed-size chunking", "",
                  _table(["retriever", "AST R@5", "Fixed R@5", "delta", "MRR AST / Fixed"], rows), ""]
    # embedding comparison
    rows = []
    for ch in config.CHUNKERS:
        c, g = cfgs.get(f"{ch}+code"), cfgs.get(f"{ch}+general")
        if c and g:
            rows.append([ch, _pct(c["recall@5"]), _pct(g["recall@5"]),
                         f"{100 * (c['recall@5'] - g['recall@5']):+.0f} pts", f"{c['mrr@10']:.3f} / {g['mrr@10']:.3f}"])
    if rows:
        emb = ret.get("embedders", {})
        lines += ["### Code-specific vs general embedding model", "",
                  f"code = `{emb.get('code', {}).get('model', '?')}`, general = `{emb.get('general', {}).get('model', '?')}`", "",
                  _table(["chunker", "code R@5", "general R@5", "delta", "MRR code / general"], rows), ""]
    # per repo
    repos = sorted(next(iter(cfgs.values()))["by_repo"]) if cfgs else []
    rows = [[f"`{n}`"] + [_pct(s["by_repo"].get(r, {}).get("recall@5")) for r in repos] for n, s in cfgs.items()]
    lines += ["### Recall@5 per repository", "", _table(["config"] + repos, rows), ""]
    # chunk stats
    rows = []
    for ch, per in ret.get("chunk_stats", {}).items():
        for repo, st in per.items():
            rows.append([ch, repo, st["n_files"], st["n_chunks"], st["avg_lines"], st["max_lines"]])
    lines += ["### Index statistics", "", _table(["chunker", "repo", "files", "chunks", "avg lines", "max lines"], rows), ""]
    if ret.get("skipped"):
        lines += ["Skipped configs: " + ", ".join(f"`{k}` ({v[:80]})" for k, v in ret["skipped"].items()), ""]
    return "\n".join(lines)


def answers_section(ans: dict) -> str:
    lines = ["## 2. Answer quality: RAG vs ReAct agent", "",
             f"Answer model: `{ans['llm']}` | judge: `{ans['judge']}` | index: `{ans['index']}` | "
             f"{ans['n_questions']} questions", ""]
    if "offline" in ans["llm"] or "heuristic" in ans["judge"]:
        lines += ["> **Note:** produced in offline mode (no LLM key). Configure an LLM in `.env` and re-run "
                  "`python -m cqa eval answers` for the real LLM-as-Judge numbers.", ""]
    rows = []
    for mode, s in ans["modes"].items():
        rows.append([mode, _pct(s["judge_correct"]), f"{s['judge_score']:.2f}", _pct(s["file_acc"]),
                     _pct(s["line_acc"]), _pct(s["top_file_acc"]), f"{s['avg_steps']:.1f}", f"{s['avg_seconds']:.1f}s"])
    lines += [_table(["mode", "Judge correct", "Judge score (1-5)", "File-location acc", "Line acc",
                      "First-citation file acc", "avg tool calls", "avg time"], rows), ""]
    for key, title in (("by_repo", "per repository"), ("by_type", "per question type")):
        groups = sorted({g for s in ans["modes"].values() for g in s[key]})
        rows = [[mode] + [f"{_pct(s[key][g]['judge_correct'])} / {_pct(s[key][g]['file_acc'])}" if g in s[key] else "-"
                          for g in groups] for mode, s in ans["modes"].items()]
        lines += [f"### Judge-correct / file-location accuracy {title}", "", _table(["mode"] + groups, rows), ""]
    wrong = [r for r in ans["rows"] if r["mode"] == "agent" and not r["judge"]["correct"]][:8]
    if wrong:
        lines += ["### Example agent failures", ""]
        lines += [f"- **{r['id']}** {r['question']} - judge: {r['judge']['reason']}" for r in wrong]
        lines.append("")
    return "\n".join(lines)


def charts(ret: dict | None, ans: dict | None) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = config.RESULTS_DIR / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if ret and ret["configs"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, s in sorted(ret["configs"].items()):
            ax.plot(ret["ks"], [s[f"recall@{k}"] for k in ret["ks"]], marker="o",
                    linestyle="-" if name.startswith("ast") else "--", label=name)
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k")
        ax.set_ylim(0, 1)
        ax.set_xticks(ret["ks"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Retrieval Recall@k (solid = AST chunks, dashed = fixed)")
        fig.tight_layout()
        fig.savefig(out_dir / "recall_at_k.png", dpi=150)
        plt.close(fig)
        made.append("charts/recall_at_k.png")
    if ans and ans["modes"]:
        metrics = [("judge_correct", "Judge correct"), ("file_acc", "File acc"), ("line_acc", "Line acc")]
        fig, ax = plt.subplots(figsize=(7, 4))
        modes = list(ans["modes"])
        w = 0.8 / len(modes)
        for i, mode in enumerate(modes):
            vals = [ans["modes"][mode][m] for m, _ in metrics]
            ax.bar([j + i * w for j in range(len(metrics))], vals, width=w, label=mode)
        ax.set_xticks([j + w * (len(modes) - 1) / 2 for j in range(len(metrics))])
        ax.set_xticklabels([t for _, t in metrics])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Answer quality: RAG vs ReAct agent")
        fig.tight_layout()
        fig.savefig(out_dir / "answers.png", dpi=150)
        plt.close(fig)
        made.append("charts/answers.png")
    return made


def main() -> str:
    ret, ans = _load("retrieval.json"), _load("answers.json")
    parts = ["# Codebase QA - Evaluation Report", ""]
    names = sorted({q for q in (ret or {}).get("chunk_stats", {}).get("ast", {})} |
                   {r["repo"] for r in (ans or {}).get("rows", [])})
    if names:
        parts += ["Repositories (pinned):", ""]
        for n in names:
            try:
                spec = get_repo(n)
                parts.append(f"- **{n}** - {spec.url} @ `{spec.ref}` ({spec.commit[:10]})")
            except KeyError:
                parts.append(f"- {n}")
        parts.append("")
    if ret:
        parts.append(retrieval_section(ret))
    if ans:
        parts.append(answers_section(ans))
    imgs = charts(ret, ans)
    if imgs:
        parts += ["## Charts", ""] + [f"![{Path(i).stem}]({i})" for i in imgs] + [""]
    if not ret and not ans:
        parts.append("No results yet. Run `python -m cqa eval all`.")
    text = "\n".join(parts)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "REPORT.md").write_text(text)
    print(f"Saved {config.RESULTS_DIR / 'REPORT.md'}")
    return text


if __name__ == "__main__":
    main()
