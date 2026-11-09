"""Command-line interface:  python -m cqa <command> ...

    setup                       clone the 3 evaluation repos at pinned versions
    add URL [--ref TAG]         clone + index any GitHub repo
    index [--repo R | --all]    build chunk + embedding indexes
    search -r R "query"         show retrieved chunks
    ask -r R "question"         answer with the ReAct agent (or --mode rag)
    serve                       start the web app
    eval retrieval|answers|report|all
    doctor                      check the environment before a demo
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time

from . import config


def _progress(stage: str, done: int, total: int) -> None:
    pct = 100 * done // max(total, 1)
    print(f"\r  {stage:<32} {pct:3d}% ({done}/{total})", end="" if done < total else "\n", flush=True)


def cmd_setup(args) -> None:
    from .repo import ensure_cloned, load_registry
    from .index import build_all
    for name, spec in load_registry().items():
        print(f"- {name}: cloning {spec.url} @ {spec.ref or 'default'}")
        ensure_cloned(spec)
        print(f"  commit {spec.commit[:10]}")
        if args.index:
            build_all(spec, progress=_progress)


def cmd_add(args) -> None:
    from .index import build_all
    from .repo import ensure_cloned, save_to_registry, spec_from_url
    spec = spec_from_url(args.url, args.ref, args.name)
    print(f"Cloning {spec.url} -> data/repos/{spec.name}")
    ensure_cloned(spec)
    save_to_registry(spec)
    stats = build_all(spec, embedders=args.embedders.split(",") if args.embedders else None, progress=_progress)
    print(json.dumps(stats, indent=2))


def cmd_index(args) -> None:
    from .index import build_all
    from .repo import ensure_cloned, get_repo, load_registry
    names = list(load_registry()) if args.all or not args.repo else [args.repo]
    for name in names:
        spec = ensure_cloned(get_repo(name))
        print(f"== {name} ({spec.commit[:10]})")
        t0 = time.time()
        stats = build_all(spec, args.chunkers.split(","),
                          [] if args.embedders == "none" else args.embedders.split(","), progress=_progress)
        for ch, st in stats.items():
            print(f"  {ch:<6} {st['n_chunks']:>5} chunks, avg {st['avg_lines']} lines, dense={st['dense']}")
        print(f"  done in {time.time() - t0:.1f}s")


def cmd_search(args) -> None:
    from .index import get_index
    from .repo import get_repo
    spec = get_repo(args.repo)
    idx = get_index(spec, args.chunker)
    for h in idx.search(args.query, args.retriever, k=args.k):
        c = h.chunk
        print(f"{h.rank:>2}. {h.score:.3f}  {c.location}  {c.kind} {c.symbol}")
        print(f"    {spec.github_url(c.path, c.start_line, c.end_line)}")


def cmd_ask(args) -> None:
    from .agent import RAGAnswerer, ReActAgent
    from .llm import make_llm
    from .repo import get_repo
    spec = get_repo(args.repo)
    if args.mode == "rag":
        res = RAGAnswerer(spec, make_llm(args.provider, args.model), args.chunker, args.retriever).run(args.question)
    else:
        agent = ReActAgent(spec, make_llm(args.provider, args.model), args.chunker, args.retriever)
        res = {}
        for ev in agent.run_stream(args.question):
            if ev["type"] == "step":
                print(f"\n[{ev['step']}] Thought: {ev['thought']}")
                print(f"    Action: {ev['action']} {json.dumps(ev['input'])}")
                print(textwrap.indent(ev["observation"][:600], "    | "))
            elif ev["type"] == "answer":
                res = ev
    print("\n" + "=" * 70 + "\n" + res["answer"] + "\n")
    for c in res["citations"]:
        print(f"  -> {c['path']}:L{c['start']}-L{c['end']}  {c['url']}")
    print(f"\n({res['llm']}, {res['steps']} tool calls, {res['seconds']}s)")


def cmd_serve(args) -> None:
    import uvicorn
    print(f"\n  Codebase QA running at  http://{args.host}:{args.port}\n")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_eval(args) -> None:
    from eval import run_answer_eval, run_retrieval_eval, report
    if args.what in ("retrieval", "all"):
        run_retrieval_eval.main(ks=(1, 3, 5, 10))
    if args.what in ("answers", "all"):
        run_answer_eval.main(modes=args.modes.split(","), limit=args.limit,
                             chunker=args.chunker, retriever=args.retriever)
    if args.what in ("report", "all", "retrieval", "answers"):
        report.main()


def cmd_doctor(args) -> None:
    from .repo import load_registry
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for mod in ("numpy", "fastembed", "fastapi", "uvicorn", "httpx", "tree_sitter_language_pack"):
        try:
            __import__(mod)
            print(f"  [ok] {mod}")
        except ImportError:
            ok = False
            print(f"  [!!] {mod} missing -> pip install -r requirements.txt")
    s = config.llm_settings()
    if s.provider == "offline":
        print("  [..] LLM: offline mode (set GEMINI_API_KEY / GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY in .env)")
    else:
        print(f"  [ok] LLM: {s.provider}:{s.model}")
        try:
            from .llm import LLM
            out = LLM(s).chat([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=200)
            print(f"       test call -> {out.strip()[:30]!r}")
        except Exception as e:
            ok = False
            print(f"  [!!] LLM call failed: {e}")
    print(f"  [..] embeddings backend: {config.EMBED_BACKEND}")
    for name, spec in load_registry().items():
        cloned = (spec.path / ".git").exists()
        idx = [ch for ch in config.CHUNKERS if (config.INDEX_DIR / name / ch / "chunks.jsonl").exists()]
        print(f"  [{'ok' if cloned else '!!'}] repo {name}: cloned={cloned} indexes={idx}")
        ok &= cloned
    print("\nAll good - run `python -m cqa serve`" if ok else "\nFix the [!!] items above (see README).")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cqa", description="Codebase question-answering assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="clone the evaluation repos")
    s.add_argument("--index", action="store_true", help="also build all indexes")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("add", help="clone and index a GitHub repo")
    s.add_argument("url")
    s.add_argument("--ref", default="")
    s.add_argument("--name", default="")
    s.add_argument("--embedders", default="")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("index", help="build indexes")
    s.add_argument("--repo", "-r")
    s.add_argument("--all", action="store_true")
    s.add_argument("--chunkers", default="ast,fixed")
    s.add_argument("--embedders", default="code,general", help="comma list or 'none'")
    s.set_defaults(fn=cmd_index)

    for name, fn in (("search", cmd_search), ("ask", cmd_ask)):
        s = sub.add_parser(name)
        s.add_argument("--repo", "-r", required=True)
        s.add_argument("--chunker", default=config.DEFAULT_CHUNKER, choices=config.CHUNKERS)
        s.add_argument("--retriever", default=config.DEFAULT_RETRIEVER, choices=config.RETRIEVERS)
        if name == "search":
            s.add_argument("query")
            s.add_argument("-k", type=int, default=5)
        else:
            s.add_argument("question")
            s.add_argument("--mode", default="agent", choices=["agent", "rag"])
            s.add_argument("--provider", default=None)
            s.add_argument("--model", default=None)
        s.set_defaults(fn=fn)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("eval", help="run the evaluation")
    s.add_argument("what", choices=["retrieval", "answers", "report", "all"])
    s.add_argument("--modes", default="rag,agent")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--chunker", default="ast")
    s.add_argument("--retriever", default="code")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("doctor", help="check environment")
    s.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
