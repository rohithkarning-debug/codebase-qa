"""Service layer: repo status, background indexing jobs, search, answering."""
from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from typing import Iterator

from cqa import config
from cqa.agent import RAGAnswerer, ReActAgent
from cqa.agent.rag import offline_rag_policy
from cqa.agent.react import offline_react_policy
from cqa.index import build_all, clear_cache, emb_filename, get_index
from cqa.llm import LLMError, make_llm
from cqa.repo import (RepoSpec, ensure_cloned, get_repo, iter_source_files, language_for, load_registry,
                      read_text, safe_join, save_to_registry, spec_from_url)


# ----------------------------------------------------------------- repos
def repo_status(spec: RepoSpec) -> dict:
    cloned = (spec.path / ".git").exists()
    indexes = {}
    for ch in config.CHUNKERS:
        meta = config.INDEX_DIR / spec.name / ch / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            dense = [k for k in config.EMBEDDERS if (config.INDEX_DIR / spec.name / ch / emb_filename(k)).exists()]
            indexes[ch] = {"n_chunks": m["n_chunks"], "created": m["created"], "dense": dense}
    return {
        "name": spec.name, "url": spec.url, "ref": spec.ref, "commit": spec.commit[:10] if spec.commit else "",
        "description": spec.description, "cloned": cloned, "indexes": indexes,
        "eval_repo": spec.name in _eval_repo_names(),
    }


def _eval_repo_names() -> set[str]:
    if config.EVAL_REPOS_FILE.exists():
        return {r["name"] for r in json.loads(config.EVAL_REPOS_FILE.read_text())}
    return set()


def list_repos() -> list[dict]:
    out = []
    for name in load_registry():
        try:
            out.append(repo_status(get_repo(name)))
        except Exception as e:  # pragma: no cover - defensive
            out.append({"name": name, "error": str(e)})
    return out


def list_files(name: str) -> list[str]:
    return [p.as_posix() for p in iter_source_files(get_repo(name))]


def read_file(name: str, path: str) -> dict:
    spec = get_repo(name)
    full = safe_join(spec, path)
    if not full.is_file():
        raise FileNotFoundError(path)
    text = read_text(full)
    return {"repo": name, "path": path, "language": language_for(path) or "text", "content": text,
            "n_lines": text.count("\n") + 1, "url": spec.github_url(path)}


# ------------------------------------------------------------------ jobs
class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, repo: str) -> dict:
        job = {"id": uuid.uuid4().hex[:10], "repo": repo, "status": "queued", "stage": "", "done": 0,
               "total": 0, "message": ""}
        with self._lock:
            self._jobs[job["id"]] = job
        return job

    def update(self, job_id: str, **kw) -> None:
        with self._lock:
            self._jobs[job_id].update(kw)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]


JOBS = JobStore()


def _run_index_job(job_id: str, spec: RepoSpec, clone: bool, embedders: list[str]) -> None:
    def progress(stage, done, total):
        JOBS.update(job_id, stage=stage, done=done, total=total)

    try:
        JOBS.update(job_id, status="running", stage="cloning")
        if clone:
            ensure_cloned(spec)
            save_to_registry(spec)
        clear_cache(spec.name)
        build_all(spec, embedders=embedders, progress=progress)
        JOBS.update(job_id, status="done", stage="ready", message=f"Indexed {spec.name}")
    except Exception as e:
        traceback.print_exc()
        JOBS.update(job_id, status="error", message=f"{type(e).__name__}: {e}")


def start_add_repo(url: str, ref: str = "", name: str = "") -> dict:
    spec = spec_from_url(url, ref, name)
    job = JOBS.create(spec.name)
    threading.Thread(target=_run_index_job, args=(job["id"], spec, True, ["code"]), daemon=True).start()
    return job


def start_index(name: str, embedders: list[str] | None = None) -> dict:
    spec = get_repo(name)
    job = JOBS.create(name)
    threading.Thread(target=_run_index_job,
                     args=(job["id"], spec, not (spec.path / ".git").exists(),
                           embedders if embedders is not None else list(config.EMBEDDERS)),
                     daemon=True).start()
    return job


# --------------------------------------------------------------- search
def search(name: str, query: str, chunker: str, retriever: str, k: int = 5) -> list[dict]:
    spec = get_repo(name)
    idx = get_index(spec, chunker)
    return [h.to_dict(spec) for h in idx.search(query, retriever, k=k)]


def gold_for(name: str, question: str) -> list[dict]:
    """If the query is one of the 50 evaluation questions, return its gold spans."""
    try:
        from eval.dataset import load_raw, resolve_gold
    except ImportError:
        return []
    for q in load_raw():
        if q["repo"] == name and q["question"].strip().lower() == question.strip().lower():
            try:
                return [resolve_gold(name, g["path"], g["symbol"]).to_dict() for g in q["gold"]]
            except Exception:
                return []
    return []


def compare(name: str, query: str, k: int = 5) -> dict:
    from eval.metrics import is_hit
    from eval.dataset import GoldSpan
    gold = gold_for(name, query)
    gold_objs = [GoldSpan(g["path"], g["symbol"], g["start"], g["end"]) for g in gold]
    results, errors = {}, {}
    for ch in config.CHUNKERS:
        for ret in config.RETRIEVERS:
            key = f"{ch}+{ret}"
            try:
                hits = search(name, query, ch, ret, k)
            except Exception as e:
                errors[key] = f"{type(e).__name__}: {e}"
                continue
            for h in hits:
                h["hit"] = bool(gold_objs) and is_hit(h["path"], h["start_line"], h["end_line"], gold_objs)
            results[key] = hits
    return {"query": query, "gold": gold, "results": results, "errors": errors}


# ------------------------------------------------------------------ ask
def llm_info() -> dict:
    s = config.llm_settings()
    available = [p for p, (_, _, env) in config.LLM_PRESETS.items()
                 if p == "offline" or (env and os.environ.get(env)) or p == "ollama" and s.provider == "ollama"]
    return {"provider": s.provider, "model": s.model, "name": f"{s.provider}:{s.model}",
            "offline": s.provider == "offline", "available": available}


def ask_stream(name: str, question: str, mode: str, chunker: str, retriever: str,
               provider: str | None = None) -> Iterator[dict]:
    try:
        spec = get_repo(name)
        get_index(spec, chunker)  # make sure the index exists before the agent starts
        if mode == "rag":
            llm = make_llm(provider, offline_policy=offline_rag_policy)
            yield {"type": "start", "question": question, "repo": name, "llm": llm.name}
            yield RAGAnswerer(spec, llm, chunker, retriever).run(question)
        else:
            llm = make_llm(provider, offline_policy=offline_react_policy)
            yield from ReActAgent(spec, llm, chunker, retriever).run_stream(question)
    except LLMError as e:
        yield {"type": "error", "message": str(e)}
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------- eval
def eval_questions() -> list[dict]:
    try:
        from eval.dataset import load_raw
        return [{k: q[k] for k in ("id", "repo", "type", "question")} for q in load_raw()]
    except Exception:
        return []


def eval_results() -> dict:
    out = {}
    for name in ("retrieval", "answers"):
        p = config.RESULTS_DIR / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if name == "retrieval":
                data.pop("per_question", None)
            out[name] = data
    return out
