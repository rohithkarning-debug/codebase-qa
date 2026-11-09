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


