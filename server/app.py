"""FastAPI web server: JSON API under /api, single-page UI from /web.

Run:  python -m cqa serve      (or: uvicorn server.app:app --reload)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cqa import __version__, config
from cqa.embeddings import embedder_info

from . import service
from .schemas import AddRepoRequest, AskRequest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Codebase QA", version=__version__,
              description="Ask questions about a GitHub repository; answers link to exact files and lines.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _known(name: str) -> None:
    if name not in {r["name"] for r in service.list_repos()}:
        raise HTTPException(404, f"Unknown repository {name!r}")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/api/config")
def get_config():
    return {
        "llm": service.llm_info(),
        "embed_backend": config.EMBED_BACKEND,
        "embedders": embedder_info(),
        "chunkers": config.CHUNKERS,
        "retrievers": config.RETRIEVERS,
        "defaults": {"chunker": config.DEFAULT_CHUNKER, "retriever": config.DEFAULT_RETRIEVER, "mode": "agent"},
        "agent_max_steps": config.AGENT_MAX_STEPS,
    }


# ------------------------------------------------------------------ repos
@app.get("/api/repos")
def repos():
    return service.list_repos()


@app.post("/api/repos")
def add_repo(req: AddRepoRequest):
    try:
        return service.start_add_repo(req.url, req.ref, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/repos/{name}/index")
def index_repo(name: str, embedders: str = Query("code,general")):
    _known(name)
    return service.start_index(name, [e for e in embedders.split(",") if e])


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = service.JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/jobs")
def jobs():
    return service.JOBS.all()


@app.get("/api/repos/{name}/files")
def files(name: str):
    _known(name)
    return service.list_files(name)


@app.get("/api/repos/{name}/file")
def file(name: str, path: str):
    _known(name)
    try:
        return service.read_file(name, path)
    except PermissionError:
        raise HTTPException(403, "path outside repository")
    except FileNotFoundError:
        raise HTTPException(404, f"file not found: {path}")


# ----------------------------------------------------------------- search
@app.get("/api/repos/{name}/search")
def search(name: str, q: str = Query(..., min_length=2), chunker: str = "ast", retriever: str = "code",
           k: int = Query(5, ge=1, le=20)):
    _known(name)
    if chunker not in config.CHUNKERS or retriever not in config.RETRIEVERS:
        raise HTTPException(400, "bad chunker/retriever")
    try:
        return service.search(name, q, chunker, retriever, k)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/repos/{name}/compare")
def compare(name: str, q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=10)):
    _known(name)
    return service.compare(name, q, k)


# -------------------------------------------------------------------- ask
@app.post("/api/ask")
def ask(req: AskRequest):
    """Streams newline-delimited JSON events: start, step*, thought?, answer | error."""
    _known(req.repo)

    def gen():
        for ev in service.ask_stream(req.repo, req.question, req.mode, req.chunker, req.retriever, req.provider):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------- eval
@app.get("/api/eval/questions")
def eval_questions(repo: str | None = None):
    qs = service.eval_questions()
    return [q for q in qs if not repo or q["repo"] == repo]


@app.get("/api/eval/results")
def eval_results():
    return service.eval_results()


# --------------------------------------------------------------- frontend
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
