"""Per-repo search index: chunks + dense embeddings (+ BM25 / hybrid baselines).

On disk:
    data/indexes/<repo>/<chunker>/meta.json
    data/indexes/<repo>/<chunker>/chunks.jsonl
    data/indexes/<repo>/<chunker>/emb_<retriever>__<model>.npy
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, chunk_repo
from .embeddings import code_tokens, get_embedder
from .repo import RepoSpec, get_repo

ProgressFn = "callable(stage: str, done: int, total: int)"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self, spec: RepoSpec | None = None) -> dict:
        d = self.chunk.to_dict()
        d.update(score=round(float(self.score), 4), rank=self.rank)
        if spec:
            d["url"] = spec.github_url(self.chunk.path, self.chunk.start_line, self.chunk.end_line)
        return d


class BM25:
    """Okapi BM25 over identifier-aware tokens (lexical baseline)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [code_tokens(d) for d in docs]
        self.doc_len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(toks) else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                self.postings[term].append((i, tf))
        n = len(toks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(code_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


def _slug(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model)


def emb_filename(key: str) -> str:
    """File name of the embedding matrix for `key` (does not load the model)."""
    model = "hashing-2048" if config.EMBED_BACKEND == "hash" else config.EMBEDDERS[key].model
    return f"emb_{key}__{_slug(model)}.npy"


class RepoIndex:
    def __init__(self, spec: RepoSpec, chunker: str, chunks: list[Chunk], folder: Path):
        self.spec, self.chunker, self.chunks, self.folder = spec, chunker, chunks, folder
        self._dense: dict[str, np.ndarray] = {}
        self._bm25: BM25 | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ build
    @classmethod
    def load_or_build(cls, spec: RepoSpec, chunker: str, progress=None, rebuild: bool = False) -> "RepoIndex":
        folder = config.INDEX_DIR / spec.name / chunker
        meta_f, chunks_f = folder / "meta.json", folder / "chunks.jsonl"
        if not rebuild and meta_f.exists() and chunks_f.exists():
            meta = json.loads(meta_f.read_text())
            if not spec.commit or meta.get("commit") == spec.commit:
                chunks = [Chunk.from_dict(json.loads(l)) for l in chunks_f.read_text().splitlines() if l]
                return cls(spec, chunker, chunks, folder)
        if progress:
            progress(f"chunking ({chunker})", 0, 1)
        if folder.exists():
            for f in folder.glob("*"):
                f.unlink()
        folder.mkdir(parents=True, exist_ok=True)
        chunks = chunk_repo(spec, chunker)
        chunks_f.write_text("\n".join(json.dumps(c.to_dict()) for c in chunks), encoding="utf-8")
        meta_f.write_text(json.dumps({
            "repo": spec.name, "chunker": chunker, "commit": spec.commit,
            "n_chunks": len(chunks), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        if progress:
            progress(f"chunking ({chunker})", 1, 1)
        return cls(spec, chunker, chunks, folder)

    # ----------------------------------------------------------------- dense
    def _emb_file(self, key: str) -> Path:
        return self.folder / emb_filename(key)

    def has_dense(self, key: str) -> bool:
        try:
            return key in self._dense or self._emb_file(key).exists()
        except Exception:
            return False

    def dense(self, key: str, progress=None) -> np.ndarray:
        with self._lock:
            if key in self._dense:
                return self._dense[key]
            f = self._emb_file(key)
            if f.exists():
                m = np.load(f)
            else:
                emb = get_embedder(key)
                cb = (lambda d, t: progress(f"embedding ({self.chunker}/{key})", d, t)) if progress else None
                m = emb.embed_documents([c.embedding_text() for c in self.chunks], progress=cb)
                np.save(f, m)
            self._dense[key] = m
            return m

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.embedding_text() for c in self.chunks])
        return self._bm25

    # ---------------------------------------------------------------- search
    def scores(self, query: str, retriever: str) -> np.ndarray:
        if retriever in config.EMBEDDERS:
            m = self.dense(retriever)
            if len(m) == 0:
                return np.zeros(0, dtype=np.float32)
            return m @ get_embedder(retriever).embed_query(query)
        if retriever == "bm25":
            return self.bm25.scores(query)
        if retriever == "hybrid":  # reciprocal-rank fusion of code embeddings + BM25
            fused = np.zeros(len(self.chunks), dtype=np.float32)
            for r in ("code", "bm25"):
                order = np.argsort(-self.scores(query, r))
                fused[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
            return fused
        raise ValueError(f"Unknown retriever {retriever!r}; choose from {config.RETRIEVERS}")

    def search(self, query: str, retriever: str = "code", k: int = 10,
               path_prefix: str = "") -> list[Hit]:
        if not self.chunks:
            return []
        s = self.scores(query, retriever)
        order = np.argsort(-s)
        hits = []
        for i in order:
            c = self.chunks[int(i)]
            if path_prefix and not c.path.startswith(path_prefix):
                continue
            hits.append(Hit(c, float(s[int(i)]), len(hits) + 1))
            if len(hits) >= k:
                break
        return hits

    def stats(self) -> dict:
        lines = [c.n_lines for c in self.chunks] or [0]
        return {
            "chunker": self.chunker, "n_chunks": len(self.chunks),
            "n_files": len({c.path for c in self.chunks}),
            "avg_lines": round(sum(lines) / len(lines), 1), "max_lines": max(lines),
            "dense": [k for k in config.EMBEDDERS if self.has_dense(k)],
        }


# -------------------------------------------------------------------- cache
_CACHE: dict[tuple[str, str], RepoIndex] = {}
_CACHE_LOCK = threading.Lock()


def get_index(repo: str | RepoSpec, chunker: str = config.DEFAULT_CHUNKER, progress=None) -> RepoIndex:
    spec = repo if isinstance(repo, RepoSpec) else get_repo(repo)
    key = (spec.name, chunker)
    with _CACHE_LOCK:
        idx = _CACHE.get(key)
        if idx is None or (spec.commit and idx.spec.commit and idx.spec.commit != spec.commit):
            idx = RepoIndex.load_or_build(spec, chunker, progress=progress)
            _CACHE[key] = idx
        return idx


def build_all(spec: RepoSpec, chunkers=None, embedders=None, progress=None) -> dict:
    """Build every (chunker x embedder) combination for one repo."""
    chunkers = chunkers or config.CHUNKERS
    embedders = embedders if embedders is not None else list(config.EMBEDDERS)
    out = {}
    for ch in chunkers:
        idx = RepoIndex.load_or_build(spec, ch, progress=progress)
        with _CACHE_LOCK:
            _CACHE[(spec.name, ch)] = idx
        for e in embedders:
            idx.dense(e, progress=progress)
        out[ch] = idx.stats()
    return out


def clear_cache(repo: str | None = None) -> None:
    with _CACHE_LOCK:
        for k in list(_CACHE):
            if repo is None or k[0] == repo:
                del _CACHE[k]
