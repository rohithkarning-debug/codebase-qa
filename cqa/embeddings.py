"""Embedding models: a code-specific model vs a general-purpose text model.

Backend: `fastembed` (ONNX runtime, CPU friendly, no PyTorch needed). Models are
downloaded from Hugging Face on first use and cached in `data/cache/models`.

An offline `HashingEmbedder` exists for unit tests and for machines without
internet access (set CQA_EMBED_BACKEND=hash). It is *not* used for the reported
results.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from functools import lru_cache

import numpy as np

from . import config

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def code_tokens(text: str) -> list[str]:
    """Split identifiers on snake_case and camelCase, lowercase everything.

    `HTTPDigestAuth.handle_401` -> http, digest, auth, handle, 401, httpdigestauth, handle_401
    """
    out: list[str] = []
    for tok in _IDENT.findall(text):
        low = tok.lower()
        parts = [p.lower() for piece in tok.split("_") for p in _CAMEL.findall(piece)]
        out.extend(parts)
        if len(parts) > 1:
            out.append(low)
    return out


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder:
    key: str = ""
    model_name: str = ""
    dim: int = 0

    def embed_documents(self, texts: list[str], batch_size: int = 32, progress=None) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    def __init__(self, spec: config.EmbedderSpec):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.spec = spec
        self.key = spec.key
        self.model_name = spec.model
        self._model = TextEmbedding(model_name=spec.model, cache_dir=str(config.CACHE_DIR / "models"))
        self._lock = threading.Lock()
        self.dim = int(self.embed_query("hello").shape[-1])

    def embed_documents(self, texts, batch_size=8, progress=None):
        out = []
        prepared = [self.spec.doc_prefix + t[: self.spec.max_chars] for t in texts]
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i + batch_size]
            with self._lock:
                out.extend(self._model.embed(batch, batch_size=batch_size))
            if progress:
                progress(min(i + batch_size, len(prepared)), len(prepared))
        if not out:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        return _normalize(np.asarray(out, dtype=np.float32))

    def embed_query(self, text):
        with self._lock:
            vec = next(iter(self._model.embed([self.spec.query_prefix + text])))
        return _normalize(np.asarray(vec, dtype=np.float32)[None, :])[0]


class HashingEmbedder(Embedder):
    """Deterministic bag-of-identifiers embedder (feature hashing + sublinear tf).

    Fast and dependency-free, used in tests / offline mode only.
    """

    def __init__(self, key: str = "hash", dim: int = 2048):
        self.key, self.model_name, self.dim = key, f"hashing-{dim}", dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, cnt in Counter(code_tokens(text)).items():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += (1.0 + math.log(cnt)) * (1 if (h >> 64) & 1 else -1)
        return v

    def embed_documents(self, texts, batch_size=32, progress=None):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        m = np.stack([self._vec(t) for t in texts])
        if progress:
            progress(len(texts), len(texts))
        return _normalize(m)

    def embed_query(self, text):
        return _normalize(self._vec(text)[None, :])[0]


_lock = threading.Lock()


@lru_cache(maxsize=None)
def _load(key: str, backend: str) -> Embedder:
    if backend == "hash":
        return HashingEmbedder(key=key)
    if key not in config.EMBEDDERS:
        raise ValueError(f"Unknown embedder {key!r}; choose from {list(config.EMBEDDERS)}")
    return FastEmbedEmbedder(config.EMBEDDERS[key])


def get_embedder(key: str) -> Embedder:
    """Load (once) and return the embedder for `code` or `general`."""
    with _lock:
        return _load(key, config.EMBED_BACKEND)


def embedder_info() -> dict:
    return {
        k: {"model": (s.model if config.EMBED_BACKEND != "hash" else "hashing (offline)"),
            "description": s.description}
        for k, s in config.EMBEDDERS.items()
    }
