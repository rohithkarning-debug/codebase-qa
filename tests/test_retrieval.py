"""Embeddings (code vs general), BM25, hybrid, index persistence and search."""
import numpy as np
import pytest

from cqa.embeddings import HashingEmbedder, code_tokens, get_embedder
from cqa.index import BM25, RepoIndex, get_index


def test_code_tokens_split_identifiers():
    toks = code_tokens("HTTPDigestAuth.handle_401 parseURL")
    for t in ("http", "digest", "auth", "handle", "401", "parse", "url", "httpdigestauth"):
        assert t in toks


def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder()
    m = e.embed_documents(["basic auth header", "gzip compression"])
    assert m.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert np.allclose(e.embed_query("basic auth"), e.embed_query("basic auth"))
    assert get_embedder("code").model_name.startswith("hashing")


def test_bm25_prefers_matching_document():
    bm = BM25(["def gzip_compress(data): pass", "def basic_auth(user, password): pass", "x = 1"])
    s = bm.scores("basic authentication password")
    assert int(np.argmax(s)) == 1


@pytest.mark.parametrize("chunker", ["ast", "fixed"])
@pytest.mark.parametrize("retriever", ["code", "general", "bm25", "hybrid"])
def test_search_finds_basic_auth(fixture_repo, chunker, retriever):
    idx = get_index(fixture_repo, chunker)
    hits = idx.search("Where is the HTTP Basic Authorization header set?", retriever, k=3)
    assert hits and [h.rank for h in hits] == [1, 2, 3][:len(hits)]
    assert any(h.chunk.path == "pkg/auth.py" for h in hits)


def test_index_is_persisted_and_reloaded(fixture_repo):
    idx = RepoIndex.load_or_build(fixture_repo, "ast", rebuild=True)
    idx.dense("code")
    assert (idx.folder / "chunks.jsonl").exists() and (idx.folder / "meta.json").exists()
    assert any(f.name.startswith("emb_code__") for f in idx.folder.glob("*.npy"))
    again = RepoIndex.load_or_build(fixture_repo, "ast")
    assert [c.id for c in again.chunks] == [c.id for c in idx.chunks]
    st = again.stats()
    assert st["n_chunks"] == len(idx.chunks) and "code" in st["dense"]


def test_unknown_retriever_raises(fixture_repo):
    with pytest.raises(ValueError):
        get_index(fixture_repo, "ast").search("x", "nope")


def test_path_prefix_filter(fixture_repo):
    hits = get_index(fixture_repo, "ast").search("router route", "bm25", k=5, path_prefix="pkg/server")
    assert hits and all(h.chunk.path.startswith("pkg/server") for h in hits)


def test_fastembed_wrapper_with_fake_model(monkeypatch):
    """Exercise the real-model code path without downloading anything."""
    import fastembed

    from cqa import config
    from cqa.embeddings import FastEmbedEmbedder

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            self.model_name = model_name

        def embed(self, texts, batch_size=8):
            for t in texts:
                yield np.array([len(t), 1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    spec = config.EmbedderSpec(key="general", model="fake/model", description="", query_prefix="Q: ", max_chars=5)
    emb = FastEmbedEmbedder(spec)
    seen = []
    m = emb.embed_documents(["a" * 50, "bb", "ccc"], batch_size=2, progress=lambda d, t: seen.append((d, t)))
    assert m.shape == (3, 3) and emb.dim == 3
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert seen == [(2, 3), (3, 3)]
    assert m[0][0] / m[0][1] == 5.0           # document text truncated to max_chars
    q = emb.embed_query("hi")                  # query prefix "Q: " applied -> length 5
    assert q.shape == (3,) and abs(q[0] / q[1] - 5.0) < 1e-5
