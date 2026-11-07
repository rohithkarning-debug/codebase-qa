"""Shared pytest fixtures.

The tests never touch the network: embeddings use the offline hashing backend
and the LLM runs in offline / scripted mode.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import textwrap

os.environ["CQA_EMBED_BACKEND"] = "hash"
os.environ["CQA_LLM_PROVIDER"] = "offline"
os.environ["CQA_LLM_CACHE"] = "0"
os.environ.setdefault("CQA_RESULTS_DIR", tempfile.mkdtemp(prefix="cqa-results-"))

import pytest  # noqa: E402

from cqa import config  # noqa: E402
from cqa.repo import RepoSpec  # noqa: E402

FIXTURE_NAME = "_pytest_fixture_repo"

AUTH_PY = textwrap.dedent('''\
    """Authentication helpers for the demo package."""
    import base64
    import hashlib

    DEFAULT_REALM = "demo"


    def _encode_credentials(username, password):
        """Return the base64 token used by HTTP Basic authentication."""
        raw = f"{username}:{password}".encode("latin1")
        return base64.b64encode(raw).decode("ascii")


    class BasicAuth:
        """Attaches HTTP Basic Authentication to a request."""

        def __init__(self, username, password):
            self.username = username
            self.password = password

        def __call__(self, request):
            request.headers["Authorization"] = "Basic " + _encode_credentials(self.username, self.password)
            return request


    @staticmethod
    def decorated_helper():
        return hashlib.sha1(b"x").hexdigest()
    ''')


def _big_class() -> str:
    methods = "\n".join(
        f"    def method_{i}(self, value):\n        \"\"\"Method number {i}.\"\"\"\n"
        f"        total = value + {i}\n        return total * 2\n" for i in range(25))
    return f'class SessionStore:\n    """Stores sessions in memory."""\n    timeout = 30\n\n{methods}'


SERVER_JS = textwrap.dedent('''\
    import express from "express";

    export function handleLogin(req, res) {
      const token = req.headers["authorization"];
      return res.send(token ? "ok" : "denied");
    }

    const normalizePath = (p) => p.replace(/\\/+$/, "");

    export class Router {
      constructor() { this.routes = []; }
      route(path, fn) { this.routes.push([normalizePath(path), fn]); }
    }
    ''')

UTILS_GO = textwrap.dedent('''\
    package utils

    // Add returns the sum of two ints.
    func Add(a int, b int) int {
        return a + b
    }
    ''')


@pytest.fixture(scope="session")
def fixture_repo() -> RepoSpec:
    """A tiny multi-language repository written to data/repos/_pytest_fixture_repo."""
    spec = RepoSpec(name=FIXTURE_NAME, url="https://github.com/example/demo", ref="v1", commit="abc123def456")
    root = spec.path
    if root.exists():
        shutil.rmtree(root)
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "auth.py").write_text(AUTH_PY)
    (root / "pkg" / "sessions.py").write_text(_big_class())
    (root / "pkg" / "server.js").write_text(SERVER_JS)
    (root / "pkg" / "utils.go").write_text(UTILS_GO)
    (root / "pkg" / "broken.py").write_text("def oops(:\n    pass\n" * 3)
    (root / "tests" / "test_auth.py").write_text("def test_x():\n    assert True\n")
    (root / "README.md").write_text("# demo\n")
    (root / ".git").mkdir()  # looks cloned, so indexing jobs never try to re-clone it
    yield spec
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(config.INDEX_DIR / FIXTURE_NAME, ignore_errors=True)


@pytest.fixture()
def registry_with_fixture(fixture_repo, monkeypatch):
    """Make the fixture repo visible through get_repo()/load_registry()."""
    import cqa.repo as repo_mod
    real = repo_mod.load_registry

    def fake():
        reg = real()
        reg[fixture_repo.name] = fixture_repo
        return reg

    monkeypatch.setattr(repo_mod, "load_registry", fake)
    try:
        import server.service as svc
        monkeypatch.setattr(svc, "load_registry", fake)
    except ImportError:
        pass
    return fixture_repo
