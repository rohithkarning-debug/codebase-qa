"""Web API: repos, files, search, compare, streaming answers, jobs, evaluation endpoints."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture()
def client(registry_with_fixture):
    return TestClient(app)


def test_health_and_config(client):
    assert client.get("/api/health").json()["ok"] is True
    cfg = client.get("/api/config").json()
    assert cfg["llm"]["offline"] is True
    assert cfg["chunkers"] == ["ast", "fixed"]
    assert set(cfg["retrievers"]) == {"code", "general", "bm25", "hybrid"}


def test_repos_list_contains_fixture(client, fixture_repo):
    names = [r["name"] for r in client.get("/api/repos").json()]
    assert fixture_repo.name in names and "requests" in names


def test_file_endpoint_and_traversal(client, fixture_repo):
    r = client.get(f"/api/repos/{fixture_repo.name}/file", params={"path": "pkg/auth.py"})
    assert r.status_code == 200 and "class BasicAuth" in r.json()["content"]
    assert r.json()["language"] == "python"
    assert client.get(f"/api/repos/{fixture_repo.name}/file", params={"path": "../../../etc/passwd"}).status_code == 403
    assert client.get(f"/api/repos/{fixture_repo.name}/file", params={"path": "pkg/missing.py"}).status_code == 404
    assert client.get("/api/repos/not-a-repo/file", params={"path": "x.py"}).status_code == 404
    files = client.get(f"/api/repos/{fixture_repo.name}/files").json()
    assert "pkg/auth.py" in files


def test_search_and_compare(client, fixture_repo):
    hits = client.get(f"/api/repos/{fixture_repo.name}/search",
                      params={"q": "basic auth header", "chunker": "fixed", "retriever": "bm25", "k": 3}).json()
    assert len(hits) == 3 and hits[0]["url"].startswith("https://github.com/example/demo")
    assert client.get(f"/api/repos/{fixture_repo.name}/search",
                      params={"q": "x y", "retriever": "bogus"}).status_code == 400
    cmp = client.get(f"/api/repos/{fixture_repo.name}/compare", params={"q": "basic auth header"}).json()
    assert set(cmp["results"]) == {f"{c}+{r}" for c in ("ast", "fixed") for r in ("code", "general", "bm25", "hybrid")}
    assert cmp["gold"] == []


def _events(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


@pytest.mark.parametrize("mode", ["agent", "rag"])
def test_ask_streams_ndjson(client, fixture_repo, mode):
    r = client.post("/api/ask", json={"repo": fixture_repo.name, "question": "Where is the Authorization header set?",
                                      "mode": mode, "chunker": "ast", "retriever": "code"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    ev = _events(r)
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "answer"
    assert ev[-1]["citations"] and ev[-1]["citations"][0]["path"].startswith("pkg/")
    if mode == "agent":
        assert any(e["type"] == "step" and e["action"] == "search_code" for e in ev)


def test_ask_validation(client, fixture_repo):
    assert client.post("/api/ask", json={"repo": fixture_repo.name, "question": "hi?", "mode": "magic"}).status_code == 422
    assert client.post("/api/ask", json={"repo": "nope", "question": "where?"}).status_code == 404


def test_add_repo_rejects_non_github(client):
    assert client.post("/api/repos", json={"url": "https://example.com/a/b"}).status_code == 400
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_index_job_runs_to_completion(client, fixture_repo):
    job = client.post(f"/api/repos/{fixture_repo.name}/index", params={"embedders": "code"}).json()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    repo = next(r for r in client.get("/api/repos").json() if r["name"] == fixture_repo.name)
    assert set(repo["indexes"]) == {"ast", "fixed"} and "code" in repo["indexes"]["ast"]["dense"]


def test_eval_endpoints(client):
    qs = client.get("/api/eval/questions").json()
    assert len(qs) == 50
    assert all(q["repo"] == "flask" for q in client.get("/api/eval/questions", params={"repo": "flask"}).json())
    assert isinstance(client.get("/api/eval/results").json(), dict)


def test_frontend_is_served(client):
    html = client.get("/").text
    for element_id in ("askForm", "question", "thread", "viewer", "compareGrid", "evalBody", "repoList"):
        assert f'id="{element_id}"' in html
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
