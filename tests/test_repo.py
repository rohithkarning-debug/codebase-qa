"""Repository ingestion: GitHub URLs, permalinks, safe paths, registry."""
import pytest

from cqa.repo import load_registry, parse_github_url, safe_join, spec_from_url


def test_github_links_and_url_parsing(fixture_repo):
    assert fixture_repo.github_url("pkg/auth.py", 10, 20) == \
        "https://github.com/example/demo/blob/abc123def456/pkg/auth.py#L10-L20"
    assert parse_github_url("https://github.com/psf/requests.git") == ("psf", "requests")
    assert spec_from_url("github.com/pallets/Flask").name == "flask"
    with pytest.raises(ValueError):
        parse_github_url("https://gitlab.com/a/b")


def test_safe_join_blocks_traversal(fixture_repo):
    assert safe_join(fixture_repo, "pkg/auth.py").name == "auth.py"
    with pytest.raises(PermissionError):
        safe_join(fixture_repo, "../../etc/passwd")


def test_registry_has_the_three_pinned_eval_repos():
    reg = load_registry()
    for name, ref in (("requests", "v2.32.3"), ("flask", "3.0.3"), ("starlette", "0.38.2")):
        assert reg[name].ref == ref and reg[name].url.startswith("https://github.com/")


def test_save_to_registry_round_trip(tmp_path, monkeypatch):
    from cqa import config
    from cqa.repo import save_to_registry
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    spec = spec_from_url("https://github.com/encode/httpx", ref="0.27.0")
    save_to_registry(spec)
    save_to_registry(spec)                       # saving twice keeps one entry
    reg = load_registry()
    assert reg["httpx"].ref == "0.27.0" and reg["httpx"].url == "https://github.com/encode/httpx"


def test_language_detection_and_link_without_lines(fixture_repo):
    from cqa.repo import language_for
    assert language_for("a/b.py") == "python" and language_for("x.tsx") == "tsx"
    assert language_for("README.md") is None
    assert fixture_repo.github_url("pkg/auth.py").endswith("/blob/abc123def456/pkg/auth.py")
    assert fixture_repo.github_url("pkg/auth.py", 7, 7).endswith("#L7")
