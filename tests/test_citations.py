"""Citations: finding path:Lx-Ly references in answers and turning them into links."""
from cqa.citations import extract_citations


def test_extract_citations_formats(fixture_repo):
    text = ("See [pkg/auth.py:L15-L24] and `auth.py:9-12`, also pkg/server.js#L3 and "
            "utils.go lines 3-5. Unknown: ghost/file.py:L1-L2")
    cites = extract_citations(text, fixture_repo)
    got = {(c.path, c.start, c.end, c.valid) for c in cites}
    assert ("pkg/auth.py", 15, 24, True) in got
    assert ("pkg/auth.py", 9, 12, True) in got          # suffix resolved to the full path
    assert ("pkg/server.js", 3, 3, True) in got
    assert ("pkg/utils.go", 3, 5, True) in got
    assert ("ghost/file.py", 1, 2, False) in got
    first = next(c for c in cites if c.start == 15)
    assert first.url.endswith("pkg/auth.py#L15-L24")


def test_citations_without_repo_keep_raw_paths_and_dedupe():
    cites = extract_citations("[a/b.py:L1-L3] then a/b.py:1-3 again, and c.go#L9")
    assert [(c.path, c.start, c.end) for c in cites] == [("a/b.py", 1, 3), ("c.go", 9, 9)]
    assert all(c.url == "" for c in cites)


def test_reversed_ranges_are_fixed_and_bare_duplicates_dropped(fixture_repo):
    cites = extract_citations("pkg/auth.py:L24-L15 and also pkg/auth.py", fixture_repo)
    assert [(c.path, c.start, c.end) for c in cites] == [("pkg/auth.py", 15, 24)]
