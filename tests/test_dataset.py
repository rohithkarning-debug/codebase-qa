"""The 50-question test set over 3 repositories."""
from collections import Counter

import pytest

from cqa.repo import get_repo
from eval.dataset import load_questions, load_raw


def test_fifty_questions_over_three_repos():
    qs = load_raw()
    assert len(qs) == 50
    per_repo = Counter(q["repo"] for q in qs)
    assert set(per_repo) == {"requests", "flask", "starlette"}
    assert min(per_repo.values()) >= 15
    assert len({q["id"] for q in qs}) == 50
    assert {q["type"] for q in qs} == {"locate", "how", "explain"}


def test_every_question_has_gold_and_reference():
    for q in load_raw():
        assert q["question"].strip().endswith(("?", "."))
        assert len(q["answer"]) > 60, q["id"]
        assert q["gold"] and all(g["path"] and g["symbol"] for g in q["gold"])
        # the reference answer names the file it points to
        assert any(g["path"].split("/")[-1] in q["answer"] or g["path"] in q["answer"] for g in q["gold"]), q["id"]


def _cloned() -> bool:
    return all((get_repo(r).path / ".git").exists() for r in ("requests", "flask", "starlette"))


@pytest.mark.skipif(not _cloned(), reason="evaluation repos not cloned (python -m cqa setup)")
def test_gold_symbols_resolve_to_line_spans():
    qs = load_questions()
    assert len(qs) == 50
    for q in qs:
        for g in q.gold:
            assert 1 <= g.start <= g.end, (q.id, g)
            src = (get_repo(q.repo).path / g.path).read_text().splitlines()
            first = src[g.start - 1].strip()
            name = g.symbol.split(".")[-1]
            assert name in first or first.startswith("@"), (q.id, g.symbol, first)


@pytest.mark.skipif(not _cloned(), reason="evaluation repos not cloned")
def test_repos_are_pinned():
    for r in ("requests", "flask", "starlette"):
        spec = get_repo(r)
        assert spec.ref and len(spec.commit) == 40
