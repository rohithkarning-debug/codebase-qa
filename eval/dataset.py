"""Load the 50-question test set and resolve gold symbols to exact line spans.

Gold locations are written as (file, symbol) pairs - e.g.
("src/requests/auth.py", "HTTPDigestAuth.handle_401") - and resolved to line
numbers with the syntax tree of the pinned commit. That keeps the test set
readable and guarantees the line numbers are right.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from cqa.chunking import symbol_table
from cqa.repo import get_repo, language_for, read_text

QUESTIONS_FILE = Path(__file__).with_name("questions.jsonl")


@dataclass
class GoldSpan:
    path: str
    symbol: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"path": self.path, "symbol": self.symbol, "start": self.start, "end": self.end}


@dataclass
class Question:
    id: str
    repo: str
    type: str
    question: str
    answer: str
    gold: list[GoldSpan] = field(default_factory=list)

    @property
    def gold_files(self) -> set[str]:
        return {g.path for g in self.gold}

    def to_dict(self) -> dict:
        return {"id": self.id, "repo": self.repo, "type": self.type, "question": self.question,
                "answer": self.answer, "gold": [g.to_dict() for g in self.gold]}


@lru_cache(maxsize=None)
def _symbols(repo: str, path: str) -> dict[str, tuple[int, int]]:
    spec = get_repo(repo)
    src = read_text(spec.path / path)
    table: dict[str, tuple[int, int]] = {}
    for sym, s, e, _kind in symbol_table(src, language_for(path)):
        # overloads (typing.overload) share a name: keep the longest = the implementation
        if sym not in table or (e - s) > (table[sym][1] - table[sym][0]):
            table[sym] = (s, e)
    return table


def resolve_gold(repo: str, path: str, symbol: str) -> GoldSpan:
    spec = get_repo(repo)
    if not (spec.path / path).exists():
        raise FileNotFoundError(f"{repo}: gold file {path} not found (run `python -m cqa setup`)")
    table = _symbols(repo, path)
    if symbol not in table:
        raise KeyError(f"{repo}: symbol {symbol!r} not found in {path}")
    s, e = table[symbol]
    return GoldSpan(path, symbol, s, e)


def load_questions(repos: list[str] | None = None, limit: int | None = None) -> list[Question]:
    out = []
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if repos and d["repo"] not in repos:
            continue
        gold = [resolve_gold(d["repo"], g["path"], g["symbol"]) for g in d["gold"]]
        out.append(Question(d["id"], d["repo"], d["type"], d["question"], d["answer"], gold))
    return out[:limit] if limit else out


def load_raw() -> list[dict]:
    """Questions without gold resolution (works before repos are cloned)."""
    return [json.loads(l) for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
