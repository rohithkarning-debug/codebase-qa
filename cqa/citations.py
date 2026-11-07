"""Find `path:Lstart-Lend` citations in an answer and turn them into links."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from . import config
from .repo import RepoSpec, iter_source_files

_EXT = "|".join(sorted({e.lstrip(".") for e in config.LANGUAGE_BY_EXT}, key=len, reverse=True))
# src/requests/auth.py:L76-L96 | auth.py:76-96 | auth.py#L76 | [auth.py] | `auth.py`
CITE_RE = re.compile(
    rf"(?P<path>[A-Za-z0-9_][\w./\\-]*\.(?:{_EXT}))\b"
    rf"(?:\s*(?::|#|,?\s+lines?)\s*L?(?P<start>\d+)(?:\s*(?:-|–|to)\s*L?(?P<end>\d+))?)?"
)


@dataclass
class Citation:
    path: str
    start: int | None
    end: int | None
    url: str = ""
    valid: bool = True

    def overlaps(self, path: str, start: int, end: int) -> bool:
        if self.path != path:
            return False
        if self.start is None:
            return False
        s, e = self.start, self.end or self.start
        return s <= end and start <= e

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve(path: str, files: list[str]) -> str | None:
    path = path.replace("\\", "/").lstrip("./")
    if path in files:
        return path
    cands = [f for f in files if f.endswith("/" + path)]
    return cands[0] if len(cands) == 1 else None


def extract_citations(answer: str, spec: RepoSpec | None = None,
                      files: list[str] | None = None) -> list[Citation]:
    if files is None and spec is not None:
        files = [p.as_posix() for p in iter_source_files(spec)]
    seen, out = set(), []
    for m in CITE_RE.finditer(answer or ""):
        raw = m.group("path")
        path =_resolve(raw, files) if files is not None else raw
        valid = path is not None
        path = path or raw
        start = int(m.group("start")) if m.group("start") else None
        end = int(m.group("end")) if m.group("end") else start
        if start and end and end < start:
            start, end = end, start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        url = spec.github_url(path, start, end) if (spec and valid) else ""
        out.append(Citation(path, start, end, url, valid))
    # prefer citations with line numbers; drop bare duplicates of the same file
    with_lines = {c.path for c in out if c.start}
    return [c for c in out if c.start or c.path not in with_lines]
