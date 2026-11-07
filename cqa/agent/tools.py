"""Tools the ReAct agent can call: semantic search, grep, open_file, find_symbol, list_dir."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .. import config
from ..chunking import symbol_table
from ..index import get_index
from ..repo import RepoSpec, iter_source_files, language_for, read_text, safe_join

MAX_OBS_CHARS = 6000
MAX_OPEN_LINES = 150

TOOL_DOCS = {
    "search_code": ('{"query": "natural language or identifiers"}',
                    "semantic search over the indexed code; returns the best matching functions/classes "
                    "with file paths and line ranges"),
    "grep": ('{"pattern": "regex", "path": "optional dir or file prefix"}',
             "regex search (case-insensitive) over source files; returns matching lines with line numbers"),
    "open_file": ('{"path": "file path", "start_line": 1, "end_line": 80}',
                  f"read a file with line numbers (at most {MAX_OPEN_LINES} lines per call)"),
    "find_symbol": ('{"name": "function, Class or Class.method"}',
                    "find where a function / class / method is defined"),
    "list_dir": ('{"path": "optional directory"}', "list source files under a directory"),
}


@dataclass
class ToolResult:
    text: str
    locations: list[dict] = field(default_factory=list)  # [{path,start,end}] touched by the tool


class RepoTools:
    def __init__(self, spec: RepoSpec, chunker: str = config.DEFAULT_CHUNKER,
                 retriever: str = config.DEFAULT_RETRIEVER):
        self.spec, self.chunker, self.retriever = spec, chunker, retriever

    # -------------------------------------------------------------- helpers
    @cached_property
    def files(self) -> list[str]:
        return [p.as_posix() for p in iter_source_files(self.spec)]

    @cached_property
    def symbols(self) -> list[tuple[str, str, int, int, str]]:
        out = []
        for rel in self.files:
            lang = language_for(rel)
            src = read_text(self.spec.path / rel)
            for sym, s, e, kind in symbol_table(src, lang):
                out.append((sym, rel, s, e, kind))
        return out

    def resolve_path(self, path: str) -> str | None:
        """Accept exact repo paths and unique suffixes (e.g. 'auth.py' -> 'src/requests/auth.py')."""
        path = path.strip().strip("`'\"").lstrip("./").replace("\\", "/")
        if not path:
            return None
        if path in self.files:
            return path
        try:
            p = safe_join(self.spec, path)
            if p.is_file():
                return path
        except PermissionError:
            return None
        matches = [f for f in self.files if f.endswith("/" + path) or f == path]
        return matches[0] if len(matches) == 1 else None

    # ---------------------------------------------------------------- tools
    def search_code(self, query: str, k: int = 5) -> ToolResult:
        idx = get_index(self.spec, self.chunker)
        hits = idx.search(query, self.retriever, k=k)
        if not hits:
            return ToolResult("No results.")
        lines, locs = [], []
        for h in hits:
            c = h.chunk
            label = f"{c.kind} {c.symbol}".strip() if c.symbol else c.kind
            lines.append(f"{h.rank}. {c.path}:L{c.start_line}-L{c.end_line}  {label}  (score {h.score:.3f})")
            snippet = [l.rstrip() for l in c.text.splitlines() if l.strip()][:6]
            lines.extend("     | " + s[:140] for s in snippet)
            locs.append({"path": c.path, "start": c.start_line, "end": c.end_line})
        return ToolResult("\n".join(lines), locs)

    def grep(self, pattern: str, path: str = "", max_results: int = 40) -> ToolResult:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        prefix = path.strip().lstrip("./")
        out, locs, total = [], [], 0
        for rel in self.files:
            if prefix and not rel.startswith(prefix):
                continue
            for no, line in enumerate(read_text(self.spec.path / rel).splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(out) < max_results:
                        out.append(f"{rel}:{no}: {line.strip()[:160]}")
                        locs.append({"path": rel, "start": no, "end": no})
        if not out:
            return ToolResult(f"No matches for /{pattern}/" + (f" under {prefix}" if prefix else ""))
        more = f"\n... {total - len(out)} more matches (narrow the pattern or path)" if total > len(out) else ""
        return ToolResult("\n".join(out) + more, locs)

    def open_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        rel = self.resolve_path(path)
        if rel is None:
            near = [f for f in self.files if Path(path).name in f][:8]
            hint = ("Did you mean: " + ", ".join(near)) if near else "Use list_dir or grep to find files."
            return ToolResult(f"File not found: {path}. {hint}")
        lines = read_text(self.spec.path / rel).splitlines()
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line else start + 99
        end = min(len(lines), end, start + MAX_OPEN_LINES - 1)
        if start > len(lines):
            return ToolResult(f"{rel} has only {len(lines)} lines.")
        body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        tail = f"\n[showing {rel} lines {start}-{end} of {len(lines)}]"
        return ToolResult(f"{rel}\n{body}{tail}", [{"path": rel, "start": start, "end": end}])

    def find_symbol(self, name: str) -> ToolResult:
        q = name.strip().strip("`()").lower()
        if not q:
            return ToolResult("Provide a symbol name.")
        exact = [s for s in self.symbols if s[0].lower() == q or s[0].lower().endswith("." + q)]
        found = exact or [s for s in self.symbols if q in s[0].lower()]
        if not found:
            return ToolResult(f"No definition named {name!r}. Try grep or search_code.")
        found = found[:15]
        text = "\n".join(f"{kind} {sym} -> {rel}:L{s}-L{e}" for sym, rel, s, e, kind in found)
        return ToolResult(text, [{"path": rel, "start": s, "end": e} for _, rel, s, e, _ in found])

    def list_dir(self, path: str = "") -> ToolResult:
        prefix = path.strip().strip("/").lstrip("./")
        files = [f for f in self.files if not prefix or f.startswith(prefix + "/") or f == prefix]
        if not files:
            return ToolResult(f"No source files under {path!r}.")
        shown = files[:80]
        extra = f"\n... and {len(files) - 80} more" if len(files) > 80 else ""
        return ToolResult("\n".join(shown) + extra)

    # ------------------------------------------------------------- dispatch
    def run(self, name: str, raw_input) -> ToolResult:
        args = parse_args(raw_input)
        name = name.strip().lower()
        try:
            if name == "search_code":
                res = self.search_code(str(args.get("query") or args.get("_") or ""))
            elif name == "grep":
                res = self.grep(str(args.get("pattern") or args.get("_") or ""), str(args.get("path", "")))
            elif name == "open_file":
                path = str(args.get("path") or args.get("_") or "")
                m = re.match(r"(.+?):L?(\d+)(?:-L?(\d+))?$", path)
                if m and not args.get("start_line"):
                    path, args["start_line"], args["end_line"] = m.group(1), m.group(2), m.group(3)
                res = self.open_file(path, _int(args.get("start_line"), 1), _int(args.get("end_line"), None))
            elif name == "find_symbol":
                res = self.find_symbol(str(args.get("name") or args.get("_") or ""))
            elif name == "list_dir":
                res = self.list_dir(str(args.get("path") or args.get("_") or ""))
            else:
                res = ToolResult(f"Unknown tool {name!r}. Available: {', '.join(TOOL_DOCS)}")
        except Exception as e:  # never crash the agent loop
            res = ToolResult(f"Tool error: {type(e).__name__}: {e}")
        if len(res.text) > MAX_OBS_CHARS:
            res.text = res.text[:MAX_OBS_CHARS] + "\n... [truncated]"
        return res


def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_args(raw) -> dict:
    """Tool input may be JSON, a JSON-ish dict, or a bare string."""
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass
    return {"_": text.strip().strip('"').strip("'")}
