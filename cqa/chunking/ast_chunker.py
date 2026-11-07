"""Syntax-tree chunking: one chunk per function / class / method.

* Python files are parsed with the standard-library `ast` module (exact line
  numbers, decorators included).
* Other languages (JS/TS, Java, Go, Rust, Ruby, C/C++, C#, PHP, Kotlin) are
  parsed with tree-sitter (`tree-sitter-language-pack`). If no grammar is
  available the file falls back to fixed-size chunking.

Rules
-----
* top-level function            -> one `function` chunk
* class <= AST_MAX_LINES lines  -> one `class` chunk
* bigger class                  -> one `class_header` chunk (signature, docstring,
                                   attributes) + one `method` chunk per method
* function > AST_SPLIT_LONG     -> split into overlapping windows that keep the
                                   symbol name
* code outside any definition   -> `module` chunks (imports, constants, ...)
"""
from __future__ import annotations

import ast
from functools import lru_cache

from .. import config
from .base import Chunk, slice_lines
from .fixed_chunker import chunk_fixed

Span = tuple[int, int, str, str]  # start, end, kind, symbol


# ==========================================================================
# Python (stdlib ast)
# ==========================================================================
def _node_span(node: ast.AST) -> tuple[int, int]:
    decos = [d.lineno for d in getattr(node, "decorator_list", [])]
    start = min(decos + [node.lineno])
    return start, node.end_lineno or node.lineno


def _split_long(start: int, end: int, kind: str, symbol: str) -> list[Span]:
    size = end - start + 1
    if size <= config.AST_SPLIT_LONG:
        return [(start, end, kind, symbol)]
    win, overlap = config.AST_MAX_LINES, 10
    spans, s, part = [], start, 1
    while s <= end:
        e = min(s + win - 1, end)
        spans.append((s, e, kind, f"{symbol} (part {part})"))
        if e == end:
            break
        s, part = e - overlap + 1, part + 1
    return spans


def _python_spans(tree: ast.Module) -> tuple[list[Span], list[tuple[int, int, str]]]:
    spans: list[Span] = []
    classes: list[tuple[int, int, str]] = []

    def visit_function(node, prefix: str, kind: str) -> None:
        s, e = _node_span(node)
        spans.extend(_split_long(s, e, kind, prefix + node.name))

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        name = prefix + node.name
        s, e = _node_span(node)
        classes.append((s, e, name))
        if e - s + 1 <= config.AST_MAX_LINES:
            spans.append((s, e, "class", name))
            return
        members = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        header_end = (_node_span(members[0])[0] - 1) if members else e
        spans.extend(_split_long(s, max(s, header_end), "class_header", name))
        for m in members:
            if isinstance(m, ast.ClassDef):
                visit_class(m, name + ".")
            else:
                visit_function(m, name + ".", "method")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(node, "", "function")
        elif isinstance(node, ast.ClassDef):
            visit_class(node, "")
    return spans, classes


# ==========================================================================
# tree-sitter (other languages)
# ==========================================================================
_TS_RULES: dict[str, dict[str, set[str]]] = {
    "javascript": {
        "func": {"function_declaration", "generator_function_declaration", "method_definition"},
        "class": {"class_declaration", "class"},
        "wrap": {"export_statement"},
        "var": {"lexical_declaration", "variable_declaration"},
        "pass": set(),
    },
    "typescript": {
        "func": {"function_declaration", "generator_function_declaration", "method_definition",
                 "function_signature", "type_alias_declaration"},
        "class": {"class_declaration", "abstract_class_declaration", "interface_declaration",
                  "enum_declaration"},
        "wrap": {"export_statement"},
        "var": {"lexical_declaration", "variable_declaration"},
        "pass": {"internal_module", "module"},
    },
    "java": {
        "func": {"method_declaration", "constructor_declaration"},
        "class": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
        "wrap": set(), "var": set(), "pass": set(),
    },
    "go": {
        "func": {"function_declaration", "method_declaration", "type_declaration"},
        "class": set(), "wrap": set(), "var": set(), "pass": set(),
    },
    "rust": {
        "func": {"function_item", "struct_item", "enum_item", "macro_definition"},
        "class": {"impl_item", "trait_item", "mod_item"},
        "wrap": set(), "var": set(), "pass": set(),
    },
    "ruby": {
        "func": {"method", "singleton_method"},
        "class": {"class", "module"},
        "wrap": set(), "var": set(), "pass": set(),
    },
    "c": {
        "func": {"function_definition", "struct_specifier"},
        "class": set(), "wrap": set(), "var": set(), "pass": set(),
    },
    "cpp": {
        "func": {"function_definition", "template_declaration"},
        "class": {"class_specifier", "struct_specifier"},
        "wrap": set(), "var": set(), "pass": {"namespace_definition", "declaration_list"},
    },
    "csharp": {
        "func": {"method_declaration", "constructor_declaration", "property_declaration"},
        "class": {"class_declaration", "interface_declaration", "struct_declaration", "record_declaration"},
        "wrap": set(), "var": set(),
        "pass": {"namespace_declaration", "file_scoped_namespace_declaration", "declaration_list"},
    },
    "php": {
        "func": {"function_definition", "method_declaration"},
        "class": {"class_declaration", "interface_declaration", "trait_declaration"},
        "wrap": set(), "var": set(), "pass": {"program"},
    },
    "kotlin": {
        "func": {"function_declaration"},
        "class": {"class_declaration", "object_declaration"},
        "wrap": set(), "var": set(), "pass": set(),
    },
}
_TS_RULES["tsx"] = _TS_RULES["typescript"]


@lru_cache(maxsize=None)
def _ts_parser(language: str):
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        return get_parser(language)
    except Exception:
        return None


def _ts_name(node) -> str:
    for field in ("name", "type", "declarator"):
        child = node.child_by_field_name(field)
        if child is None:
            continue
        if child.type in {"identifier", "field_identifier", "type_identifier", "property_identifier",
                          "constant", "name", "simple_identifier"}:
            return child.text.decode("utf8", "replace")
        if field == "declarator":
            return _ts_name(child)
        return child.text.decode("utf8", "replace").split("<")[0][:60]
    # BFS for the first identifier-ish node
    queue = list(node.named_children)[:6]
    while queue:
        c = queue.pop(0)
        if c.type.endswith("identifier") or c.type in {"name", "constant"}:
            return c.text.decode("utf8", "replace")
        queue.extend(list(c.named_children)[:4])
    return ""


def _ts_spans(root, language: str) -> tuple[list[Span], list[tuple[int, int, str]]]:
    rules = _TS_RULES[language]
    spans: list[Span] = []
    classes: list[tuple[int, int, str]] = []

    def line_span(n) -> tuple[int, int]:
        return n.start_point[0] + 1, n.end_point[0] + 1

    def var_is_function(n) -> str:
        for decl in n.named_children:
            if decl.type == "variable_declarator":
                value = decl.child_by_field_name("value")
                if value is not None and value.type in {"arrow_function", "function_expression", "function"}:
                    return _ts_name(decl)
        return ""

    def visit(node, prefix: str, in_class: bool) -> None:
        for child in node.named_children:
            t = child.type
            target, outer = child, child
            if t in rules["wrap"]:
                inner = next((c for c in child.named_children
                              if c.type in rules["func"] | rules["class"] | rules["var"]), None)
                if inner is None:
                    continue
                target, t = inner, inner.type
            if t in rules["class"]:
                name = prefix + (_ts_name(target) or t)
                s, e = line_span(outer)
                classes.append((s, e, name))
                if e - s + 1 <= config.AST_MAX_LINES:
                    spans.append((s, e, "class", name))
                else:
                    before = len(spans)
                    visit(target, name + ".", True)
                    first = min((sp[0] for sp in spans[before:]), default=e + 1)
                    spans.extend(_split_long(s, max(s, first - 1), "class_header", name))
            elif t in rules["func"]:
                s, e = line_span(outer)
                kind = "method" if in_class else (
                    "type" if any(w in t for w in ("type_", "struct", "enum")) else "function")
                spans.extend(_split_long(s, e, kind, prefix + (_ts_name(target) or t)))
            elif t in rules["var"] and not in_class:
                fname = var_is_function(target)
                if fname:
                    s, e = line_span(outer)
                    spans.extend(_split_long(s, e, "function", prefix + fname))
            elif in_class or t in rules["pass"]:
                visit(child, prefix, in_class)

    visit(root, "", False)
    return spans, classes


# ==========================================================================
# Shared: spans -> chunks (+ leftover "module" code)
# ==========================================================================
def _spans_to_chunks(spans: list[Span], classes, lines: list[str], repo: str, path: str,
                     language: str) -> list[Chunk]:
    n = len(lines)
    covered = [False] * (n + 2)
    chunks: list[Chunk] = []
    for s, e, kind, symbol in spans:
        s, e = max(1, s), min(n, e)
        if e < s:
            continue
        for i in range(s, e + 1):
            covered[i] = True
        chunks.append(Chunk(repo, path, s, e, slice_lines(lines, s, e), kind, symbol, language, "ast"))

    # leftover top-level / class-body code, grouped in contiguous runs
    i = 1
    while i <= n:
        if covered[i] or not lines[i - 1].strip():
            i += 1
            continue
        j = i
        while j + 1 <= n and not covered[j + 1]:
            j += 1
        while j > i and not lines[j - 1].strip():  # trim trailing blanks
            j -= 1
        enclosing = [c for c in classes if c[0] <= i and j <= c[1]]
        symbol = min(enclosing, key=lambda c: c[1] - c[0])[2] if enclosing else ""
        kind = "class_body" if symbol else "module"
        step = config.FIXED_CHUNK_LINES
        for s in range(i, j + 1, step):
            e = min(j, s + step - 1)
            text = slice_lines(lines, s, e)
            if text.strip():
                chunks.append(Chunk(repo, path, s, e, text, kind, symbol, language, "ast"))
        i = j + 1

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def chunk_ast(source: str, repo: str, path: str, language: str) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    if language == "python":
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return [_as_ast(c) for c in chunk_fixed(source, repo, path, language)]
        spans, classes = _python_spans(tree)
    else:
        parser = _ts_parser(language) if language in _TS_RULES else None
        if parser is None:
            return [_as_ast(c) for c in chunk_fixed(source, repo, path, language)]
        tree = parser.parse(source.encode("utf8"))
        spans, classes = _ts_spans(tree.root_node, language)
    return _spans_to_chunks(spans, classes, lines, repo, path, language)


def _as_ast(c: Chunk) -> Chunk:
    c.chunker = "ast"
    return c


def symbol_table(source: str, language: str) -> list[tuple[str, int, int, str]]:
    """(symbol, start, end, kind) for every definition - used by find_symbol tool & gold resolution."""
    if language == "python":
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return []
        out = []

        def visit(body, prefix):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    s, e = _node_span(node)
                    out.append((prefix + node.name, s, e, "method" if prefix else "function"))
                elif isinstance(node, ast.ClassDef):
                    s, e = _node_span(node)
                    out.append((prefix + node.name, s, e, "class"))
                    visit(node.body, prefix + node.name + ".")
        visit(tree.body, "")
        return out
    parser = _ts_parser(language) if language in _TS_RULES else None
    if parser is None:
        return []
    saved = config.AST_MAX_LINES, config.AST_SPLIT_LONG
    try:  # force full recursion so every method gets an entry
        config.AST_MAX_LINES, config.AST_SPLIT_LONG = 0, 10**9
        spans, classes = _ts_spans(parser.parse(source.encode("utf8")).root_node, language)
    finally:
        config.AST_MAX_LINES, config.AST_SPLIT_LONG = saved
    out = [(sym, s, e, kind) for s, e, kind, sym in spans if kind != "class_header"]
    out += [(name, s, e, "class") for s, e, name in classes]
    return out
