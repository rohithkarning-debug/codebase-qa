"""AST (syntax-tree) chunking vs fixed-size chunking."""
from cqa import config
from cqa.chunking import chunk_ast, chunk_file, chunk_fixed, chunk_repo, symbol_table
from cqa.repo import iter_source_files

from .conftest import AUTH_PY, SERVER_JS, UTILS_GO, _big_class


def _lines(src):
    return src.splitlines(keepends=True)


def test_ast_chunks_match_source_lines_exactly():
    for src, path, lang in ((AUTH_PY, "a.py", "python"), (SERVER_JS, "s.js", "javascript"),
                            (UTILS_GO, "u.go", "go"), (_big_class(), "b.py", "python")):
        lines = _lines(src)
        for c in chunk_ast(src, "r", path, lang):
            assert c.text == "".join(lines[c.start_line - 1:c.end_line]), c.symbol
            assert c.chunker == "ast"


def test_python_functions_and_classes_become_chunks():
    chunks = {c.symbol: c for c in chunk_ast(AUTH_PY, "r", "a.py", "python")}
    assert chunks["_encode_credentials"].kind == "function"
    assert chunks["BasicAuth"].kind == "class"          # small class stays whole
    assert chunks["BasicAuth"].text.startswith("class BasicAuth")
    module = [c for c in chunk_ast(AUTH_PY, "r", "a.py", "python") if c.kind == "module"]
    assert any("import base64" in c.text for c in module)


def test_decorators_are_part_of_the_function_chunk():
    c = next(c for c in chunk_ast(AUTH_PY, "r", "a.py", "python") if c.symbol == "decorated_helper")
    assert c.text.lstrip().startswith("@staticmethod")


def test_big_class_is_split_into_header_and_methods():
    src = _big_class()
    assert len(src.splitlines()) > config.AST_MAX_LINES
    chunks = chunk_ast(src, "r", "b.py", "python")
    kinds = {c.kind for c in chunks}
    assert "class_header" in kinds and "method" in kinds
    header = next(c for c in chunks if c.kind == "class_header")
    assert "timeout = 30" in header.text and "def method_0" not in header.text
    assert any(c.symbol == "SessionStore.method_24" for c in chunks)


def test_tree_sitter_languages():
    js = {c.symbol: c.kind for c in chunk_ast(SERVER_JS, "r", "s.js", "javascript")}
    assert js.get("handleLogin") == "function"
    assert js.get("normalizePath") == "function"   # arrow function assigned to const
    assert js.get("Router") == "class"
    go = {c.symbol for c in chunk_ast(UTILS_GO, "r", "u.go", "go")}
    assert "Add" in go


def test_syntax_error_falls_back_to_windows():
    chunks = chunk_ast("def oops(:\n    pass\n" * 30, "r", "x.py", "python")
    assert chunks and all(c.kind == "window" and c.chunker == "ast" for c in chunks)


def test_fixed_windows_cover_file_with_overlap():
    src = "".join(f"line {i}\n" for i in range(1, 101))
    chunks = chunk_fixed(src, "r", "x.py", "python", size=40, overlap=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (31, 70), (61, 100)]
    assert all(c.kind == "window" and c.chunker == "fixed" for c in chunks)


def test_fixed_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_fixed("a\n", "r", "x.py", "python", size=10, overlap=10)


def test_fixed_cuts_through_functions_but_ast_does_not():
    ast_chunks = chunk_ast(_big_class(), "r", "b.py", "python")
    ast_starts = {c.start_line for c in ast_chunks}
    fixed = chunk_fixed(_big_class(), "r", "b.py", "python")
    # fixed windows start in the middle of definitions; every AST method starts at its "def"
    assert any(c.start_line not in ast_starts for c in fixed[1:])
    assert all(c.text.lstrip().startswith("def ") for c in ast_chunks if c.kind == "method")


def test_symbol_table_lists_methods():
    syms = {s for s, *_ in symbol_table(AUTH_PY, "python")}
    assert {"BasicAuth", "BasicAuth.__call__", "_encode_credentials"} <= syms
    js_syms = {s for s, *_ in symbol_table(SERVER_JS, "javascript")}
    assert "Router.route" in js_syms


def test_repo_walk_skips_tests_and_non_code(fixture_repo):
    files = [p.as_posix() for p in iter_source_files(fixture_repo)]
    assert "pkg/auth.py" in files and "pkg/server.js" in files
    assert not any(f.startswith("tests/") for f in files)
    assert "README.md" not in files
    for chunker in ("ast", "fixed"):
        chunks = chunk_repo(fixture_repo, chunker)
        assert {c.path for c in chunks} == set(files)


def test_chunk_file_unknown_chunker():
    import pytest
    with pytest.raises(ValueError):
        chunk_file("x = 1\n", "r", "a.py", "nope")
