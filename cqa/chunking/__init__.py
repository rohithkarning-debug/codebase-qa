"""Chunking strategies: syntax-tree (AST) vs fixed-size windows."""
from __future__ import annotations

from ..repo import RepoSpec, iter_source_files, language_for, read_text
from .ast_chunker import chunk_ast, symbol_table
from .base import Chunk
from .fixed_chunker import chunk_fixed

CHUNKERS = {"ast": chunk_ast, "fixed": chunk_fixed}


def chunk_file(source: str, repo: str, path: str, chunker: str) -> list[Chunk]:
    language = language_for(path) or "text"
    if chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker {chunker!r}; choose from {list(CHUNKERS)}")
    return CHUNKERS[chunker](source, repo, path, language)


def chunk_repo(spec: RepoSpec, chunker: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rel in iter_source_files(spec):
        source = read_text(spec.path / rel)
        chunks.extend(chunk_file(source, spec.name, rel.as_posix(), chunker))
    return chunks


__all__ = ["Chunk", "chunk_ast", "chunk_fixed", "chunk_file", "chunk_repo", "symbol_table", "CHUNKERS"]
