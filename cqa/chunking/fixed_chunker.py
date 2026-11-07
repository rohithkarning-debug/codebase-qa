"""Fixed-size chunking baseline: sliding windows of N lines with overlap.

It ignores the syntax tree completely, so a function can be cut in half and a
chunk can contain the end of one function and the start of the next. That is
exactly the behaviour we compare the AST chunker against.
"""
from __future__ import annotations

from .. import config
from .base import Chunk, slice_lines


def chunk_fixed(source: str, repo: str, path: str, language: str,
                size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or config.FIXED_CHUNK_LINES
    overlap = config.FIXED_OVERLAP_LINES if overlap is None else overlap
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    lines = source.splitlines(keepends=True)
    n = len(lines)
    chunks: list[Chunk] = []
    start = 1
    while start <= n:
        end = min(n, start + size - 1)
        text = slice_lines(lines, start, end)
        if text.strip():
            chunks.append(Chunk(repo, path, start, end, text, "window", "", language, "fixed"))
        if end == n:
            break
        start = end - overlap + 1
    return chunks
