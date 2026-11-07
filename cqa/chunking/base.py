"""Shared chunk data structure."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    repo: str
    path: str            # repo-relative POSIX path
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    text: str            # exact source lines start..end
    kind: str            # function | method | class | class_header | module | window
    symbol: str = ""     # qualified name, e.g. "HTTPDigestAuth.handle_401"
    language: str = ""
    chunker: str = ""    # "ast" | "fixed"

    @property
    def id(self) -> str:
        raw = f"{self.repo}:{self.path}:{self.start_line}:{self.end_line}:{self.chunker}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    @property
    def location(self) -> str:
        return f"{self.path}:L{self.start_line}-L{self.end_line}"

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1

    def embedding_text(self, max_chars: int = 2000) -> str:
        """Text fed to the embedding model: a small header + the code itself.

        The header carries the file path (both chunkers) and the symbol name
        (only the AST chunker knows it) because both are strong retrieval cues.
        """
        header = f"# file: {self.path}\n"
        if self.symbol:
            header += f"# {self.kind}: {self.symbol}\n"
        return (header + self.text)[:max_chars]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def slice_lines(lines: list[str], start: int, end: int) -> str:
    """Join 1-based inclusive line range."""
    return "".join(lines[start - 1:end])
