"""Single-shot retrieval-augmented answering (baseline for the ReAct agent)."""
from __future__ import annotations

import re
import time

from .. import config
from ..citations import extract_citations
from ..index import get_index
from ..llm import LLM, make_llm
from ..repo import RepoSpec

RAG_PROMPT = """You answer questions about the `{repo}` repository using ONLY the code excerpts below.

{context}

Question: {question}

Instructions:
- Start with the direct answer: which file(s) and function/class are responsible.
- Cite every location you use as [path/to/file.ext:L<start>-L<end>] (exact path and lines from the excerpt headers; you may narrow the range).
- Then explain briefly how it works (3-8 sentences or bullets). Use `backticks` for identifiers.
- If the excerpts do not contain the answer, say so."""


def format_context(hits) -> str:
    blocks = []
    for h in hits:
        c = h.chunk
        numbered = "\n".join(f"{c.start_line + i:>5}  {line}" for i, line in enumerate(c.text.splitlines()))
        title = f"[{c.path}:L{c.start_line}-L{c.end_line}]" + (f" {c.kind} {c.symbol}" if c.symbol else "")
        blocks.append(f"### {title}\n```\n{numbered}\n```")
    return "\n\n".join(blocks)


def offline_rag_policy(messages: list[dict]) -> str:
    prompt = messages[-1]["content"]
    heads = re.findall(r"### \[(.+?):L(\d+)-L(\d+)\](.*)", prompt)
    if not heads:
        return "I could not find relevant code."
    p, s, e, label = heads[0]
    rest = ", ".join(f"[{a}:L{b}-L{c}]" for a, b, c, _ in heads[1:3])
    label = label.strip().split(" ", 1)[-1] if label.strip() else p
    ans = f"The most relevant code is `{label}` in [{p}:L{s}-L{e}]."
    if rest:
        ans += f"\n\nRelated locations: {rest}."
    return ans + "\n\n_(offline mode: no LLM API key configured - answer composed from the top search results)_"


class RAGAnswerer:
    def __init__(self, spec: RepoSpec, llm: LLM | None = None, chunker: str = config.DEFAULT_CHUNKER,
                 retriever: str = config.DEFAULT_RETRIEVER, k: int = 6):
        self.spec, self.chunker, self.retriever, self.k = spec, chunker, retriever, k
        self.llm = llm or make_llm(offline_policy=offline_rag_policy)
        if self.llm.is_offline:
            self.llm.offline_policy = offline_rag_policy

    def run(self, question: str) -> dict:
        t0 = time.time()
        idx = get_index(self.spec, self.chunker)
        hits = idx.search(question, self.retriever, k=self.k)
        prompt = RAG_PROMPT.format(repo=self.spec.name, context=format_context(hits), question=question)
        answer = self.llm.chat([{"role": "user", "content": prompt}])
        files = sorted({c.path for c in idx.chunks})
        cites = extract_citations(answer, self.spec, files=files)
        return {"type": "answer", "answer": answer.strip(), "citations": [c.to_dict() for c in cites],
                "hits": [h.to_dict(self.spec) for h in hits], "seconds": round(time.time() - t0, 2),
                "llm": self.llm.name, "steps": 0, "trace": []}
