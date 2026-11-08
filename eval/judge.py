"""LLM-as-Judge: grade a generated answer against the expert reference answer."""
from __future__ import annotations

import json
import re

from cqa.embeddings import code_tokens
from cqa.llm import LLM

JUDGE_PROMPT = """You are grading an answer to a question about the source code of the `{repo}` repository.

Question: {question}

Reference answer (written by an expert who read the code):
{reference}

Gold code locations: {gold}

Candidate answer:
\"\"\"
{candidate}
\"\"\"

Grade the candidate's technical correctness against the reference answer.
- It does not need to mention every detail, but its main claim (which code is responsible and how it works) must agree with the reference.
- Penalize wrong files/functions, invented behaviour or significant false statements.
- Ignore formatting and citation style.

Scale: 5 = fully correct and complete; 4 = correct with minor omissions; 3 = partially correct or vague;
2 = mostly wrong; 1 = wrong, irrelevant or no answer.

Return ONLY a JSON object: {{"score": <1-5>, "correct": <true if score >= 4 else false>, "reason": "<one sentence>"}}"""


def build_prompt(question, candidate: str) -> str:
    gold = ", ".join(f"{g.path}:L{g.start}-L{g.end} ({g.symbol})" for g in question.gold)
    return JUDGE_PROMPT.format(repo=question.repo, question=question.question, reference=question.answer,
                               gold=gold, candidate=(candidate or "(empty)")[:6000])


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            score = int(round(float(d.get("score", 1))))
            score = min(5, max(1, score))
            return {"score": score, "correct": score >= 4, "reason": str(d.get("reason", ""))[:300]}
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = re.search(r"score\"?\s*[:=]\s*([1-5])", text or "", re.I)
    score = int(m.group(1)) if m else 1
    return {"score": score, "correct": score >= 4, "reason": "unparsed judge output"}


def heuristic_verdict(question, candidate: str, citations: list[dict]) -> dict:
    """Offline fallback (no LLM): identifier overlap with the reference + right file.

    Clearly labelled as heuristic in the report - not a substitute for the LLM judge.
    """
    ref = {t for t in code_tokens(question.answer) if len(t) > 3}
    got = set(code_tokens(candidate))
    overlap = len(ref & got) / max(len(ref), 1)
    file_ok = any(c["path"] in question.gold_files for c in citations)
    score = 1 + 2 * file_ok + round(2 * min(1.0, overlap / 0.5))
    score = min(5, score)
    return {"score": score, "correct": score >= 4,
            "reason": f"heuristic: file_match={file_ok} keyword_overlap={overlap:.2f}"}


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm

    @property
    def name(self) -> str:
        return "heuristic (offline)" if self.llm.is_offline else self.llm.name

    def grade(self, question, candidate: str, citations: list[dict]) -> dict:
        if self.llm.is_offline:
            return heuristic_verdict(question, candidate, citations)
        out = self.llm.chat([{"role": "user", "content": build_prompt(question, candidate)}],
                            max_tokens=1500, temperature=0.0)
        return parse_verdict(out)
