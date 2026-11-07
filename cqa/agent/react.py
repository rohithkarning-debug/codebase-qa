"""ReAct agent (Reason + Act, Yao et al. 2022) that explores a repository with tools.

Loop:  Thought -> Action -> Action Input -> Observation -> ... -> Final Answer

The text protocol works with every chat model (no native tool-calling needed).
`run_stream()` yields events so the web UI can show each step live.
"""
from __future__ import annotations

import json
import re
import time
from typing import Iterator

from .. import config
from ..citations import extract_citations
from ..llm import LLM, make_llm
from ..repo import RepoSpec
from .tools import TOOL_DOCS, RepoTools, parse_args

SYSTEM_PROMPT = """You are an expert software engineer answering questions about the `{repo}` repository ({url}).
Explore the code with tools before answering. Use EXACTLY this format:

Thought: <reason about what to do next>
Action: <one of: {tool_names}>
Action Input: <JSON object with the arguments>

After each action you receive "Observation: <tool output>". Repeat as needed (max {max_steps} actions).
When you know the answer, reply with:

Thought: I can answer now.
Final Answer: <markdown answer>

Tools:
{tool_docs}

Rules for the Final Answer:
- Start with the direct answer: the file(s) and function/class responsible.
- Cite every location you rely on as [path/to/file.ext:L<start>-L<end>] with the exact path and the line numbers shown in the observations.
- Then explain briefly (3-8 sentences or bullets) how the code works. Mention key function names in `backticks`.
- Never invent files, functions or line numbers. If you cannot find it, say so.
"""

ACTION_RE = re.compile(r"Action\s*:\s*`?([A-Za-z_]+)`?", re.I)
INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.*)", re.I | re.S)
FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.I | re.S)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*:|\Z)", re.I | re.S)


def parse_react(text: str) -> dict:
    """Parse one model turn into {thought, action, input} or {thought, final}."""
    text = (text or "").split("\nObservation:")[0].strip()
    thought_m = THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = FINAL_RE.search(text)
    action_m = ACTION_RE.search(text)
    if final_m and (not action_m or final_m.start() < action_m.start()):
        return {"thought": thought, "final": final_m.group(1).strip()}
    if action_m:
        input_m = INPUT_RE.search(text, action_m.end())
        raw = input_m.group(1).strip() if input_m else ""
        return {"thought": thought, "action": action_m.group(1), "input": raw}
    return {"thought": thought or text, "error": "no Action or Final Answer found"}


class ReActAgent:
    def __init__(self, spec: RepoSpec, llm: LLM | None = None, chunker: str = config.DEFAULT_CHUNKER,
                 retriever: str = config.DEFAULT_RETRIEVER, max_steps: int = config.AGENT_MAX_STEPS,
                 seed_search: bool = True):
        self.spec = spec
        self.tools = RepoTools(spec, chunker, retriever)
        self.llm = llm or make_llm(offline_policy=offline_react_policy)
        if self.llm.is_offline and self.llm.offline_policy is None:
            self.llm.offline_policy = offline_react_policy
        self.max_steps = max_steps
        self.seed_search = seed_search

    def system_prompt(self) -> str:
        docs = "\n".join(f"- {n} {sig}: {d}" for n, (sig, d) in TOOL_DOCS.items())
        return SYSTEM_PROMPT.format(repo=self.spec.name, url=self.spec.url, tool_names=", ".join(TOOL_DOCS),
                                    max_steps=self.max_steps, tool_docs=docs)

    # ---------------------------------------------------------------- loop
    def run_stream(self, question: str) -> Iterator[dict]:
        t0 = time.time()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": f"Question: {question}"}]
        steps = 0
        yield {"type": "start", "question": question, "repo": self.spec.name, "llm": self.llm.name}

        if self.seed_search:  # give the model a head start with one semantic search
            thought = "I'll start with a semantic search over the code for the question."
            res = self.tools.run("search_code", {"query": question})
            messages.append({"role": "assistant", "content":
                             f"Thought: {thought}\nAction: search_code\nAction Input: "
                             f"{json.dumps({'query': question})}"})
            messages.append({"role": "user", "content": f"Observation: {res.text}"})
            steps += 1
            yield self._step_event(steps, thought, "search_code", {"query": question}, res)

        final = None
        while steps < self.max_steps and final is None:
            reply = self.llm.chat(messages, stop=["\nObservation:", "Observation:"])
            parsed = parse_react(reply)
            messages.append({"role": "assistant", "content": reply.strip()})
            if "final" in parsed:
                final = parsed["final"]
                yield {"type": "thought", "thought": parsed["thought"]}
                break
            if "error" in parsed:
                obs = ("Invalid format. Reply with 'Thought:', 'Action:' and 'Action Input:' lines, "
                       "or with 'Final Answer:'.")
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                steps += 1
                yield {"type": "step", "step": steps, "thought": parsed["thought"][:500], "action": "(invalid)",
                       "input": {}, "observation": obs, "locations": []}
                continue
            res = self.tools.run(parsed["action"], parsed["input"])
            messages.append({"role": "user", "content": f"Observation: {res.text}"})
            steps += 1
            yield self._step_event(steps, parsed["thought"], parsed["action"], parse_args(parsed["input"]), res)

        if final is None:  # out of budget: force an answer from what we have
            messages.append({"role": "user", "content":
                             "You have used all your actions. Give your Final Answer now, in the required format."})
            reply = self.llm.chat(messages)
            parsed = parse_react(reply)
            final = parsed.get("final") or reply.strip()

        cites = extract_citations(final, self.spec, files=self.tools.files)
        yield {"type": "answer", "answer": final, "citations": [c.to_dict() for c in cites],
               "steps": steps, "seconds": round(time.time() - t0, 2), "llm": self.llm.name}

    def run(self, question: str) -> dict:
        trace, result = [], {}
        for ev in self.run_stream(question):
            if ev["type"] == "step":
                trace.append(ev)
            elif ev["type"] == "answer":
                result = ev
        result["trace"] = trace
        return result

    @staticmethod
    def _step_event(n, thought, action, args, res) -> dict:
        return {"type": "step", "step": n, "thought": thought, "action": action, "input": args,
                "observation": res.text, "locations": res.locations}


# ==========================================================================
# Offline policy: a deterministic "model" so the full loop runs without a key
# ==========================================================================
_LOC_RE = re.compile(r"^(?:Observation:)?[ \t]*\d+\.\s+(\S+):L(\d+)-L(\d+)\s+(.*?)\s+\(score", re.M)


def offline_react_policy(messages: list[dict]) -> str:
    """search -> open the best hit -> answer from it. Mimics the ReAct text format."""
    question = next((m["content"][len("Question: "):] for m in messages
                     if m["role"] == "user" and m["content"].startswith("Question: ")), "")
    observations = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Observation:")]
    actions = [ACTION_RE.search(m["content"]).group(1) for m in messages
               if m["role"] == "assistant" and ACTION_RE.search(m["content"])]
    if messages and messages[-1]["role"] == "user" and "Give your Final Answer now" in messages[-1]["content"]:
        return "Final Answer: I could not finish exploring within the step budget."
    search_obs = next((o for o, a in zip(observations, actions) if a == "search_code"), None)
    if search_obs is None:
        return ("Thought: I should search the code first.\nAction: search_code\n"
                f"Action Input: {json.dumps({'query': question})}")
    hits = _LOC_RE.findall(search_obs)
    if not hits:
        return "Thought: Nothing relevant was found.\nFinal Answer: I could not find relevant code for this question."
    path, s, e, label = hits[0]
    if "open_file" not in actions:
        return (f"Thought: The best match is {label} in {path}. Let me read it.\nAction: open_file\n"
                f"Action Input: {json.dumps({'path': path, 'start_line': int(s), 'end_line': int(e)})}")
    opened = observations[-1]
    doc = ""
    m = re.search(r'"""(.*?)(?:"""|\n\s*\n)', opened, re.S)
    if m:
        doc = " ".join(re.sub(r"^\s*\d+\s+", "", l).strip() for l in m.group(1).splitlines()).strip()
    others = ", ".join(f"[{p}:L{a}-L{b}]" for p, a, b, _ in hits[1:3])
    label = label.split(" ", 1)[-1] if " " in label else label
    answer = (f"The most relevant code is `{label}` in [{path}:L{s}-L{e}].")
    if doc:
        answer += f"\n\nIts docstring says: {doc[:400]}"
    if others:
        answer += f"\n\nRelated locations: {others}."
    answer += "\n\n_(offline mode: no LLM API key configured - answer composed from the top search results)_"
    return f"Thought: I have read the most relevant code.\nFinal Answer: {answer}"
