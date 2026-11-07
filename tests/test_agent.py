"""ReAct agent: parsing, tools (grep / open_file / find_symbol), loop control, RAG baseline."""
from cqa import config
from cqa.agent import RAGAnswerer, ReActAgent, RepoTools, parse_react
from cqa.llm import LLM


class ScriptedLLM(LLM):
    """Returns pre-written replies in order - lets us test the loop deterministically."""

    def __init__(self, replies):
        super().__init__(config.LLMSettings(provider="offline", model="scripted"))
        self.replies = list(replies)
        self.seen = []
        self.offline_policy = self._next

    def _next(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0) if self.replies else "Final Answer: done"


# ------------------------------------------------------------------ parser
def test_parse_action_with_json_input():
    p = parse_react('Thought: look for it\nAction: grep\nAction Input: {"pattern": "Authorization"}')
    assert p == {"thought": "look for it", "action": "grep", "input": '{"pattern": "Authorization"}'}


def test_parse_final_answer():
    p = parse_react("Thought: I can answer now.\nFinal Answer: It is in [a.py:L1-L2].")
    assert p["final"] == "It is in [a.py:L1-L2]." and p["thought"] == "I can answer now."


def test_parse_ignores_hallucinated_observation():
    p = parse_react("Thought: x\nAction: grep\nAction Input: foo\nObservation: made up\nFinal Answer: no")
    assert p["action"] == "grep" and "made up" not in p["input"]


def test_parse_invalid():
    assert "error" in parse_react("I think the answer is somewhere.")


# ------------------------------------------------------------------- tools
def test_grep_reports_line_numbers(fixture_repo):
    t = RepoTools(fixture_repo)
    res = t.run("grep", {"pattern": "Authorization"})
    assert "pkg/auth.py:" in res.text
    line_no = int(res.text.split("pkg/auth.py:")[1].split(":")[0])
    src = (fixture_repo.path / "pkg/auth.py").read_text().splitlines()
    assert "Authorization" in src[line_no - 1]
    assert t.run("grep", {"pattern": "([invalid"}).text.startswith("No matches")  # bad regex -> literal


def test_open_file_numbers_lines_and_resolves_suffix(fixture_repo):
    t = RepoTools(fixture_repo)
    res = t.run("open_file", {"path": "auth.py", "start_line": 8, "end_line": 9})
    assert res.text.startswith("pkg/auth.py") and "    8  def _encode_credentials" in res.text
    assert res.locations == [{"path": "pkg/auth.py", "start": 8, "end": 9}]
    assert "lines 8-9" in t.run("open_file", "pkg/auth.py:8-9").text   # path:start-end shorthand
    assert "File not found" in t.run("open_file", {"path": "../../etc/passwd"}).text
    assert "File not found" in t.run("open_file", {"path": "nope.py"}).text


def test_find_symbol_and_list_dir(fixture_repo):
    t = RepoTools(fixture_repo)
    assert "pkg/auth.py:L" in t.run("find_symbol", {"name": "BasicAuth.__call__"}).text
    assert "SessionStore.method_3" in t.run("find_symbol", "method_3").text
    assert "No definition" in t.run("find_symbol", "doesNotExist").text
    listing = t.run("list_dir", {"path": "pkg"}).text
    assert "pkg/auth.py" in listing and "tests/" not in listing
    assert t.run("teleport", {}).text.startswith("Unknown tool")


def test_search_code_tool_format(fixture_repo):
    res = RepoTools(fixture_repo).run("search_code", {"query": "basic auth header"})
    assert res.text.startswith("1. ") and ":L" in res.text and res.locations


# -------------------------------------------------------------------- loop
def test_react_loop_executes_tools_and_validates_citations(fixture_repo):
    llm = ScriptedLLM([
        'Thought: grep it\nAction: grep\nAction Input: {"pattern": "def _encode_credentials"}',
        'Thought: read it\nAction: open_file\nAction Input: {"path": "pkg/auth.py", "start_line": 8, "end_line": 11}',
        "Thought: I can answer now.\nFinal Answer: `_encode_credentials` in [pkg/auth.py:L8-L11] base64-encodes "
        "user:password. Also see [nowhere.py:L1-L2].",
    ])
    res = ReActAgent(fixture_repo, llm, seed_search=False).run("How are basic credentials encoded?")
    assert [s["action"] for s in res["trace"]] == ["grep", "open_file"]
    assert "def _encode_credentials" in res["trace"][1]["observation"]
    cites = {c["path"]: c for c in res["citations"]}
    assert cites["pkg/auth.py"]["valid"] and cites["pkg/auth.py"]["start"] == 8
    assert not cites["nowhere.py"]["valid"]
    # the observation was fed back to the model as a user message
    assert any(m["content"].startswith("Observation:") for m in llm.seen[-1])


def test_react_recovers_from_bad_format_and_respects_step_budget(fixture_repo):
    llm = ScriptedLLM(["I am not following the format"] * 10 + ["Final Answer: gave up"])
    agent = ReActAgent(fixture_repo, llm, max_steps=3, seed_search=False)
    events = list(agent.run_stream("anything"))
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) == 3 and steps[0]["action"] == "(invalid)"
    assert events[-1]["type"] == "answer"


def test_offline_agent_end_to_end(fixture_repo):
    res = ReActAgent(fixture_repo).run("Where is the HTTP Basic Authorization header added to the request?")
    assert res["steps"] == 2
    assert res["citations"] and res["citations"][0]["path"] == "pkg/auth.py"
    assert res["citations"][0]["url"].startswith("https://github.com/example/demo/blob/")


def test_rag_offline(fixture_repo):
    res = RAGAnswerer(fixture_repo, k=4).run("basic authentication credentials")
    assert len(res["hits"]) == 4 and res["citations"]
    assert all(c["valid"] for c in res["citations"])
