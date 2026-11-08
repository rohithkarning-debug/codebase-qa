"""Evaluation metrics, LLM-as-Judge parsing and report generation."""
import json

from cqa import config
from eval import report
from eval.dataset import GoldSpan, Question
from eval.judge import Judge, build_prompt, heuristic_verdict, parse_verdict
from eval.metrics import (citation_scores, first_file_rank, first_hit_rank, is_hit, mrr, recall_at_k,
                          span_overlap)

GOLD = [GoldSpan("a.py", "f", 10, 19)]   # a 10-line function


def test_span_overlap():
    assert span_overlap(1, 10, 5, 20) == 6
    assert span_overlap(1, 4, 5, 20) == 0


def test_hit_rule_half_of_smaller_span():
    assert is_hit("a.py", 1, 40, GOLD)          # window fully contains the function
    assert is_hit("a.py", 15, 30, GOLD)         # 5 of 10 lines = 50%
    assert not is_hit("a.py", 17, 60, GOLD)     # only 3 lines
    assert not is_hit("b.py", 10, 19, GOLD)     # wrong file
    assert is_hit("a.py", 12, 14, GOLD)         # small chunk inside the function


def test_ranks_recall_mrr():
    retrieved = [("b.py", 1, 5), ("a.py", 40, 50), ("a.py", 9, 20)]
    assert first_hit_rank(retrieved, GOLD) == 3
    assert first_file_rank(retrieved, {"a.py"}) == 2
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert abs(mrr(ranks) - (1 + 1 / 3 + 1 / 2) / 4) < 1e-9


def test_citation_scores():
    s = citation_scores([{"path": "b.py", "start": 1, "end": 2}, {"path": "a.py", "start": 18, "end": 30}], GOLD)
    assert s == {"file_hit": True, "line_hit": True, "top_file": False, "precision": 0.5, "n_citations": 2}
    s = citation_scores([{"path": "a.py", "start": None, "end": None}], GOLD)
    assert s["file_hit"] and not s["line_hit"] and s["top_file"]
    assert citation_scores([], GOLD)["file_hit"] is False


def test_parse_verdict_variants():
    assert parse_verdict('{"score": 5, "correct": true, "reason": "ok"}') == {"score": 5, "correct": True, "reason": "ok"}
    assert parse_verdict('Sure!\n```json\n{"score": 3, "reason": "vague"}\n```')["correct"] is False
    assert parse_verdict('"score": 4 because...')["score"] == 4
    assert parse_verdict("garbage")["score"] == 1
    assert parse_verdict('{"score": 9}')["score"] == 5


def test_judge_prompt_and_heuristic():
    q = Question("x-1", "demo", "locate", "Where is auth?", "`BasicAuth.__call__` sets the Authorization header.", GOLD)
    prompt = build_prompt(q, "It is in a.py")
    assert "Reference answer" in prompt and "a.py:L10-L19 (f)" in prompt
    good = heuristic_verdict(q, "BasicAuth __call__ sets the Authorization header", [{"path": "a.py"}])
    bad = heuristic_verdict(q, "No idea", [])
    assert good["correct"] and not bad["correct"]


def test_offline_judge_uses_heuristic():
    from cqa.llm import make_llm
    j = Judge(make_llm("offline"))
    assert j.name.startswith("heuristic")
    q = Question("x-1", "demo", "locate", "Where?", "BasicAuth header", GOLD)
    assert "heuristic" in j.grade(q, "BasicAuth header", [{"path": "a.py"}])["reason"]


def test_report_builds_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    cfg = {f"recall@{k}": 0.5 for k in (1, 3, 5, 10)}
    cfg.update({f"file_recall@{k}": 0.6 for k in (1, 3, 5, 10)}, **{"mrr@10": 0.4, "file_acc@1": 0.6, "n": 2})
    cfg["by_repo"] = {"requests": dict(cfg)}
    cfg["by_type"] = {"locate": dict(cfg)}
    retrieval = {"ks": [1, 3, 5, 10], "n_questions": 2, "hit_rule": "rule", "embed_backend": "hash",
                 "embedders": {"code": {"model": "c"}, "general": {"model": "g"}},
                 "configs": {"ast+code": cfg, "fixed+code": cfg, "ast+general": cfg, "fixed+general": cfg},
                 "chunk_stats": {"ast": {"requests": {"n_files": 1, "n_chunks": 2, "avg_lines": 3, "max_lines": 4}}},
                 "skipped": {}}
    m = {"n": 1, "errors": 0, "judge_correct": 1.0, "judge_score": 5, "file_acc": 1.0, "line_acc": 1.0,
         "top_file_acc": 1.0, "citation_precision": 1.0, "avg_steps": 3, "avg_seconds": 2.0}
    answers = {"llm": "x:y", "judge": "x:z", "index": "ast+code", "n_questions": 1,
               "modes": {"agent": {**m, "by_repo": {"requests": m}, "by_type": {"locate": m}}},
               "rows": [{"mode": "agent", "id": "q", "repo": "requests", "question": "?", "judge": {"correct": True}}]}
    (tmp_path / "retrieval.json").write_text(json.dumps(retrieval))
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    text = report.main()
    assert "Recall@5" in text and "AST (function/class) chunking vs fixed-size chunking" in text
    assert "Code-specific vs general embedding model" in text and "Judge correct" in text
    assert (tmp_path / "REPORT.md").exists()
