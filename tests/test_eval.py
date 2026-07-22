"""Tests for the local eval harness (Milestone 8).

Judges and ask_fn are injected, so the scoring math is verified deterministically
with no LLM and no services.
"""
from src.rag.eval import evaluate
from src.rag.pipeline import Answer
from src.rag.vectorstore import Hit


def _answer(text, hits=None):
    return Answer(text=text, hits=hits or [Hit(text="ctx", score=0.9, source="d")],
                  top_score=0.9)


def test_all_pass_scores_one():
    dataset = [{"question": "q1"}, {"question": "q2"}]
    report = evaluate(
        dataset,
        ask_fn=lambda q: _answer("grounded answer"),
        faithfulness_judge=lambda a, c: True,
        relevancy_judge=lambda q, a: True,
        precision_judge=lambda q, c: True,
    )
    assert report.n == 2
    assert report.scores == {
        "faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0}


def test_half_pass_scores_half():
    # faithfulness true only for the first of two questions -> 0.5
    seen = {"n": 0}

    def faith(a, c):
        seen["n"] += 1
        return seen["n"] == 1

    report = evaluate(
        [{"question": "q1"}, {"question": "q2"}],
        ask_fn=lambda q: _answer("answer"),
        faithfulness_judge=faith,
        relevancy_judge=lambda q, a: True,
        precision_judge=lambda q, c: False,
    )
    assert report.scores["faithfulness"] == 0.5
    assert report.scores["answer_relevancy"] == 1.0
    assert report.scores["context_precision"] == 0.0


def test_per_item_records_route():
    report = evaluate(
        [{"question": "q1"}],
        ask_fn=lambda q: _answer("a"),
        faithfulness_judge=lambda a, c: True,
        relevancy_judge=lambda q, a: True,
        precision_judge=lambda q, c: True,
    )
    assert report.items[0]["route"] == "documents"
    assert report.items[0]["question"] == "q1"


def test_context_falls_back_to_sql_when_no_hits():
    captured = {}

    def precision(q, c):
        captured["ctx"] = c
        return True

    sql_answer = Answer(text="5", hits=[], top_score=0.0, route="structured",
                        sql="SELECT COUNT(*) FROM employees")
    evaluate(
        [{"question": "how many?"}],
        ask_fn=lambda q: sql_answer,
        faithfulness_judge=lambda a, c: True,
        relevancy_judge=lambda q, a: True,
        precision_judge=precision,
    )
    assert captured["ctx"] == "SELECT COUNT(*) FROM employees"
