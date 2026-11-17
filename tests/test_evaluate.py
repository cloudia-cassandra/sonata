import pytest

import sonata.evaluate as ev
import sonata.rag as rag

from test_rag import FakeEmbedder


def result(returned, retrieved=None, source="mock", message=None, dropped=(), strong=8):
    retrieved = list(range(1, 9)) if retrieved is None else retrieved
    return {"recommendations": [{"song": {"deezer_id": i, "liked": False}, "retrieval_similarity": 0.5}
                                for i in returned],
            "retrieved_ids": retrieved, "strong_matches": strong, "source": source, "message": message,
            "dropped": list(dropped)}


def test_grounding():
    assert ev.is_grounded(result([1, 2, 3]))
    assert not ev.is_grounded(result([1, 2, 99]))
    assert ev.is_grounded(result([], retrieved=[], source="rejected"))


def test_consistency_is_mean_pairwise_jaccard():
    same = [result([1, 2, 3])] * 3
    assert ev.consistency(same) == 1.0
    runs = [result([1, 2, 3]), result([1, 2, 4]), result([1, 2, 3])]
    # pairs: {123,124} 2/4, {123,123} 1, {124,123} 2/4
    assert ev.consistency(runs) == pytest.approx((0.5 + 1 + 0.5) / 3)
    assert ev.consistency([result([], source="rejected")] * 3) == 1.0


@pytest.mark.parametrize("run,kind,expected", [
    (result([1, 2, 3]), "normal", True),
    (result([1, 2, 3], source="fallback", message="Daily limit reached"), "normal", True),
    (result([1, 2, 3], source="fallback"), "normal", False),          # fallback without a notice
    (result([1, 2]), "normal", False),                                 # too few songs
    (result([1], strong=1, message="No strong match"), "normal", True),  # few strong: short answer, with a note
    (result([1], strong=1), "normal", False),                          # ...but it must say so
    (result([], strong=0, message="No strong match"), "normal", False),  # at least one song
    (result([], retrieved=[], source="rejected", message="Please type…"), "bad_input", True),
    (result([1, 2, 3], source="mock"), "bad_input", False),            # bad input was answered
    (result([1, 2, 3], source="fallback", message="Nothing matches"), "nonsense", True),
    (result([1, 2, 3], source="mock"), "nonsense", False),             # model answered gibberish
    (None, "normal", False),                                           # ask() raised
])
def test_graceful_handling(run, kind, expected):
    assert ev.handled_gracefully(run, kind) is expected


def test_average_similarity_ignores_empty_answers():
    assert ev.average_similarity([result([1, 2]), result([], source="rejected")]) == 0.5
    assert ev.average_similarity([result([], source="rejected")]) is None


def test_exceptions_are_recorded_not_raised():
    def broken(query, **kwargs):
        raise RuntimeError("boom")

    rows = ev.evaluate(broken, queries=[("anything", "normal")], runs=2)
    assert rows[0]["errors"] == ["RuntimeError: boom"] * 2 and not rows[0]["graceful"]


def test_end_to_end_with_mock_provider(monkeypatch):
    from test_rag import song
    library = [song(i, ["Reggaeton", "Rap", "Pop"][i % 3], liked=i < 4) for i in range(12)]
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.2)   # calibrated for the fake embedder
    kwargs = {"library": library, "embedder": FakeEmbedder()}
    queries = [("reggaeton songs", "normal"), ("", "bad_input"), ("qwxz vbnm", "nonsense"), ("x" * 600, "bad_input")]

    rows = ev.evaluate(rag.ask, queries=queries, use_audio=False, **kwargs)
    assert [(r["grounded"], r["consistent"], r["graceful"]) for r in rows] == [(True, True, True)] * 4
    report = ev.eval_markdown(rows, fresh=False)
    assert "**4/4**" in report and "consistency column is 100% by construction" in report
    assert "*(empty)*" in report and "(600 chars)" in report


def test_expected_calls_counts_only_normal_queries():
    normal = sum(kind == "normal" for _, kind in ev.TEST_QUERIES)
    assert ev.expected_calls(fresh=False) == 2 * normal
    assert ev.expected_calls(fresh=True) == (ev.RUNS + 1) * normal


def test_fixed_queries_cover_empty_nonsense_and_too_long():
    assert len(ev.TEST_QUERIES) == 10
    assert ("", "bad_input") in ev.TEST_QUERIES
    assert any(kind == "nonsense" for _, kind in ev.TEST_QUERIES)
    assert any(len(q) > rag.MAX_QUERY_CHARS for q, _ in ev.TEST_QUERIES)
