from assistant.search.hybrid import rrf_merge


def test_chunk_in_both_lists_wins():
    vector = [("c1", 0.9, {"path": "a"}), ("c2", 0.8, {"path": "b"})]
    bm25 = [("c3", 5.0, {"path": "c"}), ("c1", 4.0, {"path": "a"})]

    merged = rrf_merge([vector, bm25], k=60, top_k=3)
    assert merged[0][0] == "c1"
    assert merged[0][2] == {"path": "a"}


def test_top_k_limits_results():
    results = [(f"c{i}", 1.0, {}) for i in range(20)]
    assert len(rrf_merge([results], k=60, top_k=5)) == 5


def test_raw_scores_are_ignored_only_rank_matters():
    # c1 has tiny raw scores but rank 1 in both lists -> must win
    a = [("c1", 0.001, {}), ("c2", 0.0009, {})]
    b = [("c1", 0.002, {}), ("c3", 0.001, {})]
    assert rrf_merge([a, b], k=60, top_k=1)[0][0] == "c1"


def test_empty_input_returns_empty():
    assert rrf_merge([[], []], k=60, top_k=5) == []
