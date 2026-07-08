import json

from assistant.search.rerank import rerank


def make_candidates():
    # (chunk_id, score, payload) — payload needs path/symbol/text for the prompt
    return [
        ("c1", 0.9, {"path": "a.py", "symbol": "f", "text": "def f(): pass"}),
        ("c2", 0.8, {"path": "b.py", "symbol": "g", "text": "def g(): pass"}),
        ("c3", 0.7, {"path": "c.py", "symbol": "h", "text": "def h(): pass"}),
    ]


def fake_chat(ranking):
    def chat_fn(messages):
        return json.dumps({"ranking": ranking})
    return chat_fn


def test_reorders_candidates_per_model_ranking():
    candidates = make_candidates()
    # model says: c3 (index 3) most relevant, then c1 (1), then c2 (2)
    result = rerank("q", candidates, fake_chat([3, 1, 2]), top_k=3)
    assert [c[0] for c in result] == ["c3", "c1", "c2"]


def test_truncates_to_top_k():
    candidates = make_candidates()
    result = rerank("q", candidates, fake_chat([1, 2, 3]), top_k=2)
    assert len(result) == 2
    assert [c[0] for c in result] == ["c1", "c2"]


def test_falls_back_on_malformed_json():
    candidates = make_candidates()
    result = rerank("q", candidates, lambda messages: "not json at all",
                    top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_missing_ranking_key():
    candidates = make_candidates()
    chat_fn = lambda messages: json.dumps({"oops": [1, 2, 3]})
    result = rerank("q", candidates, chat_fn, top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_non_permutation_ranking():
    candidates = make_candidates()
    # duplicate 1, missing 3 — not a valid permutation of 1..3
    result = rerank("q", candidates, fake_chat([1, 1, 2]), top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_out_of_range_index():
    candidates = make_candidates()
    result = rerank("q", candidates, fake_chat([1, 2, 99]), top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_handles_fewer_candidates_than_top_k():
    candidates = make_candidates()[:2]
    result = rerank("q", candidates, fake_chat([2, 1]), top_k=5)
    assert [c[0] for c in result] == ["c2", "c1"]


def test_prompt_contains_query_and_every_candidate_path():
    candidates = make_candidates()
    seen_messages = []

    def chat_fn(messages):
        seen_messages.extend(messages)
        return json.dumps({"ranking": [1, 2, 3]})

    rerank("where is f defined", candidates, chat_fn, top_k=3)
    full_text = " ".join(m["content"] for m in seen_messages)
    assert "where is f defined" in full_text
    assert "a.py" in full_text
    assert "b.py" in full_text
    assert "c.py" in full_text
