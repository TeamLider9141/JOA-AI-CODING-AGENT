import pytest

from assistant.indexer.models import Chunk
from assistant.store.bm25_store import BM25Store, tokenize


def test_tokenize_splits_camel_case_and_snake_case():
    tokens = tokenize("JWTMiddleware read_file")
    assert "jwtmiddleware" in tokens   # whole identifier kept
    assert "jwt" in tokens             # camel parts
    assert "middleware" in tokens
    assert "read" in tokens            # snake parts
    assert "file" in tokens


def make_chunks() -> list[Chunk]:
    return [
        Chunk("auth.py", "JWTMiddleware", "class", 1, 5,
              "class JWTMiddleware:\n    def check(self): pass"),
        Chunk("db.py", "connect", "function", 1, 5,
              "def connect():\n    return engine"),
    ]


def test_exact_identifier_ranks_first():
    store = BM25Store()
    store.build(make_chunks())
    results = store.search("JWTMiddleware", top_k=2)
    assert results[0][2]["path"] == "auth.py"


def test_zero_score_results_are_dropped():
    store = BM25Store()
    store.build(make_chunks())
    assert store.search("zzz_nonexistent_zzz", top_k=5) == []


def test_save_load_roundtrip(tmp_path):
    store = BM25Store()
    store.build(make_chunks())
    store.save(tmp_path / "bm25.json")

    loaded = BM25Store.load(tmp_path / "bm25.json")
    assert loaded.search("connect", top_k=1)[0][2]["path"] == "db.py"


def test_build_with_no_chunks_raises():
    with pytest.raises(ValueError):
        BM25Store().build([])
