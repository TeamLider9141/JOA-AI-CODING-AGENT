from assistant.indexer.models import Chunk
from assistant.store.qdrant_store import QdrantStore


def make_chunks() -> list[Chunk]:
    return [
        Chunk("a.py", "f", "function", 1, 2, "def f(): pass"),
        Chunk("b.py", "g", "function", 1, 2, "def g(): pass"),
    ]


def test_upsert_and_search_roundtrip(tmp_path):
    store = QdrantStore(tmp_path / "q")
    store.reset(dim=3)
    store.upsert(make_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    chunk_id, score, payload = results[0]
    assert payload["path"] == "a.py"
    assert payload["symbol"] == "f"
    store.close()


def test_reset_clears_previous_data(tmp_path):
    store = QdrantStore(tmp_path / "q")
    store.reset(dim=3)
    store.upsert(make_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.reset(dim=3)

    assert store.search([1.0, 0.0, 0.0], top_k=5) == []
    store.close()
