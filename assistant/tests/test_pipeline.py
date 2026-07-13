import pytest

from assistant.indexer.pipeline import (
    build_bm25_index, build_index, build_vector_index, search_index,
)
from assistant.llm.ollama_client import OllamaError


def fake_embedder(texts: list[str]) -> list[list[float]]:
    # deterministic 3-dim "embedding": length signal + constants
    return [[float(len(t)), 1.0, 0.5] for t in texts]


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class JWTMiddleware:\n    def check(self):\n        return True\n")
    (repo / "db.py").write_text(
        "def connect():\n    return 'engine'\n")
    return repo


def test_build_index_persists_both_stores(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_index(repo, data, fake_embedder)

    assert n >= 2
    assert (data / "bm25.json").exists()
    assert (data / "qdrant").is_dir()


def test_search_index_hybrid_finds_exact_identifier(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    results = search_index("JWTMiddleware", data, fake_embedder)
    assert results, "expected at least one result"
    assert results[0][2]["path"] == "auth.py"


def test_search_index_vector_mode_returns_results(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    results = search_index("anything", data, fake_embedder, mode="vector")
    assert len(results) >= 1


def test_empty_repo_raises(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    with pytest.raises(ValueError, match="no indexable chunks"):
        build_index(repo, tmp_path / "data", fake_embedder)


def test_ollama_error_aborts_build(tmp_path):
    repo = make_repo(tmp_path)

    def broken_embedder(texts):
        raise OllamaError("server down")

    with pytest.raises(OllamaError):
        build_index(repo, tmp_path / "data", broken_embedder)


def test_build_bm25_index_creates_bm25_only(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_bm25_index(repo, data)

    assert n >= 2
    assert (data / "bm25.json").exists()
    assert not (data / "qdrant").exists()


def test_build_bm25_index_empty_repo_raises(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    with pytest.raises(ValueError, match="no indexable chunks"):
        build_bm25_index(repo, tmp_path / "data")


def test_build_vector_index_creates_qdrant_only(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_vector_index(repo, data, fake_embedder)

    assert n >= 2
    assert (data / "qdrant").is_dir()
    assert not (data / "bm25.json").exists()


def test_build_vector_index_respects_qdrant_dirname(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    build_vector_index(repo, data, fake_embedder, qdrant_dirname="qdrant.new")

    assert (data / "qdrant.new").is_dir()
    assert not (data / "qdrant").exists()


def test_search_index_falls_back_to_bm25_only_without_qdrant(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)  # no vector index built at all

    results = search_index("JWTMiddleware", data, fake_embedder)

    assert results, "expected at least one result"
    assert results[0][2]["path"] == "auth.py"


def test_search_index_vector_mode_without_qdrant_returns_empty(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)

    results = search_index("anything", data, fake_embedder, mode="vector")

    assert results == []


def test_search_index_hybrid_still_uses_vector_when_present(tmp_path):
    """Regression guard: once both stores exist, behavior is unchanged
    from before this task (hybrid RRF merge of both)."""
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)
    build_vector_index(repo, data, fake_embedder)

    results = search_index("JWTMiddleware", data, fake_embedder)

    assert results
    assert results[0][2]["path"] == "auth.py"


def test_search_index_falls_back_to_bm25_when_vector_store_errors(tmp_path):
    """Simulates the narrow-window race where the vector store is
    transiently broken (e.g. mid-swap by the background rebuild thread)
    — search_index must degrade to BM25 results instead of raising."""
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)
    build_vector_index(repo, data, fake_embedder)

    class BrokenQdrantStore:
        def __init__(self, path):
            pass

        def search(self, vector, top_k):
            raise ValueError("Collection code not found")

        def close(self):
            pass

    import assistant.indexer.pipeline as pipeline_module
    original = pipeline_module.QdrantStore
    pipeline_module.QdrantStore = BrokenQdrantStore
    try:
        results = search_index("JWTMiddleware", data, fake_embedder)
    finally:
        pipeline_module.QdrantStore = original

    assert results, "expected BM25 fallback results, not empty/crash"
    assert results[0][2]["path"] == "auth.py"
