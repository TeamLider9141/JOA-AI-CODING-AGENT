import pytest

from assistant.indexer.pipeline import build_index, search_index
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
