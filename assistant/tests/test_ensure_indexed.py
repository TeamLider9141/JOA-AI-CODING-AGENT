from assistant.cli import _ensure_indexed
from assistant.llm.ollama_client import OllamaError


class FakeEmbedClient:
    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_ensure_indexed_returns_true_when_index_already_exists(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    data_dir.mkdir()
    (data_dir / "bm25.json").write_text("{}")

    def confirm(_msg):
        raise AssertionError("should not prompt when index already exists")

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             lambda _o: None, confirm)
    assert result is True


def test_ensure_indexed_declined_returns_false_without_indexing(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: False)
    assert result is False
    assert not (data_dir / "bm25.json").exists()
    assert any("no index found" in o.lower() for o in out)


def test_ensure_indexed_accepted_builds_index_and_returns_true(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: True)

    assert result is True
    assert (data_dir / "bm25.json").exists()
    assert any("indekslandi" in o.lower() for o in out)


def test_ensure_indexed_bootstraps_placeholder_when_repo_is_empty(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    # repo has zero files -> build_index would normally raise ValueError;
    # _ensure_indexed should auto-create a placeholder and retry, without
    # asking for permission a second time (only the initial `confirm`
    # call — for "index this now?" — is allowed to fire).
    out = []
    confirm_calls = []

    def confirm(msg):
        confirm_calls.append(msg)
        return True

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, confirm)

    assert result is True
    assert len(confirm_calls) == 1
    assert (data_dir / "bm25.json").exists()
    placeholder = repo / ".joa-welcome.md"
    assert placeholder.exists()
    assert any("bo'sh" in o.lower() for o in out)


def test_ensure_indexed_ollama_failure_does_not_bootstrap(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    class BoomEmbedClient:
        def embed(self, texts):
            raise OllamaError("ollama is down")

    result = _ensure_indexed(repo, data_dir, BoomEmbedClient(),
                             out.append, lambda _msg: True)

    assert result is False
    assert not (data_dir / "bm25.json").exists()
    assert any("ollama is down" in o.lower() for o in out)
    # a real (non-empty) repo that failed for infra reasons must not get
    # a silently-created placeholder file
    assert not (repo / ".joa-welcome.md").exists()
