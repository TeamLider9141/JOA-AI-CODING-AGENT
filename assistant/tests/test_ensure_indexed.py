from assistant.cli import _ensure_indexed
from assistant.llm.ollama_client import OllamaError


class FakeEmbedClient:
    def embed(self, texts):
        return [[0.0] for _ in texts]


def _no_op_background(*_args, **_kwargs):
    """Injected in place of the real background-thread starter so these
    unit tests never spin a real thread or touch a real embedder."""


def test_ensure_indexed_returns_true_when_index_already_exists(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    data_dir.mkdir()
    (data_dir / "bm25.json").write_text("{}")

    def confirm(_msg):
        raise AssertionError("should not prompt when index already exists")

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             lambda _o: None, confirm,
                             start_vector_background=_no_op_background)
    assert result is True


def test_ensure_indexed_declined_returns_false_without_indexing(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: False,
                             start_vector_background=_no_op_background)
    assert result is False
    assert not (data_dir / "bm25.json").exists()
    assert any("no index found" in o.lower() for o in out)


def test_ensure_indexed_accepted_builds_bm25_and_returns_true(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: True,
                             start_vector_background=_no_op_background)

    assert result is True
    assert (data_dir / "bm25.json").exists()
    assert not (data_dir / "qdrant").exists()  # vector build was skipped
    assert any("indekslandi" in o.lower() for o in out)


def test_ensure_indexed_bootstraps_placeholder_when_repo_is_empty(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    out = []
    confirm_calls = []

    def confirm(msg):
        confirm_calls.append(msg)
        return True

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, confirm,
                             start_vector_background=_no_op_background)

    assert result is True
    assert len(confirm_calls) == 1
    assert (data_dir / "bm25.json").exists()
    placeholder = repo / ".joa-welcome.md"
    assert placeholder.exists()
    assert any("bo'sh" in o.lower() for o in out)


def test_ensure_indexed_succeeds_even_when_ollama_is_down(tmp_path):
    """BM25 doesn't call the embedder at all — indexing succeeds
    synchronously even if Ollama is unreachable. Only the background
    vector build would be affected by that (covered in
    test_build_vector_background_ollama_failure_leaves_no_qdrant_dir)."""
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    class BoomEmbedClient:
        def embed(self, texts):
            raise OllamaError("ollama is down")

    result = _ensure_indexed(repo, data_dir, BoomEmbedClient(),
                             out.append, lambda _msg: True,
                             start_vector_background=_no_op_background)

    assert result is True
    assert (data_dir / "bm25.json").exists()


def test_ensure_indexed_triggers_background_vector_start(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    calls = []

    def fake_background(repo_arg, data_dir_arg, embed_client_arg, echo_arg):
        calls.append((repo_arg, data_dir_arg))

    embed_client = FakeEmbedClient()
    _ensure_indexed(repo, data_dir, embed_client, lambda _o: None,
                    lambda _msg: True, start_vector_background=fake_background)

    assert calls == [(repo, data_dir)]


def test_ensure_indexed_already_indexed_still_triggers_background_check(tmp_path):
    """Even when BM25 already exists (no confirm prompt), the background
    starter must still be given a chance to run — it's the one that
    decides (via the manifest) whether a vector rebuild is needed."""
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    data_dir.mkdir()
    (data_dir / "bm25.json").write_text("{}")
    calls = []

    def fake_background(repo_arg, data_dir_arg, embed_client_arg, echo_arg):
        calls.append((repo_arg, data_dir_arg))

    def confirm(_msg):
        raise AssertionError("should not prompt when index already exists")

    _ensure_indexed(repo, data_dir, FakeEmbedClient(), lambda _o: None,
                    confirm, start_vector_background=fake_background)

    assert calls == [(repo, data_dir)]


def test_maybe_start_vector_background_skips_when_manifest_matches(
        tmp_path, monkeypatch):
    from assistant.cli import _maybe_start_vector_background
    from assistant.indexer.manifest import repo_fingerprint, save_manifest

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    (data_dir / "qdrant").mkdir(parents=True)
    save_manifest(data_dir, repo_fingerprint(repo))

    def boom(*_args, **_kwargs):
        raise AssertionError("should not start a thread when nothing changed")

    monkeypatch.setattr("assistant.cli.threading.Thread", boom)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)


def test_maybe_start_vector_background_starts_thread_when_stale(
        tmp_path, monkeypatch):
    from assistant.cli import _maybe_start_vector_background

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    data_dir.mkdir()
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr("assistant.cli.threading.Thread", FakeThread)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)

    assert len(started) == 1
    assert started[0][2] is True  # daemon=True


def test_maybe_start_vector_background_starts_thread_when_qdrant_missing(
        tmp_path, monkeypatch):
    """Manifest could match (e.g. copied data dir) but if qdrant/ itself
    isn't there, a rebuild is still required."""
    from assistant.cli import _maybe_start_vector_background
    from assistant.indexer.manifest import repo_fingerprint, save_manifest

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    save_manifest(data_dir, repo_fingerprint(repo))  # no qdrant/ dir made
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr("assistant.cli.threading.Thread", FakeThread)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)

    assert len(started) == 1


def test_build_vector_background_success_swaps_in_qdrant_and_saves_manifest(
        tmp_path):
    from assistant.cli import _build_vector_background
    from assistant.indexer.manifest import load_manifest, repo_fingerprint

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")

    class RealisticEmbedClient:
        def embed(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    out = []
    fingerprint = repo_fingerprint(repo)

    _build_vector_background(repo, data_dir, RealisticEmbedClient(),
                             fingerprint, out.append)

    assert (data_dir / "qdrant").is_dir()
    assert not (data_dir / "qdrant.new").exists()
    assert load_manifest(data_dir) == fingerprint
    assert any("tayyor" in o.lower() for o in out)


def test_build_vector_background_ollama_failure_leaves_no_qdrant_dir(
        tmp_path):
    from assistant.cli import _build_vector_background

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    data_dir.mkdir()
    out = []

    class BoomEmbedClient:
        def embed(self, texts):
            raise OllamaError("ollama is down")

    _build_vector_background(repo, data_dir, BoomEmbedClient(), {}, out.append)

    assert not (data_dir / "qdrant").exists()
    assert not (data_dir / "qdrant.new").exists()
    assert any("ollama is down" in o.lower() for o in out)


def test_build_vector_background_replaces_existing_qdrant_dir(tmp_path):
    from assistant.cli import _build_vector_background
    from assistant.indexer.manifest import repo_fingerprint

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")

    old_qdrant = data_dir / "qdrant"
    old_qdrant.mkdir(parents=True)
    (old_qdrant / "stale-marker.txt").write_text("old index")

    class RealisticEmbedClient:
        def embed(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    out = []
    fingerprint = repo_fingerprint(repo)

    _build_vector_background(repo, data_dir, RealisticEmbedClient(),
                             fingerprint, out.append)

    assert (data_dir / "qdrant").is_dir()
    assert not (data_dir / "qdrant" / "stale-marker.txt").exists()
    assert not (data_dir / "qdrant.old").exists()
    assert not (data_dir / "qdrant.new").exists()


def test_build_vector_background_value_error_leaves_no_qdrant_dir(tmp_path):
    from assistant.cli import _build_vector_background

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    data_dir.mkdir()
    out = []

    class AllBatchesFailEmbedClient:
        def embed(self, texts):
            raise RuntimeError("simulated non-Ollama embedding failure")

    _build_vector_background(repo, data_dir, AllBatchesFailEmbedClient(),
                             {}, out.append)

    assert not (data_dir / "qdrant").exists()
    assert not (data_dir / "qdrant.new").exists()
    assert any("muvaffaqiyatsiz" in o.lower() for o in out)
