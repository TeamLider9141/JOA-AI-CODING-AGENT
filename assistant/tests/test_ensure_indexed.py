from assistant.cli import _ensure_indexed


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

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             lambda _o: None, lambda _msg: False)
    assert result is False
    assert not (data_dir / "bm25.json").exists()


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


def test_ensure_indexed_build_failure_returns_false(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    # no indexable files in repo -> build_index raises ValueError

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             lambda _o: None, lambda _msg: True)
    assert result is False
    assert not (data_dir / "bm25.json").exists()
