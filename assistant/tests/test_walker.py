from assistant.indexer.walker import walk_repo


def test_walker_filters_gitignore_excludes_and_binaries(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.py\n")
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "secret.py").write_text("x = 2")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("x")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")

    names = {p.name for p in walk_repo(tmp_path)}
    assert names == {"app.py"}


def test_walker_skips_oversized_files(tmp_path):
    (tmp_path / "big.py").write_text("#" + "x" * 600_000)
    (tmp_path / "ok.py").write_text("x = 1")

    names = {p.name for p in walk_repo(tmp_path)}
    assert names == {"ok.py"}


def test_walker_returns_sorted_paths(tmp_path):
    (tmp_path / "b.py").write_text("x = 1")
    (tmp_path / "a.py").write_text("x = 1")

    assert [p.name for p in walk_repo(tmp_path)] == ["a.py", "b.py"]
