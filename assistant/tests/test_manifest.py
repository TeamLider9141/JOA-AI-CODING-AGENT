from assistant.indexer.manifest import (
    load_manifest, repo_fingerprint, save_manifest,
)


def test_fingerprint_stable_across_repeated_calls(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")

    fp1 = repo_fingerprint(repo)
    fp2 = repo_fingerprint(repo)

    assert fp1 == fp2


def test_fingerprint_changes_when_file_content_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "a.py"
    f.write_text("x = 1\n")
    fp1 = repo_fingerprint(repo)

    f.write_text("x = 2222222\n")  # different size, not just mtime
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2


def test_fingerprint_changes_when_file_added(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    fp1 = repo_fingerprint(repo)

    (repo / "b.py").write_text("y = 2\n")
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2
    assert "b.py" in fp2


def test_fingerprint_changes_when_file_removed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    fp1 = repo_fingerprint(repo)

    (repo / "b.py").unlink()
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2
    assert "b.py" not in fp2


def test_load_manifest_missing_file_returns_none(tmp_path):
    assert load_manifest(tmp_path) is None


def test_load_manifest_corrupt_json_returns_none(tmp_path):
    (tmp_path / "vector_manifest.json").write_text("not json{{{")
    assert load_manifest(tmp_path) is None


def test_save_then_load_round_trips(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    data_dir = tmp_path / "data"

    fp = repo_fingerprint(repo)
    save_manifest(data_dir, fp)

    assert load_manifest(data_dir) == fp
