from assistant.cli import _ensure_trusted, _load_trusted, _save_trusted


def test_load_trusted_missing_file_returns_empty_set(tmp_path):
    assert _load_trusted(tmp_path / "nope.json") == set()


def test_load_trusted_corrupt_json_returns_empty_set(tmp_path):
    path = tmp_path / "trusted_dirs.json"
    path.write_text("not json{{{")
    assert _load_trusted(path) == set()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "trusted_dirs.json"
    _save_trusted({"/a", "/b"}, path)
    assert _load_trusted(path) == {"/a", "/b"}


def test_ensure_trusted_skips_prompt_for_known_dir(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    _save_trusted({str(repo.resolve())}, trust_path)

    def read_line():
        raise AssertionError("should not prompt for an already-trusted dir")

    result = _ensure_trusted(repo, read_line, lambda _o: None,
                             trust_path=trust_path)
    assert result is True


def test_ensure_trusted_accept_saves_and_returns_true(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = iter(["1"])
    out = []

    result = _ensure_trusted(repo, lambda: next(lines), out.append,
                             trust_path=trust_path)

    assert result is True
    assert str(repo.resolve()) in _load_trusted(trust_path)
    assert any("ishonaman" in o.lower() or "1." in o for o in out)


def test_ensure_trusted_decline_returns_false_and_does_not_save(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = iter(["2"])

    result = _ensure_trusted(repo, lambda: next(lines), lambda _o: None,
                             trust_path=trust_path)

    assert result is False
    assert _load_trusted(trust_path) == set()


def test_ensure_trusted_eof_returns_false(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()

    def read_line():
        raise EOFError

    result = _ensure_trusted(repo, read_line, lambda _o: None,
                             trust_path=trust_path)
    assert result is False


def test_ensure_trusted_unknown_input_returns_false(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = iter(["blah"])

    result = _ensure_trusted(repo, lambda: next(lines), lambda _o: None,
                             trust_path=trust_path)
    assert result is False
