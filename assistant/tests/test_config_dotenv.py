import os

from assistant.config import _load_dotenv


def test_load_dotenv_sets_unset_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_TEST_VAR=hello\n# a comment\n\nOTHER=world\n")

    _load_dotenv(env_file)

    assert os.environ["SOME_TEST_VAR"] == "hello"
    assert os.environ["OTHER"] == "world"


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR", "real-value")
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_TEST_VAR=from-dotenv\n")

    _load_dotenv(env_file)

    assert os.environ["SOME_TEST_VAR"] == "real-value"


def test_load_dotenv_missing_file_is_a_noop(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")  # must not raise
