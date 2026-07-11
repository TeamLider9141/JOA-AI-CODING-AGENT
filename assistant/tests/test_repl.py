import re
import time

from typer.testing import CliRunner

from assistant.cli import app, _repl_loop
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError

runner = CliRunner()


class FakeSession:
    def __init__(self, answers):
        self._answers = list(answers)
        self.sent = []

    def send(self, task):
        self.sent.append(task)
        return self._answers.pop(0)


class FakeEmbedClient:
    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models


def test_repl_loop_sends_lines_and_exits_on_exit():
    session = FakeSession(["answer one"])
    lines = iter(["do a thing", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None)
    assert session.sent == ["do a thing"]
    assert any("answer one" in o for o in out)


def test_repl_loop_skips_blank_lines():
    session = FakeSession(["ans"])
    lines = iter(["", "   ", "real task", "quit"])
    _repl_loop(session, lambda: next(lines), lambda _o: None, None)
    assert session.sent == ["real task"]


def test_repl_loop_exits_on_eof():
    session = FakeSession([])

    def read_line():
        raise EOFError

    _repl_loop(session, read_line, lambda _o: None, None)
    assert session.sent == []


def test_repl_loop_survives_ollama_error():
    class BoomSession:
        def __init__(self):
            self.sent = []

        def send(self, task):
            self.sent.append(task)
            raise OllamaError("ollama is down")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None)
    assert session.sent == ["try this"]
    assert any("down" in o for o in out)


def test_repl_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "repl" in result.output


def test_repl_without_index_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["repl", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "index" in result.output.lower()


def test_repl_loop_echoes_elapsed_time_with_answer():
    class SlowSession:
        def __init__(self):
            self.sent = []

        def send(self, task):
            self.sent.append(task)
            time.sleep(0.05)
            return "the answer"

    session = SlowSession()
    lines = iter(["do it", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None)

    answer_line = next(o for o in out if "the answer" in o)
    match = re.search(r"\((\d+(?:\.\d+)?)s\)", answer_line)
    assert match, f"expected an elapsed-time suffix like (0.1s), got: {answer_line!r}"
    assert float(match.group(1)) >= 0.05


def test_joamodel_lists_and_switches_to_chosen_ollama_model():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(
        ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b"])
    lines = iter(["/joamodel", "2", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert isinstance(session.client, OllamaClient)
    assert session.client._model == "qwen2.5-coder:3b"
    assert any("2. qwen2.5-coder:3b" in o for o in out)
    assert any("3. gemini" in o for o in out)


def test_joamodel_closes_previous_client_on_switch():
    class FakeClosableClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    session = FakeSession([])
    old_client = FakeClosableClient()
    session.client = old_client
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "1", "exit"])
    _repl_loop(session, lambda: next(lines), lambda _o: None, embed_client)
    assert old_client.closed is True
    assert session.client is not old_client


def test_joamodel_unicode_digit_does_not_crash():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "³", "exit"])  # superscript 3
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert session.client == "initial"
    assert any("Noto'g'ri tanlov" in o for o in out)


def test_joamodel_switches_to_gemini_when_key_present(monkeypatch):
    monkeypatch.setattr("assistant.cli.config.GEMINI_API_KEY", "test-key")
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "2", "exit"])  # 1=qwen..1.5b, 2=gemini
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert isinstance(session.client, GeminiClient)
    assert any("Model: gemini" in o for o in out)


def test_joamodel_gemini_without_key_warns_and_keeps_current_client(
        monkeypatch):
    monkeypatch.setattr("assistant.cli.config.GEMINI_API_KEY", None)
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient([])
    lines = iter(["/joamodel", "1", "exit"])  # only option is "gemini"
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert session.client == "initial"
    assert any("GEMINI_API_KEY" in o for o in out)


def test_joamodel_invalid_number_keeps_current_client():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "99", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert session.client == "initial"
    assert any("Noto'g'ri tanlov" in o for o in out)


def test_joamodel_non_numeric_choice_keeps_current_client():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "abc", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client)
    assert session.client == "initial"


def test_joamodel_eof_during_selection_does_not_crash():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    calls = iter(["/joamodel"])

    def read_line():
        try:
            return next(calls)
        except StopIteration:
            raise EOFError

    _repl_loop(session, read_line, lambda _o: None, embed_client)
    assert session.client == "initial"


def test_joamodel_list_models_failure_keeps_current_client():
    class BoomEmbedClient:
        def list_models(self):
            raise OllamaError("ollama is down")

    session = FakeSession([])
    session.client = "initial"
    lines = iter(["/joamodel", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, BoomEmbedClient())
    assert session.client == "initial"
    assert any("down" in o for o in out)


def test_repl_loop_gemini_error_suggests_joamodel():
    class BoomSession:
        def __init__(self):
            self.sent = []

        def send(self, task):
            self.sent.append(task)
            raise GeminiError("rate limited")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None)
    assert any("/joamodel" in o for o in out)


def test_repl_loop_ollama_error_does_not_suggest_joamodel():
    class BoomSession:
        def __init__(self):
            self.sent = []

        def send(self, task):
            self.sent.append(task)
            raise OllamaError("ollama is down")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None)
    assert not any("/joamodel" in o for o in out)
