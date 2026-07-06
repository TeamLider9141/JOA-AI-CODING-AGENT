from typer.testing import CliRunner

from assistant.cli import app, _repl_loop
from assistant.llm.ollama_client import OllamaError

runner = CliRunner()


class FakeSession:
    def __init__(self, answers):
        self._answers = list(answers)
        self.sent = []

    def send(self, task):
        self.sent.append(task)
        return self._answers.pop(0)


def test_repl_loop_sends_lines_and_exits_on_exit():
    session = FakeSession(["answer one"])
    lines = iter(["do a thing", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append)
    assert session.sent == ["do a thing"]
    assert any("answer one" in o for o in out)


def test_repl_loop_skips_blank_lines():
    session = FakeSession(["ans"])
    lines = iter(["", "   ", "real task", "quit"])
    _repl_loop(session, lambda: next(lines), lambda _o: None)
    assert session.sent == ["real task"]


def test_repl_loop_exits_on_eof():
    session = FakeSession([])

    def read_line():
        raise EOFError

    _repl_loop(session, read_line, lambda _o: None)
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
    _repl_loop(session, lambda: next(lines), out.append)
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
