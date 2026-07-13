import re
import time
from pathlib import Path

from typer.testing import CliRunner

from assistant.cli import app, _repl_loop
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError

runner = CliRunner()


class FakeCtx:
    def __init__(self, root=None):
        self.root = root or Path(".")


class FakeSession:
    def __init__(self, answers):
        self._answers = list(answers)
        self.sent = []
        self.client = AlwaysEscalateClient()
        self.messages = [{"role": "system", "content": "agent prompt"}]
        self.ctx = FakeCtx()

    def send(self, task):
        self.sent.append(task)
        return self._answers.pop(0)


class FakeEmbedClient:
    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models


class AlwaysEscalateClient:
    """chat_stream that always answers ESCALATE — forces the agent path."""

    def chat_stream(self, messages):
        yield "ESCALATE"


class FakeStreamClient:
    """chat_stream yields the given chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def chat_stream(self, messages):
        yield from self._chunks


def test_repl_loop_sends_lines_and_exits_on_exit():
    session = FakeSession(["answer one"])
    lines = iter(["do a thing", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)
    assert session.sent == ["do a thing"]
    assert any("answer one" in o for o in out)


def test_repl_loop_skips_blank_lines():
    session = FakeSession(["ans"])
    lines = iter(["", "   ", "real task", "quit"])
    _repl_loop(session, lambda: next(lines), lambda _o: None, None, lambda _t: None)
    assert session.sent == ["real task"]


def test_repl_loop_exits_on_eof():
    session = FakeSession([])

    def read_line():
        raise EOFError

    _repl_loop(session, read_line, lambda _o: None, None, lambda _t: None)
    assert session.sent == []


def test_repl_loop_survives_ollama_error():
    class BoomSession:
        def __init__(self):
            self.sent = []
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]

        def send(self, task):
            self.sent.append(task)
            raise OllamaError("ollama is down")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)
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
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]

        def send(self, task):
            self.sent.append(task)
            time.sleep(0.05)
            return "the answer"

    session = SlowSession()
    lines = iter(["do it", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)

    answer_line = next(o for o in out if "the answer" in o)
    match = re.search(r"\((\d+(?:\.\d+)?)s", answer_line)
    assert match, f"expected an elapsed-time suffix like (0.1s), got: {answer_line!r}"
    assert float(match.group(1)) >= 0.05


def test_agent_path_footer_includes_model_name():
    session = FakeSession(["the answer"])
    session.client._model = "qwen2.5-coder:0.5b"
    lines = iter(["do it", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)

    answer_line = next(o for o in out if "the answer" in o)
    assert "qwen2.5-coder:0.5b" in answer_line


def test_fast_path_footer_includes_model_name():
    session = FakeSession([])
    session.client = FakeStreamClient(["quick answer"])
    session.client._model = "qwen2.5-coder:0.5b"
    lines = iter(["what is 2+2?", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)

    footer_line = next(o for o in out if "s ·" in o)
    assert "qwen2.5-coder:0.5b" in footer_line


def test_joamodel_lists_and_switches_to_chosen_ollama_model():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(
        ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b"])
    lines = iter(["/joamodel", "2", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)
    assert isinstance(session.client, OllamaClient)
    assert session.client._model == "qwen2.5-coder:3b"
    assert any("2." in o and "qwen2.5-coder:3b" in o for o in out)
    assert any("3." in o and "gemini" in o for o in out)


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
    _repl_loop(session, lambda: next(lines), lambda _o: None, embed_client, lambda _t: None)
    assert old_client.closed is True
    assert session.client is not old_client


def test_joamodel_unicode_digit_does_not_crash():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "³", "exit"])  # superscript 3
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)
    assert session.client == "initial"
    assert any("Noto'g'ri tanlov" in o for o in out)


def test_joamodel_switches_to_gemini_when_key_present(monkeypatch):
    monkeypatch.setattr("assistant.cli.config.GEMINI_API_KEY", "test-key")
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "2", "exit"])  # 1=qwen..1.5b, 2=gemini
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)
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
    _repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)
    assert session.client == "initial"
    assert any("GEMINI_API_KEY" in o for o in out)


def test_joamodel_invalid_number_keeps_current_client():
    session = FakeSession([])
    session.client = "initial"
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "99", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)
    assert session.client == "initial"
    assert any("Noto'g'ri tanlov" in o for o in out)


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

    _repl_loop(session, read_line, lambda _o: None, embed_client, lambda _t: None)
    assert session.client == "initial"


def test_joamodel_list_models_failure_keeps_current_client():
    class BoomEmbedClient:
        def list_models(self):
            raise OllamaError("ollama is down")

    session = FakeSession([])
    session.client = "initial"
    lines = iter(["/joamodel", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, BoomEmbedClient(), lambda _t: None)
    assert session.client == "initial"
    assert any("down" in o for o in out)


def test_repl_loop_gemini_error_suggests_joamodel():
    class BoomSession:
        def __init__(self):
            self.sent = []
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]

        def send(self, task):
            self.sent.append(task)
            raise GeminiError("rate limited")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)
    assert any("/joamodel" in o for o in out)


def test_repl_loop_ollama_error_does_not_suggest_joamodel():
    class BoomSession:
        def __init__(self):
            self.sent = []
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]

        def send(self, task):
            self.sent.append(task)
            raise OllamaError("ollama is down")

    session = BoomSession()
    lines = iter(["try this", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)
    assert not any("/joamodel" in o for o in out)


def test_fast_path_answer_skips_agent_and_shows_timing():
    session = FakeSession([])
    session.client = FakeStreamClient(["quick ", "answer"])
    lines = iter(["what is 2+2?", "exit"])
    out = []
    tokens = []
    _repl_loop(session, lambda: next(lines), out.append, None, tokens.append)
    assert session.sent == []  # agent path never ran
    assert "".join(tokens) == "quick answer"
    assert any("s ·" in o for o in out)  # timing suffix echoed


def test_escalate_falls_back_to_agent_path():
    session = FakeSession(["agent answer"])
    lines = iter(["refactor the auth module", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert session.sent == ["refactor the auth module"]
    assert any("agent answer" in o for o in out)


def test_fast_path_gemini_error_shows_hint_and_survives():
    class BoomStreamClient:
        def chat_stream(self, messages):
            raise GeminiError("rate limited")
            yield  # pragma: no cover — makes this a generator

    session = FakeSession([])
    session.client = BoomStreamClient()
    lines = iter(["hello", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert any("/joamodel" in o for o in out)
    assert session.sent == []


def test_help_lists_all_slash_commands():
    session = FakeSession([])
    lines = iter(["/help", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    joined = "\n".join(out)
    assert "/joamodel" in joined
    assert "/clear" in joined
    assert "/help" in joined
    assert "exit" in joined
    assert session.sent == []  # never reached the LLM


def test_bare_slash_also_lists_commands():
    session = FakeSession([])
    lines = iter(["/", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert any("/joamodel" in o for o in out)
    assert session.sent == []


def test_unknown_slash_command_shows_error_not_llm():
    session = FakeSession([])
    lines = iter(["/nomalum", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    joined = "\n".join(out)
    assert "/nomalum" in joined
    assert "/help" in joined
    assert session.sent == []


def test_clear_resets_history_keeping_system_prompt():
    session = FakeSession([])
    system_msg = session.messages[0]
    session.messages.append({"role": "user", "content": "old turn"})
    session.messages.append({"role": "assistant", "content": "old reply"})
    lines = iter(["/clear", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert session.messages == [system_msg]
    assert session.messages[0] is system_msg
    assert any("tozaland" in o.lower() for o in out)


def _completions(text):
    from prompt_toolkit.document import Document

    from assistant.cli import SlashCompleter

    doc = Document(text, len(text))
    return [c.text for c in SlashCompleter().get_completions(doc, None)]


def test_slash_prefix_suggests_all_commands():
    from assistant.cli import SLASH_COMMANDS

    assert set(_completions("/")) == set(SLASH_COMMANDS)


def test_partial_slash_input_filters_suggestions():
    assert _completions("/jo") == ["/joamodel"]
    assert _completions("/c") == ["/clear"]


def test_non_slash_input_suggests_nothing():
    assert _completions("hello") == []
    assert _completions("") == []


def test_bang_command_runs_directly_without_llm(monkeypatch):
    calls = []

    def fake_run_streaming(command, cwd, on_output, timeout=None):
        calls.append((command, cwd, timeout))
        on_output("hi\n")
        return 0, "hi\n", False

    monkeypatch.setattr("assistant.cli.run_streaming", fake_run_streaming)
    session = FakeSession([])
    lines = iter(["!echo hi", "exit"])
    out = []
    tokens = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               tokens.append)
    assert calls == [("echo hi", session.ctx.root, None)]
    assert "".join(tokens) == "hi\n"
    assert any("exit code: 0" in o for o in out)
    assert session.sent == []  # never reached the LLM or agent loop


def test_bang_command_shows_nonzero_exit_code(monkeypatch):
    def fake_run_streaming(command, cwd, on_output, timeout=None):
        return 1, "", False

    monkeypatch.setattr("assistant.cli.run_streaming", fake_run_streaming)
    session = FakeSession([])
    lines = iter(["!false", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert any("exit code: 1" in o for o in out)


def test_bare_bang_shows_usage_hint():
    session = FakeSession([])
    lines = iter(["!", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert any("bo'sh" in o.lower() for o in out)
    assert session.sent == []


def test_joamodel_list_marks_current_model():
    session = FakeSession([])
    session.client._model = "qwen2.5-coder:1.5b"
    embed_client = FakeEmbedClient(
        ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b"])
    lines = iter(["/joamodel", "1", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client,
               lambda _t: None)
    current_line = next(
        o for o in out if "1." in o and "qwen2.5-coder:1.5b" in o)
    other_line = next(
        o for o in out if "2." in o and "qwen2.5-coder:3b" in o)
    assert "joriy" in current_line
    assert "joriy" not in other_line


def test_joamodel_list_uses_ansi_color_codes():
    session = FakeSession([])
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "1", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, embed_client,
               lambda _t: None)
    joined = "\n".join(out)
    assert "\x1b[" in joined  # at least one ANSI escape code present


def test_keyboard_interrupt_during_fast_path_stays_in_repl():
    class InterruptingClient:
        def chat_stream(self, messages):
            raise KeyboardInterrupt
            yield  # pragma: no cover — makes this a generator function

    session = FakeSession([])
    session.client = InterruptingClient()
    lines = iter(["hello", "still here"])
    out = []

    def read_line():
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    _repl_loop(session, read_line, out.append, None, lambda _t: None)
    assert any("to'xtatildi" in o.lower() for o in out)


def test_keyboard_interrupt_during_agent_send_stays_in_repl():
    class InterruptingSession:
        def __init__(self):
            self.sent = []
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]

        def send(self, task):
            self.sent.append(task)
            raise KeyboardInterrupt

    session = InterruptingSession()
    lines = iter(["do something", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert session.sent == ["do something"]
    assert any("to'xtatildi" in o.lower() for o in out)


def test_keyboard_interrupt_during_bang_stays_in_repl(monkeypatch):
    def fake_run_streaming(command, cwd, on_output, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("assistant.cli.run_streaming", fake_run_streaming)
    session = FakeSession([])
    lines = iter(["!sleep 100", "exit"])
    out = []
    _repl_loop(session, lambda: next(lines), out.append, None,
               lambda _t: None)
    assert any("to'xtatildi" in o.lower() for o in out)


def test_joamodel_uses_injected_select_no_numeric_prompt():
    """When a `select` callable is supplied (arrow-key mode), /joamodel
    must never print the "Raqamni tanlang" numeric prompt or read a
    number from read_line."""
    session = FakeSession([])
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b", "qwen2.5-coder:3b"])
    lines = iter(["/joamodel", "exit"])  # no number typed
    out = []
    select = lambda options, current_index: 1  # pick index 1 (3b)
    _repl_loop(session, lambda: next(lines), out.append, embed_client,
               lambda _t: None, select=select)
    assert not any("Raqamni tanlang" in o for o in out)
    assert any("qwen2.5-coder:3b" in o for o in out)


def test_joamodel_injected_select_cancel_keeps_current_client():
    session = FakeSession([])
    embed_client = FakeEmbedClient(["qwen2.5-coder:1.5b"])
    lines = iter(["/joamodel", "exit"])
    select = lambda options, current_index: None  # user pressed Esc
    original_client = session.client
    _repl_loop(session, lambda: next(lines), lambda _o: None, embed_client,
               lambda _t: None, select=select)
    assert session.client is original_client


def test_arrow_select_down_down_enter_picks_third_option():
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from assistant.cli import _arrow_select

    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x1b[B\x1b[B\r")  # Down, Down, Enter
        with create_app_session(input=pipe_input, output=DummyOutput()):
            result = _arrow_select(["a", "b", "c"], current_index=0)
    assert result == 2


def test_arrow_select_escape_cancels():
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from assistant.cli import _arrow_select

    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x1b\x1b")  # Escape
        with create_app_session(input=pipe_input, output=DummyOutput()):
            result = _arrow_select(["a", "b", "c"], current_index=0)
    assert result is None


def test_arrow_confirm_ha_returns_true_and_echoes_question():
    from assistant.cli import _arrow_confirm

    out = []
    result = _arrow_confirm("Davom etamizmi?", out.append,
                            select=lambda options, current: 0)

    assert result is True
    assert out == ["Davom etamizmi?"]


def test_arrow_confirm_yoq_returns_false():
    from assistant.cli import _arrow_confirm

    result = _arrow_confirm("Davom etamizmi?", lambda _o: None,
                            select=lambda options, current: 1)

    assert result is False


def test_arrow_confirm_cancelled_returns_false():
    from assistant.cli import _arrow_confirm

    result = _arrow_confirm("Davom etamizmi?", lambda _o: None,
                            select=lambda options, current: None)

    assert result is False


def test_arrow_confirm_options_are_ha_yoq_in_order():
    from assistant.cli import _arrow_confirm

    seen = {}

    def fake_select(options, current_index):
        seen["options"] = options
        seen["current_index"] = current_index
        return 0

    _arrow_confirm("Q?", lambda _o: None, select=fake_select)

    assert seen["options"] == ["Ha", "Yo'q"]
    assert seen["current_index"] == 0
