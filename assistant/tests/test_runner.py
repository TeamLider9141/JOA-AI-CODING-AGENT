import pytest

from assistant.agent.runner import AgentSession, run_agent
from assistant.agent.tools import ToolContext


class FakeClient:
    """Returns queued replies in order; records messages it was sent."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return self._replies.pop(0)


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_immediate_final_returns_answer(tmp_path):
    client = FakeClient(['{"action": "final", "args": {}, "answer": "hi"}'])
    result = run_agent("do nothing", make_ctx(tmp_path), client)
    assert result == "hi"


def test_read_then_final(tmp_path):
    (tmp_path / "a.py").write_text("secret contents")
    client = FakeClient([
        '{"action": "read_file", "args": {"path": "a.py"}}',
        '{"action": "final", "args": {}, "answer": "found it"}',
    ])
    result = run_agent("read a.py", make_ctx(tmp_path), client)
    assert result == "found it"
    # the file contents were fed back into the second turn
    second_turn = client.calls[1]
    assert any("secret contents" in m["content"] for m in second_turn)


def test_malformed_json_is_retried_then_succeeds(tmp_path):
    client = FakeClient([
        "I refuse to emit json",
        '{"action": "final", "args": {}, "answer": "ok"}',
    ])
    result = run_agent("x", make_ctx(tmp_path), client)
    assert result == "ok"


def test_gives_up_after_two_bad_parses(tmp_path):
    client = FakeClient(["nope", "still nope", "nope again"])
    result = run_agent("x", make_ctx(tmp_path), client)
    assert "could not parse" in result.lower()


def test_unknown_action_is_reported_back_to_model(tmp_path):
    client = FakeClient([
        '{"action": "fly_to_moon", "args": {}}',
        '{"action": "final", "args": {}, "answer": "done"}',
    ])
    result = run_agent("x", make_ctx(tmp_path), client)
    assert result == "done"
    second_turn = client.calls[1]
    assert any("unknown action" in m["content"].lower() for m in second_turn)


def test_iteration_cap_stops_infinite_loop(tmp_path):
    # always asks to read, never finishes
    reply = '{"action": "read_file", "args": {"path": "a.py"}}'
    (tmp_path / "a.py").write_text("x")
    client = FakeClient([reply] * 20)
    result = run_agent("loop", make_ctx(tmp_path), client, max_iters=3)
    assert "stopped" in result.lower()
    assert len(client.calls) == 3


def test_send_caps_history_keeping_system_prompt(tmp_path):
    from assistant import config

    client = FakeClient(['{"action": "final", "args": {}, "answer": "ok"}'])
    session = AgentSession(make_ctx(tmp_path), client)
    system_msg = session.messages[0]
    for i in range(config.MAX_HISTORY_MESSAGES + 20):
        session.messages.append(
            {"role": "user", "content": f"old message {i}"})
    newest_before = session.messages[-1]

    session.send("new task")

    assert session.messages[0] is system_msg
    assert not any(
        m["content"] == "old message 0" for m in session.messages)
    assert any(
        m["content"] == newest_before["content"] for m in session.messages)
    # pin the exact slice boundary: 61 messages trimmed to 1 system +
    # 39 newest keeps "old message 21.." and drops "old message 20"
    assert not any(
        m["content"] == "old message 20" for m in session.messages)
    assert any(
        m["content"] == "old message 21" for m in session.messages)
