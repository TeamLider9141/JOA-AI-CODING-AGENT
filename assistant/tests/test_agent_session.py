from assistant.agent.runner import AgentSession
from assistant.agent.tools import ToolContext


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self._replies.pop(0)


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_send_returns_final_answer(tmp_path):
    client = FakeClient(['{"action": "final", "args": {}, "answer": "hi"}'])
    session = AgentSession(make_ctx(tmp_path), client)
    assert session.send("do nothing") == "hi"


def test_history_persists_across_sends(tmp_path):
    client = FakeClient([
        '{"action": "final", "args": {}, "answer": "one"}',
        '{"action": "final", "args": {}, "answer": "two"}',
    ])
    session = AgentSession(make_ctx(tmp_path), client)
    session.send("remember the first task about foo")
    n_before = len(client.calls)
    session.send("second task")
    second_send_first_call = client.calls[n_before]
    assert any("foo" in m["content"] for m in second_send_first_call)


def test_plan_resets_between_sends(tmp_path):
    client = FakeClient([
        '{"action": "plan", "args": {"todo": ["stepfromtaskone"]}}',
        '{"action": "final", "args": {}, "answer": "one"}',
        '{"action": "final", "args": {}, "answer": "two"}',
    ])
    session = AgentSession(make_ctx(tmp_path), client)
    session.send("task one")
    n_before = len(client.calls)
    session.send("task two")
    second_send_first_call = client.calls[n_before]
    reminders = [m["content"] for m in second_send_first_call
                 if "iterations left" in m["content"]]
    assert reminders, "expected a reminder message in the second send"
    assert "no plan yet" in reminders[0].lower()
    assert "stepfromtaskone" not in reminders[0]
