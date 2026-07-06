import inspect

from assistant.agent.runner import build_reminder, run_agent
from assistant.agent.tools import ToolContext


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages):
        # store a deep-ish copy so later mutation doesn't rewrite history
        self.calls.append([dict(m) for m in messages])
        return self._replies.pop(0)


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_build_reminder_without_plan_shows_task():
    r = build_reminder("do the thing", [], 5)
    assert "do the thing" in r
    assert "no plan" in r.lower()


def test_build_reminder_with_plan_lists_steps():
    r = build_reminder("t", ["read config", "edit code"], 3)
    assert "read config" in r
    assert "edit code" in r


def test_reminder_is_injected_before_first_turn(tmp_path):
    client = FakeClient(['{"action": "final", "args": {}, "answer": "ok"}'])
    run_agent("summarize the repo", make_ctx(tmp_path), client)
    first_turn = client.calls[0]
    assert any("summarize the repo" in m["content"] for m in first_turn)


def test_plan_action_persists_into_next_reminder(tmp_path):
    client = FakeClient([
        '{"action": "plan", "args": {"todo": ["read db.py", "commit"]}}',
        '{"action": "final", "args": {}, "answer": "done"}',
    ])
    result = run_agent("do chores", make_ctx(tmp_path), client)
    assert result == "done"
    second_turn = client.calls[1]
    assert any("read db.py" in m["content"] for m in second_turn)


def test_plan_action_with_empty_todo_is_reported_as_error(tmp_path):
    client = FakeClient([
        '{"action": "plan", "args": {"todo": []}}',
        '{"action": "final", "args": {}, "answer": "done"}',
    ])
    run_agent("x", make_ctx(tmp_path), client)
    second_turn = client.calls[1]
    assert any("error" in m["content"].lower() and "todo" in m["content"].lower()
               for m in second_turn)


def test_autofix_sequence_surfaces_failing_exit_code(tmp_path):
    # fail -> (model would fix) -> pass -> final; assert the failure was shown
    client = FakeClient([
        '{"action": "run_cmd", "args": {"command": "exit 1"}}',
        '{"action": "run_cmd", "args": {"command": "echo fixed"}}',
        '{"action": "final", "args": {}, "answer": "fixed it"}',
    ])
    result = run_agent("make it pass", make_ctx(tmp_path), client)
    assert result == "fixed it"
    # the second turn must contain the failing exit code from turn 1
    second_turn = client.calls[1]
    assert any("exit code: 1" in m["content"] for m in second_turn)


def test_default_max_iters_is_15():
    assert inspect.signature(run_agent).parameters["max_iters"].default == 15
