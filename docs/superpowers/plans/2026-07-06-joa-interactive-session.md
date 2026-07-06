# `joa` Interactive Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Type `joa` in an indexed repo to open an interactive agent session (like Claude Code) where each line is handled by the coding agent and the conversation carries across turns; `joa <args>` still works as a short form for the existing CLI.

**Architecture:** Refactor the runner's loop into a stateful `AgentSession` class that keeps conversation history across `.send()` calls (with `run_agent` staying a thin wrapper so nothing existing breaks). Add a `repl` CLI command whose read-loop is extracted into a testable `_repl_loop` function, and a `bin/joa` shell launcher that opens the repl on the current directory (no args) or passes arguments through to the CLI.

**Tech Stack:** Python 3.10, typer, existing agent loop, pytest, bash.

**Spec:** `docs/superpowers/specs/2026-07-06-joa-interactive-session-design.md`

---

## File Structure

All paths relative to repo root `/home/eaduinte/Desktop/system_llm`.

- `assistant/agent/runner.py` — MODIFY: add `AgentSession`; `run_agent` becomes a wrapper.
- `assistant/cli.py` — MODIFY: import `AgentSession`, add `_repl_loop` and the `repl` command.
- `bin/joa` — CREATE: launcher script.
- `assistant/README.md` — MODIFY: document `joa`.
- `assistant/tests/test_agent_session.py` — CREATE.
- `assistant/tests/test_repl.py` — CREATE.

Existing `test_runner.py` and `test_runner_planner.py` must stay green — `run_agent` and `build_reminder` keep their signatures and behavior.

---

### Task 1: `AgentSession` (stateful runner)

**Files:**
- Modify: `assistant/agent/runner.py`
- Test: `assistant/tests/test_agent_session.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_agent_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'AgentSession'`

- [ ] **Step 3: Refactor `assistant/agent/runner.py`**

Replace the `run_agent` function (keep `build_reminder`, `_apply_plan`,
`_parse_with_retries`, `_run_tool`, and `MAX_PARSE_RETRIES` exactly as they
are) with the class plus a thin wrapper:

```python
class AgentSession:
    """A continuing agent conversation: history persists across send() calls.

    The plan scratchpad resets per send (each user request plans fresh), but
    self.messages carries the whole conversation so later turns see earlier
    context.
    """

    def __init__(self, ctx: ToolContext, client, max_iters: int = 15):
        self.ctx = ctx
        self.client = client
        self.max_iters = max_iters
        self.messages = [
            {"role": "system", "content": build_system_prompt()},
        ]

    def send(self, task: str) -> str:
        self.messages.append({"role": "user", "content": f"Task: {task}"})
        plan: list[str] = []

        for i in range(self.max_iters):
            self.messages.append({
                "role": "user",
                "content": build_reminder(task, plan, self.max_iters - i),
            })
            reply = self.client.chat(self.messages)
            self.messages.append({"role": "assistant", "content": reply})

            action = _parse_with_retries(reply, self.messages, self.client)
            if action is None:
                return "could not parse a valid action from the model"

            name = action["action"]
            if name == "final":
                return action.get("answer", "(no answer provided)")

            if name == "plan":
                plan, result = _apply_plan(action.get("args", {}))
            else:
                result = _run_tool(name, action.get("args", {}), self.ctx)
            self.messages.append({"role": "user", "content": f"Result:\n{result}"})

        return f"stopped after {self.max_iters} iterations without a final answer"


def run_agent(task: str, ctx: ToolContext, client,
              max_iters: int = 15) -> str:
    """One-shot agent run — a fresh session that handles a single task.

    `client` needs a `.chat(messages) -> str` method (OllamaClient qualifies).
    """
    return AgentSession(ctx, client, max_iters).send(task)
```

Make sure the imports at the top of the file still include `ToolContext`
(already imported via `from assistant.agent.tools import TOOLS, ToolContext,
ToolError`).

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/pytest assistant/tests/test_agent_session.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the existing runner tests (must stay green)**

Run: `.venv/bin/pytest assistant/tests/test_runner.py assistant/tests/test_runner_planner.py -v`
Expected: all pass (6 + 7 = 13). `run_agent` and `build_reminder` are
unchanged in behavior, so every prior assertion still holds.

- [ ] **Step 6: Commit**

```bash
git add assistant/agent/runner.py assistant/tests/test_agent_session.py
git commit -m "feat: add AgentSession for continuing multi-turn conversations"
```

---

### Task 2: `repl` command and testable loop

**Files:**
- Modify: `assistant/cli.py`
- Test: `assistant/tests/test_repl.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_repl.py -v`
Expected: FAIL — `ImportError: cannot import name '_repl_loop'`

- [ ] **Step 3: Update the import in `assistant/cli.py`**

Change the runner import line from:

```python
from assistant.agent.runner import run_agent
```

to:

```python
from assistant.agent.runner import AgentSession, run_agent
```

- [ ] **Step 4: Add `_repl_loop` and the `repl` command to `assistant/cli.py`**

Add both right after the `agent` command function (before `build_prompt`):

```python
def _repl_loop(session, read_line, echo) -> None:
    """Drive an AgentSession from a line source until exit/EOF.

    `read_line()` returns the next input line (raising EOFError at end of
    input); `echo(text)` prints a line. Kept separate from the CLI command so
    the loop is testable without a live model.
    """
    echo("joa session — type 'exit' or Ctrl-D to quit")
    while True:
        try:
            line = read_line()
        except EOFError:
            return
        stripped = line.strip()
        if stripped in ("exit", "quit"):
            return
        if not stripped:
            continue
        try:
            answer = session.send(stripped)
        except OllamaError as exc:
            echo(str(exc))
            continue
        echo(answer)


@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
):
    """Interactive agent session over the repo (defaults to current dir)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=client.embed,
        confirm=lambda msg: typer.confirm(msg),
    )
    session = AgentSession(ctx, client)
    _repl_loop(session, lambda: input("joa> "), typer.echo)
```

- [ ] **Step 5: Run the new repl tests**

Run: `.venv/bin/pytest assistant/tests/test_repl.py -v`
Expected: 6 passed

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (the existing suite plus 3 from Task 1 and 6 here).

- [ ] **Step 7: Commit**

```bash
git add assistant/cli.py assistant/tests/test_repl.py
git commit -m "feat: add interactive repl command backed by AgentSession"
```

---

### Task 3: `bin/joa` launcher and README

**Files:**
- Create: `bin/joa`
- Modify: `assistant/README.md`

- [ ] **Step 1: Write `bin/joa`**

```bash
#!/usr/bin/env bash
# joa — launcher for the local coding assistant.
#   no args  → interactive agent session (repl) on the current directory
#   args     → passed straight through to the CLI, e.g. joa ask "…"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HERE/.venv/bin/python"

if [ "$#" -eq 0 ]; then
    exec "$PY" -m assistant.cli repl --repo "$(pwd)"
else
    exec "$PY" -m assistant.cli "$@"
fi
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /home/eaduinte/Desktop/system_llm/bin/joa
```

- [ ] **Step 3: Verify passthrough mode works (no model needed)**

```bash
/home/eaduinte/Desktop/system_llm/bin/joa --help
```

Expected: the CLI help text listing `index`, `search`, `ask`, `agent`, `repl`
— proving arguments pass through to `assistant.cli`.

- [ ] **Step 4: Verify no-args launches the repl and exits cleanly on EOF**

```bash
echo "exit" | /home/eaduinte/Desktop/system_llm/bin/joa
```

Run this from an indexed repo directory, e.g.:

```bash
cd /home/eaduinte/Desktop/crystal_bot && echo "exit" | /home/eaduinte/Desktop/system_llm/bin/joa
```

Expected: prints the `joa session — type 'exit' or Ctrl-D to quit` banner and
exits 0 (the piped `exit` ends the loop). If run from a non-indexed directory,
expect the "No index found" message and a non-zero exit instead — that is
correct behavior.

- [ ] **Step 5: Update `assistant/README.md`**

Add this section immediately after the `## Agent` section:

```markdown
## Interactive session (`joa`)

Put the launcher on your PATH once:

    export PATH="$HOME/Desktop/system_llm/bin:$PATH"   # add to ~/.zshrc

Then, from any indexed repo:

    cd ~/Desktop/crystal_bot
    joa                      # opens an interactive agent session
    joa> add a docstring to change_balance
    joa> now commit that change
    joa> exit

Each line is handled by the coding agent (read/write/run/search, with writes
and commands confirmed), and the conversation carries across turns so
follow-ups remember earlier context. `joa <args>` still works as a short form
for the CLI, e.g. `joa ask "how does X work"` or `joa agent "fix the bug"`.
```

- [ ] **Step 6: Full verification and commit**

```bash
.venv/bin/pytest -q
git add bin/joa assistant/README.md
git commit -m "feat: add joa launcher and document interactive session"
```

Expected: full suite green before claiming completion. (`bin/joa` itself has
no pytest coverage — it is a thin launcher verified manually in Steps 3-4.)

---

## Self-Review Notes

- **Spec coverage:** `AgentSession` with persistent history and per-send plan
  reset (Task 1), `repl` command with the exit/EOF/blank/OllamaError-survival
  behavior (Task 2), `bin/joa` no-args-repl / args-passthrough launcher and
  README PATH instructions (Task 3) — every spec section has a task.
- **run_agent unchanged:** it now delegates to `AgentSession(...).send(task)`
  but keeps its exact signature and single-shot behavior, so `test_runner.py`
  and `test_runner_planner.py` stay green (verified in Task 1 Step 5).
- **Loop testability:** `_repl_loop` is factored out of the `repl` command so
  the read/exit/blank/error behavior is unit-tested with a fake session and a
  scripted line source — no live Ollama needed.
- **Placeholder scan:** no TBDs; every code step shows complete code, every
  verification step shows the exact command and expected output.

---

## Out of scope (from spec)

Context-window trimming for long sessions, readline history/arrow-key recall,
streaming intermediate agent steps, and any install mechanism beyond a PATH
entry.
