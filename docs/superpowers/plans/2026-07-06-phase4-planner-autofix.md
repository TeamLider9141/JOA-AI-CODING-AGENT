# Phase 4 — Planner + Auto-fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 7B agent more reliable at multi-step chores by giving it a TODO scratchpad pinned in front of it each turn, letting it see command exit codes (so it can auto-fix failures), and giving commands enough time to finish.

**Architecture:** Small edits to the existing Phase 3 agent (`assistant/agent/`). The runner keeps a plan list the model sets via a new `plan` action and re-injects a compact reminder (task + plan + iterations-left) before every turn. `run_cmd` prepends its exit code to the result and reads its timeout from `config`. No new modules, no new dependencies.

**Tech Stack:** Python 3.10, existing agent loop (protocol/tools/runner), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-phase4-planner-autofix-design.md`

---

## File Structure

All paths relative to repo root `/home/eaduinte/Desktop/system_llm`.

- `assistant/config.py` — MODIFY: add `RUN_CMD_TIMEOUT = 120`.
- `assistant/agent/tools.py` — MODIFY: `run_cmd` prepends `exit code: N`, reads timeout from `config`.
- `assistant/agent/protocol.py` — MODIFY: system prompt gains the `plan` action and the auto-fix instruction.
- `assistant/agent/runner.py` — MODIFY: `build_reminder()`, plan scratchpad, per-turn reminder injection, `plan` action handling, default `max_iters=15`.
- `assistant/tests/test_tools_exitcode.py` — CREATE.
- `assistant/tests/test_protocol_plan.py` — CREATE.
- `assistant/tests/test_runner_planner.py` — CREATE.

Existing test files (`test_tools.py`, `test_protocol.py`, `test_runner.py`) must stay green — the changes are additive.

---

### Task 1: run_cmd exit code + config timeout

**Files:**
- Modify: `assistant/config.py`
- Modify: `assistant/agent/tools.py`
- Test: `assistant/tests/test_tools_exitcode.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant import config
from assistant.agent.tools import ToolContext, run_cmd


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_successful_command_reports_exit_code_zero(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "echo hi"})
    assert "exit code: 0" in out
    assert "hi" in out


def test_failing_command_reports_nonzero_exit_code(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "exit 3"})
    assert "exit code: 3" in out


def test_default_timeout_comes_from_config(tmp_path):
    assert config.RUN_CMD_TIMEOUT == 120
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_tools_exitcode.py -v`
Expected: FAIL — `AttributeError: module 'assistant.config' has no attribute 'RUN_CMD_TIMEOUT'` (and the exit-code assertions fail).

- [ ] **Step 3: Add the timeout to `assistant/config.py`**

Append after the existing `NUM_CTX` / `REQUEST_TIMEOUT` block (near the Ollama settings):

```python
# --- Agent ---
RUN_CMD_TIMEOUT = 120  # seconds; generous enough for git push on slow links
```

- [ ] **Step 4: Update `run_cmd` in `assistant/agent/tools.py`**

Change the imports and the `RUN_CMD_TIMEOUT` line at the top of the file:

```python
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from assistant import config
from assistant.agent.safety import resolve_in_root
from assistant.indexer.pipeline import Embedder, search_index

MAX_OUTPUT_CHARS = 4000  # keep tool results within the model's context
```

(The old module-level `RUN_CMD_TIMEOUT = 30` line is removed — the value now lives in `config`.)

Then replace the whole `run_cmd` function with:

```python
def run_cmd(ctx: ToolContext, args: dict,
            timeout: int | None = None) -> str:
    if timeout is None:
        timeout = config.RUN_CMD_TIMEOUT
    command = args["command"]
    if not ctx.confirm(f"run command: {command!r}?"):
        return "command cancelled by user"
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(ctx.root),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out after {timeout}s"
    output = (proc.stdout + proc.stderr) or "(no output)"
    return _truncate(f"exit code: {proc.returncode}\n{output}")
```

- [ ] **Step 5: Run the new tests and the existing tools tests**

Run: `.venv/bin/pytest assistant/tests/test_tools_exitcode.py assistant/tests/test_tools.py -v`
Expected: all pass. The existing `test_run_cmd_returns_output_when_confirmed` (asserts `"hi" in out`), `test_run_cmd_declined_does_not_run` (asserts `"cancel" in ...`), and `test_run_cmd_times_out` (asserts `"timed out" in ...`) remain green because the exit-code prefix is additive and the cancel/timeout branches are unchanged.

- [ ] **Step 6: Commit**

```bash
git add assistant/config.py assistant/agent/tools.py assistant/tests/test_tools_exitcode.py
git commit -m "feat: run_cmd reports exit code and reads timeout from config"
```

---

### Task 2: plan action + auto-fix instruction in the system prompt

**Files:**
- Modify: `assistant/agent/protocol.py`
- Test: `assistant/tests/test_protocol_plan.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.agent.protocol import build_system_prompt, parse_action


def test_system_prompt_describes_plan_action():
    prompt = build_system_prompt()
    assert "plan" in prompt
    assert "todo" in prompt


def test_system_prompt_mentions_exit_code_autofix():
    prompt = build_system_prompt()
    assert "exit code" in prompt.lower()


def test_plan_action_parses_like_any_other_action():
    action = parse_action(
        '{"action": "plan", "args": {"todo": ["a", "b"]}}')
    assert action["action"] == "plan"
    assert action["args"]["todo"] == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_protocol_plan.py -v`
Expected: FAIL — the first two assertions fail (prompt lacks "plan"/"exit code"). The third already passes (parse_action is generic), which is fine.

- [ ] **Step 3: Update the `SYSTEM_PROMPT` in `assistant/agent/protocol.py`**

Replace the entire `SYSTEM_PROMPT = """..."""` string with:

```python
SYSTEM_PROMPT = """\
You are a coding agent working inside a single repository. You act one step
at a time. On each turn you MUST reply with exactly one JSON object and
nothing else — no prose outside the JSON.

Available actions:
- {"action": "plan", "args": {"todo": ["step one", "step two"]}}
- {"action": "read_file", "args": {"path": "relative/path.py"}}
- {"action": "write_file", "args": {"path": "relative/path.py", "content": "..."}}
- {"action": "run_cmd", "args": {"command": "pytest -q"}}
- {"action": "search_code", "args": {"query": "where is X"}}
- {"action": "final", "args": {}, "answer": "your answer to the user"}

Rules:
- For any task that needs more than one step, START by emitting a "plan"
  action with an ordered todo list. Revise it with another "plan" action
  whenever the situation changes.
- Paths are always relative to the repo root. Never use absolute paths or "..".
- After each action you will be shown its result, then take the next step.
- Use search_code to locate code, read_file to inspect it, write_file to
  change it, run_cmd to run tests or commands.
- run_cmd results begin with "exit code: N". If N is not 0 the command
  failed — read the error, fix the cause, and re-run to verify before moving on.
- When the task is done, reply with the "final" action and put your answer
  in the "answer" field.
"""
```

- [ ] **Step 4: Run the new and existing protocol tests**

Run: `.venv/bin/pytest assistant/tests/test_protocol_plan.py assistant/tests/test_protocol.py -v`
Expected: all pass. The existing `test_system_prompt_lists_every_tool` still passes because read_file/write_file/run_cmd/search_code/final are all still present.

- [ ] **Step 5: Commit**

```bash
git add assistant/agent/protocol.py assistant/tests/test_protocol_plan.py
git commit -m "feat: add plan action and auto-fix guidance to system prompt"
```

---

### Task 3: plan scratchpad + per-turn reminder in the runner

**Files:**
- Modify: `assistant/agent/runner.py`
- Test: `assistant/tests/test_runner_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_runner_planner.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_reminder'`.

- [ ] **Step 3: Rewrite `assistant/agent/runner.py`**

Replace the entire file with:

```python
from assistant.agent.protocol import (
    ProtocolError, build_system_prompt, parse_action,
)
from assistant.agent.safety import PathJailError
from assistant.agent.tools import TOOLS, ToolContext, ToolError

MAX_PARSE_RETRIES = 2


def build_reminder(task: str, plan: list[str], iters_left: int) -> str:
    """A compact, always-recent nudge so the weak model keeps the goal in view."""
    if plan:
        plan_str = " ".join(f"{i + 1}.{step}" for i, step in enumerate(plan))
    else:
        plan_str = "(no plan yet)"
    return f"(Task: {task} | Plan: {plan_str} | {iters_left} iterations left)"


def run_agent(task: str, ctx: ToolContext, client,
              max_iters: int = 15) -> str:
    """Drive the plan->act->observe loop until the model says 'final'.

    `client` needs a `.chat(messages) -> str` method (OllamaClient qualifies).
    A running todo list (set via the 'plan' action) is re-shown each turn.
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Task: {task}"},
    ]
    plan: list[str] = []

    for i in range(max_iters):
        messages.append({
            "role": "user",
            "content": build_reminder(task, plan, max_iters - i),
        })
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})

        action = _parse_with_retries(reply, messages, client)
        if action is None:
            return "could not parse a valid action from the model"

        name = action["action"]
        if name == "final":
            return action.get("answer", "(no answer provided)")

        if name == "plan":
            plan, result = _apply_plan(action.get("args", {}))
        else:
            result = _run_tool(name, action.get("args", {}), ctx)
        messages.append({"role": "user", "content": f"Result:\n{result}"})

    return f"stopped after {max_iters} iterations without a final answer"


def _apply_plan(args: dict) -> tuple[list[str], str]:
    todo = args.get("todo")
    if isinstance(todo, list) and todo:
        return [str(step) for step in todo], "plan updated"
    return [], "error: plan action needs a non-empty 'todo' list"


def _parse_with_retries(reply: str, messages: list[dict], client) -> dict | None:
    for _ in range(MAX_PARSE_RETRIES):
        try:
            return parse_action(reply)
        except ProtocolError as exc:
            messages.append({
                "role": "user",
                "content": (
                    f"Your reply could not be parsed ({exc}). Reply with "
                    "exactly one JSON object and nothing else."
                ),
            })
            reply = client.chat(messages)
            messages.append({"role": "assistant", "content": reply})
    try:
        return parse_action(reply)
    except ProtocolError:
        return None


def _run_tool(name: str, args: dict, ctx: ToolContext) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"unknown action '{name}'. Valid: {', '.join(TOOLS)}, plan, final"
    try:
        return tool(ctx, args)
    except (ToolError, PathJailError) as exc:
        return f"error: {exc}"
    except KeyError as exc:
        return f"error: missing argument {exc}"
```

Note the two behavioral changes beyond the plan/reminder work: `max_iters` default is now `15`, and the unknown-action message lists `plan` as valid.

- [ ] **Step 4: Run the new planner tests**

Run: `.venv/bin/pytest assistant/tests/test_runner_planner.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the existing runner tests (must stay green)**

Run: `.venv/bin/pytest assistant/tests/test_runner.py -v`
Expected: 6 passed. Note `test_iteration_cap_stops_infinite_loop` passes `max_iters=3` explicitly, so the default change to 15 does not affect it; the reminder injection adds messages but the assertions (`"stopped" in result`, `len(client.calls) == 3`) still hold because exactly one `chat` call happens per iteration.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (65 from before + 3 + 3 + 7 new = 78).

- [ ] **Step 7: Commit**

```bash
git add assistant/agent/runner.py assistant/tests/test_runner_planner.py
git commit -m "feat: add TODO scratchpad planner with per-turn reminder to agent runner"
```

---

### Task 4: Real end-to-end — agent does a git chore

Verify the whole point of Phase 4: the agent handling a multi-step git chore on a real git repo, with the exit-code signal and the longer timeout in play. Manual verification, not a unit test.

**Do NOT run the agent against `system_llm` itself or any real project.** Use a throwaway git repo created for this test.

- [ ] **Step 1: Create a throwaway git repo and index it**

```bash
cd /home/eaduinte/Desktop/system_llm
SANDBOX=$(mktemp -d)/agent_sandbox
mkdir -p "$SANDBOX"
cd "$SANDBOX"
git init -b main
git config user.email "test@example.com"
git config user.name "Agent Test"
printf 'def add(a, b):\n    return a + b\n' > calc.py
git add calc.py && git commit -m "initial"
printf '\ndef sub(a, b):\n    return a - b\n' >> calc.py   # unstaged change
cd /home/eaduinte/Desktop/system_llm
.venv/bin/python -m assistant.cli index "$SANDBOX"
echo "SANDBOX=$SANDBOX"
```

Expected: `Indexed N chunks from <sandbox>`. Note the printed `SANDBOX` path for the next steps.

- [ ] **Step 2: Ask the agent to commit the change (answer y to confirmations)**

```bash
yes y | .venv/bin/python -m assistant.cli agent \
  "Stage all changes and commit them with the message: add sub function" \
  --repo "$SANDBOX"
```

Expected: the agent emits a `plan`, then one or more `run_cmd` steps (`git add`, `git commit`), each prompting a `run command: ...?` confirmation (auto-answered `y`), then a `final`. On CPU this is slow (minutes). The 7B model may need a few iterations; that is expected.

- [ ] **Step 3: Verify the commit landed**

```bash
git -C "$SANDBOX" log --oneline
git -C "$SANDBOX" status --short
```

Expected: a new commit (message roughly "add sub function") on top of "initial"; working tree clean. If the model didn't produce a clean commit (weak-model variance), re-run Step 2 once; note the outcome honestly either way — the mechanics (plan → run_cmd git → confirm gate → exit-code feedback) are what's being verified, not perfect model phrasing.

- [ ] **Step 4: Clean up the sandbox**

```bash
rm -rf "$SANDBOX"
rm -rf assistant/.data/agent_sandbox
```

- [ ] **Step 5: Update the assistant README**

In `assistant/README.md`, under the `## Agent` section, replace the final "Phase 4 (next, separate plan): ..." paragraph with:

```markdown
The agent maintains a TODO scratchpad (the `plan` action) that is re-shown
each turn so it keeps multi-step chores on track, and `run_cmd` results include
the exit code so it can notice a failure and re-run after fixing. Verified on a
throwaway git repo: given "stage and commit these changes", the agent planned,
ran `git add`/`git commit` via confirmed `run_cmd` steps, and produced a clean
commit.

Still deferred (separate plan): a cross-encoder reranker for retrieval quality.
```

- [ ] **Step 6: Full verification and commit**

```bash
.venv/bin/pytest -q
git add assistant/README.md
git commit -m "docs: document Phase 4 planner and auto-fix in README"
```

Expected: full suite green before claiming completion.

---

## After this plan

The only spec item still outstanding across the whole project is the
cross-encoder **reranker** (retrieval quality), which is its own small plan when
wanted: score the fused top-20 with a CPU cross-encoder, return top-5, and
measure the hit@5 change against the existing gold set.
