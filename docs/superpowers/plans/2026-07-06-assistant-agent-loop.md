# Local Coding Assistant — Agent Loop Implementation Plan (Phase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI `agent "task"` command that lets the local LLM plan and act: it receives retrieval context, emits a JSON tool call, we execute it (read/write files, run commands, search code), feed the result back, and loop until the model returns a final answer — all inside a path jail with confirmation on every side-effecting action.

**Architecture:** Hand-written prompt-based tool calling (no Ollama native `tools` API). The system prompt describes the tools and the exact JSON shape; the model returns `{"action": ..., "args": {...}}`; we parse it ourselves, execute through a tool registry, append the result to the conversation, and repeat. This makes the mechanics of an agent loop fully visible rather than hidden behind a framework. Builds directly on `search_index()`, `OllamaClient`, and `config` from the retrieval-core plan.

**Tech Stack:** Python 3.10, httpx, typer, existing retrieval core (Qdrant + BM25 + Ollama), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md` (Phase 3).

**Scope note:** Phase 4 (cross-encoder reranker, multi-step planner, auto-fix loop) is deliberately excluded — it gets its own plan once this lands. This plan stops at a single-step-per-turn tool-calling agent, which is the "real coding assistant" milestone.

---

## File Structure

All paths relative to repo root `/home/eaduinte/Desktop/system_llm`.

```
assistant/
├── llm/
│   └── ollama_client.py       # MODIFY: add chat() (non-streaming, returns full text)
├── agent/
│   ├── __init__.py            # new package
│   ├── safety.py              # PathJailError, resolve_in_root()
│   ├── tools.py               # ToolContext, ToolError, tool fns, TOOLS registry
│   ├── protocol.py            # parse_action(), ProtocolError, build_system_prompt()
│   └── runner.py              # run_agent() loop
├── cli.py                     # MODIFY: add `agent` command
└── tests/
    ├── test_ollama_chat.py    # new
    ├── test_safety.py
    ├── test_tools.py
    ├── test_protocol.py
    ├── test_runner.py
    └── test_cli_agent.py
```

Agent writes go only into the target repo root, gated by `resolve_in_root` (path jail) and a confirm callback. The agent never touches `system_llm` itself unless that is the explicit target repo.

---

### Task 1: Non-streaming chat on OllamaClient

The agent parses a full JSON response, so it needs the complete text at once, not a token stream.

**Files:**
- Modify: `assistant/llm/ollama_client.py`
- Test: `assistant/tests/test_ollama_chat.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx
import pytest

from assistant.llm.ollama_client import OllamaClient, OllamaError


def make_client(handler) -> OllamaClient:
    return OllamaClient(base_url="http://test",
                        transport=httpx.MockTransport(handler))


def test_chat_returns_full_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "hello world"},
            "done": True,
        })

    out = make_client(handler).chat([{"role": "user", "content": "hi"}])
    assert out == "hello world"


def test_chat_connect_error_is_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_ollama_chat.py -v`
Expected: FAIL — `AttributeError: 'OllamaClient' object has no attribute 'chat'`

- [ ] **Step 3: Add `chat()` to `assistant/llm/ollama_client.py`**

Insert this method immediately after `embed()` (before `chat_stream()`):

```python
    def chat(self, messages: list[dict]) -> str:
        data = self._post("/api/chat", {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": config.NUM_CTX},
        })
        return data["message"]["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_ollama_chat.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/ollama_client.py assistant/tests/test_ollama_chat.py
git commit -m "feat: add non-streaming chat() to Ollama client"
```

---

### Task 2: Path jail

Every file/command path the agent proposes must resolve to a location inside the target repo root. This is the core safety boundary.

**Files:**
- Create: `assistant/agent/__init__.py` (empty)
- Create: `assistant/agent/safety.py`
- Test: `assistant/tests/test_safety.py`

- [ ] **Step 1: Create the package init**

```bash
touch assistant/agent/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
import pytest

from assistant.agent.safety import PathJailError, resolve_in_root


def test_normal_relative_path_resolves(tmp_path):
    (tmp_path / "db.py").write_text("x = 1")
    resolved = resolve_in_root(tmp_path, "db.py")
    assert resolved == (tmp_path / "db.py").resolve()


def test_nested_path_resolves(tmp_path):
    (tmp_path / "handlers").mkdir()
    resolved = resolve_in_root(tmp_path, "handlers/user.py")
    assert resolved == (tmp_path / "handlers" / "user.py").resolve()


def test_parent_traversal_is_blocked(tmp_path):
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "../secret.py")


def test_absolute_path_outside_root_is_blocked(tmp_path):
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "/etc/passwd")


def test_symlink_escape_is_blocked(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret")
    (tmp_path / "link.py").symlink_to(outside)
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "link.py")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_safety.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_in_root'`

- [ ] **Step 4: Write `assistant/agent/safety.py`**

```python
from pathlib import Path


class PathJailError(RuntimeError):
    """A proposed path resolves outside the target repository root."""


def resolve_in_root(root: Path, rel: str) -> Path:
    """Resolve `rel` against `root` and guarantee it stays inside root.

    Uses fully-resolved (symlink-followed) real paths on both sides, so
    `..` segments, absolute paths, and symlink escapes are all rejected.
    """
    root_real = root.resolve()
    candidate = (root_real / rel).resolve()
    if candidate != root_real and root_real not in candidate.parents:
        raise PathJailError(f"path escapes repo root: {rel}")
    return candidate
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_safety.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add assistant/agent/__init__.py assistant/agent/safety.py assistant/tests/test_safety.py
git commit -m "feat: add path jail for agent file access"
```

---

### Task 3: Tools and registry

Four tools with a uniform `(ctx, args) -> str` signature. Side-effecting tools (`write_file`, `run_cmd`) go through a confirm callback so nothing mutates the repo without approval.

**Files:**
- Create: `assistant/agent/tools.py`
- Test: `assistant/tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from assistant.agent.safety import PathJailError
from assistant.agent.tools import (
    ToolContext, ToolError, TOOLS, read_file, write_file, run_cmd,
)


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_registry_exposes_all_four_tools():
    assert set(TOOLS) == {"read_file", "write_file", "run_cmd", "search_code"}


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "a.py").write_text("hello")
    assert read_file(make_ctx(tmp_path), {"path": "a.py"}) == "hello"


def test_read_file_missing_raises_tool_error(tmp_path):
    with pytest.raises(ToolError):
        read_file(make_ctx(tmp_path), {"path": "nope.py"})


def test_read_file_traversal_raises_path_jail(tmp_path):
    with pytest.raises(PathJailError):
        read_file(make_ctx(tmp_path), {"path": "../x"})


def test_write_file_creates_file_when_confirmed(tmp_path):
    result = write_file(make_ctx(tmp_path),
                        {"path": "new.py", "content": "print(1)"})
    assert (tmp_path / "new.py").read_text() == "print(1)"
    assert "new.py" in result


def test_write_file_declined_leaves_no_file(tmp_path):
    ctx = make_ctx(tmp_path, confirm=lambda _msg: False)
    result = write_file(ctx, {"path": "new.py", "content": "x"})
    assert not (tmp_path / "new.py").exists()
    assert "cancel" in result.lower()


def test_run_cmd_returns_output_when_confirmed(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "echo hi"})
    assert "hi" in out


def test_run_cmd_declined_does_not_run(tmp_path):
    ctx = make_ctx(tmp_path, confirm=lambda _msg: False)
    assert "cancel" in run_cmd(ctx, {"command": "echo hi"}).lower()


def test_run_cmd_times_out(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "sleep 5"}, timeout=1)
    assert "timed out" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_tools.py -v`
Expected: FAIL — `ImportError: cannot import name 'ToolContext'`

- [ ] **Step 3: Write `assistant/agent/tools.py`**

```python
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from assistant.agent.safety import resolve_in_root
from assistant.indexer.pipeline import Embedder, search_index

RUN_CMD_TIMEOUT = 30  # seconds
MAX_OUTPUT_CHARS = 4000  # keep tool results within the model's context


class ToolError(RuntimeError):
    """A tool failed in an expected way (missing file, bad command)."""


@dataclass
class ToolContext:
    root: Path
    data_dir: Path
    embedder: Embedder
    confirm: Callable[[str], bool]


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [truncated]"


def read_file(ctx: ToolContext, args: dict) -> str:
    path = resolve_in_root(ctx.root, args["path"])
    if not path.is_file():
        raise ToolError(f"no such file: {args['path']}")
    return _truncate(path.read_text(errors="ignore"))


def write_file(ctx: ToolContext, args: dict) -> str:
    path = resolve_in_root(ctx.root, args["path"])
    content = args.get("content", "")
    prompt = f"write {len(content)} bytes to {args['path']}?"
    if not ctx.confirm(prompt):
        return "write cancelled by user"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"wrote {len(content)} bytes to {args['path']}"


def run_cmd(ctx: ToolContext, args: dict,
            timeout: int = RUN_CMD_TIMEOUT) -> str:
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
    return _truncate((proc.stdout + proc.stderr) or "(no output)")


def search_code(ctx: ToolContext, args: dict) -> str:
    results = search_index(args["query"], ctx.data_dir, ctx.embedder)
    if not results:
        return "no matches"
    lines = [
        f"{p['path']}:{p['start_line']}-{p['end_line']}  {p['symbol']}"
        for _cid, _score, p in results
    ]
    return "\n".join(lines)


TOOLS: dict[str, Callable[[ToolContext, dict], str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
    "search_code": search_code,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_tools.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/agent/tools.py assistant/tests/test_tools.py
git commit -m "feat: add agent tools (read/write/run/search) with confirm gate"
```

---

### Task 4: JSON protocol

Parse the model's reply into an action dict, and build the system prompt that teaches it the format. Real models wrap JSON in prose or ```json fences — the parser must tolerate that.

**Files:**
- Create: `assistant/agent/protocol.py`
- Test: `assistant/tests/test_protocol.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from assistant.agent.protocol import (
    ProtocolError, build_system_prompt, parse_action,
)


def test_parses_bare_json():
    action = parse_action('{"action": "read_file", "args": {"path": "a.py"}}')
    assert action["action"] == "read_file"
    assert action["args"]["path"] == "a.py"


def test_parses_json_in_code_fence():
    text = 'Sure!\n```json\n{"action": "final", "args": {}, "answer": "done"}\n```'
    action = parse_action(text)
    assert action["action"] == "final"
    assert action["answer"] == "done"


def test_parses_json_embedded_in_prose():
    text = 'I will read it. {"action": "read_file", "args": {"path": "x"}} now.'
    assert parse_action(text)["action"] == "read_file"


def test_missing_action_key_raises():
    with pytest.raises(ProtocolError):
        parse_action('{"args": {}}')


def test_no_json_at_all_raises():
    with pytest.raises(ProtocolError):
        parse_action("I am not going to give you any json today")


def test_system_prompt_lists_every_tool():
    prompt = build_system_prompt()
    for tool in ("read_file", "write_file", "run_cmd", "search_code", "final"):
        assert tool in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_protocol.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_action'`

- [ ] **Step 3: Write `assistant/agent/protocol.py`**

```python
import json

SYSTEM_PROMPT = """\
You are a coding agent working inside a single repository. You act one step
at a time. On each turn you MUST reply with exactly one JSON object and
nothing else — no prose outside the JSON.

Available actions:
- {"action": "read_file", "args": {"path": "relative/path.py"}}
- {"action": "write_file", "args": {"path": "relative/path.py", "content": "..."}}
- {"action": "run_cmd", "args": {"command": "pytest -q"}}
- {"action": "search_code", "args": {"query": "where is X"}}
- {"action": "final", "args": {}, "answer": "your answer to the user"}

Rules:
- Paths are always relative to the repo root. Never use absolute paths or "..".
- After each action you will be shown its result, then take the next step.
- Use search_code to locate code, read_file to inspect it, write_file to
  change it, run_cmd to run tests or commands.
- When the task is done, reply with the "final" action and put your answer
  in the "answer" field.
"""


class ProtocolError(RuntimeError):
    """The model's reply did not contain a usable JSON action."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def parse_action(text: str) -> dict:
    """Extract the first balanced JSON object from the model's reply."""
    obj = _extract_json(text)
    if obj is None:
        raise ProtocolError(f"no JSON object found in reply: {text[:200]!r}")
    if "action" not in obj:
        raise ProtocolError(f"JSON is missing 'action' key: {obj}")
    return obj


def _extract_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_protocol.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/agent/protocol.py assistant/tests/test_protocol.py
git commit -m "feat: add JSON tool-call protocol parser and system prompt"
```

---

### Task 5: Agent runner loop

Ties it together: seed the conversation, loop up to `max_iters`, parse each reply (retry twice on parse failure with the error fed back), execute the tool, append the result, stop on `final`.

**Files:**
- Create: `assistant/agent/runner.py`
- Test: `assistant/tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from assistant.agent.runner import run_agent
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_agent'`

- [ ] **Step 3: Write `assistant/agent/runner.py`**

```python
from assistant.agent.protocol import (
    ProtocolError, build_system_prompt, parse_action,
)
from assistant.agent.safety import PathJailError
from assistant.agent.tools import TOOLS, ToolContext, ToolError

MAX_PARSE_RETRIES = 2


def run_agent(task: str, ctx: ToolContext, client,
              max_iters: int = 10) -> str:
    """Drive the plan→act→observe loop until the model says 'final'.

    `client` needs a `.chat(messages) -> str` method (OllamaClient qualifies).
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Task: {task}"},
    ]

    for _ in range(max_iters):
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})

        action = _parse_with_retries(reply, messages, client)
        if action is None:
            return "could not parse a valid action from the model"

        name = action["action"]
        if name == "final":
            return action.get("answer", "(no answer provided)")

        result = _run_tool(name, action.get("args", {}), ctx)
        messages.append({"role": "user", "content": f"Result:\n{result}"})

    return f"stopped after {max_iters} iterations without a final answer"


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
        return f"unknown action '{name}'. Valid: {', '.join(TOOLS)}, final"
    try:
        return tool(ctx, args)
    except (ToolError, PathJailError) as exc:
        return f"error: {exc}"
    except KeyError as exc:
        return f"error: missing argument {exc}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_runner.py -v`
Expected: 6 passed. Note the retry accounting: `test_gives_up_after_two_bad_parses` queues 3 bad replies (initial + 2 retries all fail) and expects the give-up message; if it instead raises `IndexError` from an exhausted queue, the retry count in the code and the number of queued replies in the test disagree — reconcile by counting: `run_agent` calls `chat` once, then `_parse_with_retries` calls `chat` up to `MAX_PARSE_RETRIES` more times, so 3 total bad replies must be queued.

- [ ] **Step 5: Commit**

```bash
git add assistant/agent/runner.py assistant/tests/test_runner.py
git commit -m "feat: add agent loop runner with parse-retry and iteration cap"
```

---

### Task 6: CLI `agent` command

Wire the runner to the CLI, using `typer.confirm` for the interactive approval of writes and commands.

**Files:**
- Modify: `assistant/cli.py`
- Test: `assistant/tests/test_cli_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
from typer.testing import CliRunner

from assistant.cli import app

runner = CliRunner()


def test_agent_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output


def test_agent_without_index_exits_nonzero(tmp_path):
    result = runner.invoke(
        app, ["agent", "do something", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "index" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_cli_agent.py -v`
Expected: FAIL — the `agent` command does not exist yet (help output lacks it; second test errors).

- [ ] **Step 3: Add the `agent` command to `assistant/cli.py`**

Add these imports near the top, after the existing `from assistant.llm...` line:

```python
from assistant.agent.runner import run_agent
from assistant.agent.tools import ToolContext
```

Then append this command before the `build_prompt` function definition:

```python
@app.command()
def agent(
    task: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
):
    """Run the coding agent: plan, call tools, and act on the repo."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=client.embed,
        confirm=lambda msg: typer.confirm(msg),
    )
    try:
        answer = run_agent(task, ctx, client)
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo("--- answer ---")
    typer.echo(answer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_cli_agent.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (retrieval-core suite + all new agent tests).

- [ ] **Step 6: Commit**

```bash
git add assistant/cli.py assistant/tests/test_cli_agent.py
git commit -m "feat: add agent command to CLI"
```

---

### Task 7: Real end-to-end agent run

Verify the agent actually works against a live model and a real repo. This is manual verification, not a unit test.

**Prerequisite:** a repo has been indexed (from the retrieval-core plan, `crystal_bot` is already indexed). If not, index one first. Do NOT copy any external project into `system_llm` — index it in place. If no indexed repo is available, ask the user which repo to target rather than picking one.

- [ ] **Step 1: Read-only agent task (safe, no confirmations expected)**

```bash
.venv/bin/python -m assistant.cli agent \
  "Find where the database tables are created and summarize what tables exist" \
  --repo /home/eaduinte/Desktop/crystal_bot
```

Expected: the agent issues `search_code` and/or `read_file` steps, then a `final` answer naming the tables. On CPU this is slow (each turn is a full model call — minutes per run is normal). No write/run confirmations should appear for a read-only task.

- [ ] **Step 2: Write task in a scratch area (exercises the confirm gate)**

```bash
.venv/bin/python -m assistant.cli agent \
  "Create a file notes/agent_test.txt containing the single line: hello from the agent" \
  --repo /home/eaduinte/Desktop/crystal_bot
```

Expected: a `write ... bytes to notes/agent_test.txt?` confirmation prompt appears. Answer `n` to decline first and confirm the file is NOT created; re-run and answer `y` to confirm it IS created. This proves the confirm gate works in both directions. Afterwards, remove the test file:

```bash
rm -f /home/eaduinte/Desktop/crystal_bot/notes/agent_test.txt
```

- [ ] **Step 2b: Confirm no stray writes landed in the target repo**

```bash
cd /home/eaduinte/Desktop/crystal_bot && git status --short 2>/dev/null || \
  ls -la /home/eaduinte/Desktop/crystal_bot/notes 2>/dev/null
```

Expected: the scratch file is gone; no unexpected modifications to the target repo.

- [ ] **Step 3: Update the assistant README**

In `assistant/README.md`, replace the "Agent loop (next)" section with a short "Agent" usage section:

```markdown
## Agent

    .venv/bin/python -m assistant.cli agent "task" --repo <repo-path>

The agent plans one step at a time, emitting a JSON tool call
(`read_file` / `write_file` / `run_cmd` / `search_code`) that we parse and
execute, feeding the result back until it returns a final answer. All file
access is jailed to the target repo root; writes and commands require
interactive confirmation. Loop is capped at 10 iterations.
```

- [ ] **Step 4: Commit the docs update**

```bash
git add assistant/README.md
git commit -m "docs: document the agent command"
```

---

## After this plan

Phase 4, as a future separate plan: cross-encoder reranker over the fused
top-20 (measure hit@5 gain), multi-step planner (decompose one task into an
ordered action list before executing), and an auto-fix loop (run tests →
read failure → edit → re-run). All build on the runner and tools here.
