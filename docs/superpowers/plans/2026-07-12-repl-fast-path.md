# REPL Fast Path + History Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plain questions in the `joa` REPL get one direct streaming chat call (tokens appear immediately) with `ESCALATE` falling back to the agent loop; the agent session's history is capped so long sessions stop slowing down turn by turn.

**Architecture:** A new `_fast_answer(session, line, echo_token)` helper in `assistant/cli.py` tries a single `session.client.chat_stream(...)` call with a light routing system prompt, sniffing the first ~8 chars for `ESCALATE` before streaming; `_repl_loop` calls it first and only falls back to `session.send()` on escalation. `AgentSession.send()` (in `assistant/agent/runner.py`) trims `self.messages` to `config.MAX_HISTORY_MESSAGES` at entry, always preserving the system prompt at index 0.

**Tech Stack:** Python, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-12-repl-fast-path-design.md`

---

### Task 1: History cap in `AgentSession.send()`

**Files:**
- Modify: `assistant/config.py`
- Modify: `assistant/agent/runner.py`
- Modify: `assistant/tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/tests/test_runner.py` (note: this test uses
`AgentSession` directly, so extend the existing import at the top of the
file from `from assistant.agent.runner import run_agent` to
`from assistant.agent.runner import AgentSession, run_agent`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_runner.py::test_send_caps_history_keeping_system_prompt -v`
Expected: FAIL — first with `AttributeError: module 'assistant.config' has no attribute 'MAX_HISTORY_MESSAGES'`

- [ ] **Step 3: Add the config constant**

In `assistant/config.py`, extend the `# --- Agent ---` section:

```python
# --- Agent ---
RUN_CMD_TIMEOUT = 120  # seconds; generous enough for git push on slow links
MAX_HISTORY_MESSAGES = 40  # cap session history (~20 exchanges) so CPU
                           # prompt-eval stays bounded in long REPL sessions
```

- [ ] **Step 4: Add the cap to `AgentSession.send()`**

In `assistant/agent/runner.py`, add `from assistant import config` to the
imports at the top of the file, then insert at the very start of `send()`
(before `self.messages.append(...)`):

```python
    def send(self, task: str) -> str:
        if len(self.messages) > config.MAX_HISTORY_MESSAGES:
            self.messages = [self.messages[0]] + \
                self.messages[-(config.MAX_HISTORY_MESSAGES - 1):]
        self.messages.append({"role": "user", "content": f"Task: {task}"})
        plan: list[str] = []
```

(The rest of `send()` is unchanged.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest assistant/tests/test_runner.py -v`
Expected: all PASS (existing runner tests unaffected — they never exceed
the cap).

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: 141 passed (140 baseline + 1 new).

- [ ] **Step 7: Commit**

```bash
git add assistant/config.py assistant/agent/runner.py assistant/tests/test_runner.py
git commit -m "feat: cap agent session history to keep long REPL sessions fast"
```

---

### Task 2: `_fast_answer` helper

**Files:**
- Modify: `assistant/cli.py`
- Create: `assistant/tests/test_fast_path.py`

- [ ] **Step 1: Write the failing tests**

Create `assistant/tests/test_fast_path.py`:

```python
from assistant.cli import _fast_answer


class FakeStreamClient:
    """chat_stream yields the given chunks; records the messages sent."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.calls = []

    def chat_stream(self, messages):
        self.calls.append(messages)
        yield from self._chunks


class FakeSession:
    def __init__(self, client, messages=None):
        self.client = client
        self.messages = messages or [
            {"role": "system", "content": "agent system prompt"}]


def test_plain_answer_streams_and_lands_in_history():
    client = FakeStreamClient(["The answer", " is 4."])
    session = FakeSession(client)
    tokens = []

    answer = _fast_answer(session, "what is 2+2?", tokens.append)

    assert answer == "The answer is 4."
    assert "".join(tokens) == "The answer is 4."
    assert session.messages[-2] == {
        "role": "user", "content": "what is 2+2?"}
    assert session.messages[-1] == {
        "role": "assistant", "content": "The answer is 4."}


def test_fast_prompt_replaces_agent_system_prompt():
    client = FakeStreamClient(["hi"])
    session = FakeSession(client, messages=[
        {"role": "system", "content": "agent system prompt"},
        {"role": "user", "content": "earlier turn"},
    ])

    _fast_answer(session, "hello", lambda _t: None)

    sent = client.calls[0]
    assert sent[0]["role"] == "system"
    assert "agent system prompt" not in sent[0]["content"]
    assert "ESCALATE" in sent[0]["content"]
    assert {"role": "user", "content": "earlier turn"} in sent
    assert sent[-1] == {"role": "user", "content": "hello"}


def test_escalate_returns_none_and_appends_nothing():
    client = FakeStreamClient(["ESCALATE"])
    session = FakeSession(client)
    tokens = []

    assert _fast_answer(session, "fix the bug", tokens.append) is None
    assert tokens == []
    assert len(session.messages) == 1


def test_escalate_split_across_chunks_and_lowercase():
    client = FakeStreamClient(["esc", "alate"])
    session = FakeSession(client)
    tokens = []

    assert _fast_answer(session, "fix it", tokens.append) is None
    assert tokens == []


def test_escalate_with_trailing_text_still_escalates():
    client = FakeStreamClient(["ESCALATE — this needs tools"])
    session = FakeSession(client)

    assert _fast_answer(session, "edit file", lambda _t: None) is None


def test_short_answer_smaller_than_sniff_buffer():
    client = FakeStreamClient(["4"])
    session = FakeSession(client)
    tokens = []

    answer = _fast_answer(session, "2+2?", tokens.append)

    assert answer == "4"
    assert "".join(tokens) == "4"


def test_empty_stream_returns_none():
    client = FakeStreamClient([])
    session = FakeSession(client)

    assert _fast_answer(session, "anything", lambda _t: None) is None
    assert len(session.messages) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_fast_path.py -v`
Expected: FAIL with `ImportError: cannot import name '_fast_answer'`

- [ ] **Step 3: Implement `_fast_answer` in `assistant/cli.py`**

Add after the `SYSTEM_PROMPT = (...)` block (near the top of the file):

```python
FAST_SYSTEM_PROMPT = (
    "You are a coding assistant chatting with a user inside their "
    "repository. If answering would require reading or writing files, "
    "running commands, or searching the codebase, reply with exactly "
    "ESCALATE and nothing else. Otherwise answer the question directly "
    "and concisely."
)

_SNIFF_LEN = len("ESCALATE")


def _fast_answer(session, line, echo_token):
    """Try answering `line` with one direct streaming chat call.

    Returns the full streamed answer, or None if the model escalated (or
    produced nothing) — in which case the caller should run the agent
    loop. On success the exchange is appended to session.messages so the
    agent keeps conversational context."""
    messages = (
        [{"role": "system", "content": FAST_SYSTEM_PROMPT}]
        + session.messages[1:]
        + [{"role": "user", "content": line}]
    )
    stream = session.client.chat_stream(messages)
    buffer = ""
    for chunk in stream:
        buffer += chunk
        if len(buffer) >= _SNIFF_LEN:
            break
    if buffer.strip().upper().startswith("ESCALATE"):
        return None
    if not buffer.strip():
        return None
    echo_token(buffer)
    parts = [buffer]
    for chunk in stream:
        echo_token(chunk)
        parts.append(chunk)
    answer = "".join(parts)
    session.messages.append({"role": "user", "content": line})
    session.messages.append({"role": "assistant", "content": answer})
    return answer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_fast_path.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/cli.py assistant/tests/test_fast_path.py
git commit -m "feat: add _fast_answer streaming fast path with ESCALATE routing"
```

---

### Task 3: Wire the fast path into `_repl_loop`

**Files:**
- Modify: `assistant/cli.py`
- Modify: `assistant/tests/test_repl.py`

- [ ] **Step 1: Update existing `_repl_loop` call sites for the new parameter**

`_repl_loop` gains a 5th parameter, `echo_token` (token-level printer for
streamed fast-path output). Existing tests don't exercise streaming, but
the pre-existing "answer" flow tests now go through `_fast_answer` first
— their `FakeSession` has no `client` attribute, so give them one that
always escalates. In `assistant/tests/test_repl.py`:

Add right after the existing `FakeEmbedClient` class:

```python
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
```

Then update the `FakeSession` class to carry a client (escalating by
default so every existing test keeps exercising the agent path):

```python
class FakeSession:
    def __init__(self, answers):
        self._answers = list(answers)
        self.sent = []
        self.client = AlwaysEscalateClient()
        self.messages = [{"role": "system", "content": "agent prompt"}]

    def send(self, task):
        self.sent.append(task)
        return self._answers.pop(0)
```

The two `BoomSession` classes (in
`test_repl_loop_survives_ollama_error`, `test_repl_loop_gemini_error_suggests_joamodel`,
and `test_repl_loop_ollama_error_does_not_suggest_joamodel`) also need the
same two attributes added to their `__init__`:

```python
        def __init__(self):
            self.sent = []
            self.client = AlwaysEscalateClient()
            self.messages = [{"role": "system", "content": "agent prompt"}]
```

Every `_repl_loop(...)` call in the file gains a 5th argument. There are
multiple call shapes; update each by appending `, lambda _t: None`:

- `_repl_loop(session, lambda: next(lines), out.append, None)` →
  `_repl_loop(session, lambda: next(lines), out.append, None, lambda _t: None)`
- `_repl_loop(session, lambda: next(lines), lambda _o: None, None)` →
  `_repl_loop(session, lambda: next(lines), lambda _o: None, None, lambda _t: None)`
- `_repl_loop(session, read_line, lambda _o: None, None)` →
  `_repl_loop(session, read_line, lambda _o: None, None, lambda _t: None)`
- `_repl_loop(session, lambda: next(lines), out.append, embed_client)` →
  `_repl_loop(session, lambda: next(lines), out.append, embed_client, lambda _t: None)`
- `_repl_loop(session, lambda: next(lines), lambda _o: None, embed_client)` →
  `_repl_loop(session, lambda: next(lines), lambda _o: None, embed_client, lambda _t: None)`
- `_repl_loop(session, read_line, lambda _o: None, embed_client)` →
  `_repl_loop(session, read_line, lambda _o: None, embed_client, lambda _t: None)`
- `_repl_loop(session, lambda: next(lines), out.append, BoomEmbedClient())` →
  `_repl_loop(session, lambda: next(lines), out.append, BoomEmbedClient(), lambda _t: None)`

(Use grep to find every call — don't trust this list to be exhaustive:
`grep -n "_repl_loop(" assistant/tests/test_repl.py`.)

- [ ] **Step 2: Add the new fast-path REPL tests**

Append to `assistant/tests/test_repl.py`:

```python
def test_fast_path_answer_skips_agent_and_shows_timing():
    session = FakeSession([])
    session.client = FakeStreamClient(["quick ", "answer"])
    lines = iter(["what is 2+2?", "exit"])
    out = []
    tokens = []
    _repl_loop(session, lambda: next(lines), out.append, None, tokens.append)
    assert session.sent == []  # agent path never ran
    assert "".join(tokens) == "quick answer"
    assert any("s)" in o for o in out)  # timing suffix echoed


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
```

- [ ] **Step 3: Run to verify the new tests fail**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: FAIL — `_repl_loop() takes 4 positional arguments but 5 were
given` everywhere (plus the 3 new tests failing).

- [ ] **Step 4: Update `_repl_loop` in `assistant/cli.py`**

Replace the whole `_repl_loop` function:

```python
def _repl_loop(session, read_line, echo, embed_client, echo_token) -> None:
    """Drive an AgentSession from a line source until exit/EOF.

    `read_line()` returns the next input line (raising EOFError at end of
    input); `echo(text)` prints a line; `echo_token(text)` prints a
    streamed fragment without a newline (used by the fast path).
    `embed_client` is an OllamaClient used only for `/joamodel`'s model
    listing. Each input line first tries `_fast_answer` (one direct
    streaming chat call); the agent loop only runs when the model
    escalates. Kept separate from the CLI command so the loop is testable
    without a live model.
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
        if stripped == "/joamodel":
            _handle_joamodel(session, embed_client, read_line, echo)
            continue
        start = time.perf_counter()
        try:
            answer = _fast_answer(session, stripped, echo_token)
            if answer is None:
                answer = session.send(stripped)
                elapsed = time.perf_counter() - start
                echo(f"{answer}\n({elapsed:.1f}s)")
            else:
                elapsed = time.perf_counter() - start
                echo(f"\n({elapsed:.1f}s)")
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            if isinstance(exc, GeminiError):
                echo("/joamodel bilan Ollama modeliga qayting.")
            continue
```

- [ ] **Step 5: Update the real `repl()` call site**

In `assistant/cli.py`'s `repl()` command, replace:

```python
    _repl_loop(session, lambda: input("joa> "), typer.echo, embed_client)
```

with:

```python
    _repl_loop(session, lambda: input("joa> "), typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False))
```

- [ ] **Step 6: Run the REPL and fast-path test files**

Run: `pytest assistant/tests/test_repl.py assistant/tests/test_fast_path.py -v`
Expected: all PASS (17 pre-existing repl tests updated + 3 new + 7 fast-path).

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: 151 passed (141 after Task 1 + 7 from Task 2 + 3 new).

- [ ] **Step 8: Commit**

```bash
git add assistant/cli.py assistant/tests/test_repl.py
git commit -m "feat: try streaming fast path before agent loop in REPL"
```

---

### Task 4: Manual smoke test + README

**Files:**
- Modify: `README.md`
- Modify: `assistant/README.md`

- [ ] **Step 1: Manual smoke test — fast path**

With Ollama running and an indexed repo (e.g. `assistant/llm` indexed
earlier — check `ls assistant/.data/`):

```bash
printf 'what is 2+2?\nexit\n' | timeout 60 python3 -m assistant.cli repl --repo assistant/llm
```

Expected: the answer streams out (visible as plain text before the
`(N.Ns)` timing line), noticeably faster than the old agent path, and no
`write file?`/`run command?` confirmation prompts appear.

- [ ] **Step 2: Manual smoke test — escalation**

```bash
printf 'create a file named hello.txt containing hi\nexit\n' | timeout 120 python3 -m assistant.cli repl --repo assistant/llm
```

Expected: a brief pause (the ESCALATE call), then the normal agent path
(a `write 2 bytes to hello.txt? [y/N]:` confirmation appears — piped
stdin answers `n`/EOF, which is fine; the point is the agent path was
reached). Clean up any created file if one appears.

- [ ] **Step 3: Update `README.md`**

In the `## Ishlatish` section, after the `/joamodel` block, add:

```markdown
Oddiy savollar (masalan "bu funksiya nima qiladi?") endi agent
protokolisiz, bitta streaming chaqiruv bilan javob oladi — javob token
oqib chiqadi. Fayl/buyruq talab qiladigan topshiriqlar avvalgidek to'liq
agent orqali bajariladi (model o'zi ajratadi).
```

- [ ] **Step 4: Update `assistant/README.md`**

In the `## Interactive session (joa)` section, after the `/joamodel`
paragraph, add:

```markdown
Plain questions take a fast path: one direct streaming chat call (tokens
render as they arrive) instead of the full agent protocol. The model
routes automatically — if the request needs file/command/search tools it
replies `ESCALATE` internally and the normal agent loop takes over.
Session history is capped (`MAX_HISTORY_MESSAGES` in `config.py`) so long
sessions don't slow down over time.
```

- [ ] **Step 5: Run the full suite one more time**

Run: `pytest -q`
Expected: 151 passed.

- [ ] **Step 6: Commit**

```bash
git add README.md assistant/README.md
git commit -m "docs: document REPL fast path and history cap"
```

---

## Post-implementation

Real-model verification with the user: compare `2+2` timing on
qwen2.5-coder:1.5b before/after (expect several-fold improvement), and
confirm a real task still escalates into the confirm-gated agent path.
