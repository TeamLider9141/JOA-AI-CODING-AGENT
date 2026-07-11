# `/joamodel` REPL Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user switch the active chat model — to another installed Ollama model, or to Gemini — from inside a running `joa` REPL session by typing `/joamodel`, without restarting or editing config.

**Architecture:** `OllamaClient` gains a runtime-overridable `model` and a `list_models()` method (via `GET /api/tags`). `assistant/cli.py`'s `_repl_loop` gains a 4th parameter (`embed_client`, reused purely for `.list_models()`) and a new `_handle_joamodel` helper that lists installed models + `"gemini"`, reads a number, and swaps `session.client` accordingly — leaving it unchanged on any failure (bad input, EOF, missing Gemini key, Ollama unreachable).

**Tech Stack:** Python, httpx, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-11-joamodel-repl-command-design.md`

---

### Task 1: `OllamaClient` — runtime-overridable model

**Files:**
- Modify: `assistant/llm/ollama_client.py`
- Modify: `assistant/tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/tests/test_ollama_client.py`:

```python
def test_chat_uses_overridden_model():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5-coder:1.5b"
        return httpx.Response(200, json={"message": {"content": "hi"}})

    client = OllamaClient(base_url="http://test", model="qwen2.5-coder:1.5b",
                          transport=httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "hi"}]) == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_ollama_client.py -v`
Expected: FAIL with `TypeError: OllamaClient.__init__() got an unexpected keyword argument 'model'`

- [ ] **Step 3: Add the `model` parameter and use it in `chat`/`chat_stream`**

In `assistant/llm/ollama_client.py`, replace `__init__`:

```python
    def __init__(
        self,
        base_url: str = config.OLLAMA_URL,
        model: str = config.CHAT_MODEL,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            timeout=config.REQUEST_TIMEOUT,
            transport=transport,
        )
```

Replace `chat`:

```python
    def chat(self, messages: list[dict]) -> str:
        data = self._post("/api/chat", {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": config.NUM_CTX},
        })
        return data["message"]["content"]
```

Replace `chat_stream`'s payload construction and its 404-hint line:

```python
    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": config.NUM_CTX},
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    raise OllamaError(
                        f"Ollama returned {resp.status_code} for /api/chat."
                        f" Model missing? Try: ollama pull {self._model}"
                    )
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise OllamaError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
```

`embed()` and `_post()` are unchanged — embeddings stay on `config.EMBED_MODEL`
always.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_ollama_client.py -v`
Expected: PASS (5 tests — 4 existing + 1 new). The 4 pre-existing tests
still pass unmodified: they construct `OllamaClient` without a `model`
kwarg, so it defaults to `config.CHAT_MODEL`, same as before this change.

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/ollama_client.py assistant/tests/test_ollama_client.py
git commit -m "feat: make OllamaClient's model runtime-overridable"
```

---

### Task 2: `OllamaClient.list_models()`

**Files:**
- Modify: `assistant/llm/ollama_client.py`
- Modify: `assistant/tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `assistant/tests/test_ollama_client.py`:

```python
def test_list_models_returns_sorted_names():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "qwen2.5-coder:7b"},
            {"name": "qwen2.5-coder:1.5b"},
        ]})

    assert make_client(handler).list_models() == [
        "qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]


def test_list_models_connect_error_is_actionable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        make_client(handler).list_models()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_ollama_client.py -v`
Expected: FAIL with `AttributeError: 'OllamaClient' object has no attribute 'list_models'`

- [ ] **Step 3: Implement `list_models()`**

Add to `assistant/llm/ollama_client.py`, inside `OllamaClient` (right after
`embed`):

```python
    def list_models(self) -> list[str]:
        """Names of models currently pulled into this Ollama instance."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_ollama_client.py -v`
Expected: PASS (7 tests — 5 from Task 1 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/ollama_client.py assistant/tests/test_ollama_client.py
git commit -m "feat: add OllamaClient.list_models()"
```

---

### Task 3: `/joamodel` command — full `_handle_joamodel` + wiring

**Files:**
- Modify: `assistant/cli.py`
- Modify: `assistant/tests/test_repl.py`

- [ ] **Step 1: Update the 5 existing `_repl_loop(...)` call sites in the test file for the new 4th parameter**

`_repl_loop` is about to gain a required 4th parameter, `embed_client`.
None of the existing tests exercise `/joamodel`, so they can pass `None`.

In `assistant/tests/test_repl.py`, there are 3 calls with the exact text
`_repl_loop(session, lambda: next(lines), out.append)` (in
`test_repl_loop_sends_lines_and_exits_on_exit`,
`test_repl_loop_survives_ollama_error`, and
`test_repl_loop_echoes_elapsed_time_with_answer`). Use `replace_all` to
change all 3 at once, from:

```python
    _repl_loop(session, lambda: next(lines), out.append)
```

to:

```python
    _repl_loop(session, lambda: next(lines), out.append, None)
```

Then two more, each unique, need their own edit:

In `test_repl_loop_skips_blank_lines`, change:
```python
    _repl_loop(session, lambda: next(lines), lambda _o: None)
```
to:
```python
    _repl_loop(session, lambda: next(lines), lambda _o: None, None)
```

In `test_repl_loop_exits_on_eof`, change:
```python
    _repl_loop(session, read_line, lambda _o: None)
```
to:
```python
    _repl_loop(session, read_line, lambda _o: None, None)
```

- [ ] **Step 2: Run the test file to confirm it now fails for the right reason**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: FAIL — every test now errors with something like
`TypeError: _repl_loop() takes 3 positional arguments but 4 were given`
(confirms the test file changed correctly; `_repl_loop` itself hasn't
changed yet).

- [ ] **Step 3: Write the new `/joamodel` happy-path test**

Add near the top of `assistant/tests/test_repl.py`, after the existing
`FakeSession` class:

```python
class FakeEmbedClient:
    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models
```

Then append this test:

```python
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
```

Add `from assistant.llm.ollama_client import OllamaClient` to the imports
at the top of `assistant/tests/test_repl.py` (currently it only imports
`OllamaError` from that module).

- [ ] **Step 4: Run to verify it fails**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: still FAIL — `_repl_loop` doesn't accept a 4th argument yet and
`/joamodel` isn't handled.

- [ ] **Step 5: Implement `_handle_joamodel` and wire it into `_repl_loop`**

In `assistant/cli.py`, replace the `_repl_loop` function:

```python
def _repl_loop(session, read_line, echo, embed_client) -> None:
    """Drive an AgentSession from a line source until exit/EOF.

    `read_line()` returns the next input line (raising EOFError at end of
    input); `echo(text)` prints a line. `embed_client` is an OllamaClient
    used only for `/joamodel`'s model listing (embeddings always stay on
    Ollama regardless of which chat backend is active). Kept separate from
    the CLI command so the loop is testable without a live model.
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
            answer = session.send(stripped)
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            continue
        elapsed = time.perf_counter() - start
        echo(f"{answer}\n({elapsed:.1f}s)")
```

Add `_handle_joamodel` right before `_repl_loop`:

```python
def _handle_joamodel(session, embed_client, read_line, echo) -> None:
    """List installed Ollama models plus "gemini"; switch session.client
    to whichever the user picks by number. Leaves session.client
    unchanged on any failure (bad input, EOF, missing Gemini key, or a
    failure listing Ollama's models)."""
    try:
        models = embed_client.list_models()
    except OllamaError as exc:
        echo(str(exc))
        return
    options = models + ["gemini"]
    for i, name in enumerate(options, start=1):
        echo(f"{i}. {name}")
    echo("Raqamni tanlang:")
    try:
        choice_line = read_line()
    except EOFError:
        return
    choice = choice_line.strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
        echo(f"Noto'g'ri tanlov: {choice!r}")
        return
    selected = options[int(choice) - 1]
    if selected == "gemini":
        if not config.GEMINI_API_KEY:
            echo("GEMINI_API_KEY .env'da topilmadi. Model o'zgartirilmadi.")
            return
        try:
            session.client = GeminiClient()
        except GeminiError as exc:
            echo(str(exc))
            return
    else:
        session.client = OllamaClient(model=selected)
    echo(f"✓ Model: {selected}")
```

- [ ] **Step 6: Run to verify all pass**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: PASS (8 tests — 7 existing + 1 new)

- [ ] **Step 7: Commit**

```bash
git add assistant/cli.py assistant/tests/test_repl.py
git commit -m "feat: add /joamodel REPL command (Ollama model selection)"
```

---

### Task 4: `/joamodel` — Gemini selection coverage

**Files:**
- Modify: `assistant/tests/test_repl.py`

(`_handle_joamodel`'s Gemini branch already exists from Task 3 — this
task only adds test coverage for it.)

- [ ] **Step 1: Write the tests**

Add `from assistant.llm.gemini_client import GeminiClient, GeminiError` to
the imports at the top of `assistant/tests/test_repl.py`. Append:

```python
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
```

- [ ] **Step 2: Run and confirm all pass**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: PASS (10 tests — 8 from Task 3 + 2 new). If either fails, the
`_handle_joamodel` Gemini branch from Task 3 has a bug — fix it, don't
weaken the test.

- [ ] **Step 3: Commit**

```bash
git add assistant/tests/test_repl.py
git commit -m "test: cover /joamodel Gemini selection"
```

---

### Task 5: `/joamodel` — edge-case coverage

**Files:**
- Modify: `assistant/tests/test_repl.py`

(No implementation changes — `_handle_joamodel` already handles these
cases from Task 3.)

- [ ] **Step 1: Write the tests**

Append to `assistant/tests/test_repl.py`:

```python
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
```

- [ ] **Step 2: Run and confirm all pass**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: PASS (14 tests — 10 from Task 4 + 4 new). If any fails, fix the
`_handle_joamodel` bug it reveals — don't weaken the test.

- [ ] **Step 3: Commit**

```bash
git add assistant/tests/test_repl.py
git commit -m "test: cover /joamodel edge cases (bad input, EOF, Ollama down)"
```

---

### Task 6: `GeminiError` hint line on turn errors

**Files:**
- Modify: `assistant/cli.py`
- Modify: `assistant/tests/test_repl.py`

- [ ] **Step 1: Write the failing tests**

Append to `assistant/tests/test_repl.py`:

```python
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
```

- [ ] **Step 2: Run test to verify the first one fails**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: `test_repl_loop_gemini_error_suggests_joamodel` FAILS (no hint
line yet); `test_repl_loop_ollama_error_does_not_suggest_joamodel` already
PASSES (there's nothing to suggest it yet either way — that's fine, it's
asserting an absence).

- [ ] **Step 3: Add the hint line**

In `assistant/cli.py`, inside `_repl_loop`, replace:

```python
        try:
            answer = session.send(stripped)
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            continue
```

with:

```python
        try:
            answer = session.send(stripped)
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            if isinstance(exc, GeminiError):
                echo("/joamodel bilan Ollama modeliga qayting.")
            continue
```

- [ ] **Step 4: Run test to verify both pass**

Run: `pytest assistant/tests/test_repl.py -v`
Expected: PASS (16 tests — 14 from Task 5 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/cli.py assistant/tests/test_repl.py
git commit -m "feat: hint /joamodel when a Gemini turn error occurs"
```

---

### Task 7: Wire `embed_client` into the real `repl()` command

**Files:**
- Modify: `assistant/cli.py`

- [ ] **Step 1: Update the real call site**

In `assistant/cli.py`, inside the `repl()` command, replace:

```python
    session = AgentSession(ctx, chat_client)
    _repl_loop(session, lambda: input("joa> "), typer.echo)
```

with:

```python
    session = AgentSession(ctx, chat_client)
    _repl_loop(session, lambda: input("joa> "), typer.echo, embed_client)
```

(`embed_client` already exists earlier in `repl()` — this just passes the
already-constructed instance through.)

- [ ] **Step 2: Run the full test suite**

Run: `pytest -q`
Expected: 133 passed (121 baseline before this plan + 1 new test in Task 1
+ 2 in Task 2 + 1 in Task 3 + 2 in Task 4 + 4 in Task 5 + 2 in Task 6 = 133;
Task 7 adds no new tests, just wiring).

- [ ] **Step 3: Manual smoke test**

With an already-indexed repo and Ollama running with at least 2 models
pulled (e.g. `qwen2.5-coder:1.5b` and `qwen2.5-coder:7b`):

```bash
python3 -m assistant.cli repl --repo .
```

At the `joa>` prompt type `/joamodel`, confirm the printed list matches
`ollama list`'s installed models plus a trailing `gemini` entry, pick a
number for a different Ollama model, confirm the `✓ Model: ...` line
matches your pick, then ask a question and confirm the reply comes back
(proving the swapped client actually works, not just that the field
changed). Then run `/joamodel` again and pick `gemini` — if `.env` has
`GEMINI_API_KEY` set, confirm it switches and a question gets answered via
Gemini; if not, confirm the `GEMINI_API_KEY .env'da topilmadi` warning
appears and the previous model keeps answering.

- [ ] **Step 4: Commit**

```bash
git add assistant/cli.py
git commit -m "feat: wire embed_client into repl() for /joamodel"
```

---

## Post-implementation

Update `README.md` and `assistant/README.md`'s REPL sections to mention
`/joamodel`, following the same pattern used for documenting `--backend`
earlier in this project.
