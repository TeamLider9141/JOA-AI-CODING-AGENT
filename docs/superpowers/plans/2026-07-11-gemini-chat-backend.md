# Gemini Chat Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Google Gemini as a second, opt-in chat backend (`--backend gemini` on `ask`/`agent`/`repl`), while Ollama stays the default and embeddings always stay on Ollama.

**Architecture:** A new `GeminiClient` in `assistant/llm/gemini_client.py`, duck-type compatible with `OllamaClient`'s `chat`/`chat_stream` methods, talking to the Gemini REST API directly via `httpx` (no SDK). `assistant/cli.py` gains a `--backend` flag that picks which client to construct for chat calls, while embedding always goes through a separate `OllamaClient` instance.

**Tech Stack:** Python, `httpx` (already a dependency), `typer`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-11-gemini-chat-backend-design.md`

---

### Task 1: `.env` loader + Gemini config constants

**Files:**
- Modify: `assistant/config.py`
- Create: `assistant/tests/test_config_dotenv.py`
- Modify: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Write the failing test for the `.env` loader**

```python
# assistant/tests/test_config_dotenv.py
import os

from assistant.config import _load_dotenv


def test_load_dotenv_sets_unset_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_TEST_VAR=hello\n# a comment\n\nOTHER=world\n")

    _load_dotenv(env_file)

    assert os.environ["SOME_TEST_VAR"] == "hello"
    assert os.environ["OTHER"] == "world"


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR", "real-value")
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_TEST_VAR=from-dotenv\n")

    _load_dotenv(env_file)

    assert os.environ["SOME_TEST_VAR"] == "real-value"


def test_load_dotenv_missing_file_is_a_noop(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_config_dotenv.py -v`
Expected: FAIL with `ImportError: cannot import name '_load_dotenv'`

- [ ] **Step 3: Add the loader and Gemini constants to `config.py`**

Insert at the very top of `assistant/config.py` (before the existing `from pathlib import Path` line, replace the whole file header through `EMBED_MODEL`):

```python
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ (never overriding
    a variable that's already set in the real environment)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent.parent / ".env")

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 4096            # CPU-only: keep modest, tune later
REQUEST_TIMEOUT = 300.0   # seconds; CPU inference is slow

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com"
```

Leave the rest of `config.py` (`# --- Retrieval ---` onward) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_config_dotenv.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Ensure `.env` is never committed**

Add to `.gitignore`:

```
.env
```

- [ ] **Step 6: Add a `.env.example` template**

```
# Copy to .env and fill in your key to use --backend gemini.
# Get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY=
```

- [ ] **Step 7: Run the full suite to confirm nothing else broke**

Run: `pytest -q`
Expected: 99 passed (96 existing + 3 new)

- [ ] **Step 8: Commit**

```bash
git add assistant/config.py assistant/tests/test_config_dotenv.py .gitignore .env.example
git commit -m "feat: add .env loader and Gemini config constants"
```

---

### Task 2: `GeminiClient` — message translation helper

**Files:**
- Create: `assistant/llm/gemini_client.py`
- Create: `assistant/tests/test_gemini_client.py`

- [ ] **Step 1: Write the failing test for message translation**

```python
# assistant/tests/test_gemini_client.py
from assistant.llm.gemini_client import _to_gemini_contents


def test_translates_user_and_assistant_roles():
    contents, system_instruction = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]
    assert system_instruction is None


def test_folds_system_messages_into_system_instruction():
    contents, system_instruction = _to_gemini_contents([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ])
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert system_instruction == {"parts": [{"text": "You are helpful."}]}


def test_joins_multiple_system_messages():
    contents, system_instruction = _to_gemini_contents([
        {"role": "system", "content": "First."},
        {"role": "system", "content": "Second."},
        {"role": "user", "content": "hi"},
    ])
    assert system_instruction == {"parts": [{"text": "First.\n\nSecond."}]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.llm.gemini_client'`

- [ ] **Step 3: Create `assistant/llm/gemini_client.py` with the translation helper**

```python
from assistant import config


def _to_gemini_contents(
    messages: list[dict],
) -> tuple[list[dict], dict | None]:
    """Translate OllamaClient-shaped messages (role/content) into Gemini's
    request shape. `system` messages are folded into a single
    systemInstruction rather than sent as a contents turn; `assistant`
    becomes `model` (Gemini's name for the model turn)."""
    contents = []
    system_parts = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]}
        if system_parts else None
    )
    return contents, system_instruction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/gemini_client.py assistant/tests/test_gemini_client.py
git commit -m "feat: add Gemini message translation helper"
```

---

### Task 3: `GeminiClient` — missing API key fails fast

**Files:**
- Modify: `assistant/llm/gemini_client.py`
- Modify: `assistant/tests/test_gemini_client.py`

- [ ] **Step 1: Write the failing test**

```python
# append to assistant/tests/test_gemini_client.py
import pytest

from assistant.llm.gemini_client import GeminiClient, GeminiError


def test_missing_api_key_raises_without_request():
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: FAIL with `ImportError: cannot import name 'GeminiClient'`

- [ ] **Step 3: Add `GeminiError` and `GeminiClient.__init__` to `gemini_client.py`**

Add to `assistant/llm/gemini_client.py` (after the imports, before
`_to_gemini_contents`):

```python
import httpx

MISSING_KEY_MSG = (
    "GEMINI_API_KEY not set. Add it to a .env file in the repo root, e.g.\n"
    "GEMINI_API_KEY=your-key-here\n"
    "(get one at https://aistudio.google.com/apikey)"
)
UNREACHABLE_MSG = "Gemini API unreachable at {url}. Check your network connection."


class GeminiError(RuntimeError):
    """Gemini API key missing/invalid, rate-limited, unreachable, or errored."""
```

Add at the end of the file:

```python
class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.GEMINI_MODEL,
        base_url: str = config.GEMINI_URL,
        transport: httpx.BaseTransport | None = None,
    ):
        api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        if not api_key:
            raise GeminiError(MISSING_KEY_MSG)
        self._model = model
        self._base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            timeout=config.REQUEST_TIMEOUT,
            transport=transport,
            headers={"x-goog-api-key": api_key},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/gemini_client.py assistant/tests/test_gemini_client.py
git commit -m "feat: GeminiClient fails fast on missing API key"
```

---

### Task 4: `GeminiClient.chat()` — happy path

**Files:**
- Modify: `assistant/llm/gemini_client.py`
- Modify: `assistant/tests/test_gemini_client.py`

- [ ] **Step 1: Write the failing test**

```python
# append to assistant/tests/test_gemini_client.py
import json


def make_client(handler) -> GeminiClient:
    return GeminiClient(
        api_key="test-key",
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )


def test_chat_posts_translated_contents_and_returns_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-3-flash:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]}
        ]
        return httpx.Response(200, json={
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "hello"}]}}
            ]
        })

    out = make_client(handler).chat([{"role": "user", "content": "hi"}])
    assert out == "hello"


def test_chat_sends_system_instruction_separately():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]}
        ]
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        })

    make_client(handler).chat([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ])
```

Add `import httpx` near the top of the test file (alongside the existing
imports) if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: FAIL with `AttributeError: 'GeminiClient' object has no attribute 'chat'`

- [ ] **Step 3: Implement `chat()` and the response-parsing helper**

Add to `assistant/llm/gemini_client.py`, inside `GeminiClient` (after
`__init__`):

```python
    def chat(self, messages: list[dict]) -> str:
        contents, system_instruction = _to_gemini_contents(messages)
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        data = self._post(
            f"/v1beta/models/{self._model}:generateContent", payload)
        return _extract_text(data)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise GeminiError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
        if resp.status_code >= 400:
            raise _http_error(resp)
        return resp.json()
```

Add module-level helpers at the end of the file:

```python
def _http_error(resp: httpx.Response) -> GeminiError:
    if resp.status_code == 429:
        return GeminiError(
            "Gemini rate limit hit (429). Try --backend ollama or wait "
            "and retry."
        )
    if resp.status_code in (400, 401, 403):
        return GeminiError(
            f"Gemini rejected the request ({resp.status_code}): "
            f"{resp.text}. Check GEMINI_API_KEY."
        )
    return GeminiError(f"Gemini API returned {resp.status_code}: {resp.text}")


def _extract_text(data: dict, allow_empty: bool = False) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        if allow_empty:
            return ""
        raise GeminiError(f"Gemini response had no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/gemini_client.py assistant/tests/test_gemini_client.py
git commit -m "feat: implement GeminiClient.chat()"
```

---

### Task 5: `GeminiClient` — HTTP error mapping

**Files:**
- Modify: `assistant/tests/test_gemini_client.py`

(`_http_error` and the `ConnectError` branch already exist from Task 4 —
this task only adds coverage.)

- [ ] **Step 1: Write the failing/uncovered tests**

```python
# append to assistant/tests/test_gemini_client.py
def test_401_raises_key_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error": "bad auth"}')

    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_403_raises_key_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text='{"error": "permission denied"}')

    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_429_raises_rate_limit_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": "quota exceeded"}')

    with pytest.raises(GeminiError, match="rate limit"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_connect_error_becomes_actionable_gemini_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(GeminiError, match="unreachable"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_empty_candidates_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    with pytest.raises(GeminiError, match="no candidates"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run and confirm all pass (implementation already exists)**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: PASS (11 tests) — if any fail, the `_http_error`/`_post` code from
Task 4 has a bug; fix it before continuing.

- [ ] **Step 3: Commit**

```bash
git add assistant/tests/test_gemini_client.py
git commit -m "test: cover GeminiClient HTTP error mapping"
```

---

### Task 6: `GeminiClient.chat_stream()`

**Files:**
- Modify: `assistant/llm/gemini_client.py`
- Modify: `assistant/tests/test_gemini_client.py`

- [ ] **Step 1: Write the failing test**

```python
# append to assistant/tests/test_gemini_client.py
def test_chat_stream_concatenates_sse_chunks():
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n'
        "\n"
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == \
            "/v1beta/models/gemini-3-flash:streamGenerateContent"
        assert request.url.params["alt"] == "sse"
        return httpx.Response(200, text=body)

    out = "".join(make_client(handler).chat_stream(
        [{"role": "user", "content": "hi"}]))
    assert out == "Hello"


def test_chat_stream_skips_metadata_only_chunks():
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n'
        "\n"
        'data: {"usageMetadata": {"totalTokenCount": 5}}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    out = "".join(make_client(handler).chat_stream(
        [{"role": "user", "content": "hi"}]))
    assert out == "Hi"


def test_chat_stream_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": "quota exceeded"}')

    with pytest.raises(GeminiError, match="rate limit"):
        list(make_client(handler).chat_stream(
            [{"role": "user", "content": "hi"}]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: FAIL with `AttributeError: 'GeminiClient' object has no attribute 'chat_stream'`

- [ ] **Step 3: Implement `chat_stream()`**

Add `from collections.abc import Iterator` and `import json` to the top of
`assistant/llm/gemini_client.py` if not already imported, then add inside
`GeminiClient` (after `chat`):

```python
    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        contents, system_instruction = _to_gemini_contents(messages)
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        path = f"/v1beta/models/{self._model}:streamGenerateContent"
        try:
            with self._client.stream(
                "POST", path, json=payload, params={"alt": "sse"}
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise _http_error(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[len("data: "):])
                    text = _extract_text(chunk, allow_empty=True)
                    if text:
                        yield text
        except httpx.ConnectError as exc:
            raise GeminiError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_gemini_client.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 113 passed (99 from Task 1 + 14 new)

- [ ] **Step 6: Commit**

```bash
git add assistant/llm/gemini_client.py assistant/tests/test_gemini_client.py
git commit -m "feat: implement GeminiClient.chat_stream()"
```

---

### Task 7: CLI wiring — `--backend` flag on `ask`/`agent`/`repl`

**Files:**
- Modify: `assistant/cli.py`
- Create: `assistant/tests/test_cli_backend.py`

- [ ] **Step 1: Write the failing test for the client-selection helper**

```python
# assistant/tests/test_cli_backend.py
import pytest

from assistant.cli import _chat_client
from assistant.llm.gemini_client import GeminiClient
from assistant.llm.ollama_client import OllamaClient


def test_ollama_backend_returns_ollama_client():
    assert isinstance(_chat_client("ollama"), OllamaClient)


def test_gemini_backend_returns_gemini_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "assistant.cli.config.GEMINI_API_KEY", "test-key")
    assert isinstance(_chat_client("gemini"), GeminiClient)


def test_gemini_backend_without_key_raises(monkeypatch):
    monkeypatch.setattr("assistant.cli.config.GEMINI_API_KEY", None)
    from assistant.llm.gemini_client import GeminiError
    with pytest.raises(GeminiError):
        _chat_client("gemini")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest assistant/tests/test_cli_backend.py -v`
Expected: FAIL with `ImportError: cannot import name '_chat_client'`

- [ ] **Step 3: Add the import and helper to `cli.py`**

In `assistant/cli.py`, change the import block (currently lines 1-10) by
adding one import line after the existing `ollama_client` import:

```python
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError
```

Add this helper function right after the `app = typer.Typer(...)` line:

```python
def _chat_client(backend: str):
    if backend == "gemini":
        return GeminiClient()
    return OllamaClient()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest assistant/tests/test_cli_backend.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire `--backend` into the `ask` command**

Replace the `ask` command's signature and body in `assistant/cli.py`:

```python
@app.command()
def ask(
    question: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
):
    """Ask a question about the indexed repository."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
        results = search_index(question, data_dir, embed_client.embed)
        typer.echo("--- sources ---")
        for _chunk_id, _score, p in results:
            typer.echo(
                f"  {p['path']}:{p['start_line']}-{p['end_line']}"
                f"  {p['symbol']}")
        typer.echo("--- answer ---")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, results)},
        ]
        for token in chat_client.chat_stream(messages):
            typer.echo(token, nl=False)
        typer.echo()
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
```

- [ ] **Step 6: Wire `--backend` into the `agent` command**

Replace the `agent` command:

```python
@app.command()
def agent(
    task: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
):
    """Run the coding agent: plan, call tools, and act on the repo."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
        ctx = ToolContext(
            root=repo.resolve(),
            data_dir=data_dir,
            embedder=embed_client.embed,
            confirm=lambda msg: typer.confirm(msg),
        )
        answer = run_agent(task, ctx, chat_client)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo("--- answer ---")
    typer.echo(answer)
```

- [ ] **Step 7: Wire `--backend` into the `repl` command**

Replace the `repl` command:

```python
@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
):
    """Interactive agent session over the repo (defaults to current dir)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=embed_client.embed,
        confirm=lambda msg: typer.confirm(msg),
    )
    session = AgentSession(ctx, chat_client)
    _repl_loop(session, lambda: input("joa> "), typer.echo)
```

Note `_repl_loop` (unchanged) still catches `OllamaError` inside its
per-turn loop at line ~140 — widen that too:

```python
        try:
            answer = session.send(stripped)
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            continue
```

- [ ] **Step 8: Run the full CLI test suite**

Run: `pytest assistant/tests/test_cli.py assistant/tests/test_cli_agent.py assistant/tests/test_cli_backend.py -v`
Expected: all PASS, no regressions (the existing "without index exits
nonzero" tests still fail before any client is constructed, so they're
unaffected by this change).

- [ ] **Step 9: Run the full suite**

Run: `pytest -q`
Expected: 116 passed (113 from Task 6 + 3 new)

- [ ] **Step 10: Commit**

```bash
git add assistant/cli.py assistant/tests/test_cli_backend.py
git commit -m "feat: add --backend ollama|gemini flag to ask/agent/repl"
```

---

### Task 8: Manual smoke test with a real Gemini key

**Files:** none (manual verification only — no code changes)

- [ ] **Step 1: Add a real key**

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` with a real key
from https://aistudio.google.com/apikey.

- [ ] **Step 2: Confirm the model id is live**

Run:
```bash
python3 -c "
from assistant.llm.gemini_client import GeminiClient
c = GeminiClient()
print(c.chat([{'role': 'user', 'content': 'Say OK.'}]))
"
```
Expected: prints a short reply, no `GeminiError`. If it raises
`GeminiError` with a 404/`model not found`-style message, the
`gemini-3-flash` model id has changed — run
`curl -H "x-goog-api-key: $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1beta/models`
to list current model ids, and update `GEMINI_MODEL` in
`assistant/config.py` to match (Task 1's value is a best guess as of
2026-07-11, not a verified live id).

- [ ] **Step 3: Smoke test through the CLI**

Run (against an already-indexed repo — see `_require_index`):
```bash
python3 -m assistant.cli ask "what does this project do?" --repo . --backend gemini
```
Expected: streams an answer using Gemini instead of Ollama, same output
shape as `--backend ollama` (the default).

- [ ] **Step 4: Confirm default (no flag) still uses Ollama**

Run:
```bash
python3 -m assistant.cli ask "what does this project do?" --repo .
```
Expected: unchanged behavior from before this feature — uses Ollama, no
`--backend` needed.

No commit for this task — it's verification, not code.

---

## Post-implementation

Update `README.md`'s usage section to mention `--backend gemini` and the
`.env` / `GEMINI_API_KEY` setup, if the README documents CLI flags at that
level of detail (check current README structure before adding — keep it
consistent with how other flags, like `--rerank`, are or aren't documented
there).
