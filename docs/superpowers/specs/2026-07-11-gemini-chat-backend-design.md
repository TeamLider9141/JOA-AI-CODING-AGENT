# Gemini Chat Backend Design Spec

**Date:** 2026-07-11
**Status:** Approved (design reviewed in session)

## Goal

Add Google Gemini as an optional second chat backend, selectable per
invocation, so the CPU-only Ollama setup can be swapped for a faster cloud
model when desired (and compared for speed/quality). Ollama stays the
default; nothing changes for users who never pass `--backend gemini`.

## Decisions (made during brainstorming)

1. **Backend selection: CLI flag (`--backend ollama|gemini`, default
   `ollama`)** on `ask`, `agent`, `repl`. Rejected a `config.py` constant or
   `.env`-only toggle — the whole point is comparing backends run-to-run
   (e.g. `qwen2.5-coder:1.5b` vs `3b` vs Gemini) without editing files each
   time.
2. **Implementation: raw `httpx` REST calls**, mirroring `OllamaClient`,
   rather than the `google-genai` SDK or an abstraction layer like LiteLLM.
   No new dependency, matches the project's hand-written-core ethos (the
   same reasoning that drove the original LlamaIndex → hand-written pivot).
3. **Embeddings stay on Ollama always.** The index is built with
   `nomic-embed-text`; swapping the embedding backend would require
   reindexing and buys nothing here. `--backend` only affects chat/agent
   calls, never `index`/`search` (which only embed).
4. **Default model: Gemini 3 Flash.** Free tier confirmed at 1,500
   requests/day, 10 RPM, 250k TPM — wide enough for interactive use.
   (Gemini 2.5 Pro's 50 RPD/5 RPM free tier was rejected as too tight for
   quick comparisons.)
5. **No silent fallback.** If the Gemini backend errors (missing key,
   rate-limited, network), the command exits with a clear error — same
   pattern as `OllamaError` today. The user re-runs with `--backend ollama`.

## Constraints

- No new Python dependencies (`httpx` already in `requirements.txt`).
- `GeminiClient` must be duck-type compatible with `OllamaClient`'s chat
  surface (`chat(messages) -> str`, `chat_stream(messages) -> Iterator[str]`)
  so `AgentSession`, `run_agent`, and the `ask`/`repl` commands don't need to
  branch on backend beyond client construction.
- API key must never be hardcoded; read from environment, loadable from a
  `.env` file in the repo root.

## Architecture

```
CLI command (ask/agent/repl)
        │
        ├── embed_client = OllamaClient()          (always, for index/embed)
        │
        └── chat_client = OllamaClient()            if --backend ollama (default)
                         | GeminiClient()            if --backend gemini
                              │
                    chat_client.chat(messages) -> str
                    chat_client.chat_stream(messages) -> Iterator[str]
```

## Components

### `assistant/llm/gemini_client.py` (new)

```python
class GeminiError(RuntimeError):
    """Gemini unreachable, key missing/invalid, rate-limited, or API error."""

class GeminiClient:
    def __init__(self, api_key=config.GEMINI_API_KEY, model=config.GEMINI_MODEL,
                 transport=None): ...
    def chat(self, messages: list[dict]) -> str: ...
    def chat_stream(self, messages: list[dict]) -> Iterator[str]: ...
```

- No `.embed()` method — intentionally absent, since embeddings never route
  through Gemini (see Decision 3).
- `messages` uses the same `{"role": "user"|"assistant"|"system", "content":
  str}` shape as `OllamaClient`. Internally translated to Gemini's
  `contents: [{role, parts: [{text}]}]` schema (`assistant` → `model`;
  `system` messages folded into the request's `system_instruction` field
  rather than sent as a turn).
- `chat` posts to `:generateContent`; `chat_stream` posts to
  `:streamGenerateContent` and yields text chunks as they arrive (SSE/JSON
  lines, parsed the same incremental way `OllamaClient.chat_stream` parses
  Ollama's NDJSON stream).
- Missing/empty API key → `GeminiError` raised eagerly in `__init__`, with a
  message pointing at `GEMINI_API_KEY` in `.env`.
- HTTP error mapping (mirrors `OllamaClient._post`):
  - `401`/`403` → `GeminiError` "invalid API key".
  - `429` → `GeminiError` "rate limited, try --backend ollama or wait".
  - `httpx.ConnectError` → `GeminiError` "Gemini API unreachable, check network".
  - Other 4xx/5xx → `GeminiError` with the response body.
- Model id defaults to `"gemini-3-flash"`. If the live API 404s on that id
  at implementation time (aliases/dated ids shift), swap in whatever
  `GET /v1beta/models` lists for the current Flash model — the id itself is
  the only thing that changes, not the client shape.

### `.env` loading

A small loader (no `python-dotenv` dependency) added at the top of
`assistant/config.py`: on import, if a `.env` file exists in the repo root,
parse `KEY=VALUE` lines (skip blank/`#` lines) and set them into
`os.environ` only if not already set, so real environment variables always
win. `GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")` is read after this.

### `assistant/config.py` additions

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com"
```

### CLI (`assistant/cli.py`)

- `ask`, `agent`, `repl` each gain `backend: str = typer.Option("ollama",
  "--backend", help="ollama | gemini")`.
- Each command keeps `embed_client = OllamaClient()` unconditionally, and
  builds `chat_client = OllamaClient() if backend == "ollama" else
  GeminiClient()` for the chat-facing calls (`chat_stream` in `ask`; the
  `client` passed to `run_agent`/`AgentSession` in `agent`/`repl`).
- `index`/`search` are unchanged (embed-only, no `--backend`).
- Exception handling widens from `except OllamaError` to `except
  (OllamaError, GeminiError)` everywhere a chat call can now fail, same
  `typer.Exit(1)` behavior.

## Data flow

```
user runs: joa ask "..." --repo . --backend gemini
        → embed_client (Ollama) embeds query for retrieval (unchanged)
        → chat_client = GeminiClient()
        → chat_client.chat_stream(messages) streams tokens to stdout
        → GeminiError (if any) → printed, exit 1
```

## Error handling

- `GeminiError` mirrors `OllamaError`'s role: actionable message, caught at
  the CLI boundary, non-zero exit, no silent fallback to the other backend.
- A missing `GEMINI_API_KEY` fails fast in `GeminiClient.__init__`, before
  any network call.

## Testing

- `test_gemini_client.py`, mirroring `test_ollama_client.py`'s
  `httpx.MockTransport` pattern:
  - `chat` posts the translated `contents` payload and returns the parsed
    text.
  - `chat_stream` concatenates streamed chunks correctly.
  - Missing API key raises `GeminiError` without making a request.
  - `401`/`429`/connect-error responses raise `GeminiError` with the
    expected actionable hint.
  - `system` role messages are folded into `system_instruction`, not sent
    as a `contents` turn.
- CLI tests: `--backend gemini` on `ask`/`agent`/`repl` constructs a
  `GeminiClient` instead of a second `OllamaClient` (verified via
  monkeypatch/spy), while `embed_client` stays `OllamaClient` regardless.
- No test hits the real Gemini API.

## Out of scope

Streaming tool-call support beyond what the agent loop already does,
automatic backend fallback, per-request cost/quota tracking, any backend
other than Ollama/Gemini, changing how `index`/`search` work.
