# `/joamodel` REPL Command Design Spec

**Date:** 2026-07-11
**Status:** Approved (design reviewed in session)

## Goal

Let a user switch the active chat model — either to a different installed
Ollama model (e.g. `qwen2.5-coder:1.5b` vs `3b` vs `7b`, for speed
comparison) or to Gemini — from inside an active `joa` REPL session,
without restarting the session or editing `assistant/config.py`.

## Decisions (made during brainstorming)

1. **Scope: REPL only (`joa`'s interactive session), not `ask`/`agent`.**
   Those are one-shot commands; there's no persistent session to switch
   the model on. Only `_repl_loop` gains this behavior.
2. **Command name: `/joamodel`, no arguments.** Always shows the list;
   there is no `/joamodel <name>` direct-jump shortcut (rejected in favor
   of the simpler two-step "list, then type a number" flow).
3. **Ollama model list is discovered dynamically**, not hardcoded — `GET
   /api/tags` against the running Ollama server, so the list always
   matches what's actually `ollama pull`ed, and never goes stale the way
   a fixed list embedded in code would.
4. **Gemini is appended as the last option in the same list** (not a
   separate `/backend` command) — one unified command covers both "which
   local model" and "or use Gemini instead" in a single mental model.
5. **Selection is number-only.** The user types `/joamodel`, sees a
   numbered list, then types just the number on the next line.
6. **Missing `GEMINI_API_KEY` is checked eagerly, at selection time, not
   deferred to the next chat call.** If the user picks "gemini" and no key
   is configured, they see a warning immediately and the session's model
   is left unchanged (does *not* switch to a Gemini client that would just
   fail on the next message).
7. **A `GeminiError` during a normal chat turn (e.g. rate limit, blocked
   response) gets an extra hint line appended** pointing back at
   `/joamodel` as the way to switch backends — makes the recovery path
   discoverable without the user needing to remember the command exists.

## Constraints

- No new dependencies.
- `OllamaClient`'s existing hardcoded `config.CHAT_MODEL` usage becomes
  overridable via a constructor parameter, defaulting to the same value —
  existing callers (`index`, `search`, `ask`, `agent`'s `embed_client`, and
  any `OllamaClient()` construction elsewhere) are unaffected.
- Must not crash the REPL on invalid input (bad number, EOF mid-selection,
  Ollama unreachable while listing models) — always fall back to "nothing
  changed, keep going."

## Architecture

```
joa> /joamodel
        │
        ├── embed_client.list_models()  (GET /api/tags on Ollama)
        │        │
        │        ├── success → [installed models...] + ["gemini"]
        │        └── OllamaError → print it, abort (session.client unchanged)
        │
        ├── print numbered list, prompt "Raqamni tanlang:"
        ├── read one more line via read_line()
        │        └── EOFError → abort silently (next loop iteration exits REPL as usual)
        │
        ├── invalid number / non-digit → print error, abort (unchanged)
        │
        ├── selected == "gemini"
        │        ├── no GEMINI_API_KEY → warn, abort (unchanged)
        │        └── has key → session.client = GeminiClient()
        │
        └── selected == <ollama model name>
                 └── session.client = OllamaClient(model=<name>)
```

Normal turn error path (unchanged control flow, new hint line):
```
joa> <message>
        → session.send(...) raises GeminiError
        → echo(str(exc))
        → echo("/joamodel bilan Ollama modeliga qayting.")   # only for GeminiError
        → continue loop
```

## Components

### `assistant/llm/ollama_client.py` — model becomes overridable

```python
class OllamaClient:
    def __init__(
        self,
        base_url: str = config.OLLAMA_URL,
        model: str = config.CHAT_MODEL,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url
        self._model = model
        self._client = httpx.Client(...)
```

`chat`, `chat_stream`, and the 404-hint message all switch from reading
`config.CHAT_MODEL` directly to reading `self._model`. `embed()` is
unaffected (`config.EMBED_MODEL` stays hardcoded — embeddings are out of
scope here, same invariant as the Gemini backend work).

### `assistant/llm/ollama_client.py` — new `list_models()`

```python
def list_models(self) -> list[str]:
    """Installed Ollama model names (`ollama pull`ed), sorted."""
```

Implemented via `GET /api/tags`, parsing `{"models": [{"name": ...}, ...]}`
into a sorted `list[str]`. Reuses the same `OllamaError` conversion pattern
as `_post` (`ConnectError` → actionable "start Ollama" message;
`HTTPStatusError` → status+body message) via a small `_get` helper mirroring
the existing `_post`.

### `assistant/cli.py` — `_repl_loop` and new `_handle_joamodel`

`_repl_loop` gains a 4th parameter, `embed_client` (the `OllamaClient`
instance `repl()` already constructs unconditionally for embeddings — reused
here purely to call `.list_models()`, regardless of which client
`session.client` currently points at).

```python
def _repl_loop(session, read_line, echo, embed_client) -> None:
    ...
    if stripped == "/joamodel":
        _handle_joamodel(session, embed_client, read_line, echo)
        continue
    ...
    except (OllamaError, GeminiError) as exc:
        echo(str(exc))
        if isinstance(exc, GeminiError):
            echo("/joamodel bilan Ollama modeliga qayting.")
        continue
```

```python
def _handle_joamodel(session, embed_client, read_line, echo) -> None:
    """List installed Ollama models + "gemini"; switch session.client to
    the chosen one. Any failure (unreachable Ollama, bad input, EOF,
    missing Gemini key) leaves session.client unchanged."""
```

`session.client` is reassigned directly (`AgentSession.send()` always reads
`self.client` fresh on each call, so swapping it between turns is safe with
no other state to reconcile).

`repl()` passes `embed_client` into `_repl_loop` at its existing call site.

## Error handling

- Ollama unreachable while listing models → `OllamaError` from
  `list_models()` is caught inside `_handle_joamodel`, printed, no crash,
  no model change.
- Non-numeric or out-of-range selection → printed error, no model change.
- `EOFError` while reading the selection line → silently aborts
  `_handle_joamodel`; the outer loop's next `read_line()` call raises
  `EOFError` again and exits the REPL normally (same as today's Ctrl-D
  behavior).
- Gemini selected without `GEMINI_API_KEY` → warning printed, no model
  change (never constructs a `GeminiClient` that would just fail later).
- `GeminiClient()` construction itself raising `GeminiError` (shouldn't
  happen given the eager key check, but defensive) → caught, printed, no
  model change.

## Testing

- `OllamaClient.list_models()` returns a sorted list from a mocked
  `/api/tags` response; raises `OllamaError` on connect failure, matching
  the existing `_post` error-conversion tests' pattern.
- `OllamaClient.chat()`/`chat_stream()` use a constructor-provided `model`
  override instead of `config.CHAT_MODEL` when one is passed.
- `_handle_joamodel` (or `_repl_loop` end-to-end with a scripted
  `read_line`): selecting a valid Ollama model number swaps
  `session.client` to a new `OllamaClient` with the right `_model`;
  selecting "gemini" with `GEMINI_API_KEY` set swaps to `GeminiClient`;
  selecting "gemini" without a key leaves `session.client` unchanged and
  prints a warning; an invalid number leaves `session.client` unchanged;
  `list_models()` raising `OllamaError` leaves `session.client` unchanged.
- A `GeminiError` raised from `session.send()` during a normal turn causes
  the `/joamodel` hint line to be echoed; an `OllamaError` does not.

## Out of scope

`--model`/`--backend`-style CLI flags for `ask`/`agent` (already exists for
backend via `--backend`; no `--model` flag is being added — this feature is
REPL-only). A `/joamodel <name>` direct-jump shortcut. Persisting the
chosen model across REPL sessions. Listing Gemini's own available models
(only Ollama's installed list + the single "gemini" entry).
