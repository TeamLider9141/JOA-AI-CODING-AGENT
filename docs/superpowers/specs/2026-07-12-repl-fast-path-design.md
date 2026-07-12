# REPL Fast Path + History Cap Design Spec

**Date:** 2026-07-12
**Status:** Approved (design reviewed in session)

## Goal

Make the `joa` REPL feel as fast as chatting with the model directly:

1. **Fast path:** a plain question ("2+2?", "what does this function do?")
   gets one direct streaming chat call — tokens appear immediately —
   instead of paying the agent protocol's cost (~390-token system prompt,
   full history re-eval, 2-3 sequential JSON round-trips, no streaming).
2. **History cap:** the agent session's message history stops growing
   unboundedly, so long sessions don't get slower turn by turn — this
   speeds up the *task* path too, not just chit-chat.

Observed baseline on qwen2.5-coder:1.5b (CPU): the Ollama app answers
"2+2" near-instantly; the same model through `joa` takes 15-20s. Gemini
through `joa` is also markedly slower than raw Gemini for the same
structural reasons (sequential round-trips, no streaming).

## Decisions (made during brainstorming)

1. **Routing: the model decides.** Every REPL line first goes to a single
   streaming chat call with a light system prompt: answer directly, OR
   reply exactly `ESCALATE` if the request needs file/command/search
   tools. `ESCALATE` → the line is re-run through the existing agent loop
   unchanged. Rejected: a manual prefix (`? question`) — works but makes
   the user do the routing every time.
2. **Streaming with a small sniff buffer.** The first ~8 characters are
   buffered to check for `ESCALATE`; everything after that streams to the
   terminal as it arrives. This avoids printing half an answer and then
   "un-printing" it on escalation.
3. **Shared history.** Fast-path Q/A pairs are appended to
   `session.messages`, so a later task ("now add that to the file") still
   sees the earlier conversation. The agent's own system prompt stays at
   index 0 and is *replaced* (not stacked) by the fast prompt for
   fast-path calls.
4. **History cap lives in `AgentSession.send()`**, not the REPL — both
   the fast path and the agent loop benefit, and one-shot `agent` runs
   are unaffected in practice (they rarely hit the cap).
5. Known trade-off, accepted: a weak model may occasionally answer a
   *task* conversationally instead of escalating (no files changed). The
   user rephrases more explicitly; the reverse error (escalating a plain
   question) costs only speed, not correctness.

## Constraints

- No new dependencies.
- Works identically for both chat backends (`OllamaClient` and
  `GeminiClient` both expose `chat_stream(messages) -> Iterator[str]`).
- Existing error UX preserved: `OllamaError`/`GeminiError` during the
  fast path print the same way as agent-turn errors, including the
  `/joamodel` hint on `GeminiError`.
- `_repl_loop` stays testable without a live model (fake session/client).

## Architecture

```
joa> <line>
        │
        ├── /joamodel, exit, blank … (existing handling, unchanged)
        │
        ├── _fast_answer(session, line, echo_token)
        │        │  messages = [fast system prompt] + session.messages[1:]
        │        │             + [user line]
        │        │  stream = session.client.chat_stream(messages)
        │        │
        │        ├── first ≥8 chars == "ESCALATE" → drain/abort, return None
        │        ├── empty stream → return None (treat as escalate)
        │        └── else: sniffed prefix + remaining tokens → echo_token(...)
        │             append user line + full answer to session.messages
        │             return answer
        │
        ├── answer is not None → echo timing suffix, next prompt
        └── answer is None → session.send(line)   (existing agent loop)
```

History cap (in `AgentSession.send`, before the iteration loop):

```
if len(self.messages) > config.MAX_HISTORY_MESSAGES:
    self.messages = [self.messages[0]] + \
        self.messages[-(config.MAX_HISTORY_MESSAGES - 1):]
```

`messages[0]` is always the agent system prompt; the newest messages are
kept, the oldest turns are dropped.

## Components

### `assistant/config.py`

```python
MAX_HISTORY_MESSAGES = 40   # ~20 exchanges; keeps CPU prompt-eval bounded
```

### `assistant/cli.py` — `FAST_SYSTEM_PROMPT` + `_fast_answer`

```python
FAST_SYSTEM_PROMPT = (
    "You are a coding assistant chatting with a user inside their "
    "repository. If answering would require reading or writing files, "
    "running commands, or searching the codebase, reply with exactly "
    "ESCALATE and nothing else. Otherwise answer the question directly "
    "and concisely."
)

def _fast_answer(session, line, echo_token) -> str | None:
    """One direct streaming chat call. Returns the streamed answer, or
    None if the model escalated (or produced nothing) — caller then runs
    the full agent loop. Raises OllamaError/GeminiError like any chat."""
```

- Builds `[{"role": "system", "content": FAST_SYSTEM_PROMPT}]` +
  `session.messages[1:]` (agent history minus its system prompt) +
  the new user line.
- Iterates `session.client.chat_stream(...)`: accumulates until the
  buffer holds ≥ 8 characters (or the stream ends). If
  `buffer.strip().upper().startswith("ESCALATE")` → exhausts/abandons the
  stream and returns `None`. Otherwise emits the buffered prefix via
  `echo_token`, then streams every subsequent chunk through `echo_token`.
- On success appends `{"role": "user", "content": line}` and
  `{"role": "assistant", "content": answer}` to `session.messages`,
  returns the full answer string.
- Empty/whitespace-only stream → `None`.

### `assistant/cli.py` — `_repl_loop` integration

- Gains an `echo_token` parameter (defaulting at the `repl()` call site to
  `lambda t: typer.echo(t, nl=False)`); tests pass a collector.
- Per line (after the existing exit/blank//joamodel handling):

```python
start = time.perf_counter()
try:
    answer = _fast_answer(session, stripped, echo_token)
    if answer is None:
        answer = session.send(stripped)
        echo(f"{answer}\n({time.perf_counter() - start:.1f}s)")
    else:
        echo(f"\n({time.perf_counter() - start:.1f}s)")
except (OllamaError, GeminiError) as exc:
    echo(str(exc))
    if isinstance(exc, GeminiError):
        echo("/joamodel bilan Ollama modeliga qayting.")
    continue
```

(The fast path already streamed the answer via `echo_token`, so only the
timing suffix is echoed after it; the agent path echoes answer + timing
as today.)

### `assistant/agent/runner.py` — history cap

At the top of `AgentSession.send()`:

```python
if len(self.messages) > config.MAX_HISTORY_MESSAGES:
    self.messages = [self.messages[0]] + \
        self.messages[-(config.MAX_HISTORY_MESSAGES - 1):]
```

(`assistant/agent/runner.py` gains a `from assistant import config`
import.)

## Error handling

- `OllamaError`/`GeminiError` raised inside `_fast_answer` propagate to
  `_repl_loop`'s existing handler (same message + `/joamodel` hint for
  Gemini). Partial streamed output before the error is acceptable (same
  as `ask`'s behavior today).
- Escalation is silent — the user just sees the agent path's normal
  output (plan/tool confirmations) after a brief pause.

## Testing

- `_fast_answer` with a fake client: plain answer streams through
  `echo_token` and lands in `session.messages`; `ESCALATE` (exact, with
  trailing text, lowercase) returns `None` and appends nothing; empty
  stream returns `None`; short answers (< 8 chars, e.g. "4") still work
  (stream ends before the sniff buffer fills).
- `_repl_loop`: fast answer skips `session.send`; `None` falls through to
  `session.send`; timing suffix present in both paths; errors during the
  fast path keep the loop alive and show the Gemini hint only for
  `GeminiError`; `/joamodel` and `exit` handling unaffected.
- `AgentSession` history cap: seeding a session with > cap messages and
  calling `send` keeps `messages[0]` (system prompt), keeps the newest
  messages, and stays within the cap (fake client returning a `final`
  action immediately).

## Out of scope

Trimming the agent's own system prompt (risks weak-model accuracy),
summarizing dropped history instead of discarding it, streaming inside
the agent loop's JSON turns, any change to the one-shot `ask`/`agent`
CLI commands.
