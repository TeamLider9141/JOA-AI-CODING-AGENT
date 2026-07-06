# `joa` Interactive Session Design Spec

**Date:** 2026-07-06
**Status:** Approved (design reviewed in session)
**Builds on:** `docs/superpowers/plans/2026-07-06-assistant-agent-loop.md` (agent loop) and `docs/superpowers/plans/2026-07-06-phase4-planner-autofix.md` (planner)

## Goal

Make the assistant pleasant to drive from the terminal the way Claude Code is:
type `joa` in a repo and get an interactive session where each line you enter
is handled by the coding agent, with the conversation carried across turns so
follow-ups ("now add a function to that file") remember what came before.

## Decisions (made during brainstorming)

1. **Session mode: full agent.** Every line runs through the agent (read/write/
   run/search), not just Q&A. Writes and commands still require confirmation.
2. **Target repo: current directory.** `joa` with no arguments operates on the
   repo in `$(pwd)` (which must already be indexed), matching how Claude Code
   attaches to the working directory.
3. **Conversation continuity: yes.** The whole session is one continuous
   conversation — message history persists across turns so later turns see
   earlier context.

## Architecture

Two pieces: a reusable `AgentSession` that holds conversation state, and a thin
REPL command plus launcher script that drive it.

```
bin/joa (shell launcher)
  ├─ no args → assistant.cli repl --repo "$(pwd)"
  └─ args    → assistant.cli <args>     (e.g. joa ask "…", joa agent "…")

assistant.cli repl
  → AgentSession(ctx, client)
  → loop: read line → session.send(line) → print answer → until exit/EOF

AgentSession (in agent/runner.py)
  → persistent self.messages (system prompt + growing history)
  → .send(task) runs the plan→act→observe loop, appending to history
```

## Components

### `AgentSession` (refactor of the runner)

The existing `run_agent()` loop is refactored into a class that keeps
`messages` and other state as instance attributes, so state survives between
calls:

```python
class AgentSession:
    def __init__(self, ctx, client, max_iters=15): ...
    def send(self, task: str) -> str: ...   # runs one full agent turn
```

- `__init__` seeds `self.messages` with the system prompt once.
- `send(task)` appends the new user task, runs the same plan→act→observe loop
  (per-turn reminder, JSON parse with retry, tool execution, `plan`/`final`
  handling, iteration cap) that exists today, appending everything to
  `self.messages`, and returns the final answer string.
- The plan scratchpad resets at the start of each `send()` — each user request
  plans fresh, but the message history (context) carries across requests. This
  gives continuity of context without a stale "done" plan from a prior request
  lingering in the reminder.

**`run_agent()` stays as a thin wrapper** — `return AgentSession(ctx, client,
max_iters).send(task)` — so its signature and behavior are unchanged and all
existing runner tests keep passing.

### `repl` CLI command

`assistant.cli repl --repo <path>` (default `--repo .`):
- Resolves and requires the index for the repo (same `_require_index` check as
  the other commands).
- Builds the `ToolContext` (root = repo, `typer.confirm` gate) and an
  `AgentSession`.
- Loops: read a line (`exit`/`quit`/EOF ends the session; blank lines are
  skipped), call `session.send(line)`, print the answer. An `OllamaError`
  during a turn prints the error and keeps the session alive rather than
  crashing it.

### `bin/joa` launcher

A small shell script at `bin/joa`:
- With no arguments → runs `repl` against the current directory.
- With arguments → passes them straight through to `assistant.cli` (so
  `joa ask "…"`, `joa agent "…"`, `joa search "…" --repo …` all work as
  short forms).
- Locates the project's `.venv` relative to its own path, so it works from any
  directory once on `PATH`.

The user puts `~/Desktop/system_llm/bin` on their `PATH` (documented in the
README); no change to their shell config is made by us.

## Data flow

```
cd ~/Desktop/crystal_bot && joa
  → repl --repo /home/.../crystal_bot
  → AgentSession created
  → "add a docstring to change_balance"  → send() → agent acts → answer
  → "now commit that change"             → send() → sees prior turn → acts
  → exit
```

## Error handling

- Repo not indexed → the existing "No index found…" message, exit non-zero,
  before the loop starts.
- `OllamaError` mid-session → print the actionable message, keep the session
  running (the user can retry or exit).
- EOF (Ctrl-D) or `exit`/`quit` → clean exit, status 0.

## Testing

- `AgentSession.send` runs a full turn and returns the final answer (fake
  client), matching `run_agent`'s behavior.
- Continuity: two successive `send()` calls share history — the second turn's
  messages include content from the first turn.
- Plan resets per `send()` — a plan set in the first turn does not appear in
  the reminder of the second turn's first iteration.
- `run_agent()` still behaves identically (existing runner tests unchanged).
- `repl` command is registered and exits non-zero when the repo has no index.
- `bin/joa` is verified manually: no-args launches the repl, args pass through.

## Out of scope

- Context-window trimming: a long session's history can eventually exceed
  `NUM_CTX` on CPU. v1 accepts this (sessions are short); trimming/summarizing
  old turns is future work.
- Readline niceties (history file, arrow-key recall), streaming the agent's
  intermediate steps, and any packaging/installation beyond a `PATH` entry.
