# Phase 4 — Planner + Auto-fix Design Spec

**Date:** 2026-07-06
**Status:** Approved (design reviewed in session)
**Builds on:** `docs/superpowers/plans/2026-07-06-assistant-agent-loop.md` (agent loop, Phase 3)

## Goal

Make the 7B agent more reliable at multi-step chores — edit→commit→push,
run→read error→fix→rerun — without changing the model. The levers are all in
the scaffold: keep the goal and a running plan pinned in front of the model so
it doesn't lose track over a long conversation, let it see whether commands
actually succeeded, and give commands enough time to finish.

The model stays `qwen2.5-coder:7b` on CPU. This is not an attempt to match a
frontier model's reliability; it is squeezing more out of a weak model with
better scaffolding.

## Decisions (made during brainstorming)

1. **Planner: TODO scratchpad.** The agent writes an ordered todo list and the
   runner re-injects a compact reminder (task + plan + iterations left) before
   every turn. Flexible — the plan can be revised mid-run. Chosen over an
   upfront rigid plan (brittle when the weak model plans wrong or a step fails)
   and over no-planner (agent loses direction on multi-step chores).
2. **Reranker deferred.** It addresses retrieval quality, not agent
   reliability, so it is out of scope here and becomes its own small plan later
   if wanted.

## Constraints

- Model: `qwen2.5-coder:7b`, CPU-only. Weak at JSON and multi-step chaining, so
  added protocol structure must stay minimal and forgiving.
- No new dependencies. Everything builds on the existing `assistant/agent/`
  modules and `config.py`.

## Components

### 1. TODO scratchpad (planner)

A new agent action:

```json
{"action": "plan", "args": {"todo": ["read config.py", "edit change_balance", "commit"]}}
```

- The runner stores the todo list (`list[str]`) and returns `"plan updated"` as
  the tool result, so the loop continues normally.
- **Before every turn**, the runner appends a compact reminder as a user
  message: `(Task: <task> | Plan: 1. … 2. … | N iterations left)`. This is the
  core of the design — a weak model drifts as history grows, so the goal and
  the current plan are kept in the most recent position (recency) every turn,
  not left to be re-read from far back in the transcript.
- The model may emit `plan` at any time to set or revise the list. The system
  prompt instructs it to start multi-step tasks with a `plan`.

### 2. Auto-fix enabler

The key change is that **`run_cmd` returns the exit code** alongside output:

```
exit code: 1
Traceback (most recent call last): ...
```

Today `run_cmd` returns only stdout+stderr, so the model cannot tell whether a
command passed or failed. With the exit code visible, the model can see a
failure, read the error, fix the cause, and re-run to verify. The system prompt
gains one instruction: "if run_cmd fails (exit code ≠ 0), read the error, fix
the cause, and re-run to verify." No new control flow — auto-fix is the existing
loop plus this signal and guidance.

### 3. run_cmd timeout

Raise the timeout from 30s to 120s and move it into `config.py` as
`RUN_CMD_TIMEOUT` (single source of truth, matching the rest of the project's
config). This keeps `git push` on a slow/flaky connection from being killed
mid-transfer.

### 4. Iteration cap

Raise the runner's default `max_iters` from 10 to 15, giving multi-step chores
(plan + several actions + verify) room to finish.

## Files

Small, focused edits to existing files — no new modules:

- `assistant/config.py` — add `RUN_CMD_TIMEOUT = 120`.
- `assistant/agent/tools.py` — `run_cmd` prepends `exit code: <n>` to its
  result and reads the timeout from `config.RUN_CMD_TIMEOUT`.
- `assistant/agent/protocol.py` — add the `plan` action and the auto-fix
  instruction to the system prompt.
- `assistant/agent/runner.py` — maintain the plan scratchpad, inject the
  per-turn reminder, handle the `plan` action, default `max_iters=15`.

## Data flow

```
task → runner seeds messages
loop (≤15):
    append reminder (task + plan + iters left)
    chat → JSON action
    ├─ plan       → store todo, result "plan updated"
    ├─ run_cmd    → "exit code: N\n<output>"   (model sees pass/fail)
    ├─ read/write/search → as before
    └─ final      → return answer
    append result
```

## Error handling

- `plan` action missing the `todo` arg → error string fed back to the model
  (same pattern as other missing-arg errors).
- `run_cmd` timeout → still returns the "timed out after Ns" message, now with
  the config-driven duration.
- Everything else unchanged from Phase 3 (path jail, parse-retry, confirm gate).

## Testing

- `run_cmd` on a failing command shows `exit code:` with a non-zero value; on a
  succeeding command shows `exit code: 0`.
- `plan` action updates the scratchpad and returns "plan updated".
- The per-turn reminder is present and contains the task and current plan
  (verified via a fake client inspecting the messages it receives).
- Auto-fix scenario end-to-end with a scripted fake client: `run_cmd` fails →
  model reads the error → writes a fix → re-runs → `final`.
- Timeout value comes from `config.RUN_CMD_TIMEOUT`.

## Out of scope

Cross-encoder reranker (separate retrieval-quality plan), any change to the
model or embeddings, parallel/multi-agent execution, streaming the agent's
intermediate steps to the UI.
