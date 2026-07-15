# Local Coding Assistant — Retrieval Core

CPU-only coding assistant core: tree-sitter AST chunking, embedded Qdrant +
BM25 hybrid retrieval (RRF), Ollama for embeddings and chat. Chat can
optionally run against Gemini instead of Ollama via `--backend gemini` —
embeddings always stay on Ollama regardless of chat backend.

## Install (Linux, one-liner)

    curl -fsSL https://raw.githubusercontent.com/TeamLider9141/JOA-AI-CODING-AGENT/main/install.sh | bash

Clones into `~/.joa` (or `$JOA_HOME`), creates its own venv, installs
deps, and symlinks `bin/joa` into `~/.local/bin/joa` so `joa` works from
any directory afterward. Re-running the same command updates an existing
install (`git pull --ff-only`).

Both the installer and every `joa` REPL launch print the same JOA ASCII
banner in blue (`typer.secho(..., fg=typer.colors.BLUE)` /
ANSI `\033[34m` in `install.sh`).

## Setup (manual, for working on this repo itself)

    python3 -m venv .venv                      # from repo root
    .venv/bin/pip install -r assistant/requirements.txt
    ollama pull qwen2.5-coder:0.5b              # fastest/lightest; or :1.5b / :3b / :7b
    ollama pull nomic-embed-text

Optional — to use `--backend gemini`:

    cp .env.example .env
    # add GEMINI_API_KEY to .env (get one at https://aistudio.google.com/apikey)

## Usage

    .venv/bin/python -m assistant.cli index <repo-path>
    .venv/bin/python -m assistant.cli search "query" --repo <repo-path>
    .venv/bin/python -m assistant.cli ask "question" --repo <repo-path>
    .venv/bin/python -m assistant.cli ask "question" --repo <repo-path> --backend gemini

`--backend` (`ollama` | `gemini`, default `ollama`) works on `ask`, `agent`,
and `repl`. `index`/`search` have no `--backend` — they only ever embed, and
embedding always uses Ollama's `nomic-embed-text`.

## Tests and eval

    .venv/bin/pytest
    .venv/bin/python -m assistant.eval.run_eval --repo <repo-path>

## Status

Retrieval core (indexing, hybrid search, `ask`/`search`/`index` CLI) is built,
tested, and verified end-to-end — 35 unit tests passing, plus a real run
against `~/Desktop/crystal_bot` (indexed in place, read-only): `search` and
`ask` both return correct file:line results, and the 10-question gold eval
scores vector hit@5: 10/10, hybrid hit@5: 10/10.

## Design

See `docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md`.
Models and retrieval parameters live in `assistant/config.py` only.

## Agent

    .venv/bin/python -m assistant.cli agent "task" --repo <repo-path>

The agent plans one step at a time, emitting a JSON tool call
(`read_file` / `write_file` / `run_cmd` / `search_code`) that we parse and
execute, feeding the result back until it returns a final answer. All file
access is jailed to the target repo root; writes and commands require
interactive confirmation. Loop is capped at 15 iterations.

Verified end-to-end on `crystal_bot`: a read-only task correctly located
`init_db` and summarized the schema; a write task exercised the confirm gate
in both directions (declined → no file; accepted → file created).

The agent maintains a TODO scratchpad (the `plan` action) that is re-shown
each turn so it keeps multi-step chores on track, and `run_cmd` results include
the exit code so it can notice a failure and re-run after fixing. Verified on a
throwaway git repo: given "stage and commit these changes", the agent planned,
ran `git add`/`git commit` via confirmed `run_cmd` steps, and produced a clean
commit.

Still deferred (separate plan): a cross-encoder reranker for retrieval quality.

## Interactive session (`joa`)

Put the launcher on your PATH once:

    export PATH="$HOME/Desktop/system_llm/bin:$PATH"   # add to ~/.zshrc

Then, from any indexed repo:

    cd ~/Desktop/crystal_bot
    joa                      # opens an interactive agent session
    joa> add a docstring to change_balance
    joa> now commit that change
    joa> exit

The first time `joa` runs in a given directory (interactive terminal
only), it shows a Claude Code-style workspace-trust prompt before
touching anything — an arrow-key Ha/Yo'q menu (Up/Down/Enter, same
`_arrow_select` widget `/joamodel` uses) rather than typed input. Accept
once and that directory is remembered in `~/.config/joa/trusted_dirs.json`,
no more prompts for it. If the directory hasn't been indexed yet, `joa`
offers to index it right there (same arrow-key menu) instead of just
erroring — declining, or running non-interactively, falls back to the old
`No index found` message (and non-interactive stdin falls back to typed
"1"/number input throughout, so piped/scripted use is unaffected).

Indexing itself is two-phase: a BM25 (lexical) index builds synchronously
(sub-second — no embedding call, so it's unaffected by Ollama's speed) and
the REPL is usable immediately after; a vector (semantic) index then
builds in a background daemon thread using the same embedder. Search
transparently falls back to BM25-only until the vector index is ready,
then upgrades to full hybrid RRF search automatically — no user action
needed. A fingerprint of the repo's files (path, mtime, size) is saved
once the vector build succeeds (`vector_manifest.json` in the repo's data
directory); subsequent `joa` launches skip rebuilding the vector index
entirely when nothing has changed. The background build writes into a
temp `qdrant.new` directory and atomically swaps it in on success, so a
concurrent search against the live index is never disturbed. The
standalone `joa index <repo>` command is unaffected — it still blocks
until both BM25 and vector indexes are fully built, which is the point
for scripted/CI use.

Each line is handled by the coding agent (read/write/run/search, with writes
and commands confirmed), and the conversation carries across turns so
follow-ups remember earlier context. `joa <args>` still works as a short form
for the CLI, e.g. `joa ask "how does X work"` or `joa agent "fix the bug"`.
`joa --backend gemini` opens the same REPL against Gemini instead of Ollama.

Mid-session, type `/joamodel` to switch models without restarting: it lists
whatever Ollama models are actually installed (`ollama pull`ed) plus
`gemini` as a last option, then swaps the active chat client to whichever
one you pick. In an interactive terminal, picking is done with an inline
arrow-key menu (Up/Down to move, Enter to select, Esc/Ctrl-C to cancel) —
no number typing, pre-highlighted on the currently active model. Piped/
scripted stdin falls back to typing a number instead (same isatty-guard
pattern as the trust screen and auto-index prompt). Picking `gemini`
without `GEMINI_API_KEY` set just warns and leaves the current model in
place instead of switching to something that would immediately fail. The
list is colorized (Ollama models cyan, `gemini` magenta) and the currently
active model is marked green with a `(joriy)` suffix — colors auto-disable
on piped/non-terminal output.

Plain questions take a fast path: one direct chat call instead of the
full agent protocol, printed once the full answer is ready (not
token-by-token — buffering the whole reply first lets LaTeX cleanup run
on the complete text, see below). The model routes automatically — if
the request needs file/command/search tools it replies `ESCALATE`
internally and the normal agent loop takes over. Session history is
capped (`MAX_HISTORY_MESSAGES` in `config.py`) so long sessions don't
slow down over time. Every reply's timing footer now includes which
model produced it: `(2.3s · qwen2.5-coder:0.5b)`.

Both the fast path and the agent's final answer run through
`clean_latex()` (`assistant/latex_clean.py`) before display — small local
models often answer math questions in raw LaTeX (`\frac{1}{2}`,
`\alpha`, `x^2`), which is unreadable escape-sequence soup in a plain
terminal. It's a best-effort regex-based converter (Greek letters, common
operators, `\frac{a}{b}` → `a/b`, `^`/`_` → Unicode super/subscript where
a mapping exists) — it doesn't implement the full LaTeX grammar, and
unrecognized commands are left as-is rather than risk corrupting the
text. `FAST_SYSTEM_PROMPT` also nudges the model to flag when it isn't
confident in a calculation rather than stating a guess as fact — small
models (`0.5b`/`1.5b`) are tuned for code, not general knowledge or math,
and can be confidently wrong; switch to a larger model or Gemini via
`/joamodel` for those questions.

### Watch list: Bonsai 27B

[PrismML's Bonsai 27B](https://docs.prismml.com/models/bonsai-27b)
(1-bit/ternary, 27B params, Apache 2.0, released 2026-07-14) is worth
tracking — the 1-bit variant is ~3.5GB (down from 54GB at full
precision), small enough for a laptop or even a phone. It's not usable
via Joa yet: Ollama itself can't load its Q1_0 GGUF (Ollama's bundled
`ggml` build doesn't include that tensor type). Once Ollama adds
support, Joa needs no code changes — `/joamodel` auto-discovers whatever
you `ollama pull`.

Other slash commands: `/` or `/help` lists every command; `/clear` resets
the conversation context to zero (keeps only the system prompt). Anything
starting with `/` is handled locally and never sent to the LLM — unknown
commands get an error pointing at `/help` instead of confusing the model.

`!command` runs a shell command directly, bypassing the LLM/agent loop
entirely, with live streaming output — a bare carriage return (`\r`)
passes through untouched so progress bars (`!ollama pull ...`) redraw in
place instead of spamming new lines. No timeout, since you're watching
and can Ctrl-C. The agent's own `run_cmd` tool also streams live now
(via `ToolContext.output_sink`) instead of going silent until a
long-running command finishes.

Ctrl-C stops whatever's currently running — a streaming answer, the
agent loop, or a `!command` — without leaving the REPL; it's caught
around each of those, prints a short "stopped" notice, and returns to
the `joa>` prompt. Only `exit`/`quit`/Ctrl-D end the session itself.

In a real terminal, typing `/` pops a live completion dropdown
(prompt_toolkit): suggestions filter as you type, Tab/arrows select.
Piped/scripted stdin falls back to plain `input()` automatically, so
tests and shell pipelines are unaffected.
