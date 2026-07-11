# Local Coding Assistant — Retrieval Core

CPU-only coding assistant core: tree-sitter AST chunking, embedded Qdrant +
BM25 hybrid retrieval (RRF), Ollama for embeddings and chat. Chat can
optionally run against Gemini instead of Ollama via `--backend gemini` —
embeddings always stay on Ollama regardless of chat backend.

## Setup

    python3 -m venv .venv                      # from repo root
    .venv/bin/pip install -r assistant/requirements.txt
    ollama pull qwen2.5-coder:7b
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

Each line is handled by the coding agent (read/write/run/search, with writes
and commands confirmed), and the conversation carries across turns so
follow-ups remember earlier context. `joa <args>` still works as a short form
for the CLI, e.g. `joa ask "how does X work"` or `joa agent "fix the bug"`.
`joa --backend gemini` opens the same REPL against Gemini instead of Ollama.

Mid-session, type `/joamodel` to switch models without restarting: it lists
whatever Ollama models are actually installed (`ollama pull`ed) plus
`gemini` as a last option, then swaps the active chat client to whichever
number you pick. Picking `gemini` without `GEMINI_API_KEY` set just warns
and leaves the current model in place instead of switching to something
that would immediately fail.
