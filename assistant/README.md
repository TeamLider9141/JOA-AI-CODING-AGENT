# Local Coding Assistant — Retrieval Core

CPU-only coding assistant core: tree-sitter AST chunking, embedded Qdrant +
BM25 hybrid retrieval (RRF), Ollama for embeddings and chat.

## Setup

    python3 -m venv .venv                      # from repo root
    .venv/bin/pip install -r assistant/requirements.txt
    ollama pull qwen2.5-coder:7b
    ollama pull nomic-embed-text

## Usage

    .venv/bin/python -m assistant.cli index <repo-path>
    .venv/bin/python -m assistant.cli search "query" --repo <repo-path>
    .venv/bin/python -m assistant.cli ask "question" --repo <repo-path>

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
interactive confirmation. Loop is capped at 10 iterations.

Verified end-to-end on `crystal_bot`: a read-only task correctly located
`init_db` and summarized the schema; a write task exercised the confirm gate
in both directions (declined → no file; accepted → file created).

Phase 4 (next, separate plan): cross-encoder reranker, multi-step planner,
and an auto-fix loop (run tests → read failure → edit → re-run).
