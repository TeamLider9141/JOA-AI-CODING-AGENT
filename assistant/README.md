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

Retrieval core (indexing, hybrid search, `ask`/`search`/`index` CLI) is built
and tested — 35 tests passing. Not yet run against a real repo or evaluated
end-to-end: that requires Ollama installed with `qwen2.5-coder:7b` and
`nomic-embed-text` pulled (~6.5 GB), deferred pending user go-ahead. The
`eval/` module (gold-question hit@5 comparison) is part of the next work
session, once Ollama is in place.

## Design

See `docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md`.
Models and retrieval parameters live in `assistant/config.py` only.

## Agent loop (next)

Tool-calling agent (spec Phases 3–4) is a separate, not-yet-written plan:
`read_file`/`write_file`/`run_cmd`/`search_code` tools, path jail, JSON
tool-call protocol, diff-confirmation on writes. Builds on `search_index()`
and `OllamaClient` from this plan.
