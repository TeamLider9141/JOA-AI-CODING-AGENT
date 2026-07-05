# Local Coding Assistant — Design Spec

**Date:** 2026-07-05
**Status:** Approved (design reviewed in session)
**Location:** `system_llm/assistant/`

## Goal

Build a local, CPU-only coding assistant in the spirit of Cursor / Claude Code:
index a real codebase with AST-aware chunking, retrieve with hybrid search, answer
questions with citations, and execute multi-step coding tasks through a tool-calling
agent loop. Every core layer is written by hand (no RAG framework) so each design
decision is understood, not inherited.

This replaces the earlier LlamaIndex `SimpleDirectoryReader` RAG project, which was
a useful learning exercise but framework-bound and fragile across LlamaIndex API
changes.

## Decisions (made during brainstorming)

1. **Approach: vertical slice.** Each phase ends with a working end-to-end system;
   layers are deepened incrementally rather than built breadth-first.
2. **Interface: CLI first.** Typer-based CLI (`index`, `search`, `ask`, `agent`).
   VS Code extension is deferred to v2.
3. **Framework: core written by hand.** Tree-sitter chunker, hybrid search (RRF),
   agent loop, and tool calling are our own code. Qdrant is used via its client
   directly, in embedded (local, serverless) mode. LlamaIndex is not a dependency.

## Constraints

- **Hardware:** AMD Ryzen 7 5800U (8c/16t), 16 GB RAM (~14 GiB usable), Vega 8
  (not usable for inference) → CPU-only inference.
- **Runtime:** Python 3.10, Linux. No Docker (Qdrant runs embedded).
- **Models (via Ollama):** chat `qwen2.5-coder:7b` (Q4, ~4.7 GB), embeddings
  `nomic-embed-text` (~274 MB). Start with `num_ctx=4096`; tune later.
- **Test corpus:** `~/Desktop/uzbek_ai` repository (real, familiar codebase).

## Architecture

```
                    ┌─────────── INDEX time ────────────┐
repo files → tree-sitter AST → code chunks (function/class level)
                                      │
                          Ollama embeddings (nomic-embed-text)
                                      │
                        ┌─────────────┴─────────────┐
                     Qdrant (embedded)          BM25 index
                        └─────────────┬─────────────┘
                    ┌─────────── QUERY time ────────────┐
query → vector search + BM25 search → RRF merge → [reranker] → top-k chunks
                                      │
                          context + Ollama chat (qwen2.5-coder:7b)
                                      │
                     ┌────────────────┴────────────────┐
                  `ask` (answer + citations)    `agent` (tool loop)
```

## Project structure

```
system_llm/assistant/
├── indexer/          # repo walker (gitignore-aware), tree-sitter parser, chunker
├── store/            # qdrant embedded wrapper + BM25 store (persisted)
├── search/           # hybrid search: RRF merge, later reranker
├── llm/              # ollama client (chat, embed, streaming)
├── agent/            # tool schema, executor, agent loop
├── cli.py            # typer entrypoint: index / search / ask / agent
├── config.py         # model names, paths, top_k — single source of truth
└── tests/
```

## Components

### Indexer
- Repo walker honors `.gitignore` plus a hard exclude list (`venv`, `node_modules`,
  `__pycache__`, `.git`, binary files).
- Tree-sitter grammars: Python and JavaScript/TypeScript initially; grammar registry
  designed so adding a language is one entry.
- Chunking at function/class boundaries. Each chunk carries metadata:
  `path`, `symbol_name`, `kind` (function/class/method/module), `start_line`,
  `end_line`, and the parent class header for methods (so a method chunk is
  self-describing).
- Files with no grammar (markdown, config) fall back to plain sliding-window chunks.
- Chunk id = stable hash of `path + symbol + content` (enables incremental reindex
  later).

### Store
- Qdrant in embedded local mode (`QdrantClient(path=...)`) — one collection,
  cosine distance, payload = chunk metadata + text.
- BM25 (`rank_bm25`) over tokenized chunk text, persisted to disk alongside a
  chunk-id list. Both stores share the same chunk ids.

### Hybrid search
- Vector top-40 and BM25 top-40, merged with Reciprocal Rank Fusion (k=60),
  return top-10.
- Rationale: exact identifiers (`JWTMiddleware`) are BM25's strength; semantic
  questions ("where is authentication handled?") are the vector side's strength.
- Phase 4 adds a CPU cross-encoder reranker (e.g. `bge-reranker-base`) applied to
  the fused top-20 → top-5; kept optional behind config since CPU cost is real.

### LLM client
- Thin `httpx` client for the Ollama REST API: `/api/chat` (streaming),
  `/api/embed`. No SDK dependency.
- Model names, timeouts, and `num_ctx` come from `config.py` only — the earlier
  project's duplicate-`Settings.llm` bug class is designed out.
- Clear failure mode: if Ollama is unreachable, print how to start it and exit
  non-zero.

### Agent
- Tools: `read_file`, `write_file`, `run_cmd`, `search_code` (reuses hybrid
  search).
- Safety: all paths resolved and jailed to the target project root (path traversal
  blocked); `write_file` shows a diff and asks for confirmation; `run_cmd` has a
  timeout and an allowlist-first posture; hard cap of 10 loop iterations.
- Protocol: the LLM returns a JSON tool call; malformed JSON is retried twice with
  the parse error fed back, then the run aborts with the raw output shown.
- Loop: task → (RAG context) → LLM → tool call → execute → result appended to
  conversation → repeat until the LLM returns a final answer or the cap is hit.

## Phases (each ends with a working result)

| Phase | Builds | Done when |
|-------|--------|-----------|
| 0 | Setup: Ollama install, model pulls, venv, deps | `ollama list` shows both models; `pytest` runs |
| 1 | Indexer + Qdrant + vector search + `ask` | End-to-end: question → answer with file:line citations on uzbek_ai |
| 2 | BM25 + RRF hybrid + retrieval eval | Gold-question set shows hybrid hit@5 ≥ vector-only hit@5 |
| 3 | Agent loop + tools + safety | `agent "…"` completes a real multi-step task on a sandbox copy |
| 4 | Reranker, multi-step planner, auto-fix loop | Measured retrieval gain; agent fixes a failing script unaided |
| v2 | VS Code extension, incremental reindex, git tools | out of scope for this spec |

## Testing & evaluation

- pytest per component (chunker boundaries, RRF math, path jail, JSON retry).
- Retrieval quality is measured, not vibed: a gold set of ~15 questions about
  uzbek_ai with expected file paths; report hit@5 for vector-only vs hybrid vs
  reranked.
- Agent safety tests: path traversal attempts, oversized outputs, malformed JSON.

## Error handling

- Ollama down → actionable message, exit non-zero.
- Model missing → suggest exact `ollama pull` command.
- Index absent when `ask`/`search` runs → tell user to run `index` first.
- Embedding batch failures → retry with backoff, then skip-and-log the chunk.

## Out of scope (v1)

VS Code extension, multi-agent workflows, GPU runtimes (llama.cpp/vLLM), graph RAG,
watch-mode incremental indexing, remote Qdrant.
