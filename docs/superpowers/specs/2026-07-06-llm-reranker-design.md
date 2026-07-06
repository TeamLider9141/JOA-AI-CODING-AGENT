# LLM Reranker Design Spec

**Date:** 2026-07-06
**Status:** Approved (design reviewed in session)
**Builds on:** `docs/superpowers/plans/2026-07-05-assistant-retrieval-core.md` (hybrid retrieval)

## Goal

Improve ranking quality of the hybrid (vector + BM25 + RRF) search results by
asking the existing chat model to reorder the top candidates by actual
relevance to the query. This is the last item deferred from the original
design spec (`docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md`,
Phase 4).

The goal is not just "the right file appears in the top 5" (the current gold
eval already scores 10/10 hit@5 on `crystal_bot` — that metric is saturated
and won't show a reranker's benefit). The goal is **improving the rank of the
right result** — pushing a correct-but-buried match from #4 up to #1.

## Decisions (made during brainstorming)

1. **Approach: LLM-as-reranker**, using the existing `qwen2.5-coder:7b` chat
   model via a single listwise prompt, rather than a trained cross-encoder
   (e.g. `sentence-transformers` + `ms-marco-MiniLM`). Chosen because it adds
   no new dependency (no `torch`, no ~2GB pull), stays consistent with the
   project's hand-written-core ethos, and teaches how LLM-based reranking
   actually works. Trade-off accepted: slower (one CPU chat call per query,
   ~30-60s) and likely less precise than a purpose-trained cross-encoder.
2. **Measurement: add MRR alongside hit@5.** Because hit@5 is already
   saturated on the current gold set, a reranker that only reshuffles within
   the top 5 would show no hit@5 change even if it meaningfully improves
   ranking. Mean Reciprocal Rank (average of 1/rank of the first correct
   result) is added to the eval so a real rank improvement is visible, and so
   the eval can honestly report if the reranker provides no benefit on this
   small corpus.

## Constraints

- CPU-only, no new dependencies — reuses `OllamaClient.chat`.
- Reranking must never make results worse than doing nothing: any parse
  failure or malformed model output falls back to the pre-rerank order.

## Architecture

```
hybrid search (existing) → top-20 candidates
                                  │
                    rerank(query, candidates, chat_fn)
                                  │
                    LLM ranks candidates by relevance
                    (single listwise chat call, JSON response)
                                  │
                    parse ranking → reorder candidates
                    (any failure → return original order unchanged)
                                  │
                            top-5 final results
```

## Components

### `assistant/search/rerank.py` (new)

```python
def rerank(query, candidates, chat_fn, top_k=5) -> list[candidate]
```

- `candidates`: the existing `list[tuple[str, float, dict]]` shape used
  throughout the retrieval pipeline (chunk_id, score, payload).
- `chat_fn`: a `(messages) -> str` callable — the CLI will pass
  `OllamaClient.chat` (or a wrapper); the reranker module itself has no
  dependency on `OllamaClient` directly, matching how `embedder` is injected
  elsewhere in the pipeline.
- Builds a single prompt: the query, followed by each candidate numbered
  1..N with `path:symbol` and a short text excerpt (~160 chars — long enough
  to judge relevance, short enough to keep the CPU prompt small for N=20).
  Asks the model to return `{"ranking": [3, 1, 7, ...]}` — a permutation of
  the candidate numbers, most relevant first.
- Parses the JSON, validates it's a permutation of `1..len(candidates)` (or a
  usable subset — see Error Handling), and returns candidates reordered
  accordingly, truncated to `top_k`.
- **Fallback is the core safety property:** if the model's output can't be
  parsed into a valid ranking, `rerank` returns `candidates[:top_k]` in their
  original (pre-rerank) order. The reranker can only help; it can never make
  results worse than skipping it.

### Pipeline integration

`search_index(query, data_dir, embedder, mode="hybrid", reranker=None)` gains
an optional `reranker` parameter: a `(query, candidates) -> candidates`
callable, applied to the fused hybrid results before the final top-k slice.
When `reranker` is `None` (the default), behavior is unchanged from today.

### CLI

`search` and `ask` gain a `--rerank` flag. When set, a reranker callable
(closing over `OllamaClient.chat`) is constructed and passed to
`search_index`. Without the flag, behavior is unchanged — reranking is opt-in
given its CPU cost.

### Config

`assistant/config.py` gains:
- `RERANK_CANDIDATES = 20` — how many fused hybrid results are shown to the
  reranker (must be ≥ `FINAL_TOP_K` used elsewhere; RRF's existing `top_k` for
  this call is raised to this value when reranking is active).
- `RERANK_TOP_K = 5` — how many the reranker keeps after reordering.

### Eval: MRR alongside hit@5

`assistant/eval/run_eval.py` adds a `hybrid+rerank` mode alongside the
existing `vector`/`hybrid` modes, and computes **MRR** (mean of `1/rank` of
the first result whose path matches `expect_path_contains`, across the gold
set; 0 if no match in the candidate list) in addition to hit@5. This makes a
reranker's actual value — or lack of it, on a small corpus — visible and
honestly reported rather than hidden by a saturated hit@5 metric.

## Data flow

```
query → embed → vector search (top VECTOR_TOP_K)
      → BM25 search (top BM25_TOP_K)
      → RRF merge (top RERANK_CANDIDATES if reranking, else FINAL_TOP_K)
      → [rerank if requested] → top RERANK_TOP_K / FINAL_TOP_K
```

## Error handling

- Model returns non-JSON, JSON without a `"ranking"` key, a ranking that
  isn't a valid permutation of the candidate indices, or an empty response →
  `rerank` logs nothing fatal and returns the original top-`top_k` candidates
  unchanged.
- `OllamaError` (server down) during reranking is **not** swallowed — it
  propagates, consistent with how `embed`/`chat` failures are handled
  elsewhere in the CLI (clear error, non-zero exit), since a dead Ollama
  server is a real failure the user needs to know about, not a "just skip
  reranking" situation.
- Fewer candidates than `top_k` → return all of them, reordered.

## Testing

- `rerank` reorders candidates according to a fake `chat_fn`'s ranking.
- `rerank` falls back to original order when `chat_fn` returns malformed
  JSON, non-permutation indices, or empty output.
- `rerank` handles `len(candidates) < top_k` without error.
- The prompt built by `rerank` contains the query and every candidate's path.
- `search_index` applies the `reranker` callable when provided, and is
  unchanged when it is not.
- Eval's MRR calculation is correct for known rank positions (first match at
  rank 1 → MRR contribution 1.0; at rank 3 → 1/3; no match → 0).

## Out of scope

Trained cross-encoder reranking, reranking for the agent's `search_code`
tool (kept simple/fast for the agent loop), any change to the RRF merge
logic itself, batching multiple queries into one rerank call.
