# LLM Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in LLM-based reranker that reorders hybrid search's fused top-N candidates by asking the existing chat model to rank them, improving where the correct result lands (not just whether it's in the top 5) — with a safe fallback to the original order on any parse failure.

**Architecture:** A new pure function `rerank(query, candidates, chat_fn, top_k)` in `assistant/search/rerank.py`, injected into `search_index()` as an optional callable (same dependency-injection pattern as `embedder`). CLI gains a `--rerank` flag that wires `OllamaClient.chat` into it. The eval script gains a `hybrid+rerank` mode and an MRR metric so a saturated hit@5 doesn't hide the reranker's real effect.

**Tech Stack:** Python 3.10, existing `OllamaClient`, typer, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-06-llm-reranker-design.md`

---

## File Structure

All paths relative to repo root `/home/eaduinte/Desktop/system_llm`.

- `assistant/search/rerank.py` — CREATE: `rerank(query, candidates, chat_fn, top_k)`.
- `assistant/config.py` — MODIFY: add `RERANK_CANDIDATES`, `RERANK_TOP_K`.
- `assistant/indexer/pipeline.py` — MODIFY: `search_index` gains `reranker` param; raises the RRF candidate count when reranking so there's something to rerank.
- `assistant/cli.py` — MODIFY: `search` and `ask` gain `--rerank`.
- `assistant/eval/run_eval.py` — MODIFY: add `hybrid+rerank` mode and MRR.
- `assistant/tests/test_rerank.py` — CREATE.
- `assistant/tests/test_pipeline_rerank.py` — CREATE.
- `assistant/tests/test_eval_mrr.py` — CREATE.

Existing tests (`test_pipeline.py`, `test_cli.py`, `test_hybrid.py`) must stay green — all changes are additive/optional-parameter.

---

### Task 1: `rerank()` core function

**Files:**
- Create: `assistant/search/rerank.py`
- Test: `assistant/tests/test_rerank.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from assistant.search.rerank import rerank


def make_candidates():
    # (chunk_id, score, payload) — payload needs path/symbol/text for the prompt
    return [
        ("c1", 0.9, {"path": "a.py", "symbol": "f", "text": "def f(): pass"}),
        ("c2", 0.8, {"path": "b.py", "symbol": "g", "text": "def g(): pass"}),
        ("c3", 0.7, {"path": "c.py", "symbol": "h", "text": "def h(): pass"}),
    ]


def fake_chat(ranking):
    def chat_fn(messages):
        return json.dumps({"ranking": ranking})
    return chat_fn


def test_reorders_candidates_per_model_ranking():
    candidates = make_candidates()
    # model says: c3 (index 3) most relevant, then c1 (1), then c2 (2)
    result = rerank("q", candidates, fake_chat([3, 1, 2]), top_k=3)
    assert [c[0] for c in result] == ["c3", "c1", "c2"]


def test_truncates_to_top_k():
    candidates = make_candidates()
    result = rerank("q", candidates, fake_chat([1, 2, 3]), top_k=2)
    assert len(result) == 2
    assert [c[0] for c in result] == ["c1", "c2"]


def test_falls_back_on_malformed_json():
    candidates = make_candidates()
    result = rerank("q", candidates, lambda messages: "not json at all",
                    top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_missing_ranking_key():
    candidates = make_candidates()
    chat_fn = lambda messages: json.dumps({"oops": [1, 2, 3]})
    result = rerank("q", candidates, chat_fn, top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_non_permutation_ranking():
    candidates = make_candidates()
    # duplicate 1, missing 3 — not a valid permutation of 1..3
    result = rerank("q", candidates, fake_chat([1, 1, 2]), top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_falls_back_on_out_of_range_index():
    candidates = make_candidates()
    result = rerank("q", candidates, fake_chat([1, 2, 99]), top_k=3)
    assert [c[0] for c in result] == ["c1", "c2", "c3"]


def test_handles_fewer_candidates_than_top_k():
    candidates = make_candidates()[:2]
    result = rerank("q", candidates, fake_chat([2, 1]), top_k=5)
    assert [c[0] for c in result] == ["c2", "c1"]


def test_prompt_contains_query_and_every_candidate_path():
    candidates = make_candidates()
    seen_messages = []

    def chat_fn(messages):
        seen_messages.extend(messages)
        return json.dumps({"ranking": [1, 2, 3]})

    rerank("where is f defined", candidates, chat_fn, top_k=3)
    full_text = " ".join(m["content"] for m in seen_messages)
    assert "where is f defined" in full_text
    assert "a.py" in full_text
    assert "b.py" in full_text
    assert "c.py" in full_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'assistant.search.rerank'`

- [ ] **Step 3: Write `assistant/search/rerank.py`**

```python
import json

EXCERPT_CHARS = 160  # keep the listwise prompt small for ~20 candidates on CPU

RERANK_SYSTEM_PROMPT = (
    "You are ranking search results by relevance to a query. Reply with "
    "ONLY a JSON object: {\"ranking\": [n1, n2, ...]}, listing every result "
    "number exactly once, most relevant first. No other text."
)


def rerank(
    query: str,
    candidates: list[tuple[str, float, dict]],
    chat_fn,
    top_k: int = 5,
) -> list[tuple[str, float, dict]]:
    """Reorder `candidates` by asking `chat_fn` to rank them by relevance.

    Falls back to the original order (truncated to top_k) on any parse
    failure — reranking can only improve results, never make them worse
    than skipping it.
    """
    if not candidates:
        return []

    prompt = _build_prompt(query, candidates)
    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    reply = chat_fn(messages)

    order = _parse_ranking(reply, len(candidates))
    if order is None:
        return candidates[:top_k]

    return [candidates[i - 1] for i in order][:top_k]


def _build_prompt(query: str, candidates: list[tuple[str, float, dict]]) -> str:
    lines = [f"Query: {query}", "", "Results:"]
    for i, (_cid, _score, payload) in enumerate(candidates, start=1):
        excerpt = payload["text"][:EXCERPT_CHARS].replace("\n", " ")
        lines.append(
            f"{i}. {payload['path']}:{payload['symbol']} — {excerpt}")
    return "\n".join(lines)


def _parse_ranking(reply: str, n: int) -> list[int] | None:
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "ranking" not in data:
        return None
    ranking = data["ranking"]
    if not isinstance(ranking, list):
        return None
    if sorted(ranking) != list(range(1, n + 1)):
        return None  # not a valid permutation of 1..n
    return ranking
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_rerank.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/search/rerank.py assistant/tests/test_rerank.py
git commit -m "feat: add LLM-based reranker with fallback to original order"
```

---

### Task 2: Config additions

**Files:**
- Modify: `assistant/config.py`

- [ ] **Step 1: Add reranker settings**

Append to `assistant/config.py` after the `# --- Retrieval ---` block (before `# --- Paths ---`):

```python
RERANK_CANDIDATES = 20  # how many fused hybrid results to show the reranker
RERANK_TOP_K = 5        # how many the reranker keeps after reordering
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/bin/python -c "from assistant import config; print(config.RERANK_CANDIDATES, config.RERANK_TOP_K)"`
Expected: `20 5`

- [ ] **Step 3: Commit**

```bash
git add assistant/config.py
git commit -m "feat: add RERANK_CANDIDATES and RERANK_TOP_K config"
```

---

### Task 3: Wire reranker into `search_index`

**Files:**
- Modify: `assistant/indexer/pipeline.py`
- Test: `assistant/tests/test_pipeline_rerank.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from assistant.indexer.pipeline import build_index, search_index
from assistant.search.rerank import rerank


def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [[float(len(t)), 1.0, 0.5] for t in texts]


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class JWTMiddleware:\n    def check(self):\n        return True\n")
    (repo / "db.py").write_text(
        "def connect():\n    return 'engine'\n")
    return repo


def test_search_index_applies_reranker_when_provided(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    # a reranker that reverses whatever it's given, so we can prove it ran
    def reverse_chat(messages):
        # figure out how many candidates were shown from the prompt content
        prompt = messages[-1]["content"]
        n = prompt.count("\n") - 2  # "Query:", "", "Results:" header lines
        n = max(n, 1)
        return json.dumps({"ranking": list(range(n, 0, -1))})

    def reranker(query, candidates):
        return rerank(query, candidates, reverse_chat, top_k=len(candidates))

    without = search_index("JWTMiddleware", data, fake_embedder)
    with_rerank = search_index(
        "JWTMiddleware", data, fake_embedder, reranker=reranker)

    assert [c[0] for c in with_rerank] == list(reversed([c[0] for c in without]))


def test_search_index_unchanged_when_reranker_is_none(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    results = search_index("JWTMiddleware", data, fake_embedder)
    assert results[0][2]["path"] == "auth.py"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_pipeline_rerank.py -v`
Expected: FAIL — `TypeError: search_index() got an unexpected keyword argument 'reranker'`

- [ ] **Step 3: Modify `search_index` in `assistant/indexer/pipeline.py`**

Replace the existing `search_index` function (the last function in the file)
with:

```python
def search_index(
    query: str,
    data_dir: Path,
    embedder: Embedder,
    mode: str = "hybrid",
    reranker=None,
) -> list[tuple[str, float, dict]]:
    qvec = embedder([query])[0]
    store = QdrantStore(data_dir / "qdrant")
    vector_results = store.search(qvec, config.VECTOR_TOP_K)
    store.close()
    if mode == "vector":
        return vector_results[:config.FINAL_TOP_K]

    bm25 = BM25Store.load(data_dir / "bm25.json")
    bm25_results = bm25.search(query, config.BM25_TOP_K)
    # BM25 first: on an RRF score tie (symmetric rank swap between the two
    # retrievers), dict insertion order decides the winner. Exact lexical
    # matches should win those ties over vector-similarity noise.
    top_k = config.RERANK_CANDIDATES if reranker else config.FINAL_TOP_K
    fused = rrf_merge(
        [bm25_results, vector_results],
        k=config.RRF_K,
        top_k=top_k,
    )
    if reranker is None:
        return fused
    return reranker(query, fused)
```

- [ ] **Step 4: Run the new tests and the existing pipeline tests**

Run: `.venv/bin/pytest assistant/tests/test_pipeline_rerank.py assistant/tests/test_pipeline.py -v`
Expected: all pass. Existing pipeline tests are unaffected since `reranker`
defaults to `None`, preserving prior behavior exactly.

- [ ] **Step 5: Commit**

```bash
git add assistant/indexer/pipeline.py assistant/tests/test_pipeline_rerank.py
git commit -m "feat: add optional reranker parameter to search_index"
```

---

### Task 4: CLI `--rerank` flag

**Files:**
- Modify: `assistant/cli.py`

- [ ] **Step 1: Write the failing test**

Create `assistant/tests/test_cli_rerank.py`:

```python
from typer.testing import CliRunner

from assistant.cli import app

runner = CliRunner()


def test_search_help_mentions_rerank_flag():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "--rerank" in result.output


def test_ask_help_mentions_rerank_flag():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--rerank" in result.output
```

Run: `.venv/bin/pytest assistant/tests/test_cli_rerank.py -v`
Expected: FAIL — `--rerank` not in help output for either command.

- [ ] **Step 2: Add the import**

In `assistant/cli.py`, add this import after the existing
`from assistant.search... ` — there is no such import yet, so add it after
`from assistant.indexer.pipeline import build_index, search_index`:

```python
from assistant.search.rerank import rerank
```

- [ ] **Step 3: Update the `search` command**

Replace the existing `search` command function with:

```python
@app.command()
def search(
    query: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    mode: str = typer.Option("hybrid", help="hybrid | vector"),
    rerank_flag: bool = typer.Option(
        False, "--rerank", help="rerank results with the LLM"),
):
    """Search the index and print matching chunks (debug view)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    reranker = _make_reranker(client) if rerank_flag else None
    try:
        results = search_index(
            query, data_dir, client.embed, mode=mode, reranker=reranker)
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    for _chunk_id, score, p in results:
        typer.echo(
            f"{score:.4f}  {p['path']}:{p['start_line']}-{p['end_line']}"
            f"  {p['symbol']}")
```

- [ ] **Step 4: Update the `ask` command**

Replace the existing `ask` command function with:

```python
@app.command()
def ask(
    question: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    rerank_flag: bool = typer.Option(
        False, "--rerank", help="rerank results with the LLM"),
):
    """Ask a question about the indexed repository."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    reranker = _make_reranker(client) if rerank_flag else None
    try:
        results = search_index(
            question, data_dir, client.embed, reranker=reranker)
        typer.echo("--- sources ---")
        for _chunk_id, _score, p in results:
            typer.echo(
                f"  {p['path']}:{p['start_line']}-{p['end_line']}"
                f"  {p['symbol']}")
        typer.echo("--- answer ---")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, results)},
        ]
        for token in client.chat_stream(messages):
            typer.echo(token, nl=False)
        typer.echo()
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
```

- [ ] **Step 5: Add the `_make_reranker` helper**

Add this function right after `_require_index` (before the `index` command):

```python
def _make_reranker(client: OllamaClient):
    def reranker(query: str, candidates: list) -> list:
        return rerank(query, candidates, client.chat,
                      top_k=config.RERANK_TOP_K)
    return reranker
```

- [ ] **Step 6: Run the rerank CLI tests and the full existing CLI tests**

Run: `.venv/bin/pytest assistant/tests/test_cli_rerank.py assistant/tests/test_cli.py assistant/tests/test_cli_agent.py -v`
Expected: all pass.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (78 pre-existing + 9 + 2 + 2 = 91).

- [ ] **Step 8: Commit**

```bash
git add assistant/cli.py assistant/tests/test_cli_rerank.py
git commit -m "feat: add --rerank flag to search and ask commands"
```

---

### Task 5: Eval — MRR metric and `hybrid+rerank` mode

**Files:**
- Modify: `assistant/eval/run_eval.py`
- Test: `assistant/tests/test_eval_mrr.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.eval.run_eval import compute_mrr


def test_mrr_full_credit_when_match_is_first():
    results = [
        {"path": "a.py"}, {"path": "b.py"}, {"path": "c.py"},
    ]
    assert compute_mrr(results, "a.py") == 1.0


def test_mrr_partial_credit_when_match_is_third():
    results = [
        {"path": "x.py"}, {"path": "y.py"}, {"path": "a.py"},
    ]
    assert compute_mrr(results, "a.py") == 1 / 3


def test_mrr_zero_when_no_match():
    results = [{"path": "x.py"}, {"path": "y.py"}]
    assert compute_mrr(results, "a.py") == 0.0


def test_mrr_matches_by_substring_like_hit_at_5():
    results = [{"path": "handlers/admin.py"}]
    assert compute_mrr(results, "handlers/admin.py") == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_eval_mrr.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_mrr'`

- [ ] **Step 3: Rewrite `assistant/eval/run_eval.py`**

Replace the whole file with:

```python
"""Retrieval quality eval: hit@5 and MRR across vector / hybrid / hybrid+rerank.

Usage:
    .venv/bin/python -m assistant.eval.run_eval --repo ~/Desktop/crystal_bot
"""
from pathlib import Path

import typer
import yaml

from assistant import config
from assistant.indexer.pipeline import search_index
from assistant.llm.ollama_client import OllamaClient
from assistant.search.rerank import rerank

GOLD_PATH = Path(__file__).parent / "gold.yaml"
MODES = ("vector", "hybrid", "hybrid+rerank")


def compute_mrr(payloads: list[dict], expect_path_contains: str) -> float:
    """1/rank of the first matching payload (1-indexed); 0 if no match."""
    for rank, payload in enumerate(payloads, start=1):
        if expect_path_contains in payload["path"]:
            return 1.0 / rank
    return 0.0


def main(repo: Path = typer.Option(..., "--repo", exists=True)):
    gold = yaml.safe_load(GOLD_PATH.read_text())
    data_dir = config.DATA_DIR / repo.resolve().name
    client = OllamaClient()

    def reranker(query: str, candidates: list) -> list:
        return rerank(query, candidates, client.chat,
                      top_k=config.RERANK_TOP_K)

    for mode in MODES:
        search_mode = "vector" if mode == "vector" else "hybrid"
        use_reranker = reranker if mode == "hybrid+rerank" else None

        hits = 0
        mrr_total = 0.0
        for item in gold:
            results = search_index(
                item["question"], data_dir, client.embed,
                mode=search_mode, reranker=use_reranker)
            payloads = [p for _cid, _s, p in results[:5]]
            paths = [p["path"] for p in payloads]
            if any(item["expect_path_contains"] in path for path in paths):
                hits += 1
            mrr_total += compute_mrr(payloads, item["expect_path_contains"])

        n = len(gold)
        print(f"{mode:14s} hit@5: {hits}/{n}   MRR: {mrr_total / n:.3f}")


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 4: Run the new MRR tests**

Run: `.venv/bin/pytest assistant/tests/test_eval_mrr.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (91 pre-existing + 4 = 95).

- [ ] **Step 6: Commit**

```bash
git add assistant/eval/run_eval.py assistant/tests/test_eval_mrr.py
git commit -m "feat: add MRR metric and hybrid+rerank mode to eval"
```

---

### Task 6: Real run — measure the reranker honestly

Run the eval against a real indexed repo and report the numbers as they
actually come out — including if reranking shows no improvement on this
small corpus (the spec explicitly anticipates and accepts this outcome).

**Use whichever repo is already indexed from prior work** (`crystal_bot`, per
`assistant/.data/crystal_bot/`). Do not index a new external project without
asking the user first.

- [ ] **Step 1: Confirm the existing index is present**

```bash
ls assistant/.data/crystal_bot/bm25.json
```

Expected: file exists. If not, ask the user before indexing anything.

- [ ] **Step 2: Run the eval**

```bash
.venv/bin/python -m assistant.eval.run_eval --repo /home/eaduinte/Desktop/crystal_bot
```

Expected output shape (numbers will vary — report what actually prints):

```
vector         hit@5: 10/10   MRR: 0.850
hybrid         hit@5: 10/10   MRR: 0.900
hybrid+rerank  hit@5: 10/10   MRR: 0.XXX
```

This step is slow — one LLM chat call per gold question for the rerank row
(CPU, ~30-60s each).

- [ ] **Step 3: Update `assistant/README.md`**

Add a new subsection right after the existing `## Agent` section content
(after the "Still deferred (separate plan): a cross-encoder reranker..."
line — replace that line with the paragraph below, since the reranker is no
longer deferred):

```markdown
## Reranker

    .venv/bin/python -m assistant.cli search "query" --repo <repo-path> --rerank
    .venv/bin/python -m assistant.cli ask "question" --repo <repo-path> --rerank

An opt-in LLM reranker (`assistant/search/rerank.py`) asks the chat model to
reorder hybrid search's fused top-20 candidates by relevance in a single
listwise call, then keeps the top 5. Any parse failure (malformed JSON, not a
valid permutation) falls back to the original order — reranking can only
help, never hurt. Measured with `assistant/eval/run_eval.py`, which reports
both hit@5 and MRR (mean reciprocal rank of the first correct result) across
vector / hybrid / hybrid+rerank, since hit@5 alone saturates on small repos
and can't show a reranker's real effect. Actual measured result on
`crystal_bot`: **[fill in from the Task 6 Step 2 run — report the real
numbers here, including if rerank showed no improvement]**.
```

Replace the bracketed sentence with the actual numbers from Step 2 before
committing — do not leave it as a placeholder.

- [ ] **Step 4: Commit**

```bash
git add assistant/README.md
git commit -m "docs: document reranker and report measured eval results"
```

---

## Self-Review Notes

- **Spec coverage:** `rerank()` module (Task 1), config additions (Task 2),
  `search_index` integration (Task 3), CLI `--rerank` (Task 4), eval MRR +
  `hybrid+rerank` mode (Task 5), honest real-world measurement (Task 6) — all
  spec sections have a task.
- **Fallback semantics:** matches the spec's "never worse than skipping it" —
  `_parse_ranking` requires an exact permutation of `1..n`; any deviation
  (wrong key, wrong type, duplicate/missing/out-of-range index) returns
  `None`, and `rerank` falls back to `candidates[:top_k]`.
- **OllamaError propagation:** unchanged from existing behavior — `chat_fn`
  is `client.chat`, which already raises `OllamaError` on a dead server; nothing
  in `rerank` catches it, so it propagates to the CLI's existing
  `except OllamaError` handler exactly as embedding/chat failures do today.
