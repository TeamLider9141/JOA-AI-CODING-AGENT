# Arrow-key confirm + fast (BM25-first) indexing

Date: 2026-07-13
Status: approved, not yet implemented

## Context

Measured on real hardware (5800U, CPU-only, `crystal_bot` repo: 20 files,
278 chunks): `build_index()` takes **67.45s**, of which **99.9% is
embedding** (`nomic-embed-text` via Ollama, sequential batches of 32).
Chunking is 0.09s; BM25 build + Qdrant upsert are both sub-second.
Parallelizing the embedding calls (2 or 4 worker threads) gave **no
speedup** — Ollama serves this model with `num_parallel=1` by default and
the model is already CPU-bound, so concurrent requests just queue.

Two separate user-facing problems this spec addresses:

1. `/joamodel`'s selection UI moved to an arrow-key menu (already shipped,
   commit `c5c120e`). The **index-now?** confirm (`_ensure_indexed`, via
   `typer.confirm`, y/N) and the **workspace-trust** screen
   (`_ensure_trusted`, via typed `"1"`/`"2"`) still require typing —
   inconsistent with the rest of the REPL's interactive UX now.
2. First-time indexing a repo blocks the REPL for up to ~70s (scales with
   chunk count) before you can ask anything, because embedding — the only
   slow step — runs synchronously before the prompt appears.

## Decisions (from brainstorming Q&A)

- Arrow-key confirm applies to **both** the index-now prompt and the
  workspace-trust screen — consistent UX, not just one of them.
- Indexing strategy: **BM25 built immediately (sync), vector index built
  in the background (automatic, no extra command)**. The REPL becomes
  usable in ~0.1s; semantic (vector) search phases in once the background
  build finishes. Rejected alternatives: BM25-only forever (loses semantic
  search permanently), manual `--vector` flag (extra step user has to
  remember), fastembed/ONNX (new dependency, unverified speedup, doesn't
  address the "blocks the REPL" problem on its own — still worth
  revisiting later but out of scope here).
- Re-indexing after file changes must stay cheap. Since BM25 rebuild is
  already ~0.1s, the only thing worth guarding is *not* re-running the
  67s vector embed when the repo hasn't actually changed.

## A. Arrow-key confirm (`_arrow_confirm`)

New helper in `assistant/cli.py`, built on the existing `_arrow_select`:

```python
def _arrow_confirm(question: str, echo, select=_arrow_select) -> bool:
    """Two-option arrow-key menu: Ha / Yo'q. `echo(question)` first,
    then the menu. Returns True for "Ha", False for "Yo'q" or a
    cancelled selection (Esc/Ctrl-C -> None from select)."""
    echo(question)
    index = select(["Ha", "Yo'q"], 0)
    return index == 0
```

Wiring (both call sites gate on `sys.stdin.isatty()`, matching the
existing guard pattern used throughout `cli.py`):

- `_ensure_indexed(repo, data_dir, embed_client, echo, confirm)` — the
  `confirm` param's contract doesn't change (still `Callable[[str], bool]`),
  so `repl()` passes `lambda msg: _arrow_confirm(msg, typer.echo)` when
  interactive, `typer.confirm` otherwise. **No signature change needed**
  in `_ensure_indexed` itself — this is a pure call-site swap in `repl()`.
- `_ensure_trusted(repo, read_line, echo, trust_path=...)` — gets an
  optional `select=None` parameter. When `select` is given, it renders the
  trust screen text via `echo` (unchanged) then replaces the final
  `"1"`/`"2"` numeric read with
  `_arrow_confirm("Ishonasizmi?", echo, select) `... concretely: the two
  option lines currently printed as
  `" 1. Ha, bu papkaga ishonaman"` / `" 2. Yo'q, chiqish"` are replaced by
  the arrow menu when `select` is provided; the surrounding banner/warning
  text is unchanged. `repl()` passes `select=_arrow_select` when
  `sys.stdin.isatty()`, else `None` (existing numeric path, keeps all
  current non-interactive tests green).

Both `_ensure_indexed`'s and `_ensure_trusted`'s **existing signatures
still accept plain callables**, so every existing test (which drives them
with fake `confirm`/`read_line` functions, no real terminal) keeps passing
unmodified. New tests inject a fake `select` to cover the arrow path,
mirroring how `/joamodel`'s arrow tests were written.

## B. BM25-first, vector-in-background indexing

### B.1 `assistant/indexer/pipeline.py` split

```python
def build_bm25_index(repo: Path, data_dir: Path) -> int:
    """Walk + chunk + BM25 build/save. No embedding — sub-second."""
    ...

def build_vector_index(repo: Path, data_dir: Path, embedder: Embedder) -> int:
    """Walk + chunk + embed + Qdrant upsert. The slow path."""
    ...

def build_index(repo: Path, data_dir: Path, embedder: Embedder) -> int:
    """Existing blocking behavior: BM25 + vector, sequentially.
    Kept for `joa index` (explicit CLI command) and as the function
    used when a caller wants a fully-synchronous, complete index."""
    n = build_bm25_index(repo, data_dir)
    build_vector_index(repo, data_dir, embedder)
    return n
```

Chunking (`walk_repo` + `chunk_file`) currently happens inline in
`build_index`; it gets extracted so both new functions call the same
chunk list without walking the repo twice per full build. The empty-repo
`ValueError("no indexable chunks found in {repo}")` is raised from
`build_bm25_index` (it's the one that always runs first) so
`_ensure_indexed`'s existing empty-repo bootstrap logic (catches that
exact message, creates `.joa-welcome.md`, retries) needs **no changes**.

### B.2 Vector staleness guard — `assistant/indexer/manifest.py` (new)

```python
def repo_fingerprint(repo: Path) -> dict[str, tuple[float, int]]:
    """path (relative, str) -> (mtime, size) for every indexable file,
    using the same walk_repo() filter as chunking."""

def load_manifest(data_dir: Path) -> dict | None:
    """Read vector_manifest.json; None if missing/corrupt (never raises)."""

def save_manifest(data_dir: Path, fingerprint: dict) -> None:
    """Write vector_manifest.json."""
```

Whole-repo fingerprint, not per-file incremental embedding — matches the
YAGNI framing agreed in brainstorming: the goal is "don't redo 67s of
work when nothing changed," not partial re-embedding of individual edited
files. If the fingerprint differs at all, the entire vector index is
rebuilt from scratch (same as today, just skipped when unnecessary).

### B.3 Background trigger — `assistant/cli.py`, `_ensure_indexed`

```python
def _ensure_indexed(repo, data_dir, embed_client, echo, confirm) -> bool:
    if (data_dir / "bm25.json").exists():
        _maybe_start_vector_background(repo, data_dir, embed_client, echo)
        return True
    if not confirm(...):
        ...  # unchanged
    echo(f"Indekslanmoqda: {repo} ...")
    try:
        n = build_bm25_index(repo, data_dir)
    except ValueError as exc:
        ...  # unchanged bootstrap-on-empty-repo logic, calls build_bm25_index on retry
    echo(f"✓ Indekslandi (BM25): {n} chunk")
    _maybe_start_vector_background(repo, data_dir, embed_client, echo)
    return True
```

`_maybe_start_vector_background(repo, data_dir, embed_client, echo)`:
- Computes `repo_fingerprint(repo)`, compares to `load_manifest(data_dir)`.
- If equal *and* `data_dir / "qdrant"` already exists → nothing to do
  (already up to date), returns immediately.
- Otherwise spawns `threading.Thread(target=_build_vector_background,
  args=(...), daemon=True).start()` and returns immediately — never blocks
  the REPL prompt.
- `daemon=True` so an interrupted/exited REPL doesn't hang on this thread.

`_build_vector_background(repo, data_dir, embed_client, echo)` (runs on
the background thread):
1. Builds into a **sibling temp path** `data_dir / "qdrant.new"` (a fresh
   `QdrantStore` there), not `data_dir / "qdrant"` directly — the embedded
   Qdrant client only allows one live client per path, and the foreground
   `search_index()` may open `data_dir / "qdrant"` concurrently while this
   is running. `build_vector_index` gains an optional
   `qdrant_dirname: str = "qdrant"` param so this path can be overridden
   without duplicating its logic.
2. On success: `os.replace` swaps `qdrant.new` → `qdrant` (atomic on the
   same filesystem), then `save_manifest(data_dir, fingerprint)`, then
   `echo("✓ Semantik qidiruv ham tayyor.")`.
3. On `OllamaError` (server down / model missing mid-build): swallow —
   log via `echo` (`"Semantik indekslash muvaffaqiyatsiz bo'ldi: {exc}"`)
   but leave the existing (or absent) `qdrant/` untouched; BM25 keeps
   working either way. No retry loop — next `joa` launch tries again
   (fingerprint won't match a saved manifest since none was saved).
4. `echo` calls from a background thread interleave with the REPL's own
   output; this is accepted as-is (same tradeoff the existing live
   `run_cmd`/`!bang` streaming already has) — no output-locking added,
   since REPL is a single-user local terminal, not a concern users are
   likely to notice given the message is short and infrequent.

### B.4 `assistant/indexer/pipeline.py` — `search_index` fallback

```python
def search_index(query, data_dir, embedder, mode="hybrid"):
    bm25 = BM25Store.load(data_dir / "bm25.json")
    bm25_results = bm25.search(query, config.BM25_TOP_K)
    if mode == "vector" and not (data_dir / "qdrant").exists():
        return []  # explicit vector-only request with no vector index yet
    if not (data_dir / "qdrant").exists():
        return bm25_results[:config.FINAL_TOP_K]
    qvec = embedder([query])[0]
    store = QdrantStore(data_dir / "qdrant")
    try:
        vector_results = store.search(qvec, config.VECTOR_TOP_K)
    finally:
        store.close()
    if mode == "vector":
        return vector_results[:config.FINAL_TOP_K]
    return rrf_merge([bm25_results, vector_results], k=config.RRF_K,
                     top_k=config.FINAL_TOP_K)
```

Existing behavior (both stores present) is unchanged; the only new branch
is "no `qdrant/` directory yet" → BM25-only, silently. No warning printed
on every search call (would be noisy) — the one-time background-thread
message from B.3 step 2/3 is the only signal.

### B.5 What stays untouched

- `joa index <repo>` (the standalone CLI command, non-REPL) keeps calling
  `build_index()` — fully synchronous, both stores guaranteed present when
  it returns. No behavior change there; it's for scripted/CI use where
  "returns immediately" isn't the goal.
- `ask`/`search`/`agent` commands: unaffected by the background-thread
  logic (that only lives in `repl()`'s `_ensure_indexed` call site);
  `search_index`'s new fallback benefits them too, automatically.

## Testing

- `test_pipeline.py`: `build_bm25_index` alone (no embedder arg needed,
  fast); `build_vector_index` alone against a fake embedder;
  `search_index` falls back to BM25-only when no `qdrant/` dir exists;
  `search_index` behaves exactly as before when both exist (regression).
- `test_manifest.py`: fingerprint stable across repeated calls on
  unchanged files; changes when a file's content/mtime/size changes or a
  file is added/removed; `load_manifest` returns `None` on missing/corrupt
  JSON (never raises).
- `test_repl.py` / `test_ensure_indexed.py`: `_maybe_start_vector_background`
  is called with a fake/no-op background-starter injected (so tests never
  spin up a real thread or call a real embedder) — verifies it's invoked
  exactly when expected (fresh index, and pre-existing-but-stale index),
  and skipped when the manifest already matches. `_arrow_confirm` tested
  directly with a fake `select`; `_ensure_trusted`'s new `select` param
  tested the same way (Ha/Yo'q/cancel), existing numeric-path tests
  untouched.
- Real end-to-end (manual, like every prior feature in this session): time
  from `joa` launch to first usable prompt on an unindexed repo (expect
  ~0.1s instead of ~67s); confirm a background "✓ Semantik qidiruv ham
  tayyor" message appears ~70s later; confirm a second `joa` launch on the
  same unchanged repo starts no background thread at all (manifest match).

## Docs

`README.md` / `assistant/README.md`: replace the "auto-index-on-missing"
description with the two-phase (BM25-instant, vector-background)
behavior; document the arrow-key trust/index confirms replacing typed
`1`/`y`; note `joa index` (standalone command) is unaffected and still
fully synchronous.
