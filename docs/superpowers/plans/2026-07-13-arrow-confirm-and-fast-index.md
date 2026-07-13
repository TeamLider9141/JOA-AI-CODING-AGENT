# Arrow-key confirm + BM25-first background indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the last two typed-input prompts in `joa` (index-now confirm, workspace-trust confirm) with the existing arrow-key menu, and make first-time indexing return control to the REPL in ~0.1s instead of ~67s by building BM25 synchronously and the vector (semantic) index in a background thread.

**Architecture:** `assistant/indexer/pipeline.py` splits into `build_bm25_index` (fast, no embedding) and `build_vector_index` (slow, the existing embedding path); `build_index` becomes a thin wrapper calling both for the standalone `joa index` command. A new `assistant/indexer/manifest.py` fingerprints a repo's files (path→mtime/size) so a background vector rebuild is skipped when nothing changed since the last one. `assistant/cli.py` gains `_arrow_confirm` (built on the already-shipped `_arrow_select`) and wires it into `_ensure_trusted` and `_ensure_indexed`; `_ensure_indexed` now builds BM25 only and kicks off vector indexing via an injectable `start_vector_background` callable, defaulting to a real daemon-thread starter in production and a fake in tests.

**Tech Stack:** Python, prompt_toolkit (already a dependency), threading (stdlib), existing BM25Store/QdrantStore.

**Reference spec:** `docs/superpowers/specs/2026-07-13-arrow-confirm-and-fast-index-design.md`

---

## Baseline

Full current file contents you'll be editing are reproduced inline in each task below — you don't need to go hunting for context. Run the full suite once before starting to confirm a clean baseline:

```bash
cd /home/eaduinte/Desktop/system_llm
.venv/bin/python -m pytest -q
```

Expected: `193 passed`.

All work happens on branch `feature/retrieval-core` (already checked out and in sync with `main` as of commit `111e993`). Every task ends with a commit; do not switch branches mid-plan. After the final task, fast-forward `main` and push both branches (Task 9).

---

### Task 1: Repo fingerprint manifest (`assistant/indexer/manifest.py`)

**Files:**
- Create: `assistant/indexer/manifest.py`
- Test: `assistant/tests/test_manifest.py`

This is the staleness guard the background vector build uses to skip re-embedding an unchanged repo. It has no dependency on anything else in this plan, so it goes first.

- [ ] **Step 1: Write the failing tests**

Create `assistant/tests/test_manifest.py`:

```python
from assistant.indexer.manifest import (
    load_manifest, repo_fingerprint, save_manifest,
)


def test_fingerprint_stable_across_repeated_calls(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")

    fp1 = repo_fingerprint(repo)
    fp2 = repo_fingerprint(repo)

    assert fp1 == fp2


def test_fingerprint_changes_when_file_content_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "a.py"
    f.write_text("x = 1\n")
    fp1 = repo_fingerprint(repo)

    f.write_text("x = 2222222\n")  # different size, not just mtime
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2


def test_fingerprint_changes_when_file_added(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    fp1 = repo_fingerprint(repo)

    (repo / "b.py").write_text("y = 2\n")
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2
    assert "b.py" in fp2


def test_fingerprint_changes_when_file_removed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    fp1 = repo_fingerprint(repo)

    (repo / "b.py").unlink()
    fp2 = repo_fingerprint(repo)

    assert fp1 != fp2
    assert "b.py" not in fp2


def test_load_manifest_missing_file_returns_none(tmp_path):
    assert load_manifest(tmp_path) is None


def test_load_manifest_corrupt_json_returns_none(tmp_path):
    (tmp_path / "vector_manifest.json").write_text("not json{{{")
    assert load_manifest(tmp_path) is None


def test_save_then_load_round_trips(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    data_dir = tmp_path / "data"

    fp = repo_fingerprint(repo)
    save_manifest(data_dir, fp)

    assert load_manifest(data_dir) == fp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_manifest.py -v`
Expected: `ModuleNotFoundError: No module named 'assistant.indexer.manifest'` (or collection error) — the module doesn't exist yet.

- [ ] **Step 3: Write the implementation**

Create `assistant/indexer/manifest.py`:

```python
import json
from pathlib import Path

from assistant.indexer.walker import walk_repo

MANIFEST_FILENAME = "vector_manifest.json"


def repo_fingerprint(repo: Path) -> dict:
    """{relative_path: [mtime, size]} for every file walk_repo() would
    index right now. JSON-serializable — used directly as the on-disk
    manifest and compared for equality to detect any change (content,
    add, remove) since the last vector build."""
    fingerprint: dict[str, list] = {}
    for path in walk_repo(repo):
        rel = str(path.relative_to(repo))
        stat = path.stat()
        fingerprint[rel] = [stat.st_mtime, stat.st_size]
    return fingerprint


def load_manifest(data_dir: Path) -> dict | None:
    path = data_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(data_dir: Path, fingerprint: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MANIFEST_FILENAME).write_text(json.dumps(fingerprint))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest assistant/tests/test_manifest.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/indexer/manifest.py assistant/tests/test_manifest.py
git commit -m "feat: add repo fingerprint manifest for vector-index staleness checks"
```

---

### Task 2: Split `build_index` into `build_bm25_index` + `build_vector_index`

**Files:**
- Modify: `assistant/indexer/pipeline.py` (full file, shown below)
- Modify: `assistant/tests/test_pipeline.py`

Current `assistant/indexer/pipeline.py` in full (for reference — you're replacing this):

```python
import sys
import time
from collections.abc import Callable
from pathlib import Path

from assistant import config
from assistant.indexer.chunker import chunk_file
from assistant.indexer.models import Chunk
from assistant.indexer.walker import walk_repo
from assistant.llm.ollama_client import OllamaError
from assistant.search.hybrid import rrf_merge
from assistant.store.bm25_store import BM25Store
from assistant.store.qdrant_store import QdrantStore

Embedder = Callable[[list[str]], list[list[float]]]

BATCH_SIZE = 32


def build_index(repo: Path, data_dir: Path, embedder: Embedder) -> int:
    files = walk_repo(repo)
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(f, repo))
    if not chunks:
        raise ValueError(f"no indexable chunks found in {repo}")

    kept: list[Chunk] = []
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [f"{c.path} {c.symbol}\n{c.text}" for c in batch]
        result = _embed_with_retry(embedder, texts)
        if result is None:
            print(f"warning: skipped {len(batch)} chunks "
                  f"(embedding failed after retries)", file=sys.stderr)
            continue
        kept.extend(batch)
        vectors.extend(result)

    if not kept:
        raise ValueError("all embedding batches failed — nothing indexed")

    store = QdrantStore(data_dir / "qdrant")
    store.reset(dim=len(vectors[0]))
    store.upsert(kept, vectors)
    store.close()

    bm25 = BM25Store()
    bm25.build(kept)
    bm25.save(data_dir / "bm25.json")
    return len(kept)


def _embed_with_retry(embedder: Embedder, texts: list[str],
                      attempts: int = 3):
    for attempt in range(attempts):
        try:
            return embedder(texts)
        except OllamaError:
            raise
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def search_index(
    query: str,
    data_dir: Path,
    embedder: Embedder,
    mode: str = "hybrid",
) -> list[tuple[str, float, dict]]:
    qvec = embedder([query])[0]
    store = QdrantStore(data_dir / "qdrant")
    vector_results = store.search(qvec, config.VECTOR_TOP_K)
    store.close()
    if mode == "vector":
        return vector_results[:config.FINAL_TOP_K]

    bm25 = BM25Store.load(data_dir / "bm25.json")
    bm25_results = bm25.search(query, config.BM25_TOP_K)
    return rrf_merge(
        [bm25_results, vector_results],
        k=config.RRF_K,
        top_k=config.FINAL_TOP_K,
    )
```

Note: `search_index`'s BM25-fallback change is Task 3, not this task — this task only splits the build side. `_collect_chunks` is a new private helper both `build_bm25_index` and `build_vector_index` call; each does its own `walk_repo`+`chunk_file` pass. That means `build_index` (used only by the standalone `joa index` command) now walks/chunks the repo twice instead of once — measured cost is ~0.09s for a 278-chunk repo, negligible next to the 67s embedding step, so this is accepted for the simplicity of having two fully independent, separately-testable public functions instead of a shared chunk-passing internal API.

- [ ] **Step 1: Write the failing tests**

Add to `assistant/tests/test_pipeline.py` (append — keep all existing tests and the existing `import pytest` / `fake_embedder` / `make_repo` at the top):

```python
from assistant.indexer.pipeline import build_bm25_index, build_vector_index
```

Add this import line to the existing `from assistant.indexer.pipeline import build_index, search_index` line so it becomes:

```python
from assistant.indexer.pipeline import (
    build_bm25_index, build_index, build_vector_index, search_index,
)
```

Then append these new test functions to the end of the file:

```python
def test_build_bm25_index_creates_bm25_only(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_bm25_index(repo, data)

    assert n >= 2
    assert (data / "bm25.json").exists()
    assert not (data / "qdrant").exists()


def test_build_bm25_index_empty_repo_raises(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    with pytest.raises(ValueError, match="no indexable chunks"):
        build_bm25_index(repo, tmp_path / "data")


def test_build_vector_index_creates_qdrant_only(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_vector_index(repo, data, fake_embedder)

    assert n >= 2
    assert (data / "qdrant").is_dir()
    assert not (data / "bm25.json").exists()


def test_build_vector_index_respects_qdrant_dirname(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    build_vector_index(repo, data, fake_embedder, qdrant_dirname="qdrant.new")

    assert (data / "qdrant.new").is_dir()
    assert not (data / "qdrant").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_pipeline.py -v`
Expected: `ImportError: cannot import name 'build_bm25_index'`

- [ ] **Step 3: Replace `assistant/indexer/pipeline.py` with the split implementation**

```python
import sys
import time
from collections.abc import Callable
from pathlib import Path

from assistant import config
from assistant.indexer.chunker import chunk_file
from assistant.indexer.models import Chunk
from assistant.indexer.walker import walk_repo
from assistant.llm.ollama_client import OllamaError
from assistant.search.hybrid import rrf_merge
from assistant.store.bm25_store import BM25Store
from assistant.store.qdrant_store import QdrantStore

Embedder = Callable[[list[str]], list[list[float]]]

BATCH_SIZE = 32


def _collect_chunks(repo: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in walk_repo(repo):
        chunks.extend(chunk_file(f, repo))
    if not chunks:
        raise ValueError(f"no indexable chunks found in {repo}")
    return chunks


def build_bm25_index(repo: Path, data_dir: Path) -> int:
    """Walk + chunk + BM25 build/save. No embedding call — sub-second
    even on large repos, since it's pure CPU tokenization."""
    chunks = _collect_chunks(repo)
    bm25 = BM25Store()
    bm25.build(chunks)
    bm25.save(data_dir / "bm25.json")
    return len(chunks)


def build_vector_index(repo: Path, data_dir: Path, embedder: Embedder,
                       qdrant_dirname: str = "qdrant") -> int:
    """Walk + chunk + embed + Qdrant upsert — the slow path (embedding
    dominates: ~99.9% of total time on CPU-only hardware). `qdrant_dirname`
    lets a caller build into a temp directory (e.g. "qdrant.new") and swap
    it in atomically once complete, so a concurrent reader of the live
    "qdrant" directory is never disturbed mid-build."""
    chunks = _collect_chunks(repo)
    kept: list[Chunk] = []
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [f"{c.path} {c.symbol}\n{c.text}" for c in batch]
        result = _embed_with_retry(embedder, texts)
        if result is None:
            print(f"warning: skipped {len(batch)} chunks "
                  f"(embedding failed after retries)", file=sys.stderr)
            continue
        kept.extend(batch)
        vectors.extend(result)

    if not kept:
        raise ValueError("all embedding batches failed — nothing indexed")

    store = QdrantStore(data_dir / qdrant_dirname)
    store.reset(dim=len(vectors[0]))
    store.upsert(kept, vectors)
    store.close()
    return len(kept)


def build_index(repo: Path, data_dir: Path, embedder: Embedder) -> int:
    """Full synchronous build: BM25 then vector, in sequence. Used by the
    standalone `joa index` CLI command, where blocking until both stores
    are ready is the desired behavior (unlike the REPL's background-vector
    flow in `assistant/cli.py`)."""
    n = build_bm25_index(repo, data_dir)
    build_vector_index(repo, data_dir, embedder)
    return n


def _embed_with_retry(embedder: Embedder, texts: list[str],
                      attempts: int = 3):
    for attempt in range(attempts):
        try:
            return embedder(texts)
        except OllamaError:
            raise  # server down / model missing: abort the whole run
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def search_index(
    query: str,
    data_dir: Path,
    embedder: Embedder,
    mode: str = "hybrid",
) -> list[tuple[str, float, dict]]:
    qvec = embedder([query])[0]
    store = QdrantStore(data_dir / "qdrant")
    vector_results = store.search(qvec, config.VECTOR_TOP_K)
    store.close()
    if mode == "vector":
        return vector_results[:config.FINAL_TOP_K]

    bm25 = BM25Store.load(data_dir / "bm25.json")
    bm25_results = bm25.search(query, config.BM25_TOP_K)
    return rrf_merge(
        [bm25_results, vector_results],
        k=config.RRF_K,
        top_k=config.FINAL_TOP_K,
    )
```

(`search_index` is unchanged from the current file in this step — the BM25-fallback behavior is Task 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest assistant/tests/test_pipeline.py -v`
Expected: all tests pass, including the pre-existing `test_build_index_persists_both_stores`, `test_search_index_hybrid_finds_exact_identifier`, `test_search_index_vector_mode_returns_results`, `test_empty_repo_raises`, `test_ollama_error_aborts_build` (unchanged — `build_index` still raises `OllamaError` from `build_vector_index`).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `.venv/bin/python -m pytest -q`
Expected: `204 passed` (193 baseline + 7 manifest + 4 new pipeline tests)

- [ ] **Step 6: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/indexer/pipeline.py assistant/tests/test_pipeline.py
git commit -m "refactor: split build_index into build_bm25_index + build_vector_index"
```

---

### Task 3: `search_index` falls back to BM25-only when no vector index exists

**Files:**
- Modify: `assistant/indexer/pipeline.py:search_index` (function from Task 2)
- Modify: `assistant/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `assistant/tests/test_pipeline.py`:

```python
def test_search_index_falls_back_to_bm25_only_without_qdrant(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)  # no vector index built at all

    results = search_index("JWTMiddleware", data, fake_embedder)

    assert results, "expected at least one result"
    assert results[0][2]["path"] == "auth.py"


def test_search_index_vector_mode_without_qdrant_returns_empty(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)

    results = search_index("anything", data, fake_embedder, mode="vector")

    assert results == []


def test_search_index_hybrid_still_uses_vector_when_present(tmp_path):
    """Regression guard: once both stores exist, behavior is unchanged
    from before this task (hybrid RRF merge of both)."""
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_bm25_index(repo, data)
    build_vector_index(repo, data, fake_embedder)

    results = search_index("JWTMiddleware", data, fake_embedder)

    assert results
    assert results[0][2]["path"] == "auth.py"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_pipeline.py -k "bm25_only or vector_mode_without or hybrid_still" -v`
Expected: `test_search_index_falls_back_to_bm25_only_without_qdrant` and `test_search_index_vector_mode_without_qdrant_returns_empty` FAIL — current `search_index` unconditionally opens `QdrantStore(data_dir / "qdrant")`, which raises because the directory was never created (embedded Qdrant client errors on an empty/missing collection path in this codebase's usage — confirm by reading the actual error in the test output, but the expected failure mode is an exception, not a silent empty result).

- [ ] **Step 3: Update `search_index` in `assistant/indexer/pipeline.py`**

Replace the existing `search_index` function (the one from Task 2, unchanged since) with:

```python
def search_index(
    query: str,
    data_dir: Path,
    embedder: Embedder,
    mode: str = "hybrid",
) -> list[tuple[str, float, dict]]:
    has_vector = (data_dir / "qdrant").exists()
    if mode == "vector" and not has_vector:
        return []

    bm25 = BM25Store.load(data_dir / "bm25.json")
    bm25_results = bm25.search(query, config.BM25_TOP_K)
    if not has_vector:
        return bm25_results[:config.FINAL_TOP_K]

    qvec = embedder([query])[0]
    store = QdrantStore(data_dir / "qdrant")
    vector_results = store.search(qvec, config.VECTOR_TOP_K)
    store.close()
    if mode == "vector":
        return vector_results[:config.FINAL_TOP_K]

    # BM25 first: on an RRF score tie (symmetric rank swap between the two
    # retrievers), dict insertion order decides the winner. Exact lexical
    # matches should win those ties over vector-similarity noise.
    return rrf_merge(
        [bm25_results, vector_results],
        k=config.RRF_K,
        top_k=config.FINAL_TOP_K,
    )
```

(The tie-breaking comment is carried over from the original file's git history — preserve it since it documents a non-obvious ordering decision.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest assistant/tests/test_pipeline.py -v`
Expected: all pass (existing + 3 new).

- [ ] **Step 5: Full suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: `207 passed`

- [ ] **Step 6: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/indexer/pipeline.py assistant/tests/test_pipeline.py
git commit -m "feat: search_index falls back to BM25-only when no vector index exists yet"
```

---

### Task 4: `_arrow_confirm` helper

**Files:**
- Modify: `assistant/cli.py` (add function after `_arrow_select`, currently ending at line 371)
- Modify: `assistant/tests/test_repl.py`

- [ ] **Step 1: Write the failing tests**

Append to `assistant/tests/test_repl.py` (the file already imports `from assistant.cli import app, _repl_loop` at the top — these new tests import `_arrow_confirm` directly at point of use since it's a one-off, matching how the file's existing `_arrow_select` tests do it):

```python
def test_arrow_confirm_ha_returns_true_and_echoes_question():
    from assistant.cli import _arrow_confirm

    out = []
    result = _arrow_confirm("Davom etamizmi?", out.append,
                            select=lambda options, current: 0)

    assert result is True
    assert out == ["Davom etamizmi?"]


def test_arrow_confirm_yoq_returns_false():
    from assistant.cli import _arrow_confirm

    result = _arrow_confirm("Davom etamizmi?", lambda _o: None,
                            select=lambda options, current: 1)

    assert result is False


def test_arrow_confirm_cancelled_returns_false():
    from assistant.cli import _arrow_confirm

    result = _arrow_confirm("Davom etamizmi?", lambda _o: None,
                            select=lambda options, current: None)

    assert result is False


def test_arrow_confirm_options_are_ha_yoq_in_order():
    from assistant.cli import _arrow_confirm

    seen = {}

    def fake_select(options, current_index):
        seen["options"] = options
        seen["current_index"] = current_index
        return 0

    _arrow_confirm("Q?", lambda _o: None, select=fake_select)

    assert seen["options"] == ["Ha", "Yo'q"]
    assert seen["current_index"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_repl.py -k arrow_confirm -v`
Expected: `ImportError: cannot import name '_arrow_confirm'`

- [ ] **Step 3: Add `_arrow_confirm` to `assistant/cli.py`**

Insert immediately after the `_arrow_select` function (which currently ends at line 371 with `return app.run()`), before `def _handle_joamodel(...)`:

```python
def _arrow_confirm(question: str, echo, select=_arrow_select) -> bool:
    """Two-option arrow-key menu: Ha / Yo'q. Prints `question` via `echo`
    first, then the menu. Returns True for "Ha", False for "Yo'q" or a
    cancelled selection (Esc/Ctrl-C, which `select` reports as None)."""
    echo(question)
    index = select(["Ha", "Yo'q"], 0)
    return index == 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest assistant/tests/test_repl.py -k arrow_confirm -v`
Expected: `4 passed`

- [ ] **Step 5: Full suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: `211 passed`

- [ ] **Step 6: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/cli.py assistant/tests/test_repl.py
git commit -m "feat: add _arrow_confirm (Ha/Yo'q arrow-key menu) built on _arrow_select"
```

---

### Task 5: Wire `_arrow_confirm` into `_ensure_trusted`

**Files:**
- Modify: `assistant/cli.py:171-202` (the `_ensure_trusted` function)
- Modify: `assistant/cli.py:527-571` (the `repl()` command — trust call site)
- Modify: `assistant/tests/test_trust.py`

Current `_ensure_trusted` (lines 171-202 of `assistant/cli.py`):

```python
def _ensure_trusted(repo: Path, read_line, echo,
                    trust_path: Path = config.TRUST_FILE) -> bool:
    """Ask the user to trust `repo` (like Claude Code's workspace-trust
    screen), unless it's already trusted. Returns True to proceed, False
    to abort. A "1" answer is remembered in `trust_path`; anything else
    (including EOF) is treated as decline and never saved."""
    resolved = str(repo.resolve())
    trusted = _load_trusted(trust_path)
    if resolved in trusted:
        return True
    echo("─" * 60)
    echo(" JOA — workspace'ga kirish:")
    echo("")
    echo(f"   {resolved}")
    echo("")
    echo(" Xavfsizlik tekshiruvi: bu papka o'zingiz yaratgan yoki")
    echo(" ishonchli loyihami? JOA bu yerda fayllarni o'qiy, tahrirlay")
    echo(" va buyruq bajara oladi.")
    echo("")
    echo(" 1. Ha, bu papkaga ishonaman")
    echo(" 2. Yo'q, chiqish")
    echo("─" * 60)
    echo("Raqamni tanlang:")
    try:
        choice = read_line().strip()
    except EOFError:
        return False
    if choice != "1":
        return False
    trusted.add(resolved)
    _save_trusted(trusted, trust_path)
    return True
```

- [ ] **Step 1: Write the failing tests**

Append to `assistant/tests/test_trust.py` (existing top-of-file import `from assistant.cli import _ensure_trusted, _load_trusted, _save_trusted` stays as-is):

```python
def _unused_read_line():
    raise AssertionError("arrow path must not read numeric input")


def test_ensure_trusted_arrow_accept_saves_and_returns_true(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _ensure_trusted(repo, _unused_read_line, lambda _o: None,
                             trust_path=trust_path,
                             select=lambda options, current: 0)

    assert result is True
    assert str(repo.resolve()) in _load_trusted(trust_path)


def test_ensure_trusted_arrow_decline_returns_false_and_does_not_save(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _ensure_trusted(repo, _unused_read_line, lambda _o: None,
                             trust_path=trust_path,
                             select=lambda options, current: 1)

    assert result is False
    assert _load_trusted(trust_path) == set()


def test_ensure_trusted_arrow_cancel_returns_false(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _ensure_trusted(repo, _unused_read_line, lambda _o: None,
                             trust_path=trust_path,
                             select=lambda options, current: None)

    assert result is False


def test_ensure_trusted_arrow_skips_prompt_for_known_dir(tmp_path):
    trust_path = tmp_path / "trusted_dirs.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    _save_trusted({str(repo.resolve())}, trust_path)

    def boom_select(options, current):
        raise AssertionError("should not prompt for an already-trusted dir")

    result = _ensure_trusted(repo, _unused_read_line, lambda _o: None,
                             trust_path=trust_path, select=boom_select)
    assert result is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_trust.py -k arrow -v`
Expected: `TypeError: _ensure_trusted() got an unexpected keyword argument 'select'`

- [ ] **Step 3: Update `_ensure_trusted` in `assistant/cli.py`**

Replace the function (lines 171-202) with:

```python
def _ensure_trusted(repo: Path, read_line, echo,
                    trust_path: Path = config.TRUST_FILE,
                    select=None) -> bool:
    """Ask the user to trust `repo` (like Claude Code's workspace-trust
    screen), unless it's already trusted. Returns True to proceed, False
    to abort. Interactive terminals get an arrow-key Ha/Yo'q menu (pass
    `select`, e.g. `_arrow_select`); piped/scripted input falls back to
    typing "1" (anything else, including EOF, is decline). Only "Ha" /
    typed "1" is remembered in `trust_path`."""
    resolved = str(repo.resolve())
    trusted = _load_trusted(trust_path)
    if resolved in trusted:
        return True
    echo("─" * 60)
    echo(" JOA — workspace'ga kirish:")
    echo("")
    echo(f"   {resolved}")
    echo("")
    echo(" Xavfsizlik tekshiruvi: bu papka o'zingiz yaratgan yoki")
    echo(" ishonchli loyihami? JOA bu yerda fayllarni o'qiy, tahrirlay")
    echo(" va buyruq bajara oladi.")
    echo("")
    if select is not None:
        echo("─" * 60)
        trust = _arrow_confirm("Bu papkaga ishonasizmi?", echo, select)
    else:
        echo(" 1. Ha, bu papkaga ishonaman")
        echo(" 2. Yo'q, chiqish")
        echo("─" * 60)
        echo("Raqamni tanlang:")
        try:
            choice = read_line().strip()
        except EOFError:
            return False
        trust = choice == "1"
    if not trust:
        return False
    trusted.add(resolved)
    _save_trusted(trusted, trust_path)
    return True
```

Note: `_arrow_confirm` is defined earlier in the file (Task 4), before `_ensure_trusted` — no reordering needed, `_ensure_trusted` already sits after it in the current file (`_ensure_trusted` is at line 171, `_arrow_select`/`_arrow_confirm` will be at ~300-378 — wait, check actual order).

**Important ordering check:** in the current file, `_ensure_trusted` (line 171) is defined *before* `_arrow_select` (line 330) and `_arrow_confirm` (added in Task 4, right after `_arrow_select`). Since Python resolves names inside a function body at *call* time, not at *definition* time, `_ensure_trusted` referencing `_arrow_confirm` before it's defined lower in the file is fine — by the time `_ensure_trusted` is actually called (from `repl()`, which is defined even later), `_arrow_confirm` already exists at module scope. No reordering of the file is required.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest assistant/tests/test_trust.py -v`
Expected: all pass — the 4 new arrow tests plus all pre-existing numeric-path tests (`test_ensure_trusted_skips_prompt_for_known_dir`, `test_ensure_trusted_accept_saves_and_returns_true`, `test_ensure_trusted_decline_returns_false_and_does_not_save`, `test_ensure_trusted_eof_returns_false`, `test_ensure_trusted_unknown_input_returns_false`, plus the `_load_trusted`/`_save_trusted` tests) — none of those pass `select`, so they exercise the unchanged `select is None` branch.

- [ ] **Step 5: Wire `repl()`'s trust call site to pass `select` when interactive**

Current call site (`assistant/cli.py`, inside `repl()`):

```python
    typer.secho(JOA_BANNER, fg=typer.colors.BLUE)
    if sys.stdin.isatty():
        if not _ensure_trusted(repo, lambda: input(""), typer.echo):
            raise typer.Exit(0)
```

Replace with:

```python
    typer.secho(JOA_BANNER, fg=typer.colors.BLUE)
    interactive = sys.stdin.isatty()
    select = _arrow_select if interactive else None
    if interactive:
        if not _ensure_trusted(repo, lambda: input(""), typer.echo,
                               select=select):
            raise typer.Exit(0)
```

(This introduces the `interactive`/`select` locals that Task 7 will also reuse further down in the same function — don't remove them after this step. Task 6, which comes next, does not touch `repl()` at all — it only touches `_ensure_indexed` and the module's imports.)

- [ ] **Step 6: Run the repl command tests**

Run: `.venv/bin/python -m pytest assistant/tests/test_repl.py -k "repl_command or repl_without_index" -v`
Expected: `2 passed` (these use `CliRunner`, which runs with non-interactive stdin, so `interactive` is `False` and behavior is unchanged).

- [ ] **Step 7: Full suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: `215 passed`

- [ ] **Step 8: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/cli.py assistant/tests/test_trust.py
git commit -m "feat: workspace-trust screen uses arrow-key Ha/Yo'q menu in interactive terminals"
```

---

### Task 6: BM25-first + background vector indexing in `_ensure_indexed`

**Files:**
- Modify: `assistant/cli.py:1-23` (imports)
- Modify: `assistant/cli.py:112-150` (the `_ensure_indexed` function)
- Modify: `assistant/tests/test_ensure_indexed.py`

This is the core of the speed fix. Current `_ensure_indexed` (lines 112-150):

```python
def _ensure_indexed(repo: Path, data_dir: Path, embed_client, echo,
                    confirm) -> bool:
    """If `repo` has no index yet, ask (via `confirm`) whether to build
    one now. Returns True once an index exists (already did, or just
    built), False if the user declined or the build itself failed."""
    if (data_dir / "bm25.json").exists():
        return True
    if not confirm(f"'{repo}' indekslanmagan. Hozir indekslaymanmi?"):
        echo("No index found. Run first: python -m assistant.cli index <repo>")
        return False
    echo(f"Indekslanmoqda: {repo} ...")
    try:
        n = build_index(repo, data_dir, embed_client.embed)
    except ValueError as exc:
        if "no indexable chunks found" not in str(exc):
            echo(f"Indekslash muvaffaqiyatsiz bo'ldi: {exc}")
            return False
        placeholder = repo / ".joa-welcome.md"
        placeholder.write_text(
            "# JOA\n\n"
            "Bu papka bo'sh edi — JOA birinchi ishga tushishda shu faylni "
            "avtomatik yaratdi (indekslash uchun kamida bitta fayl kerak). "
            "Xohlasangiz o'chirib, o'z fayllaringizni qo'shishingiz "
            "mumkin.\n")
        echo(f"Papka bo'sh edi — {placeholder.name} avtomatik yaratildi.")
        try:
            n = build_index(repo, data_dir, embed_client.embed)
        except (OllamaError, ValueError) as retry_exc:
            echo(f"Indekslash muvaffaqiyatsiz bo'ldi: {retry_exc}")
            return False
    except OllamaError as exc:
        echo(f"Indekslash muvaffaqiyatsiz bo'ldi: {exc}")
        return False
    echo(f"✓ Indekslandi: {n} chunk")
    return True
```

**Behavior change this task intentionally makes:** `build_index` (BM25 + vector, blocking) is replaced with `build_bm25_index` (BM25 only). Because `build_bm25_index` never calls the embedder, `_ensure_indexed` can no longer fail with an `OllamaError` from indexing — BM25 succeeds even if Ollama is completely unreachable. The `except OllamaError` branch (both outer and in the empty-repo retry) is therefore removed — it would be dead code. Ollama failures now only affect the background vector build (handled separately, see Step 3's `_build_vector_background`), and don't block `_ensure_indexed`'s synchronous return.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `assistant/tests/test_ensure_indexed.py` with:

```python
from assistant.cli import _ensure_indexed
from assistant.llm.ollama_client import OllamaError


class FakeEmbedClient:
    def embed(self, texts):
        return [[0.0] for _ in texts]


def _no_op_background(*_args, **_kwargs):
    """Injected in place of the real background-thread starter so these
    unit tests never spin a real thread or touch a real embedder."""


def test_ensure_indexed_returns_true_when_index_already_exists(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    data_dir.mkdir()
    (data_dir / "bm25.json").write_text("{}")

    def confirm(_msg):
        raise AssertionError("should not prompt when index already exists")

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             lambda _o: None, confirm,
                             start_vector_background=_no_op_background)
    assert result is True


def test_ensure_indexed_declined_returns_false_without_indexing(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: False,
                             start_vector_background=_no_op_background)
    assert result is False
    assert not (data_dir / "bm25.json").exists()
    assert any("no index found" in o.lower() for o in out)


def test_ensure_indexed_accepted_builds_bm25_and_returns_true(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, lambda _msg: True,
                             start_vector_background=_no_op_background)

    assert result is True
    assert (data_dir / "bm25.json").exists()
    assert not (data_dir / "qdrant").exists()  # vector build was skipped
    assert any("indekslandi" in o.lower() for o in out)


def test_ensure_indexed_bootstraps_placeholder_when_repo_is_empty(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    out = []
    confirm_calls = []

    def confirm(msg):
        confirm_calls.append(msg)
        return True

    result = _ensure_indexed(repo, data_dir, FakeEmbedClient(),
                             out.append, confirm,
                             start_vector_background=_no_op_background)

    assert result is True
    assert len(confirm_calls) == 1
    assert (data_dir / "bm25.json").exists()
    placeholder = repo / ".joa-welcome.md"
    assert placeholder.exists()
    assert any("bo'sh" in o.lower() for o in out)


def test_ensure_indexed_succeeds_even_when_ollama_is_down(tmp_path):
    """BM25 doesn't call the embedder at all — indexing succeeds
    synchronously even if Ollama is unreachable. Only the background
    vector build would be affected by that (covered in
    test_build_vector_background_ollama_failure_leaves_no_qdrant_dir)."""
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    out = []

    class BoomEmbedClient:
        def embed(self, texts):
            raise OllamaError("ollama is down")

    result = _ensure_indexed(repo, data_dir, BoomEmbedClient(),
                             out.append, lambda _msg: True,
                             start_vector_background=_no_op_background)

    assert result is True
    assert (data_dir / "bm25.json").exists()


def test_ensure_indexed_triggers_background_vector_start(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    calls = []

    def fake_background(repo_arg, data_dir_arg, embed_client_arg, echo_arg):
        calls.append((repo_arg, data_dir_arg))

    embed_client = FakeEmbedClient()
    _ensure_indexed(repo, data_dir, embed_client, lambda _o: None,
                    lambda _msg: True, start_vector_background=fake_background)

    assert calls == [(repo, data_dir)]


def test_ensure_indexed_already_indexed_still_triggers_background_check(tmp_path):
    """Even when BM25 already exists (no confirm prompt), the background
    starter must still be given a chance to run — it's the one that
    decides (via the manifest) whether a vector rebuild is needed."""
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    data_dir.mkdir()
    (data_dir / "bm25.json").write_text("{}")
    calls = []

    def fake_background(repo_arg, data_dir_arg, embed_client_arg, echo_arg):
        calls.append((repo_arg, data_dir_arg))

    def confirm(_msg):
        raise AssertionError("should not prompt when index already exists")

    _ensure_indexed(repo, data_dir, FakeEmbedClient(), lambda _o: None,
                    confirm, start_vector_background=fake_background)

    assert calls == [(repo, data_dir)]
```

(The old `test_ensure_indexed_ollama_failure_does_not_bootstrap` is intentionally removed — its assumption, that an `OllamaError` during indexing makes `_ensure_indexed` return `False`, is no longer true now that BM25 doesn't touch the embedder. Its replacement is `test_ensure_indexed_succeeds_even_when_ollama_is_down` above, plus a new direct test of `_build_vector_background`'s own Ollama-failure handling in Step 6 below.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest assistant/tests/test_ensure_indexed.py -v`
Expected: `TypeError: _ensure_indexed() got an unexpected keyword argument 'start_vector_background'` (all tests fail at call time).

- [ ] **Step 3: Update imports in `assistant/cli.py`**

Current top-of-file imports (lines 1-23):

```python
import json
import sys
import time
from enum import Enum
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from assistant import config
from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError
from assistant.agent.runner import AgentSession, run_agent
from assistant.agent.tools import ToolContext
from assistant.agent.proc import run_streaming
```

Replace with:

```python
import json
import os
import shutil
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from assistant import config
from assistant.indexer.manifest import (
    load_manifest, repo_fingerprint, save_manifest,
)
from assistant.indexer.pipeline import (
    build_bm25_index, build_index, build_vector_index, search_index,
)
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError
from assistant.agent.runner import AgentSession, run_agent
from assistant.agent.tools import ToolContext
from assistant.agent.proc import run_streaming
```

- [ ] **Step 4: Replace `_ensure_indexed` in `assistant/cli.py`**

Replace the function (lines 112-150 in the original) with:

```python
def _ensure_indexed(repo: Path, data_dir: Path, embed_client, echo,
                    confirm, start_vector_background=None) -> bool:
    """If `repo` has no BM25 index yet, ask (via `confirm`) whether to
    build one now. BM25 builds synchronously (sub-second — no embedding
    call). The vector (semantic) index is always given a chance to start
    afterward via `start_vector_background(repo, data_dir, embed_client,
    echo)` — production callers get `_maybe_start_vector_background` (a
    real background thread, skipped if the repo is unchanged since the
    last build); tests inject a no-op/fake so no real thread or embedder
    call ever happens in the unit test suite. Returns True once a BM25
    index exists (already did, or just built), False if the user
    declined or the BM25 build itself failed."""
    if start_vector_background is None:
        start_vector_background = _maybe_start_vector_background
    if (data_dir / "bm25.json").exists():
        start_vector_background(repo, data_dir, embed_client, echo)
        return True
    if not confirm(f"'{repo}' indekslanmagan. Hozir indekslaymanmi?"):
        echo("No index found. Run first: python -m assistant.cli index <repo>")
        return False
    echo(f"Indekslanmoqda: {repo} ...")
    try:
        n = build_bm25_index(repo, data_dir)
    except ValueError as exc:
        if "no indexable chunks found" not in str(exc):
            echo(f"Indekslash muvaffaqiyatsiz bo'ldi: {exc}")
            return False
        placeholder = repo / ".joa-welcome.md"
        placeholder.write_text(
            "# JOA\n\n"
            "Bu papka bo'sh edi — JOA birinchi ishga tushishda shu faylni "
            "avtomatik yaratdi (indekslash uchun kamida bitta fayl kerak). "
            "Xohlasangiz o'chirib, o'z fayllaringizni qo'shishingiz "
            "mumkin.\n")
        echo(f"Papka bo'sh edi — {placeholder.name} avtomatik yaratildi.")
        try:
            n = build_bm25_index(repo, data_dir)
        except ValueError as retry_exc:
            echo(f"Indekslash muvaffaqiyatsiz bo'ldi: {retry_exc}")
            return False
    echo(f"✓ Indekslandi (BM25): {n} chunk")
    start_vector_background(repo, data_dir, embed_client, echo)
    return True
```

- [ ] **Step 5: Add `_maybe_start_vector_background` and `_build_vector_background` to `assistant/cli.py`**

Insert these two new functions immediately after `_ensure_indexed` (before `_load_trusted`):

```python
def _maybe_start_vector_background(repo: Path, data_dir: Path, embed_client,
                                   echo) -> None:
    """Kick off vector (semantic) indexing in a background daemon thread,
    unless the repo is unchanged since the last successful vector build
    (per the saved fingerprint manifest) — in which case do nothing.
    Never blocks the caller either way."""
    fingerprint = repo_fingerprint(repo)
    if (load_manifest(data_dir) == fingerprint
            and (data_dir / "qdrant").is_dir()):
        return
    threading.Thread(
        target=_build_vector_background,
        args=(repo, data_dir, embed_client, fingerprint, echo),
        daemon=True,
    ).start()


def _build_vector_background(repo: Path, data_dir: Path, embed_client,
                             fingerprint: dict, echo) -> None:
    """Runs on the background thread started by
    `_maybe_start_vector_background`. Builds into a temp directory first
    (embedded Qdrant only allows one live client per path, and the
    foreground `search_index()` may be reading the live "qdrant"
    directory concurrently) then atomically swaps it in on success."""
    tmp_dirname = "qdrant.new"
    try:
        build_vector_index(repo, data_dir, embed_client.embed,
                           qdrant_dirname=tmp_dirname)
    except OllamaError as exc:
        echo(f"Semantik indekslash muvaffaqiyatsiz bo'ldi: {exc}")
        shutil.rmtree(data_dir / tmp_dirname, ignore_errors=True)
        return
    final_path = data_dir / "qdrant"
    if final_path.exists():
        shutil.rmtree(final_path)
    os.replace(data_dir / tmp_dirname, final_path)
    save_manifest(data_dir, fingerprint)
    echo("✓ Semantik qidiruv ham tayyor.")
```

- [ ] **Step 6: Add tests for `_maybe_start_vector_background` and `_build_vector_background`**

Append to `assistant/tests/test_ensure_indexed.py`:

```python
def test_maybe_start_vector_background_skips_when_manifest_matches(
        tmp_path, monkeypatch):
    from assistant.cli import _maybe_start_vector_background
    from assistant.indexer.manifest import repo_fingerprint, save_manifest

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    (data_dir / "qdrant").mkdir(parents=True)
    save_manifest(data_dir, repo_fingerprint(repo))

    def boom(*_args, **_kwargs):
        raise AssertionError("should not start a thread when nothing changed")

    monkeypatch.setattr("assistant.cli.threading.Thread", boom)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)


def test_maybe_start_vector_background_starts_thread_when_stale(
        tmp_path, monkeypatch):
    from assistant.cli import _maybe_start_vector_background

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    data_dir.mkdir()
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr("assistant.cli.threading.Thread", FakeThread)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)

    assert len(started) == 1
    assert started[0][2] is True  # daemon=True


def test_maybe_start_vector_background_starts_thread_when_qdrant_missing(
        tmp_path, monkeypatch):
    """Manifest could match (e.g. copied data dir) but if qdrant/ itself
    isn't there, a rebuild is still required."""
    from assistant.cli import _maybe_start_vector_background
    from assistant.indexer.manifest import repo_fingerprint, save_manifest

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    save_manifest(data_dir, repo_fingerprint(repo))  # no qdrant/ dir made
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr("assistant.cli.threading.Thread", FakeThread)

    _maybe_start_vector_background(repo, data_dir, FakeEmbedClient(),
                                   lambda _o: None)

    assert len(started) == 1


def test_build_vector_background_success_swaps_in_qdrant_and_saves_manifest(
        tmp_path):
    from assistant.cli import _build_vector_background
    from assistant.indexer.manifest import load_manifest, repo_fingerprint

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")

    class RealisticEmbedClient:
        def embed(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    out = []
    fingerprint = repo_fingerprint(repo)

    _build_vector_background(repo, data_dir, RealisticEmbedClient(),
                             fingerprint, out.append)

    assert (data_dir / "qdrant").is_dir()
    assert not (data_dir / "qdrant.new").exists()
    assert load_manifest(data_dir) == fingerprint
    assert any("tayyor" in o.lower() for o in out)


def test_build_vector_background_ollama_failure_leaves_no_qdrant_dir(
        tmp_path):
    from assistant.cli import _build_vector_background

    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    data_dir.mkdir()
    out = []

    class BoomEmbedClient:
        def embed(self, texts):
            raise OllamaError("ollama is down")

    _build_vector_background(repo, data_dir, BoomEmbedClient(), {}, out.append)

    assert not (data_dir / "qdrant").exists()
    assert not (data_dir / "qdrant.new").exists()
    assert any("ollama is down" in o.lower() for o in out)
```

- [ ] **Step 7: Run all `_ensure_indexed`/background tests**

Run: `.venv/bin/python -m pytest assistant/tests/test_ensure_indexed.py -v`
Expected: `12 passed` (7 from Step 1 + 5 from Step 6).

- [ ] **Step 8: Full suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, zero failures. (The old `test_ensure_indexed.py` had 5 tests; it's now 12 — a net +7 on top of the 215 total from Task 5. Trust `pytest`'s actual printed count over any arithmetic here — the only thing that matters is zero failed/errored.)

- [ ] **Step 9: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/cli.py assistant/tests/test_ensure_indexed.py
git commit -m "feat: BM25 builds synchronously, vector index builds in a background thread"
```

---

### Task 7: Wire arrow-confirm into `repl()`'s index-now prompt

**Files:**
- Modify: `assistant/cli.py:527-571` (the `repl()` command)

By this point, Task 5's Step 5 already introduced `interactive` and `select` locals near the top of `repl()`. This task uses them for the index confirm too, and passes `select` through to `_repl_loop` (replacing the old standalone `select = _arrow_select if sys.stdin.isatty() else None` line that currently sits right before the `_repl_loop` call).

Current full `repl()` command body (after Task 5's Step 5 edit, before this task):

```python
@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
    backend: Backend = typer.Option(
        Backend.ollama, "--backend",
        help="ollama | gemini (gemini needs GEMINI_API_KEY in .env)"),
):
    """Interactive agent session over the repo (defaults to current dir)."""
    typer.secho(JOA_BANNER, fg=typer.colors.BLUE)
    interactive = sys.stdin.isatty()
    select = _arrow_select if interactive else None
    if interactive:
        if not _ensure_trusted(repo, lambda: input(""), typer.echo,
                               select=select):
            raise typer.Exit(0)
    data_dir = _data_dir(repo)
    embed_client = OllamaClient()
    if sys.stdin.isatty():
        if not _ensure_indexed(repo, data_dir, embed_client, typer.echo,
                               typer.confirm):
            raise typer.Exit(1)
    else:
        _require_index(data_dir)
    try:
        chat_client = _chat_client(backend)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=embed_client.embed,
        confirm=lambda msg: typer.confirm(msg),
        output_sink=lambda t: typer.echo(t, nl=False),
    )
    session = AgentSession(ctx, chat_client)
    if sys.stdin.isatty():
        prompt_session = PromptSession(
            "joa> ", completer=SlashCompleter(),
            complete_while_typing=True)
        read_line = prompt_session.prompt
    else:
        # piped/scripted input: plain input(), no interactive dropdown
        read_line = lambda: input("joa> ")  # noqa: E731
    select = _arrow_select if sys.stdin.isatty() else None
    _repl_loop(session, read_line, typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False), select=select)
```

- [ ] **Step 1: Replace the index-confirm block and drop the now-redundant `select` recomputation**

Replace:

```python
    data_dir = _data_dir(repo)
    embed_client = OllamaClient()
    if sys.stdin.isatty():
        if not _ensure_indexed(repo, data_dir, embed_client, typer.echo,
                               typer.confirm):
            raise typer.Exit(1)
    else:
        _require_index(data_dir)
```

with:

```python
    data_dir = _data_dir(repo)
    embed_client = OllamaClient()
    if interactive:
        confirm = lambda msg: _arrow_confirm(msg, typer.echo, select)  # noqa: E731
        if not _ensure_indexed(repo, data_dir, embed_client, typer.echo,
                               confirm):
            raise typer.Exit(1)
    else:
        _require_index(data_dir)
```

And replace:

```python
    select = _arrow_select if sys.stdin.isatty() else None
    _repl_loop(session, read_line, typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False), select=select)
```

with (drop the redundant recomputation — `select` was already set once near the top of the function):

```python
    _repl_loop(session, read_line, typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False), select=select)
```

Also replace the two remaining `sys.stdin.isatty()` calls further down (the `PromptSession` branch) with `interactive` for consistency:

```python
    session = AgentSession(ctx, chat_client)
    if interactive:
        prompt_session = PromptSession(
            "joa> ", completer=SlashCompleter(),
            complete_while_typing=True)
        read_line = prompt_session.prompt
    else:
        # piped/scripted input: plain input(), no interactive dropdown
        read_line = lambda: input("joa> ")  # noqa: E731
```

Final full `repl()` body after this task:

```python
@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
    backend: Backend = typer.Option(
        Backend.ollama, "--backend",
        help="ollama | gemini (gemini needs GEMINI_API_KEY in .env)"),
):
    """Interactive agent session over the repo (defaults to current dir)."""
    typer.secho(JOA_BANNER, fg=typer.colors.BLUE)
    interactive = sys.stdin.isatty()
    select = _arrow_select if interactive else None
    if interactive:
        if not _ensure_trusted(repo, lambda: input(""), typer.echo,
                               select=select):
            raise typer.Exit(0)
    data_dir = _data_dir(repo)
    embed_client = OllamaClient()
    if interactive:
        confirm = lambda msg: _arrow_confirm(msg, typer.echo, select)  # noqa: E731
        if not _ensure_indexed(repo, data_dir, embed_client, typer.echo,
                               confirm):
            raise typer.Exit(1)
    else:
        _require_index(data_dir)
    try:
        chat_client = _chat_client(backend)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=embed_client.embed,
        confirm=lambda msg: typer.confirm(msg),
        output_sink=lambda t: typer.echo(t, nl=False),
    )
    session = AgentSession(ctx, chat_client)
    if interactive:
        prompt_session = PromptSession(
            "joa> ", completer=SlashCompleter(),
            complete_while_typing=True)
        read_line = prompt_session.prompt
    else:
        # piped/scripted input: plain input(), no interactive dropdown
        read_line = lambda: input("joa> ")  # noqa: E731
    _repl_loop(session, read_line, typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False), select=select)
```

- [ ] **Step 2: Run the repl command tests**

Run: `.venv/bin/python -m pytest assistant/tests/test_repl.py -v`
Expected: all pass, including `test_repl_command_is_registered` and `test_repl_without_index_exits_nonzero` (both use `CliRunner`, non-interactive, so `interactive` is `False` — same code path as before this task, `_require_index` unchanged).

- [ ] **Step 3: Full suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, count unchanged from Task 6 (no new tests in this task — it's pure call-site wiring, exercised by the existing `CliRunner` tests plus manual verification in Task 9).

- [ ] **Step 4: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add assistant/cli.py
git commit -m "feat: wire arrow-key confirm into repl()'s index-now prompt"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `assistant/README.md`

- [ ] **Step 1: Update `README.md`**

Find this paragraph (added in the previous session's work, currently right after the "Tezkor o'rnatish" code block):

```
`joa` yangi papkada birinchi marta ochilganda Claude Code'dagidek
workspace-trust ekrani chiqadi — bir marta tasdiqlaysiz, keyin
o'sha papka uchun so'ramaydi. Papka hali indekslanmagan bo'lsa, shu yerda
hoziroq indekslashni ham taklif qiladi (`~` kabi katta/aralash papkalar
uchun emas — aniq loyiha papkasi uchun mo'ljallangan).
```

Replace with:

```
`joa` yangi papkada birinchi marta ochilganda Claude Code'dagidek
workspace-trust ekrani chiqadi — strelka tugmalari bilan (↑/↓, Enter)
"Ha"/"Yo'q" tanlaysiz, keyin o'sha papka uchun qayta so'ramaydi. Papka
hali indekslanmagan bo'lsa, shu yerda hoziroq indekslashni ham taklif
qiladi — xuddi shu arrow-key menyu bilan (`~` kabi katta/aralash papkalar
uchun emas — aniq loyiha papkasi uchun mo'ljallangan).

Indekslash ikki bosqichda ishlaydi: **leksik (BM25) qism darhol** quriladi
(taxminan 0.1s — REPL shu zahoti ishlatishga tayyor), **semantik (vektor)
qism esa fonda**, sizga xalaqit bermay. Fon tugagach
`✓ Semantik qidiruv ham tayyor.` deb xabar chiqadi — o'shangacha qidiruv
faqat leksik (aniq so'z) ishlaydi. Fayllar o'zgarmagan bo'lsa, keyingi
`joa` ochilishlarida semantik qism qayta qurilmaydi (avtomatik aniqlanadi).
```

- [ ] **Step 2: Update `assistant/README.md`**

Find this paragraph:

```
The first time `joa` runs in a given directory (interactive terminal
only), it shows a Claude Code-style workspace-trust prompt before
touching anything — accept once and that directory is remembered in
`~/.config/joa/trusted_dirs.json`, no more prompts for it. If the
directory hasn't been indexed yet, `joa` offers to index it right there
(interactive only) instead of just erroring — declining, or running
non-interactively, falls back to the old `No index found` message.
```

Replace with:

```
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
```

- [ ] **Step 3: Commit**

```bash
cd /home/eaduinte/Desktop/system_llm
git add README.md assistant/README.md
git commit -m "docs: document arrow-key confirms and two-phase (BM25-first) indexing"
```

---

### Task 9: Full verification, merge, push

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/eaduinte/Desktop/system_llm
.venv/bin/python -m pytest -q
```

Expected: all tests pass, zero failures. Note the final count for your own sanity-check but don't block on matching the exact numbers estimated in earlier tasks — what matters is 0 failed/errored.

- [ ] **Step 2: Live smoke test — unindexed repo, timing**

```bash
cd /home/eaduinte/Desktop/system_llm
rm -rf /tmp/joa-speed-test && mkdir -p /tmp/joa-speed-test
cp assistant/indexer/*.py /tmp/joa-speed-test/  # a few real files to index
time (echo -e "exit" | .venv/bin/python -m assistant.cli repl --repo /tmp/joa-speed-test)
```

This runs non-interactively (piped stdin), so it takes the `_require_index` path on first run and exits 1 (expected — no index exists yet and non-tty never offers to build one, matching existing behavior). This step is really about confirming the command runs without crashing after all the refactoring; the actual interactive timing check is Step 3.

- [ ] **Step 3: Live smoke test — interactive, via pty (arrow-key confirms + background indexing timing)**

```bash
cd /home/eaduinte/Desktop/system_llm
python3 -c "
import pty, os, time, select

pid, fd = pty.fork()
if pid == 0:
    os.execvp('.venv/bin/python', ['.venv/bin/python', '-m', 'assistant.cli',
                                    'repl', '--repo', '/tmp/joa-speed-test'])
else:
    def read_avail(t=1.0):
        deadline = time.time() + t
        buf = b''
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.1)
            if fd in r:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
        return buf

    t0 = time.time()
    print('BOOT:', read_avail(2.0))
    os.write(fd, b'\x1b\r')  # trust: Enter on default-highlighted 'Ha'
    print('TRUST:', read_avail(1.5))
    os.write(fd, b'\r')      # index-now confirm: Enter on default 'Ha'
    print('INDEX-CONFIRM:', read_avail(2.0))
    print(f'TIME TO PROMPT READY: {time.time() - t0:.2f}s')
    os.write(fd, b'exit\r')
    print('EXIT:', read_avail(1.0))
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
"
```

Expected: `TIME TO PROMPT READY` well under 5s (BM25-only build on a handful of files) — confirms the REPL is usable almost immediately instead of waiting ~67s. Read through the printed output to confirm: the trust screen and index-now prompt both rendered as arrow menus (no "Raqamni tanlang:" numeric prompt text anywhere in the captured output), and the BM25 success message appeared (`✓ Indekslandi (BM25)`).

- [ ] **Step 4: Live smoke test — background vector build completes and search upgrades**

```bash
cd /home/eaduinte/Desktop/system_llm
python3 -c "
import pty, os, time, select

pid, fd = pty.fork()
if pid == 0:
    os.execvp('.venv/bin/python', ['.venv/bin/python', '-m', 'assistant.cli',
                                    'repl', '--repo', '/tmp/joa-speed-test'])
else:
    def read_avail(t=1.0):
        deadline = time.time() + t
        buf = b''
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.1)
            if fd in r:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
        return buf

    print('BOOT:', read_avail(2.0))
    os.write(fd, b'\x1b\r')
    read_avail(1.5)
    os.write(fd, b'\r')
    read_avail(2.0)
    # wait for the background vector build to finish (small repo, should
    # be done well within 60s — a handful of files, not the full
    # crystal_bot corpus)
    out = read_avail(60.0)
    print('WAITED-FOR-BACKGROUND:', out)
    assert 'tayyor' in out.decode(errors='replace').lower(), \
        'expected background vector-ready message within 60s'
    os.write(fd, b'exit\r')
    read_avail(1.0)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    print('OK: background vector build completed and was announced')
"
```

Expected: prints `OK: background vector build completed and was announced` with no `AssertionError`. If this fails, check whether Ollama is running (`curl -s http://localhost:11434/api/tags`) before assuming a code bug.

- [ ] **Step 5: Live smoke test — second launch skips the vector rebuild**

```bash
cd /home/eaduinte/Desktop/system_llm
python3 -c "
import pty, os, time, select

pid, fd = pty.fork()
if pid == 0:
    os.execvp('.venv/bin/python', ['.venv/bin/python', '-m', 'assistant.cli',
                                    'repl', '--repo', '/tmp/joa-speed-test'])
else:
    def read_avail(t=1.0):
        deadline = time.time() + t
        buf = b''
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.1)
            if fd in r:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
        return buf

    print('BOOT:', read_avail(2.5))  # repo already trusted+indexed by now
    out = read_avail(5.0)
    print('WAIT-5S-NO-REBUILD:', out)
    assert 'tayyor' not in out.decode(errors='replace').lower(), \
        'vector index should NOT rebuild when repo is unchanged'
    os.write(fd, b'exit\r')
    read_avail(1.0)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    print('OK: unchanged repo did not trigger a vector rebuild')
"
```

Expected: `OK: unchanged repo did not trigger a vector rebuild`.

- [ ] **Step 6: Clean up the smoke-test scratch directory**

```bash
rm -rf /tmp/joa-speed-test
```

- [ ] **Step 7: Push feature branch, fast-forward main, push main**

```bash
cd /home/eaduinte/Desktop/system_llm
git branch --show-current   # confirm: feature/retrieval-core
git push origin feature/retrieval-core
git checkout main
git merge --ff-only feature/retrieval-core
git push origin main
git log --oneline -10
```

Expected: `main` and `feature/retrieval-core` both point at the same final commit; both pushes succeed.
