# Local Coding Assistant — Retrieval Core Implementation Plan (Phases 0–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working CLI (`index` / `search` / `ask`) that AST-chunks a repository with tree-sitter, stores it in embedded Qdrant + BM25, retrieves with hybrid RRF search, and answers questions via Ollama with file:line citations.

**Architecture:** Hand-written pipeline, no RAG framework. Index time: repo walker → tree-sitter chunker → Ollama embeddings → Qdrant (embedded) + BM25. Query time: vector + BM25 → Reciprocal Rank Fusion → context prompt → streaming chat. Retrieval quality is measured with a gold-question eval (hit@5).

**Tech Stack:** Python 3.10, httpx, typer, qdrant-client (embedded mode, no Docker), rank-bm25, tree-sitter + tree-sitter-language-pack, pathspec, pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md`

**Scope note:** This plan covers spec Phases 0–2 (setup, indexing, vector RAG, hybrid + eval). The agent loop (spec Phases 3–4) gets its own plan once this lands — it consumes `search_index()` built here.

---

## File Structure

All paths relative to repo root `/home/eaduinte/Desktop/system_llm`.

```
assistant/
├── __init__.py
├── config.py                  # single source of truth: models, URLs, top_k, paths
├── cli.py                     # typer entrypoint: index / search / ask
├── requirements.txt
├── llm/
│   ├── __init__.py
│   └── ollama_client.py       # httpx client: embed(), chat_stream(); OllamaError
├── indexer/
│   ├── __init__.py
│   ├── models.py              # Chunk dataclass, chunk_id, payload()
│   ├── walker.py              # gitignore-aware repo walker
│   ├── chunker.py             # tree-sitter AST chunking + text fallback
│   └── pipeline.py            # build_index(), search_index()
├── store/
│   ├── __init__.py
│   ├── qdrant_store.py        # embedded Qdrant wrapper
│   └── bm25_store.py          # BM25 + code-aware tokenizer, JSON persist
├── search/
│   ├── __init__.py
│   └── hybrid.py              # rrf_merge()
├── eval/
│   ├── __init__.py
│   ├── gold.yaml              # gold questions for uzbek_ai
│   └── run_eval.py            # hit@5: vector vs hybrid
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_ollama_client.py
    ├── test_walker.py
    ├── test_chunker.py
    ├── test_qdrant_store.py
    ├── test_bm25_store.py
    ├── test_hybrid.py
    ├── test_pipeline.py
    └── test_cli.py
pytest.ini                     # testpaths + pythonpath
.gitignore
```

Index data lives under `assistant/.data/<repo-name>/` (gitignored). BM25 state is persisted as JSON (not pickle) — it is plain strings/dicts and JSON avoids arbitrary-code-execution risk on load.

---

### Task 1: Project scaffolding

**Files:**
- Create: `assistant/__init__.py`, `assistant/llm/__init__.py`, `assistant/indexer/__init__.py`, `assistant/store/__init__.py`, `assistant/search/__init__.py`, `assistant/eval/__init__.py`, `assistant/tests/__init__.py`
- Create: `assistant/config.py`
- Create: `assistant/requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`

- [ ] **Step 1: Create package directories and empty `__init__.py` files**

```bash
cd /home/eaduinte/Desktop/system_llm
mkdir -p assistant/{llm,indexer,store,search,eval,tests}
touch assistant/__init__.py assistant/llm/__init__.py assistant/indexer/__init__.py \
      assistant/store/__init__.py assistant/search/__init__.py \
      assistant/eval/__init__.py assistant/tests/__init__.py
```

- [ ] **Step 2: Write `assistant/config.py`**

```python
from pathlib import Path

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 4096            # CPU-only: keep modest, tune later
REQUEST_TIMEOUT = 300.0   # seconds; CPU inference is slow

# --- Retrieval ---
VECTOR_TOP_K = 40
BM25_TOP_K = 40
RRF_K = 60
FINAL_TOP_K = 10

# --- Paths ---
DATA_DIR = Path(__file__).parent / ".data"
```

- [ ] **Step 3: Write `assistant/requirements.txt`**

```
httpx>=0.27
typer>=0.12
qdrant-client>=1.10
rank-bm25>=0.2.2
tree-sitter>=0.23
tree-sitter-language-pack>=0.7
pathspec>=0.12
pyyaml>=6.0
pytest>=8.0
```

- [ ] **Step 4: Write `pytest.ini`** (repo root)

```ini
[pytest]
testpaths = assistant/tests
pythonpath = .
```

- [ ] **Step 5: Write `.gitignore`** (repo root)

```
.venv/
__pycache__/
*.pyc
assistant/.data/
```

- [ ] **Step 6: Commit**

```bash
git add assistant pytest.ini .gitignore
git commit -m "feat: scaffold assistant package with config and deps"
```

---

### Task 2: Virtualenv and dependencies

**Files:** none created (environment only)

- [ ] **Step 1: Create venv and install dependencies**

```bash
cd /home/eaduinte/Desktop/system_llm
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r assistant/requirements.txt
```

Expected: all packages install without error. `tree-sitter-language-pack` ships prebuilt wheels — no compiler needed.

- [ ] **Step 2: Sanity-check pytest and tree-sitter**

```bash
.venv/bin/pytest --collect-only -q
.venv/bin/python -c "from tree_sitter_language_pack import get_parser; get_parser('python'); print('ok')"
```

Expected: pytest reports "no tests ran" (collection works); tree-sitter prints `ok`.

---

### Task 3: Chunk model

**Files:**
- Create: `assistant/indexer/models.py`
- Test: `assistant/tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.indexer.models import Chunk


def make(text="def f(): pass") -> Chunk:
    return Chunk(path="a.py", symbol="f", kind="function",
                 start_line=1, end_line=2, text=text)


def test_chunk_id_is_deterministic():
    assert make().chunk_id == make().chunk_id


def test_chunk_id_changes_with_content():
    assert make().chunk_id != make(text="def f(): return 1").chunk_id


def test_chunk_id_is_uuid_shaped():
    # Qdrant point ids must be UUIDs (or ints)
    import uuid
    uuid.UUID(make().chunk_id)  # raises if not a valid UUID


def test_payload_contains_all_metadata():
    p = make().payload()
    assert p == {
        "path": "a.py", "symbol": "f", "kind": "function",
        "start_line": 1, "end_line": 2, "text": "def f(): pass",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'Chunk'`

- [ ] **Step 3: Write `assistant/indexer/models.py`**

```python
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    path: str          # relative to indexed repo root
    symbol: str        # e.g. "UserService.login" or "lines-1" for text
    kind: str          # function | method | class | text
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    text: str

    @property
    def chunk_id(self) -> str:
        raw = f"{self.path}:{self.symbol}:{self.text}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def payload(self) -> dict:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/indexer/models.py assistant/tests/test_models.py
git commit -m "feat: add Chunk model with stable UUID ids"
```

---

### Task 4: Ollama client

**Files:**
- Create: `assistant/llm/ollama_client.py`
- Test: `assistant/tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx
import pytest

from assistant.llm.ollama_client import OllamaClient, OllamaError


def make_client(handler) -> OllamaClient:
    return OllamaClient(base_url="http://test",
                        transport=httpx.MockTransport(handler))


def test_embed_posts_model_and_returns_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        assert body["input"] == ["hello"]
        assert "model" in body
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    assert make_client(handler).embed(["hello"]) == [[0.1, 0.2]]


def test_chat_stream_concatenates_content():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": True}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, text="\n".join(lines))

    out = "".join(make_client(handler).chat_stream(
        [{"role": "user", "content": "hi"}]))
    assert out == "Hello"


def test_connect_error_becomes_actionable_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        make_client(handler).embed(["x"])


def test_missing_model_404_suggests_pull():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(OllamaError, match="ollama pull"):
        make_client(handler).embed(["x"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_ollama_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'OllamaClient'`

- [ ] **Step 3: Write `assistant/llm/ollama_client.py`**

```python
import json
from collections.abc import Iterator

import httpx

from assistant import config

UNREACHABLE_MSG = (
    "Ollama is not reachable at {url}. Start it with: ollama serve "
    "(install: https://ollama.com/download)"
)


class OllamaError(RuntimeError):
    """Ollama unreachable, model missing, or server-side error."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = config.OLLAMA_URL,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            timeout=config.REQUEST_TIMEOUT,
            transport=transport,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed",
                          {"model": config.EMBED_MODEL, "input": texts})
        return data["embeddings"]

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": config.NUM_CTX},
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    raise OllamaError(
                        f"Ollama returned {resp.status_code} for /api/chat."
                        f" Model missing? Try: ollama pull {config.CHAT_MODEL}"
                    )
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise OllamaError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as exc:
            raise OllamaError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
        except httpx.HTTPStatusError as exc:
            hint = ""
            if exc.response.status_code == 404:
                hint = f" Model missing? Try: ollama pull {payload.get('model')}"
            raise OllamaError(
                f"Ollama returned {exc.response.status_code}: "
                f"{exc.response.text}.{hint}"
            ) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_ollama_client.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/llm/ollama_client.py assistant/tests/test_ollama_client.py
git commit -m "feat: add Ollama client with streaming chat and embeddings"
```

---

### Task 5: Repo walker

**Files:**
- Create: `assistant/indexer/walker.py`
- Test: `assistant/tests/test_walker.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.indexer.walker import walk_repo


def test_walker_filters_gitignore_excludes_and_binaries(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.py\n")
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "secret.py").write_text("x = 2")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("x")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")

    names = {p.name for p in walk_repo(tmp_path)}
    assert names == {"app.py"}


def test_walker_skips_oversized_files(tmp_path):
    (tmp_path / "big.py").write_text("#" + "x" * 600_000)
    (tmp_path / "ok.py").write_text("x = 1")

    names = {p.name for p in walk_repo(tmp_path)}
    assert names == {"ok.py"}


def test_walker_returns_sorted_paths(tmp_path):
    (tmp_path / "b.py").write_text("x = 1")
    (tmp_path / "a.py").write_text("x = 1")

    assert [p.name for p in walk_repo(tmp_path)] == ["a.py", "b.py"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_walker.py -v`
Expected: FAIL — `ImportError: cannot import name 'walk_repo'`

- [ ] **Step 3: Write `assistant/indexer/walker.py`**

```python
from pathlib import Path

import pathspec

HARD_EXCLUDES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".data", "storage", "dist", "build", ".mypy_cache", ".pytest_cache",
}

TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".html", ".css", ".sh", ".sql",
}

MAX_FILE_BYTES = 512 * 1024


def walk_repo(root: Path) -> list[Path]:
    gitignore = root / ".gitignore"
    spec = None
    if gitignore.exists():
        spec = pathspec.PathSpec.from_lines(
            "gitwildmatch", gitignore.read_text().splitlines())

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in HARD_EXCLUDES for part in rel.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        if spec is not None and spec.match_file(str(rel)):
            continue
        files.append(path)
    return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_walker.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/indexer/walker.py assistant/tests/test_walker.py
git commit -m "feat: add gitignore-aware repo walker"
```

---

### Task 6: Tree-sitter chunker

**Files:**
- Create: `assistant/indexer/chunker.py`
- Test: `assistant/tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.indexer.chunker import chunk_file

SAMPLE = '''\
import os


def top_level(a, b):
    return a + b


class UserService:
    """Service for users."""

    def login(self, user):
        return True

    def logout(self):
        return False
'''


def write(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def test_functions_and_methods_become_chunks(tmp_path):
    f = write(tmp_path, "svc.py", SAMPLE)
    symbols = {c.symbol for c in chunk_file(f, tmp_path)}
    assert {"top_level", "UserService.login", "UserService.logout"} <= symbols


def test_method_chunk_carries_class_header_and_real_lines(tmp_path):
    f = write(tmp_path, "svc.py", SAMPLE)
    login = next(c for c in chunk_file(f, tmp_path)
                 if c.symbol == "UserService.login")
    assert login.kind == "method"
    assert login.text.startswith("class UserService:")
    assert login.start_line == 11  # actual def line in SAMPLE
    assert login.path == "svc.py"


def test_class_without_methods_is_one_chunk(tmp_path):
    f = write(tmp_path, "cfg.py", "class Config:\n    DEBUG = True\n")
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) == 1
    assert chunks[0].kind == "class"
    assert chunks[0].symbol == "Config"


def test_unknown_extension_falls_back_to_text_windows(tmp_path):
    f = write(tmp_path, "notes.md",
              "\n".join(f"line {i}" for i in range(200)))
    chunks = chunk_file(f, tmp_path)
    assert all(c.kind == "text" for c in chunks)
    assert len(chunks) >= 2


def test_decorated_function_is_found(tmp_path):
    f = write(tmp_path, "app.py",
              "@app.route('/x')\ndef handler():\n    return 1\n")
    symbols = {c.symbol for c in chunk_file(f, tmp_path)}
    assert "handler" in symbols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_chunker.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_file'`

- [ ] **Step 3: Write `assistant/indexer/chunker.py`**

```python
from pathlib import Path

from tree_sitter_language_pack import get_parser

from assistant.indexer.models import Chunk

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

FUNC_NODES = {"function_definition", "function_declaration", "method_definition"}
CLASS_NODES = {"class_definition", "class_declaration"}
WRAPPER_NODES = {"decorated_definition", "export_statement"}

TEXT_WINDOW = 80   # lines per fallback chunk
TEXT_OVERLAP = 15  # lines shared between consecutive fallback chunks


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    rel = str(path.relative_to(root))
    try:
        source = path.read_text(errors="ignore")
    except OSError:
        return []
    if not source.strip():
        return []

    lang = LANGUAGES.get(path.suffix.lower())
    if lang is None:
        return _chunk_text(rel, source)

    tree = get_parser(lang).parse(source.encode())
    chunks: list[Chunk] = []
    _collect(tree.root_node, source.encode(), rel, chunks)
    return chunks or _chunk_text(rel, source)


def _collect(node, src: bytes, rel: str, chunks: list[Chunk]) -> None:
    for child in node.children:
        if child.type in WRAPPER_NODES:
            _collect(child, src, rel, chunks)
        elif child.type in CLASS_NODES:
            _collect_class(child, src, rel, chunks)
        elif child.type in FUNC_NODES:
            chunks.append(
                _make_chunk(child, src, rel, _name(child, src), "function"))


def _collect_class(class_node, src: bytes, rel: str,
                   chunks: list[Chunk]) -> None:
    class_name = _name(class_node, src)
    header = _text(class_node, src).split("\n", 1)[0]
    body = class_node.child_by_field_name("body")

    methods = []
    for child in (body.children if body is not None else []):
        target = child
        if child.type in WRAPPER_NODES:
            target = next(
                (c for c in child.children if c.type in FUNC_NODES), child)
        if target.type in FUNC_NODES:
            methods.append(target)

    if not methods:
        chunks.append(_make_chunk(class_node, src, rel, class_name, "class"))
        return

    for m in methods:
        chunks.append(Chunk(
            path=rel,
            symbol=f"{class_name}.{_name(m, src)}",
            kind="method",
            start_line=m.start_point[0] + 1,
            end_line=m.end_point[0] + 1,
            text=f"{header}\n{_text(m, src)}",
        ))


def _make_chunk(node, src: bytes, rel: str, symbol: str, kind: str) -> Chunk:
    return Chunk(
        path=rel,
        symbol=symbol,
        kind=kind,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        text=_text(node, src),
    )


def _name(node, src: bytes) -> str:
    name_node = node.child_by_field_name("name")
    return _text(name_node, src) if name_node is not None else "anonymous"


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode(errors="ignore")


def _chunk_text(rel: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    step = TEXT_WINDOW - TEXT_OVERLAP
    for start in range(0, len(lines), step):
        window = lines[start:start + TEXT_WINDOW]
        if not window:
            break
        chunks.append(Chunk(
            path=rel,
            symbol=f"lines-{start + 1}",
            kind="text",
            start_line=start + 1,
            end_line=start + len(window),
            text="\n".join(window),
        ))
        if start + TEXT_WINDOW >= len(lines):
            break
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_chunker.py -v`
Expected: 5 passed. If `test_method_chunk_carries_class_header_and_real_lines` fails on `start_line == 11`, count the actual `def login` line in SAMPLE and fix the assertion (the line number, not the chunker) — the invariant is "start_line = the method's own def line".

- [ ] **Step 5: Commit**

```bash
git add assistant/indexer/chunker.py assistant/tests/test_chunker.py
git commit -m "feat: add tree-sitter AST chunker with text fallback"
```

---

### Task 7: Qdrant store (embedded)

**Files:**
- Create: `assistant/store/qdrant_store.py`
- Test: `assistant/tests/test_qdrant_store.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.indexer.models import Chunk
from assistant.store.qdrant_store import QdrantStore


def make_chunks() -> list[Chunk]:
    return [
        Chunk("a.py", "f", "function", 1, 2, "def f(): pass"),
        Chunk("b.py", "g", "function", 1, 2, "def g(): pass"),
    ]


def test_upsert_and_search_roundtrip(tmp_path):
    store = QdrantStore(tmp_path / "q")
    store.reset(dim=3)
    store.upsert(make_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    chunk_id, score, payload = results[0]
    assert payload["path"] == "a.py"
    assert payload["symbol"] == "f"
    store.close()


def test_reset_clears_previous_data(tmp_path):
    store = QdrantStore(tmp_path / "q")
    store.reset(dim=3)
    store.upsert(make_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.reset(dim=3)

    assert store.search([1.0, 0.0, 0.0], top_k=5) == []
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_qdrant_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'QdrantStore'`

- [ ] **Step 3: Write `assistant/store/qdrant_store.py`**

```python
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from assistant.indexer.models import Chunk

COLLECTION = "code"


class QdrantStore:
    """Embedded (serverless) Qdrant — data lives in a local directory.

    Note: embedded mode allows only ONE live client per path. Open, use,
    then close(); don't hold two stores on the same path.
    """

    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))

    def reset(self, dim: int) -> None:
        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        self._client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=c.chunk_id, vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(COLLECTION, points)

    def search(self, vector: list[float],
               top_k: int) -> list[tuple[str, float, dict]]:
        hits = self._client.query_points(
            COLLECTION, query=vector, limit=top_k).points
        return [(str(h.id), h.score, h.payload) for h in hits]

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_qdrant_store.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/store/qdrant_store.py assistant/tests/test_qdrant_store.py
git commit -m "feat: add embedded Qdrant store"
```

---

### Task 8: BM25 store with code-aware tokenizer

**Files:**
- Create: `assistant/store/bm25_store.py`
- Test: `assistant/tests/test_bm25_store.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from assistant.indexer.models import Chunk
from assistant.store.bm25_store import BM25Store, tokenize


def test_tokenize_splits_camel_case_and_snake_case():
    tokens = tokenize("JWTMiddleware read_file")
    assert "jwtmiddleware" in tokens   # whole identifier kept
    assert "jwt" in tokens             # camel parts
    assert "middleware" in tokens
    assert "read" in tokens            # snake parts
    assert "file" in tokens


def make_chunks() -> list[Chunk]:
    return [
        Chunk("auth.py", "JWTMiddleware", "class", 1, 5,
              "class JWTMiddleware:\n    def check(self): pass"),
        Chunk("db.py", "connect", "function", 1, 5,
              "def connect():\n    return engine"),
    ]


def test_exact_identifier_ranks_first():
    store = BM25Store()
    store.build(make_chunks())
    results = store.search("JWTMiddleware", top_k=2)
    assert results[0][2]["path"] == "auth.py"


def test_zero_score_results_are_dropped():
    store = BM25Store()
    store.build(make_chunks())
    assert store.search("zzz_nonexistent_zzz", top_k=5) == []


def test_save_load_roundtrip(tmp_path):
    store = BM25Store()
    store.build(make_chunks())
    store.save(tmp_path / "bm25.json")

    loaded = BM25Store.load(tmp_path / "bm25.json")
    assert loaded.search("connect", top_k=1)[0][2]["path"] == "db.py"


def test_build_with_no_chunks_raises():
    with pytest.raises(ValueError):
        BM25Store().build([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_bm25_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'BM25Store'`

- [ ] **Step 3: Write `assistant/store/bm25_store.py`**

```python
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from assistant.indexer.models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
# split on: underscores, lower->Upper boundary, ACRONYMWord boundary
_SPLIT_RE = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        tokens.append(tok.lower())
        parts = [p.lower() for p in _SPLIT_RE.split(tok) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Store:
    """BM25 over chunk text. State is persisted as JSON (safe to load)."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._payloads: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build BM25 index from zero chunks")
        self._ids = [c.chunk_id for c in chunks]
        self._payloads = [c.payload() for c in chunks]
        self._corpus = [
            tokenize(f"{c.path} {c.symbol} {c.text}") for c in chunks
        ]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str,
               top_k: int) -> list[tuple[str, float, dict]]:
        if self._bm25 is None:
            raise ValueError("BM25 index empty — call build() or load() first")
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)),
                       key=lambda i: scores[i], reverse=True)
        return [
            (self._ids[i], float(scores[i]), self._payloads[i])
            for i in order[:top_k]
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"ids": self._ids, "payloads": self._payloads,
                 "corpus": self._corpus}
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        state = json.loads(path.read_text())
        store = cls()
        store._ids = state["ids"]
        store._payloads = state["payloads"]
        store._corpus = state["corpus"]
        store._bm25 = BM25Okapi(store._corpus)
        return store
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_bm25_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/store/bm25_store.py assistant/tests/test_bm25_store.py
git commit -m "feat: add BM25 store with camelCase/snake_case tokenizer"
```

---

### Task 9: Reciprocal Rank Fusion

**Files:**
- Create: `assistant/search/hybrid.py`
- Test: `assistant/tests/test_hybrid.py`

- [ ] **Step 1: Write the failing tests**

```python
from assistant.search.hybrid import rrf_merge


def test_chunk_in_both_lists_wins():
    vector = [("c1", 0.9, {"path": "a"}), ("c2", 0.8, {"path": "b"})]
    bm25 = [("c3", 5.0, {"path": "c"}), ("c1", 4.0, {"path": "a"})]

    merged = rrf_merge([vector, bm25], k=60, top_k=3)
    assert merged[0][0] == "c1"
    assert merged[0][2] == {"path": "a"}


def test_top_k_limits_results():
    results = [(f"c{i}", 1.0, {}) for i in range(20)]
    assert len(rrf_merge([results], k=60, top_k=5)) == 5


def test_raw_scores_are_ignored_only_rank_matters():
    # c1 has tiny raw scores but rank 1 in both lists -> must win
    a = [("c1", 0.001, {}), ("c2", 0.0009, {})]
    b = [("c1", 0.002, {}), ("c3", 0.001, {})]
    assert rrf_merge([a, b], k=60, top_k=1)[0][0] == "c1"


def test_empty_input_returns_empty():
    assert rrf_merge([[], []], k=60, top_k=5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_hybrid.py -v`
Expected: FAIL — `ImportError: cannot import name 'rrf_merge'`

- [ ] **Step 3: Write `assistant/search/hybrid.py`**

```python
def rrf_merge(
    result_lists: list[list[tuple[str, float, dict]]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float, dict]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank).

    Raw retrieval scores are intentionally ignored — vector cosine and BM25
    scores live on incomparable scales; rank is the only shared currency.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for results in result_lists:
        for rank, (chunk_id, _score, payload) in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            payloads[chunk_id] = payload
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, score, payloads[cid]) for cid, score in ranked[:top_k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_hybrid.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/search/hybrid.py assistant/tests/test_hybrid.py
git commit -m "feat: add reciprocal rank fusion merge"
```

---

### Task 10: Index and search pipeline

**Files:**
- Create: `assistant/indexer/pipeline.py`
- Test: `assistant/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaError


def fake_embedder(texts: list[str]) -> list[list[float]]:
    # deterministic 3-dim "embedding": length signal + constants
    return [[float(len(t)), 1.0, 0.5] for t in texts]


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "class JWTMiddleware:\n    def check(self):\n        return True\n")
    (repo / "db.py").write_text(
        "def connect():\n    return 'engine'\n")
    return repo


def test_build_index_persists_both_stores(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"

    n = build_index(repo, data, fake_embedder)

    assert n >= 2
    assert (data / "bm25.json").exists()
    assert (data / "qdrant").is_dir()


def test_search_index_hybrid_finds_exact_identifier(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    results = search_index("JWTMiddleware", data, fake_embedder)
    assert results, "expected at least one result"
    assert results[0][2]["path"] == "auth.py"


def test_search_index_vector_mode_returns_results(tmp_path):
    repo = make_repo(tmp_path)
    data = tmp_path / "data"
    build_index(repo, data, fake_embedder)

    results = search_index("anything", data, fake_embedder, mode="vector")
    assert len(results) >= 1


def test_empty_repo_raises(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    with pytest.raises(ValueError, match="no indexable chunks"):
        build_index(repo, tmp_path / "data", fake_embedder)


def test_ollama_error_aborts_build(tmp_path):
    repo = make_repo(tmp_path)

    def broken_embedder(texts):
        raise OllamaError("server down")

    with pytest.raises(OllamaError):
        build_index(repo, tmp_path / "data", broken_embedder)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_index'`

- [ ] **Step 3: Write `assistant/indexer/pipeline.py`**

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
        # prefix path+symbol so the embedding carries location semantics
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
    # BM25 first: on an RRF score tie (symmetric rank swap between the two
    # retrievers), dict insertion order decides the winner. Exact lexical
    # matches should win those ties over vector-similarity noise.
    return rrf_merge(
        [bm25_results, vector_results],
        k=config.RRF_K,
        top_k=config.FINAL_TOP_K,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (models, client, walker, chunker, stores, hybrid, pipeline)

- [ ] **Step 6: Commit**

```bash
git add assistant/indexer/pipeline.py assistant/tests/test_pipeline.py
git commit -m "feat: add index/search pipeline wiring chunker, stores, RRF"
```

---

### Task 11: CLI (index / search / ask)

**Files:**
- Create: `assistant/cli.py`
- Test: `assistant/tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from typer.testing import CliRunner

from assistant.cli import app, build_prompt

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "ask"):
        assert cmd in result.output


def test_build_prompt_contains_citations_and_question():
    results = [("id1", 0.5, {
        "path": "auth.py", "start_line": 3, "end_line": 9,
        "kind": "class", "symbol": "JWTMiddleware",
        "text": "class JWTMiddleware: ...",
    })]
    prompt = build_prompt("where is auth?", results)
    assert "auth.py:3-9" in prompt
    assert "JWTMiddleware" in prompt
    assert "where is auth?" in prompt


def test_search_without_index_exits_nonzero(tmp_path):
    result = runner.invoke(
        app, ["search", "query", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "index" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest assistant/tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'assistant.cli'`

- [ ] **Step 3: Write `assistant/cli.py`**

```python
from pathlib import Path

import typer

from assistant import config
from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaClient, OllamaError

app = typer.Typer(no_args_is_help=True, add_completion=False)

SYSTEM_PROMPT = (
    "You are a coding assistant. Answer the question using ONLY the provided "
    "context chunks. Cite sources as path:start_line-end_line. If the context "
    "is insufficient, say what is missing instead of guessing."
)


def _data_dir(repo: Path) -> Path:
    return config.DATA_DIR / repo.resolve().name


def _require_index(data_dir: Path) -> None:
    if not (data_dir / "bm25.json").exists():
        typer.echo(
            "No index found. Run first: python -m assistant.cli index <repo>",
            err=True)
        raise typer.Exit(1)


@app.command()
def index(repo: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Index a repository: tree-sitter chunks -> Qdrant + BM25."""
    client = OllamaClient()
    try:
        n = build_index(repo, _data_dir(repo), client.embed)
    except (OllamaError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"Indexed {n} chunks from {repo}")


@app.command()
def search(
    query: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    mode: str = typer.Option("hybrid", help="hybrid | vector"),
):
    """Search the index and print matching chunks (debug view)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    try:
        results = search_index(query, data_dir, client.embed, mode=mode)
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    for _chunk_id, score, p in results:
        typer.echo(
            f"{score:.4f}  {p['path']}:{p['start_line']}-{p['end_line']}"
            f"  {p['symbol']}")


@app.command()
def ask(
    question: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
):
    """Ask a question about the indexed repository."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    try:
        results = search_index(question, data_dir, client.embed)
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


def build_prompt(question: str,
                 results: list[tuple[str, float, dict]]) -> str:
    blocks = []
    for i, (_chunk_id, _score, p) in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {p['path']}:{p['start_line']}-{p['end_line']} "
            f"({p['kind']} {p['symbol']})\n{p['text']}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest assistant/tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/cli.py assistant/tests/test_cli.py
git commit -m "feat: add typer CLI with index, search, ask commands"
```

---

### Task 12: Ollama install and models — ⚠ REQUIRES USER CONFIRMATION

Downloads: Ollama install script (ollama.com), `qwen2.5-coder:7b` (~4.7 GB), `nomic-embed-text` (~274 MB). **Do not run without the user's explicit OK in chat.** RAM note: close heavy apps before first inference (16 GB total).

- [ ] **Step 1: Install Ollama** (after user confirmation)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Expected: installer finishes; `ollama --version` prints a version. The installer usually registers and starts a systemd service; if not, run `ollama serve` in a separate terminal.

- [ ] **Step 2: Pull models** (after user confirmation)

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

Expected: both downloads complete.

- [ ] **Step 3: Verify**

```bash
ollama list
curl -s http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": ["hello"]}' | head -c 200
```

Expected: `ollama list` shows both models; curl returns JSON starting with `{"model":"nomic-embed-text","embeddings":[[...`.

---

### Task 13: Real run on uzbek_ai + retrieval eval

**Files:**
- Create: `assistant/eval/gold.yaml`
- Create: `assistant/eval/run_eval.py`

- [ ] **Step 1: Index the real repo**

```bash
cd /home/eaduinte/Desktop/system_llm
.venv/bin/python -m assistant.cli index /home/eaduinte/Desktop/uzbek_ai
```

Expected: `Indexed N chunks from /home/eaduinte/Desktop/uzbek_ai` with N in the hundreds. First run takes minutes (CPU embedding).

- [ ] **Step 2: Sanity-check search**

```bash
.venv/bin/python -m assistant.cli search "voice provider selection" \
  --repo /home/eaduinte/Desktop/uzbek_ai
.venv/bin/python -m assistant.cli search "VOICE_PROVIDER" \
  --repo /home/eaduinte/Desktop/uzbek_ai --mode vector
```

Expected: hybrid results include `voice_ai/` paths; compare vector-only vs hybrid output by eye.

- [ ] **Step 3: Sanity-check ask**

```bash
.venv/bin/python -m assistant.cli ask \
  "How does the system decide between the OpenAI relay and the local HF pipeline?" \
  --repo /home/eaduinte/Desktop/uzbek_ai
```

Expected: streamed answer citing `voice_ai/...` paths. Slow on CPU — up to a couple of minutes is normal.

- [ ] **Step 4: Write `assistant/eval/gold.yaml`**

Starter set below. **Before saving, verify each `expect_path_contains` prefix actually exists in uzbek_ai** (`ls /home/eaduinte/Desktop/uzbek_ai/<prefix>`), fix any that don't, and extend to at least 10 questions by inspecting `backend/app/`, `frontend/`, `voice_ai/` and asking "where/how" questions whose answering file you can name.

```yaml
- question: Where is the complaint classifier implemented?
  expect_path_contains: backend/
- question: How does the voice server choose between the OpenAI relay and the local HF pipeline?
  expect_path_contains: voice_ai/
- question: Where are tickets created in the backend API?
  expect_path_contains: backend/
- question: Where does the frontend show the tickets dashboard?
  expect_path_contains: frontend/
- question: How is the knowledge graph of the repo built?
  expect_path_contains: graphify/
```

- [ ] **Step 5: Write `assistant/eval/run_eval.py`**

```python
"""Retrieval quality eval: hit@5 for vector-only vs hybrid.

Usage:
    .venv/bin/python -m assistant.eval.run_eval --repo ~/Desktop/uzbek_ai
"""
from pathlib import Path

import typer
import yaml

from assistant import config
from assistant.indexer.pipeline import search_index
from assistant.llm.ollama_client import OllamaClient

GOLD_PATH = Path(__file__).parent / "gold.yaml"


def main(repo: Path = typer.Option(..., "--repo", exists=True)):
    gold = yaml.safe_load(GOLD_PATH.read_text())
    data_dir = config.DATA_DIR / repo.resolve().name
    client = OllamaClient()

    for mode in ("vector", "hybrid"):
        hits = 0
        for item in gold:
            results = search_index(
                item["question"], data_dir, client.embed, mode=mode)
            paths = [p["path"] for _cid, _s, p in results[:5]]
            if any(item["expect_path_contains"] in path for path in paths):
                hits += 1
        print(f"{mode:7s} hit@5: {hits}/{len(gold)}")


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 6: Run the eval**

```bash
.venv/bin/python -m assistant.eval.run_eval --repo /home/eaduinte/Desktop/uzbek_ai
```

Expected output shape:

```
vector  hit@5: 6/10
hybrid  hit@5: 8/10
```

Success criterion from spec: hybrid hit@5 ≥ vector hit@5. If hybrid is worse, inspect which questions regressed (`search` command with both modes) before touching parameters.

- [ ] **Step 7: Commit**

```bash
git add assistant/eval/gold.yaml assistant/eval/run_eval.py
git commit -m "feat: add retrieval eval with gold questions (hit@5)"
```

---

### Task 14: README and wrap-up

**Files:**
- Create: `assistant/README.md`

- [ ] **Step 1: Write `assistant/README.md`**

```markdown
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

## Design

See `docs/superpowers/specs/2026-07-05-local-coding-assistant-design.md`.
Models and retrieval parameters live in `assistant/config.py` only.
```

- [ ] **Step 2: Full verification**

```bash
.venv/bin/pytest -q
```

Expected: all tests pass. Only claim completion with this output in hand (verification-before-completion).

- [ ] **Step 3: Commit**

```bash
git add assistant/README.md
git commit -m "docs: add assistant README"
```

---

## After this plan

Next plan (separate document): **agent loop** (spec Phases 3–4) — tool schema (`read_file`, `write_file`, `run_cmd`, `search_code`), path jail, JSON tool-call protocol with retry, loop cap, diff-confirmation on writes, then reranker + multi-step planner. It builds directly on `search_index()` and `OllamaClient` from this plan.
