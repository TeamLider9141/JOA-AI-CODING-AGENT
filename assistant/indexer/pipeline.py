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
