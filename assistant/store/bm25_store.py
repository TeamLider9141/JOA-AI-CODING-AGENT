import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

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
    """BM25 over chunk text. State is persisted as JSON (safe to load).

    Uses BM25Plus rather than classic BM25Okapi: Okapi's IDF term is exactly
    zero whenever a token's document frequency equals N/2, which silently
    zeroes out otherwise-relevant results (verified: a 2-doc corpus where a
    term appears in 1 doc triggers this every time). BM25Plus's IDF floor
    is always positive.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._payloads: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build BM25 index from zero chunks")
        self._ids = [c.chunk_id for c in chunks]
        self._payloads = [c.payload() for c in chunks]
        self._corpus = [
            tokenize(f"{c.path} {c.symbol} {c.text}") for c in chunks
        ]
        self._bm25 = BM25Plus(self._corpus)

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
        store._bm25 = BM25Plus(store._corpus)
        return store
