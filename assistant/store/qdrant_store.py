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
