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
