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
