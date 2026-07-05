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
