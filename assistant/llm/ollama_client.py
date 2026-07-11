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
        model: str = config.CHAT_MODEL,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            timeout=config.REQUEST_TIMEOUT,
            transport=transport,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed",
                          {"model": config.EMBED_MODEL, "input": texts})
        return data["embeddings"]

    def list_models(self) -> list[str]:
        """Names of models currently pulled into this Ollama instance."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))

    def close(self) -> None:
        self._client.close()

    def chat(self, messages: list[dict]) -> str:
        data = self._post("/api/chat", {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": config.NUM_CTX},
        })
        return data["message"]["content"]

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": config.NUM_CTX},
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    raise OllamaError(
                        f"Ollama returned {resp.status_code} for /api/chat."
                        f" Model missing? Try: ollama pull {self._model}"
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
