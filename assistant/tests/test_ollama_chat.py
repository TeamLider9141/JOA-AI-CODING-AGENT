import json

import httpx
import pytest

from assistant.llm.ollama_client import OllamaClient, OllamaError


def make_client(handler) -> OllamaClient:
    return OllamaClient(base_url="http://test",
                        transport=httpx.MockTransport(handler))


def test_chat_returns_full_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "hello world"},
            "done": True,
        })

    out = make_client(handler).chat([{"role": "user", "content": "hi"}])
    assert out == "hello world"


def test_chat_connect_error_is_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])
