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


def test_chat_uses_overridden_model():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5-coder:1.5b"
        return httpx.Response(200, json={"message": {"content": "hi"}})

    client = OllamaClient(base_url="http://test", model="qwen2.5-coder:1.5b",
                          transport=httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "hi"}]) == "hi"
