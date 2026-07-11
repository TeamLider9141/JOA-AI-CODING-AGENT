import pytest

from assistant.cli import _chat_client
from assistant.llm.gemini_client import GeminiClient
from assistant.llm.ollama_client import OllamaClient


def test_ollama_backend_returns_ollama_client():
    assert isinstance(_chat_client("ollama"), OllamaClient)


def test_gemini_backend_returns_gemini_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "assistant.cli.config.GEMINI_API_KEY", "test-key")
    assert isinstance(_chat_client("gemini"), GeminiClient)


def test_gemini_backend_without_key_raises(monkeypatch):
    monkeypatch.setattr("assistant.cli.config.GEMINI_API_KEY", None)
    from assistant.llm.gemini_client import GeminiError
    with pytest.raises(GeminiError):
        _chat_client("gemini")
