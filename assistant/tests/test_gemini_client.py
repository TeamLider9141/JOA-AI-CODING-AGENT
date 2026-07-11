import json

import httpx
import pytest

from assistant.llm.gemini_client import GeminiClient, GeminiError, _to_gemini_contents


def test_translates_user_and_assistant_roles():
    contents, system_instruction = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]
    assert system_instruction is None


def test_folds_system_messages_into_system_instruction():
    contents, system_instruction = _to_gemini_contents([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ])
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert system_instruction == {"parts": [{"text": "You are helpful."}]}


def test_joins_multiple_system_messages():
    contents, system_instruction = _to_gemini_contents([
        {"role": "system", "content": "First."},
        {"role": "system", "content": "Second."},
        {"role": "user", "content": "hi"},
    ])
    assert system_instruction == {"parts": [{"text": "First.\n\nSecond."}]}


def test_missing_api_key_raises_without_request():
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="")


def make_client(handler) -> GeminiClient:
    return GeminiClient(
        api_key="test-key",
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )


def test_chat_posts_translated_contents_and_returns_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-3-flash:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]}
        ]
        return httpx.Response(200, json={
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "hello"}]}}
            ]
        })

    out = make_client(handler).chat([{"role": "user", "content": "hi"}])
    assert out == "hello"


def test_chat_sends_system_instruction_separately():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]}
        ]
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        })

    make_client(handler).chat([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ])
