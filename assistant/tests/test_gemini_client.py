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
        assert request.url.path == "/v1beta/models/gemini-flash-latest:generateContent"
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


def test_404_raises_model_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error": "model not found"}')

    with pytest.raises(GeminiError, match="GEMINI_MODEL"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_401_raises_key_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error": "bad auth"}')

    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_403_raises_key_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text='{"error": "permission denied"}')

    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_429_raises_rate_limit_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": "quota exceeded"}')

    with pytest.raises(GeminiError, match="rate limit"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_connect_error_becomes_actionable_gemini_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(GeminiError, match="unreachable"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_empty_candidates_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    with pytest.raises(GeminiError, match="no candidates"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_chat_stream_concatenates_sse_chunks():
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n'
        "\n"
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == \
            "/v1beta/models/gemini-flash-latest:streamGenerateContent"
        assert request.url.params["alt"] == "sse"
        return httpx.Response(200, text=body)

    out = "".join(make_client(handler).chat_stream(
        [{"role": "user", "content": "hi"}]))
    assert out == "Hello"


def test_chat_stream_skips_metadata_only_chunks():
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n'
        "\n"
        'data: {"usageMetadata": {"totalTokenCount": 5}}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    out = "".join(make_client(handler).chat_stream(
        [{"role": "user", "content": "hi"}]))
    assert out == "Hi"


def test_chat_stream_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": "quota exceeded"}')

    with pytest.raises(GeminiError, match="rate limit"):
        list(make_client(handler).chat_stream(
            [{"role": "user", "content": "hi"}]))


def test_chat_stream_raises_on_safety_block():
    body = (
        'data: {"promptFeedback": {"blockReason": "SAFETY"}}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(GeminiError, match="blocked"):
        list(make_client(handler).chat_stream(
            [{"role": "user", "content": "hi"}]))


def test_chat_raises_on_finish_reason_safety():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"finishReason": "SAFETY", "content": {}}]
        })

    with pytest.raises(GeminiError, match="SAFETY"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_chat_stream_raises_on_finish_reason_safety():
    body = 'data: {"candidates":[{"finishReason":"SAFETY","content":{}}]}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(GeminiError, match="SAFETY"):
        list(make_client(handler).chat_stream(
            [{"role": "user", "content": "hi"}]))


def test_chat_stream_connect_error_becomes_actionable_gemini_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(GeminiError, match="unreachable"):
        list(make_client(handler).chat_stream(
            [{"role": "user", "content": "hi"}]))


def test_close_closes_underlying_http_client():
    client = make_client(lambda r: httpx.Response(200))
    client.close()
    assert client._client.is_closed
