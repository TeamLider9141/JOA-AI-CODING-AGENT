import json
from collections.abc import Iterator

import httpx

from assistant import config

MISSING_KEY_MSG = (
    "GEMINI_API_KEY not set. Add it to a .env file in the repo root, e.g.\n"
    "GEMINI_API_KEY=your-key-here\n"
    "(get one at https://aistudio.google.com/apikey)"
)
UNREACHABLE_MSG = "Gemini API unreachable at {url}. Check your network connection."


class GeminiError(RuntimeError):
    """Gemini API key missing/invalid, rate-limited, unreachable, or errored."""


def _to_gemini_contents(
    messages: list[dict],
) -> tuple[list[dict], dict | None]:
    """Translate OllamaClient-shaped messages (role/content) into Gemini's
    request shape. `system` messages are folded into a single
    systemInstruction rather than sent as a contents turn; `assistant`
    becomes `model` (Gemini's name for the model turn)."""
    contents = []
    system_parts = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]}
        if system_parts else None
    )
    return contents, system_instruction


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.GEMINI_MODEL,
        base_url: str = config.GEMINI_URL,
        transport: httpx.BaseTransport | None = None,
    ):
        api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        if not api_key:
            raise GeminiError(MISSING_KEY_MSG)
        self._model = model
        self._base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            timeout=config.REQUEST_TIMEOUT,
            transport=transport,
            headers={"x-goog-api-key": api_key},
        )

    def chat(self, messages: list[dict]) -> str:
        contents, system_instruction = _to_gemini_contents(messages)
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        data = self._post(
            f"/v1beta/models/{self._model}:generateContent", payload)
        return _extract_text(data)

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        contents, system_instruction = _to_gemini_contents(messages)
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        path = f"/v1beta/models/{self._model}:streamGenerateContent"
        try:
            with self._client.stream(
                "POST", path, json=payload, params={"alt": "sse"}
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise _http_error(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line.removeprefix("data: "))
                    text = _extract_text(chunk, allow_empty=True)
                    if text:
                        yield text
        except httpx.ConnectError as exc:
            raise GeminiError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise GeminiError(
                UNREACHABLE_MSG.format(url=self._base_url)) from exc
        if resp.status_code >= 400:
            raise _http_error(resp)
        return resp.json()

    def close(self) -> None:
        self._client.close()


def _http_error(resp: httpx.Response) -> GeminiError:
    if resp.status_code == 429:
        return GeminiError(
            "Gemini rate limit hit (429). Try --backend ollama or wait "
            "and retry."
        )
    if resp.status_code == 404:
        return GeminiError(
            f"Gemini API returned 404 (model not found). config.GEMINI_MODEL "
            f"may be stale — check available models at "
            f"https://generativelanguage.googleapis.com/v1beta/models and "
            f"update it if needed. Response: {resp.text}"
        )
    if resp.status_code in (400, 401, 403):
        return GeminiError(
            f"Gemini rejected the request ({resp.status_code}): "
            f"{resp.text}. Check GEMINI_API_KEY or the request payload."
        )
    return GeminiError(f"Gemini API returned {resp.status_code}: {resp.text}")


def _extract_text(data: dict, allow_empty: bool = False) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise GeminiError(f"Gemini blocked the request: {block_reason}")
        if allow_empty:
            return ""
        raise GeminiError(f"Gemini response had no candidates: {data}")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    finish_reason = candidate.get("finishReason")
    if not text and finish_reason not in (None, "STOP", "MAX_TOKENS"):
        raise GeminiError(f"Gemini stopped without output: {finish_reason}")
    return text
