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
