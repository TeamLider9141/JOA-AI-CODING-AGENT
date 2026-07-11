from assistant import config


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
