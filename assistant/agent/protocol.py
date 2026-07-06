import json

SYSTEM_PROMPT = """\
You are a coding agent working inside a single repository. You act one step
at a time. On each turn you MUST reply with exactly one JSON object and
nothing else — no prose outside the JSON.

Available actions:
- {"action": "read_file", "args": {"path": "relative/path.py"}}
- {"action": "write_file", "args": {"path": "relative/path.py", "content": "..."}}
- {"action": "run_cmd", "args": {"command": "pytest -q"}}
- {"action": "search_code", "args": {"query": "where is X"}}
- {"action": "final", "args": {}, "answer": "your answer to the user"}

Rules:
- Paths are always relative to the repo root. Never use absolute paths or "..".
- After each action you will be shown its result, then take the next step.
- Use search_code to locate code, read_file to inspect it, write_file to
  change it, run_cmd to run tests or commands.
- When the task is done, reply with the "final" action and put your answer
  in the "answer" field.
"""


class ProtocolError(RuntimeError):
    """The model's reply did not contain a usable JSON action."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def parse_action(text: str) -> dict:
    """Extract the first balanced JSON object from the model's reply."""
    obj = _extract_json(text)
    if obj is None:
        raise ProtocolError(f"no JSON object found in reply: {text[:200]!r}")
    if "action" not in obj:
        raise ProtocolError(f"JSON is missing 'action' key: {obj}")
    return obj


def _extract_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
