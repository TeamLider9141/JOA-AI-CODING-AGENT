from assistant.llm.gemini_client import _to_gemini_contents


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
