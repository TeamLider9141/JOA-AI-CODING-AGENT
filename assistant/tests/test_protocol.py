import pytest

from assistant.agent.protocol import (
    ProtocolError, build_system_prompt, parse_action,
)


def test_parses_bare_json():
    action = parse_action('{"action": "read_file", "args": {"path": "a.py"}}')
    assert action["action"] == "read_file"
    assert action["args"]["path"] == "a.py"


def test_parses_json_in_code_fence():
    text = 'Sure!\n```json\n{"action": "final", "args": {}, "answer": "done"}\n```'
    action = parse_action(text)
    assert action["action"] == "final"
    assert action["answer"] == "done"


def test_parses_json_embedded_in_prose():
    text = 'I will read it. {"action": "read_file", "args": {"path": "x"}} now.'
    assert parse_action(text)["action"] == "read_file"


def test_missing_action_key_raises():
    with pytest.raises(ProtocolError):
        parse_action('{"args": {}}')


def test_no_json_at_all_raises():
    with pytest.raises(ProtocolError):
        parse_action("I am not going to give you any json today")


def test_system_prompt_lists_every_tool():
    prompt = build_system_prompt()
    for tool in ("read_file", "write_file", "run_cmd", "search_code", "final"):
        assert tool in prompt
