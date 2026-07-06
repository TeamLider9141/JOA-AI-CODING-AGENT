from assistant.agent.protocol import build_system_prompt, parse_action


def test_system_prompt_describes_plan_action():
    prompt = build_system_prompt()
    assert "plan" in prompt
    assert "todo" in prompt


def test_system_prompt_mentions_exit_code_autofix():
    prompt = build_system_prompt()
    assert "exit code" in prompt.lower()


def test_plan_action_parses_like_any_other_action():
    action = parse_action(
        '{"action": "plan", "args": {"todo": ["a", "b"]}}')
    assert action["action"] == "plan"
    assert action["args"]["todo"] == ["a", "b"]
