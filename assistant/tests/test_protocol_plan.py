from assistant.agent.protocol import build_system_prompt, parse_action


def test_system_prompt_describes_plan_action():
    prompt = build_system_prompt()
    assert "plan" in prompt
    assert "todo" in prompt


def test_system_prompt_mentions_exit_code_autofix():
    prompt = build_system_prompt()
    assert "exit code" in prompt.lower()


def test_system_prompt_discourages_tools_for_plain_questions():
    prompt = build_system_prompt()
    normalized = " ".join(prompt.lower().split())
    assert "final" in prompt
    assert "do not write files or run commands" in normalized


def test_plan_action_parses_like_any_other_action():
    action = parse_action(
        '{"action": "plan", "args": {"todo": ["a", "b"]}}')
    assert action["action"] == "plan"
    assert action["args"]["todo"] == ["a", "b"]
