from assistant.agent.protocol import (
    ProtocolError, build_system_prompt, parse_action,
)
from assistant.agent.safety import PathJailError
from assistant.agent.tools import TOOLS, ToolContext, ToolError

MAX_PARSE_RETRIES = 2


def build_reminder(task: str, plan: list[str], iters_left: int) -> str:
    """A compact, always-recent nudge so the weak model keeps the goal in view."""
    if plan:
        plan_str = " ".join(f"{i + 1}.{step}" for i, step in enumerate(plan))
    else:
        plan_str = "(no plan yet)"
    return f"(Task: {task} | Plan: {plan_str} | {iters_left} iterations left)"


def run_agent(task: str, ctx: ToolContext, client,
              max_iters: int = 15) -> str:
    """Drive the plan->act->observe loop until the model says 'final'.

    `client` needs a `.chat(messages) -> str` method (OllamaClient qualifies).
    A running todo list (set via the 'plan' action) is re-shown each turn.
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Task: {task}"},
    ]
    plan: list[str] = []

    for i in range(max_iters):
        messages.append({
            "role": "user",
            "content": build_reminder(task, plan, max_iters - i),
        })
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})

        action = _parse_with_retries(reply, messages, client)
        if action is None:
            return "could not parse a valid action from the model"

        name = action["action"]
        if name == "final":
            return action.get("answer", "(no answer provided)")

        if name == "plan":
            plan, result = _apply_plan(action.get("args", {}))
        else:
            result = _run_tool(name, action.get("args", {}), ctx)
        messages.append({"role": "user", "content": f"Result:\n{result}"})

    return f"stopped after {max_iters} iterations without a final answer"


def _apply_plan(args: dict) -> tuple[list[str], str]:
    todo = args.get("todo")
    if isinstance(todo, list) and todo:
        return [str(step) for step in todo], "plan updated"
    return [], "error: plan action needs a non-empty 'todo' list"


def _parse_with_retries(reply: str, messages: list[dict], client) -> dict | None:
    for _ in range(MAX_PARSE_RETRIES):
        try:
            return parse_action(reply)
        except ProtocolError as exc:
            messages.append({
                "role": "user",
                "content": (
                    f"Your reply could not be parsed ({exc}). Reply with "
                    "exactly one JSON object and nothing else."
                ),
            })
            reply = client.chat(messages)
            messages.append({"role": "assistant", "content": reply})
    try:
        return parse_action(reply)
    except ProtocolError:
        return None


def _run_tool(name: str, args: dict, ctx: ToolContext) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"unknown action '{name}'. Valid: {', '.join(TOOLS)}, plan, final"
    try:
        return tool(ctx, args)
    except (ToolError, PathJailError) as exc:
        return f"error: {exc}"
    except KeyError as exc:
        return f"error: missing argument {exc}"
