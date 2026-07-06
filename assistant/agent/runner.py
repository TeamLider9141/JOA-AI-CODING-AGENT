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


class AgentSession:
    """A continuing agent conversation: history persists across send() calls.

    The plan scratchpad resets per send (each user request plans fresh), but
    self.messages carries the whole conversation so later turns see earlier
    context.
    """

    def __init__(self, ctx: ToolContext, client, max_iters: int = 15):
        self.ctx = ctx
        self.client = client
        self.max_iters = max_iters
        self.messages = [
            {"role": "system", "content": build_system_prompt()},
        ]

    def send(self, task: str) -> str:
        self.messages.append({"role": "user", "content": f"Task: {task}"})
        plan: list[str] = []

        for i in range(self.max_iters):
            self.messages.append({
                "role": "user",
                "content": build_reminder(task, plan, self.max_iters - i),
            })
            reply = self.client.chat(self.messages)
            self.messages.append({"role": "assistant", "content": reply})

            action = _parse_with_retries(reply, self.messages, self.client)
            if action is None:
                return "could not parse a valid action from the model"

            name = action["action"]
            if name == "final":
                return action.get("answer", "(no answer provided)")

            if name == "plan":
                plan, result = _apply_plan(action.get("args", {}))
            else:
                result = _run_tool(name, action.get("args", {}), self.ctx)
            self.messages.append({"role": "user", "content": f"Result:\n{result}"})

        return f"stopped after {self.max_iters} iterations without a final answer"


def run_agent(task: str, ctx: ToolContext, client,
              max_iters: int = 15) -> str:
    """One-shot agent run — a fresh session that handles a single task.

    `client` needs a `.chat(messages) -> str` method (OllamaClient qualifies).
    """
    return AgentSession(ctx, client, max_iters).send(task)


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
