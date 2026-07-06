import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from assistant.agent.safety import resolve_in_root
from assistant.indexer.pipeline import Embedder, search_index

RUN_CMD_TIMEOUT = 30  # seconds
MAX_OUTPUT_CHARS = 4000  # keep tool results within the model's context


class ToolError(RuntimeError):
    """A tool failed in an expected way (missing file, bad command)."""


@dataclass
class ToolContext:
    root: Path
    data_dir: Path
    embedder: Embedder
    confirm: Callable[[str], bool]


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [truncated]"


def read_file(ctx: ToolContext, args: dict) -> str:
    path = resolve_in_root(ctx.root, args["path"])
    if not path.is_file():
        raise ToolError(f"no such file: {args['path']}")
    return _truncate(path.read_text(errors="ignore"))


def write_file(ctx: ToolContext, args: dict) -> str:
    path = resolve_in_root(ctx.root, args["path"])
    content = args.get("content", "")
    prompt = f"write {len(content)} bytes to {args['path']}?"
    if not ctx.confirm(prompt):
        return "write cancelled by user"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"wrote {len(content)} bytes to {args['path']}"


def run_cmd(ctx: ToolContext, args: dict,
            timeout: int = RUN_CMD_TIMEOUT) -> str:
    command = args["command"]
    if not ctx.confirm(f"run command: {command!r}?"):
        return "command cancelled by user"
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(ctx.root),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out after {timeout}s"
    return _truncate((proc.stdout + proc.stderr) or "(no output)")


def search_code(ctx: ToolContext, args: dict) -> str:
    results = search_index(args["query"], ctx.data_dir, ctx.embedder)
    if not results:
        return "no matches"
    lines = [
        f"{p['path']}:{p['start_line']}-{p['end_line']}  {p['symbol']}"
        for _cid, _score, p in results
    ]
    return "\n".join(lines)


TOOLS: dict[str, Callable[[ToolContext, dict], str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
    "search_code": search_code,
}
