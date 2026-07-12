import sys
import time
from enum import Enum
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from assistant import config
from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError
from assistant.agent.runner import AgentSession, run_agent
from assistant.agent.tools import ToolContext
from assistant.agent.proc import run_streaming


class Backend(str, Enum):
    ollama = "ollama"
    gemini = "gemini"


app = typer.Typer(no_args_is_help=True, add_completion=False)


def _chat_client(backend: Backend):
    if backend == Backend.gemini:
        return GeminiClient()
    return OllamaClient()


SYSTEM_PROMPT = (
    "You are a coding assistant. Answer the question using ONLY the provided "
    "context chunks. Cite sources as path:start_line-end_line. If the context "
    "is insufficient, say what is missing instead of guessing."
)

FAST_SYSTEM_PROMPT = (
    "You are a coding assistant chatting with a user inside their "
    "repository. If answering would require reading or writing files, "
    "running commands, or searching the codebase, reply with exactly "
    "ESCALATE and nothing else. Otherwise answer the question directly "
    "and concisely."
)

_SNIFF_LEN = len("ESCALATE")


def _fast_answer(session, line, echo_token):
    """Try answering `line` with one direct streaming chat call.

    Returns the full streamed answer, or None if the model escalated (or
    produced nothing) — in which case the caller should run the agent
    loop. On success the exchange is appended to session.messages so the
    agent keeps conversational context."""
    messages = (
        [{"role": "system", "content": FAST_SYSTEM_PROMPT}]
        + session.messages[1:]
        + [{"role": "user", "content": line}]
    )
    stream = session.client.chat_stream(messages)
    buffer = ""
    for chunk in stream:
        buffer += chunk
        if len(buffer.strip()) >= _SNIFF_LEN:
            break
    if buffer.strip().upper().startswith("ESCALATE"):
        return None
    if not buffer.strip():
        return None
    echo_token(buffer)
    parts = [buffer]
    for chunk in stream:
        echo_token(chunk)
        parts.append(chunk)
    answer = "".join(parts)
    session.messages.append({"role": "user", "content": line})
    session.messages.append({"role": "assistant", "content": answer})
    return answer


def _data_dir(repo: Path) -> Path:
    return config.DATA_DIR / repo.resolve().name


def _require_index(data_dir: Path) -> None:
    if not (data_dir / "bm25.json").exists():
        typer.echo(
            "No index found. Run first: python -m assistant.cli index <repo>",
            err=True)
        raise typer.Exit(1)


@app.command()
def index(repo: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Index a repository: tree-sitter chunks -> Qdrant + BM25."""
    client = OllamaClient()
    try:
        n = build_index(repo, _data_dir(repo), client.embed)
    except (OllamaError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"Indexed {n} chunks from {repo}")


@app.command()
def search(
    query: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    mode: str = typer.Option("hybrid", help="hybrid | vector"),
):
    """Search the index and print matching chunks (debug view)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    try:
        results = search_index(query, data_dir, client.embed, mode=mode)
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    for _chunk_id, score, p in results:
        typer.echo(
            f"{score:.4f}  {p['path']}:{p['start_line']}-{p['end_line']}"
            f"  {p['symbol']}")


@app.command()
def ask(
    question: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    backend: Backend = typer.Option(
        Backend.ollama, "--backend",
        help="ollama | gemini (gemini needs GEMINI_API_KEY in .env)"),
):
    """Ask a question about the indexed repository."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
        results = search_index(question, data_dir, embed_client.embed)
        typer.echo("--- sources ---")
        for _chunk_id, _score, p in results:
            typer.echo(
                f"  {p['path']}:{p['start_line']}-{p['end_line']}"
                f"  {p['symbol']}")
        typer.echo("--- answer ---")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, results)},
        ]
        for token in chat_client.chat_stream(messages):
            typer.echo(token, nl=False)
        typer.echo()
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


@app.command()
def agent(
    task: str,
    repo: Path = typer.Option(..., "--repo", exists=True, file_okay=False),
    backend: Backend = typer.Option(
        Backend.ollama, "--backend",
        help="ollama | gemini (gemini needs GEMINI_API_KEY in .env)"),
):
    """Run the coding agent: plan, call tools, and act on the repo."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
        ctx = ToolContext(
            root=repo.resolve(),
            data_dir=data_dir,
            embedder=embed_client.embed,
            confirm=lambda msg: typer.confirm(msg),
            output_sink=lambda t: typer.echo(t, nl=False),
        )
        answer = run_agent(task, ctx, chat_client)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo("--- answer ---")
    typer.echo(answer)


def _handle_joamodel(session, embed_client, read_line, echo) -> None:
    """List installed Ollama models plus "gemini"; switch session.client
    to whichever the user picks by number. Leaves session.client
    unchanged on any failure (bad input, EOF, missing Gemini key, or a
    failure listing Ollama's models)."""
    try:
        models = embed_client.list_models()
    except OllamaError as exc:
        echo(str(exc))
        return
    options = models + ["gemini"]
    for i, name in enumerate(options, start=1):
        echo(f"{i}. {name}")
    echo("Raqamni tanlang:")
    try:
        choice_line = read_line()
    except EOFError:
        return
    choice = choice_line.strip()
    try:
        index = int(choice)
    except ValueError:
        echo(f"Noto'g'ri tanlov: {choice!r}")
        return
    if not (1 <= index <= len(options)):
        echo(f"Noto'g'ri tanlov: {choice!r}")
        return
    selected = options[index - 1]
    if selected == "gemini":
        if not config.GEMINI_API_KEY:
            echo("GEMINI_API_KEY .env'da topilmadi. Model o'zgartirilmadi.")
            return
        try:
            new_client = GeminiClient()
        except GeminiError as exc:
            echo(str(exc))
            return
    else:
        new_client = OllamaClient(model=selected)
    if hasattr(session.client, "close"):
        session.client.close()
    session.client = new_client
    echo(f"✓ Model: {selected}")


SLASH_COMMANDS = {
    "/joamodel": "modelni almashtirish (Ollama modellari / Gemini)",
    "/clear": "suhbat kontekstini tozalash (tarix 0 dan boshlanadi)",
    "/help": "shu ro'yxat",
}


class SlashCompleter(Completer):
    """Live dropdown of slash commands while typing — only when the line
    starts with "/", so normal questions get no suggestion noise."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for name, desc in SLASH_COMMANDS.items():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text),
                                 display_meta=desc)


def _show_help(echo) -> None:
    echo("Buyruqlar:")
    for name, desc in SLASH_COMMANDS.items():
        echo(f"  {name:<10} — {desc}")
    echo("  !buyruq    — buyruqni to'g'ridan-to'g'ri bajarish (LLM'siz, "
         "jonli chiqish)")
    echo("  exit, quit — sessiyadan chiqish")


def _run_bang(session, command, echo, echo_token) -> None:
    """Run a shell command directly, bypassing the LLM entirely, with
    live output (progress bars render in place) and no timeout — the
    user is watching and can Ctrl-C. Never touches session.messages."""
    returncode, _output, _timed_out = run_streaming(
        command, session.ctx.root, echo_token, timeout=None)
    echo(f"\n(exit code: {returncode})")


def _repl_loop(session, read_line, echo, embed_client, echo_token) -> None:
    """Drive an AgentSession from a line source until exit/EOF.

    `read_line()` returns the next input line (raising EOFError at end of
    input); `echo(text)` prints a line; `echo_token(text)` prints a
    streamed fragment without a newline (used by the fast path).
    `embed_client` is an OllamaClient used only for `/joamodel`'s model
    listing. Lines starting with "/" are slash commands and never reach
    the LLM. Every other line first tries `_fast_answer` (one direct
    streaming chat call); the agent loop only runs when the model
    escalates. Kept separate from the CLI command so the loop is testable
    without a live model.
    """
    echo("joa session — type 'exit' or Ctrl-D to quit "
         "('/' — buyruqlar, '!' — shell buyrug'i)")
    while True:
        try:
            line = read_line()
        except EOFError:
            return
        stripped = line.strip()
        if stripped in ("exit", "quit"):
            return
        if not stripped:
            continue
        if stripped.startswith("/"):
            if stripped == "/joamodel":
                _handle_joamodel(session, embed_client, read_line, echo)
            elif stripped == "/clear":
                session.messages = session.messages[:1]
                echo("✓ Suhbat tozalandi — kontekst 0 dan boshlanadi.")
            elif stripped in ("/", "/help"):
                _show_help(echo)
            else:
                echo(f"Noma'lum buyruq: {stripped!r}. Ro'yxat uchun: /help")
            continue
        if stripped.startswith("!"):
            command = stripped[1:].strip()
            if not command:
                echo("Bo'sh buyruq. Masalan: "
                     "!ollama pull qwen2.5-coder:0.5b")
            else:
                _run_bang(session, command, echo, echo_token)
            continue
        start = time.perf_counter()
        try:
            answer = _fast_answer(session, stripped, echo_token)
            if answer is None:
                answer = session.send(stripped)
                elapsed = time.perf_counter() - start
                echo(f"{answer}\n({elapsed:.1f}s)")
            else:
                elapsed = time.perf_counter() - start
                echo(f"\n({elapsed:.1f}s)")
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            if isinstance(exc, GeminiError):
                echo("/joamodel bilan Ollama modeliga qayting.")
            continue


@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
    backend: Backend = typer.Option(
        Backend.ollama, "--backend",
        help="ollama | gemini (gemini needs GEMINI_API_KEY in .env)"),
):
    """Interactive agent session over the repo (defaults to current dir)."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    embed_client = OllamaClient()
    try:
        chat_client = _chat_client(backend)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    ctx = ToolContext(
        root=repo.resolve(),
        data_dir=data_dir,
        embedder=embed_client.embed,
        confirm=lambda msg: typer.confirm(msg),
        output_sink=lambda t: typer.echo(t, nl=False),
    )
    session = AgentSession(ctx, chat_client)
    if sys.stdin.isatty():
        prompt_session = PromptSession(
            "joa> ", completer=SlashCompleter(),
            complete_while_typing=True)
        read_line = prompt_session.prompt
    else:
        # piped/scripted input: plain input(), no interactive dropdown
        read_line = lambda: input("joa> ")  # noqa: E731
    _repl_loop(session, read_line, typer.echo, embed_client,
               lambda t: typer.echo(t, nl=False))


def build_prompt(question: str,
                 results: list[tuple[str, float, dict]]) -> str:
    blocks = []
    for i, (_chunk_id, _score, p) in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {p['path']}:{p['start_line']}-{p['end_line']} "
            f"({p['kind']} {p['symbol']})\n{p['text']}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


if __name__ == "__main__":
    app()
