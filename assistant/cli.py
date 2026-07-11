import time
from pathlib import Path

import typer

from assistant import config
from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaClient, OllamaError
from assistant.llm.gemini_client import GeminiClient, GeminiError
from assistant.agent.runner import AgentSession, run_agent
from assistant.agent.tools import ToolContext

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _chat_client(backend: str):
    if backend == "gemini":
        return GeminiClient()
    return OllamaClient()


SYSTEM_PROMPT = (
    "You are a coding assistant. Answer the question using ONLY the provided "
    "context chunks. Cite sources as path:start_line-end_line. If the context "
    "is insufficient, say what is missing instead of guessing."
)


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
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
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
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
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
        )
        answer = run_agent(task, ctx, chat_client)
    except (OllamaError, GeminiError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo("--- answer ---")
    typer.echo(answer)


def _repl_loop(session, read_line, echo) -> None:
    """Drive an AgentSession from a line source until exit/EOF.

    `read_line()` returns the next input line (raising EOFError at end of
    input); `echo(text)` prints a line. Kept separate from the CLI command so
    the loop is testable without a live model.
    """
    echo("joa session — type 'exit' or Ctrl-D to quit")
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
        start = time.perf_counter()
        try:
            answer = session.send(stripped)
        except (OllamaError, GeminiError) as exc:
            echo(str(exc))
            continue
        elapsed = time.perf_counter() - start
        echo(f"{answer}\n({elapsed:.1f}s)")


@app.command()
def repl(
    repo: Path = typer.Option(Path("."), "--repo", exists=True,
                              file_okay=False),
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | gemini"),
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
    )
    session = AgentSession(ctx, chat_client)
    _repl_loop(session, lambda: input("joa> "), typer.echo)


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
