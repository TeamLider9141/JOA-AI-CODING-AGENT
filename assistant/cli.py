from pathlib import Path

import typer

from assistant import config
from assistant.indexer.pipeline import build_index, search_index
from assistant.llm.ollama_client import OllamaClient, OllamaError

app = typer.Typer(no_args_is_help=True, add_completion=False)

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
):
    """Ask a question about the indexed repository."""
    data_dir = _data_dir(repo)
    _require_index(data_dir)
    client = OllamaClient()
    try:
        results = search_index(question, data_dir, client.embed)
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
        for token in client.chat_stream(messages):
            typer.echo(token, nl=False)
        typer.echo()
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


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
