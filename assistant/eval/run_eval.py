"""Retrieval quality eval: hit@5 for vector-only vs hybrid.

Usage:
    .venv/bin/python -m assistant.eval.run_eval --repo ~/Desktop/crystal_bot
"""
from pathlib import Path

import typer
import yaml

from assistant import config
from assistant.indexer.pipeline import search_index
from assistant.llm.ollama_client import OllamaClient

GOLD_PATH = Path(__file__).parent / "gold.yaml"


def main(repo: Path = typer.Option(..., "--repo", exists=True)):
    gold = yaml.safe_load(GOLD_PATH.read_text())
    data_dir = config.DATA_DIR / repo.resolve().name
    client = OllamaClient()

    for mode in ("vector", "hybrid"):
        hits = 0
        for item in gold:
            results = search_index(
                item["question"], data_dir, client.embed, mode=mode)
            paths = [p["path"] for _cid, _s, p in results[:5]]
            if any(item["expect_path_contains"] in path for path in paths):
                hits += 1
        print(f"{mode:7s} hit@5: {hits}/{len(gold)}")


if __name__ == "__main__":
    typer.run(main)
