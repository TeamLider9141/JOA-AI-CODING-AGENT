from pathlib import Path

import pathspec

HARD_EXCLUDES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".data", "storage", "dist", "build", ".mypy_cache", ".pytest_cache",
}

TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".html", ".css", ".sh", ".sql",
}

MAX_FILE_BYTES = 512 * 1024


def walk_repo(root: Path) -> list[Path]:
    gitignore = root / ".gitignore"
    spec = None
    if gitignore.exists():
        spec = pathspec.PathSpec.from_lines(
            "gitignore", gitignore.read_text().splitlines())

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in HARD_EXCLUDES for part in rel.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        if spec is not None and spec.match_file(str(rel)):
            continue
        files.append(path)
    return files
