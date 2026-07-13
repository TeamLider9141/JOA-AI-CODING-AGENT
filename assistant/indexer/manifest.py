import json
from pathlib import Path

from assistant.indexer.walker import walk_repo

MANIFEST_FILENAME = "vector_manifest.json"


def repo_fingerprint(repo: Path) -> dict:
    """{relative_path: [mtime, size]} for every file walk_repo() would
    index right now. JSON-serializable — used directly as the on-disk
    manifest and compared for equality to detect any change (content,
    add, remove) since the last vector build."""
    fingerprint: dict[str, list] = {}
    for path in walk_repo(repo):
        rel = str(path.relative_to(repo))
        stat = path.stat()
        fingerprint[rel] = [stat.st_mtime, stat.st_size]
    return fingerprint


def load_manifest(data_dir: Path) -> dict | None:
    path = data_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(data_dir: Path, fingerprint: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MANIFEST_FILENAME).write_text(json.dumps(fingerprint))
