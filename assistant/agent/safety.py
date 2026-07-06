from pathlib import Path


class PathJailError(RuntimeError):
    """A proposed path resolves outside the target repository root."""


def resolve_in_root(root: Path, rel: str) -> Path:
    """Resolve `rel` against `root` and guarantee it stays inside root.

    Uses fully-resolved (symlink-followed) real paths on both sides, so
    `..` segments, absolute paths, and symlink escapes are all rejected.
    """
    root_real = root.resolve()
    candidate = (root_real / rel).resolve()
    if candidate != root_real and root_real not in candidate.parents:
        raise PathJailError(f"path escapes repo root: {rel}")
    return candidate
