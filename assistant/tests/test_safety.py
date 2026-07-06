import pytest

from assistant.agent.safety import PathJailError, resolve_in_root


def test_normal_relative_path_resolves(tmp_path):
    (tmp_path / "db.py").write_text("x = 1")
    resolved = resolve_in_root(tmp_path, "db.py")
    assert resolved == (tmp_path / "db.py").resolve()


def test_nested_path_resolves(tmp_path):
    (tmp_path / "handlers").mkdir()
    resolved = resolve_in_root(tmp_path, "handlers/user.py")
    assert resolved == (tmp_path / "handlers" / "user.py").resolve()


def test_parent_traversal_is_blocked(tmp_path):
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "../secret.py")


def test_absolute_path_outside_root_is_blocked(tmp_path):
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "/etc/passwd")


def test_symlink_escape_is_blocked(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret")
    (tmp_path / "link.py").symlink_to(outside)
    with pytest.raises(PathJailError):
        resolve_in_root(tmp_path, "link.py")
