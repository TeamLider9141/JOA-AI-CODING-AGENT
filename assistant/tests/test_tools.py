import pytest

from assistant.agent.safety import PathJailError
from assistant.agent.tools import (
    ToolContext, ToolError, TOOLS, read_file, write_file, run_cmd,
)


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_registry_exposes_all_four_tools():
    assert set(TOOLS) == {"read_file", "write_file", "run_cmd", "search_code"}


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "a.py").write_text("hello")
    assert read_file(make_ctx(tmp_path), {"path": "a.py"}) == "hello"


def test_read_file_missing_raises_tool_error(tmp_path):
    with pytest.raises(ToolError):
        read_file(make_ctx(tmp_path), {"path": "nope.py"})


def test_read_file_traversal_raises_path_jail(tmp_path):
    with pytest.raises(PathJailError):
        read_file(make_ctx(tmp_path), {"path": "../x"})


def test_write_file_creates_file_when_confirmed(tmp_path):
    result = write_file(make_ctx(tmp_path),
                        {"path": "new.py", "content": "print(1)"})
    assert (tmp_path / "new.py").read_text() == "print(1)"
    assert "new.py" in result


def test_write_file_declined_leaves_no_file(tmp_path):
    ctx = make_ctx(tmp_path, confirm=lambda _msg: False)
    result = write_file(ctx, {"path": "new.py", "content": "x"})
    assert not (tmp_path / "new.py").exists()
    assert "cancel" in result.lower()


def test_run_cmd_returns_output_when_confirmed(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "echo hi"})
    assert "hi" in out


def test_run_cmd_declined_does_not_run(tmp_path):
    ctx = make_ctx(tmp_path, confirm=lambda _msg: False)
    assert "cancel" in run_cmd(ctx, {"command": "echo hi"}).lower()


def test_run_cmd_times_out(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "sleep 5"}, timeout=1)
    assert "timed out" in out.lower()
