from assistant import config
from assistant.agent.tools import ToolContext, run_cmd


def make_ctx(tmp_path, confirm=lambda _msg: True):
    return ToolContext(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        embedder=lambda texts: [[0.0] for _ in texts],
        confirm=confirm,
    )


def test_successful_command_reports_exit_code_zero(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "echo hi"})
    assert "exit code: 0" in out
    assert "hi" in out


def test_failing_command_reports_nonzero_exit_code(tmp_path):
    out = run_cmd(make_ctx(tmp_path), {"command": "exit 3"})
    assert "exit code: 3" in out


def test_default_timeout_comes_from_config(tmp_path):
    assert config.RUN_CMD_TIMEOUT == 120
