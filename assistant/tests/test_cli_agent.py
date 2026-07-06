from typer.testing import CliRunner

from assistant.cli import app

runner = CliRunner()


def test_agent_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output


def test_agent_without_index_exits_nonzero(tmp_path):
    result = runner.invoke(
        app, ["agent", "do something", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "index" in result.output.lower()
