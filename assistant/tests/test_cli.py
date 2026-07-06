from typer.testing import CliRunner

from assistant.cli import app, build_prompt

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "ask"):
        assert cmd in result.output


def test_build_prompt_contains_citations_and_question():
    results = [("id1", 0.5, {
        "path": "auth.py", "start_line": 3, "end_line": 9,
        "kind": "class", "symbol": "JWTMiddleware",
        "text": "class JWTMiddleware: ...",
    })]
    prompt = build_prompt("where is auth?", results)
    assert "auth.py:3-9" in prompt
    assert "JWTMiddleware" in prompt
    assert "where is auth?" in prompt


def test_search_without_index_exits_nonzero(tmp_path):
    result = runner.invoke(
        app, ["search", "query", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "index" in result.output.lower()
