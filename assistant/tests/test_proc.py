from pathlib import Path

import pytest

from assistant.agent.proc import run_streaming


def test_run_streaming_captures_output_and_streams_chunks(tmp_path):
    chunks = []
    returncode, output, timed_out = run_streaming(
        "echo hi", tmp_path, chunks.append)
    assert returncode == 0
    assert "hi" in output
    assert "hi" in "".join(chunks)
    assert timed_out is False


def test_run_streaming_reports_nonzero_exit(tmp_path):
    returncode, _output, _timed_out = run_streaming(
        "exit 3", tmp_path, lambda _c: None)
    assert returncode == 3


def test_run_streaming_times_out_and_kills_process(tmp_path):
    returncode, _output, timed_out = run_streaming(
        "sleep 5", tmp_path, lambda _c: None, timeout=1)
    assert timed_out is True
    assert returncode != 0


def test_run_streaming_delivers_carriage_return_progress_as_is(tmp_path):
    chunks = []
    run_streaming(
        "printf 'a\\rb\\rc'", tmp_path, chunks.append)
    joined = "".join(chunks)
    # raw \r must survive untouched — not translated to \n, which would
    # break in-place progress-bar redraws in a real terminal
    assert "\r" in joined
    assert "\n" not in joined
    assert joined == "a\rb\rc"


def test_run_streaming_runs_in_given_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    chunks = []
    run_streaming("ls", tmp_path, chunks.append)
    assert "marker.txt" in "".join(chunks)


def test_run_streaming_kills_process_and_reraises_on_keyboard_interrupt(
        tmp_path, monkeypatch):
    import subprocess as subprocess_module

    original_wait = subprocess_module.Popen.wait
    calls = {"count": 0}

    def fake_wait(self, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyboardInterrupt
        return original_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess_module.Popen, "wait", fake_wait)

    with pytest.raises(KeyboardInterrupt):
        run_streaming("sleep 5", tmp_path, lambda _c: None)
    # a second (real) wait() call means our code actually called
    # proc.kill() before re-raising -- otherwise this would hang for
    # the full 5s (or longer) instead of returning almost immediately
    assert calls["count"] == 2
