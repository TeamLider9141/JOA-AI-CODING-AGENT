import subprocess
import threading
from collections.abc import Callable
from pathlib import Path


def run_streaming(
    command: str,
    cwd: Path,
    on_output: Callable[[str], None],
    timeout: float | None = None,
) -> tuple[int, str, bool]:
    """Run `command` in a shell under `cwd`, calling `on_output(chunk)` as
    raw output arrives. Returns (returncode, full_output, timed_out).

    Reads binary chunks (not text-mode lines) so a bare carriage return
    from a progress bar passes through untouched instead of being
    translated to a newline — the terminal handles \\r itself, giving a
    real in-place progress bar when `on_output` prints without a trailing
    newline.
    """
    proc = subprocess.Popen(
        command, shell=True, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    chunks: list[bytes] = []

    def _read() -> None:
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            chunks.append(chunk)
            on_output(chunk.decode(errors="replace"))

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        timed_out = True
    reader.join(timeout=2)
    output = b"".join(chunks).decode(errors="replace")
    return proc.returncode, output, timed_out
