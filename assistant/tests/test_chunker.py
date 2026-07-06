from assistant.indexer.chunker import chunk_file

SAMPLE = '''\
import os


def top_level(a, b):
    return a + b


class UserService:
    """Service for users."""

    def login(self, user):
        return True

    def logout(self):
        return False
'''


def write(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def test_functions_and_methods_become_chunks(tmp_path):
    f = write(tmp_path, "svc.py", SAMPLE)
    symbols = {c.symbol for c in chunk_file(f, tmp_path)}
    assert {"top_level", "UserService.login", "UserService.logout"} <= symbols


def test_method_chunk_carries_class_header_and_real_lines(tmp_path):
    f = write(tmp_path, "svc.py", SAMPLE)
    login = next(c for c in chunk_file(f, tmp_path)
                 if c.symbol == "UserService.login")
    assert login.kind == "method"
    assert login.text.startswith("class UserService:")
    assert login.start_line == 11  # actual def line in SAMPLE
    assert login.path == "svc.py"


def test_class_without_methods_is_one_chunk(tmp_path):
    f = write(tmp_path, "cfg.py", "class Config:\n    DEBUG = True\n")
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) == 1
    assert chunks[0].kind == "class"
    assert chunks[0].symbol == "Config"


def test_unknown_extension_falls_back_to_text_windows(tmp_path):
    f = write(tmp_path, "notes.md",
              "\n".join(f"line {i}" for i in range(200)))
    chunks = chunk_file(f, tmp_path)
    assert all(c.kind == "text" for c in chunks)
    assert len(chunks) >= 2


def test_decorated_function_is_found(tmp_path):
    f = write(tmp_path, "app.py",
              "@app.route('/x')\ndef handler():\n    return 1\n")
    symbols = {c.symbol for c in chunk_file(f, tmp_path)}
    assert "handler" in symbols
