from assistant.indexer.models import Chunk


def make(text="def f(): pass") -> Chunk:
    return Chunk(path="a.py", symbol="f", kind="function",
                 start_line=1, end_line=2, text=text)


def test_chunk_id_is_deterministic():
    assert make().chunk_id == make().chunk_id


def test_chunk_id_changes_with_content():
    assert make().chunk_id != make(text="def f(): return 1").chunk_id


def test_chunk_id_is_uuid_shaped():
    # Qdrant point ids must be UUIDs (or ints)
    import uuid
    uuid.UUID(make().chunk_id)  # raises if not a valid UUID


def test_payload_contains_all_metadata():
    p = make().payload()
    assert p == {
        "path": "a.py", "symbol": "f", "kind": "function",
        "start_line": 1, "end_line": 2, "text": "def f(): pass",
    }
