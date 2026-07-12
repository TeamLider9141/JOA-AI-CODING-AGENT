from assistant.cli import _fast_answer


class FakeStreamClient:
    """chat_stream yields the given chunks; records the messages sent."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.calls = []

    def chat_stream(self, messages):
        self.calls.append(messages)
        yield from self._chunks


class FakeSession:
    def __init__(self, client, messages=None):
        self.client = client
        self.messages = messages or [
            {"role": "system", "content": "agent system prompt"}]


def test_plain_answer_streams_and_lands_in_history():
    client = FakeStreamClient(["The answer", " is 4."])
    session = FakeSession(client)
    tokens = []

    answer = _fast_answer(session, "what is 2+2?", tokens.append)

    assert answer == "The answer is 4."
    assert "".join(tokens) == "The answer is 4."
    assert session.messages[-2] == {
        "role": "user", "content": "what is 2+2?"}
    assert session.messages[-1] == {
        "role": "assistant", "content": "The answer is 4."}


def test_fast_prompt_replaces_agent_system_prompt():
    client = FakeStreamClient(["hi"])
    session = FakeSession(client, messages=[
        {"role": "system", "content": "agent system prompt"},
        {"role": "user", "content": "earlier turn"},
    ])

    _fast_answer(session, "hello", lambda _t: None)

    sent = client.calls[0]
    assert sent[0]["role"] == "system"
    assert "agent system prompt" not in sent[0]["content"]
    assert "ESCALATE" in sent[0]["content"]
    assert {"role": "user", "content": "earlier turn"} in sent
    assert sent[-1] == {"role": "user", "content": "hello"}


def test_escalate_returns_none_and_appends_nothing():
    client = FakeStreamClient(["ESCALATE"])
    session = FakeSession(client)
    tokens = []

    assert _fast_answer(session, "fix the bug", tokens.append) is None
    assert tokens == []
    assert len(session.messages) == 1


def test_escalate_split_across_chunks_and_lowercase():
    client = FakeStreamClient(["esc", "alate"])
    session = FakeSession(client)
    tokens = []

    assert _fast_answer(session, "fix it", tokens.append) is None
    assert tokens == []


def test_escalate_with_trailing_text_still_escalates():
    client = FakeStreamClient(["ESCALATE — this needs tools"])
    session = FakeSession(client)

    assert _fast_answer(session, "edit file", lambda _t: None) is None


def test_short_answer_smaller_than_sniff_buffer():
    client = FakeStreamClient(["4"])
    session = FakeSession(client)
    tokens = []

    answer = _fast_answer(session, "2+2?", tokens.append)

    assert answer == "4"
    assert "".join(tokens) == "4"


def test_empty_stream_returns_none():
    client = FakeStreamClient([])
    session = FakeSession(client)

    assert _fast_answer(session, "anything", lambda _t: None) is None
    assert len(session.messages) == 1
