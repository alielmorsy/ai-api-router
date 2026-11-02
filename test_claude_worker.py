import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from claude_cli import ClaudeCodeCLI


class FakeClaudeProcess:
    def __init__(self):
        self.returncode = None
        self.turns = []
        self.events = asyncio.Queue()
        self.stdin = self
        self.stdout = self
        self.stderr = type("Stderr", (), {"read": self._stderr})()

    async def _stderr(self):
        return b""

    def write(self, data):
        self.turns.append(json.loads(data))
        answer = f"turn {len(self.turns)}"
        for event in (
            {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": answer}}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}},
            {"type": "result", "subtype": "success", "is_error": False},
        ):
            self.events.put_nowait((json.dumps(event) + "\n").encode())

    async def drain(self):
        pass

    def close(self):
        self.events.put_nowait(b"")

    async def readline(self):
        return await self.events.get()

    def kill(self):
        self.returncode = -9
        self.events.put_nowait(b"")

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_claude_worker_reuses_session_and_streams_once(monkeypatch):
    processes = []
    commands = []

    async def start(*args, **kwargs):
        commands.append(args)
        process = FakeClaudeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = ClaudeCodeCLI(timeout=30000)
    try:
        first = [chunk async for chunk in cli.run_completion("Human: one", session_id="s", incremental_prompt="Human: one", model="opus", effort="high")]
        second = [chunk async for chunk in cli.run_completion("Human: one\n\nAssistant: turn 1\n\nHuman: two", session_id="s", incremental_prompt="Human: two", model="opus", effort="high")]
        assert len(processes) == 1
        assert "--input-format" in commands[0] and "--output-format" in commands[0]
        assert "--verbose" in commands[0]
        assert commands[0][commands[0].index("--model") + 1] == "opus"
        assert commands[0][commands[0].index("--effort") + 1] == "high"
        assert [turn["message"]["content"] for turn in processes[0].turns] == ["Human: one", "Human: two"]
        assert cli.parse_claude_message(first) == "turn 1"
        assert cli.parse_claude_message(second) == "turn 2"
    finally:
        await cli.shutdown()


@pytest.mark.asyncio
async def test_claude_worker_restarts_for_changed_settings_and_closes_idle(monkeypatch):
    processes = []

    async def start(*args, **kwargs):
        process = FakeClaudeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = ClaudeCodeCLI(timeout=30000)
    try:
        for effort in ("low", "high"):
            _ = [chunk async for chunk in cli.run_completion("full history", session_id="s", incremental_prompt="new turn", model="sonnet", effort=effort)]
        assert len(processes) == 2
        assert processes[0].returncode == -9
        assert processes[1].turns[0]["message"]["content"] == "full history"
        await cli.reap_idle(0)
        assert processes[1].returncode == -9
    finally:
        await cli.shutdown()


def test_claude_api_lists_aliases_and_passes_effort_to_worker(monkeypatch):
    import main

    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        yield {"content": [{"type": "text", "text": "answer"}]}

    monkeypatch.setattr(main, "BACKEND", "claude")
    monkeypatch.setattr(main.claude_worker_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_worker_cli, "parse_claude_message", lambda chunks: "answer")
    monkeypatch.setattr(main.claude_cli, "list_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_cli, "shutdown", AsyncMock())
    with TestClient(main.app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert [model["id"] for model in models.json()["data"]] == ["opus", "sonnet", "haiku"]
        for message in ("one", "two"):
            response = client.post("/v1/chat/completions", json={
                "model": "sonnet", "reasoning_effort": "high", "session_id": "claude-worker-test",
                "messages": [{"role": "user", "content": message}],
            })
            assert response.status_code == 200, response.text
    assert [call["effort"] for call in calls] == ["high", "high"]
    assert [call["session_id"] for call in calls] == ["claude-worker-test", "claude-worker-test"]
    assert calls[1]["incremental_prompt"] == "Human: two"


def test_claude_alias_routes_directly_without_backend_switch(monkeypatch):
    import main

    claude_calls = []
    agy_calls = []

    async def claude_completion(**kwargs):
        claude_calls.append(kwargs)
        yield {"content": [{"type": "text", "text": "Claude answer"}]}

    async def agy_completion(**kwargs):
        agy_calls.append(kwargs)
        yield {"content": [{"type": "text", "text": "Agy answer"}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", agy_completion)
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_cli, "list_models", AsyncMock(return_value=[
        {"id": "agy-default", "object": "model", "owned_by": "antigravity"}
    ]))
    monkeypatch.setattr(main, "claude_worker_cli", ClaudeCodeCLI(), raising=False)
    monkeypatch.setattr(main.claude_worker_cli, "run_completion", claude_completion)
    monkeypatch.setattr(main.claude_worker_cli, "parse_claude_message", lambda chunks: "Claude answer")
    with TestClient(main.app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert [model["id"] for model in models.json()["data"]] == ["agy-default", "opus", "sonnet", "haiku"]
        response = client.post("/v1/chat/completions", json={
            "model": "haiku", "effort": "low", "session_id": "claude-session",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "Claude answer"
        response = client.post("/v1/chat/completions", json={
            "model": "agy-default", "session_id": "agy-session",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "Agy answer"
    assert len(claude_calls) == 1
    assert claude_calls[0]["model"] == "haiku"
    assert claude_calls[0]["effort"] == "low"
    assert len(agy_calls) == 1
    assert agy_calls[0]["model"] == "agy-default"
    assert agy_calls[0]["session_id"] == "agy-session"


def test_claude_alias_stream_preserves_spaces(monkeypatch):
    import main

    async def completion(**kwargs):
        for part in ("Hello", " ", "world", "\n"):
            yield {"content": [{"type": "text", "text": part}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_worker_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "haiku", "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        })

    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: {")]
    fragments = [event["choices"][0]["delta"]["content"] for event in events
                 if "content" in event["choices"][0]["delta"]]
    assert "".join(fragments) == "Hello world\n"


def test_explicit_backend_setting_does_not_hide_agy_sessions(monkeypatch):
    import main

    async def completion(**kwargs):
        yield {"content": [{"type": "text", "text": "Agy answer"}]}

    monkeypatch.setattr(main, "BACKEND", "claude")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_cli, "list_models", AsyncMock(return_value=[
        {"id": "agy-default", "object": "model", "owned_by": "antigravity"}
    ]))
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (False, {"errors": ["missing"]}))
    with TestClient(main.app) as client:
        models = client.get("/v1/models")
        assert [model["id"] for model in models.json()["data"]] == ["agy-default", "opus", "sonnet", "haiku"]
        response = client.post("/v1/chat/completions", json={
            "model": "agy-default", "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "Agy answer"
