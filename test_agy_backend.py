import asyncio
import json
import logging
import os
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


def test_agy_is_default_backend():
    env = os.environ.copy()
    env.pop("AGENT_BACKEND", None)
    result = subprocess.run(
        [sys.executable, "-c", "import dotenv; dotenv.load_dotenv = lambda: None; import main; print(main.BACKEND)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "agy"


@pytest.mark.asyncio
async def test_agy_completion_uses_json_result_and_default_model(monkeypatch):
    from agy_cli import AgyCLI

    seen = {}

    class Output:
        def __init__(self, lines):
            self.lines = iter(lines)

        async def readline(self):
            return next(self.lines, b"")

        async def read(self):
            return b""

    class Input:
        def write(self, data):
            seen["input"] = data

        async def drain(self):
            pass

        def close(self):
            pass

    class Process:
        returncode = 0
        stdin = Input()
        stdout = Output([
            json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "pong"}}).encode() + b"\n",
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "pong"}}).encode() + b"\n",
        ])
        stderr = Output([])

        async def wait(self):
            return 0

    async def start(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    backend = AgyCLI(timeout=30000)
    chunks = [chunk async for chunk in backend.run_completion("ping", model="agy-default", effort="high")]

    assert backend.parse_claude_message(chunks) == "pong"
    assert "--model" not in seen["args"]
    assert seen["args"][seen["args"].index("--effort") + 1] == "high"
    assert "--input-format" in seen["args"]
    assert "--sandbox" in seen["args"]
    assert "--dangerously-skip-permissions" in seen["args"]
    assert "--agent" not in seen["args"]
    settings = json.loads(seen["args"][seen["args"].index("--settings") + 1])
    assert settings["artifactReviewPolicy"] == "always-proceed"
    assert settings["permissions"] == {}
    assert json.loads(seen["input"])["message"]["content"] == "ping"
    assert seen["kwargs"]["cwd"] == str(backend.cwd)


@pytest.mark.asyncio
async def test_agy_lists_installed_cli_models(monkeypatch):
    from agy_cli import AgyCLI

    class Process:
        returncode = 0

        async def communicate(self):
            return (b"gemini-3.8-flash-high\tGemini 3.8 Flash (High)\nclaude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n", b"Fetching available models...\n")

    async def start(*args, **kwargs):
        assert args == ("agy", "models")
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    models = await AgyCLI().list_models()
    assert models == [
        {"id": "gemini-3.8-flash-high", "object": "model", "name": "Gemini 3.8 Flash (High)", "owned_by": "antigravity"},
        {"id": "claude-sonnet-4-6", "object": "model", "name": "Claude Sonnet 4.6 (Thinking)", "owned_by": "antigravity"},
    ]


def test_chat_request_accepts_effort_names():
    from models import ChatCompletionRequest

    base = {"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "Hi"}]}
    assert ChatCompletionRequest.model_validate({**base, "effort": "high"}).agy_effort == "high"
    assert ChatCompletionRequest.model_validate({**base, "reasoning_effort": "medium"}).agy_effort == "medium"
    with pytest.raises(ValueError):
        ChatCompletionRequest.model_validate({**base, "effort": "invalid"})


@pytest.mark.asyncio
async def test_agy_failure_is_not_returned_as_assistant_text(monkeypatch):
    from agy_cli import AgyCLI

    class Process:
        returncode = 1
        stdin = type("Input", (), {"write": lambda self, data: None, "drain": AsyncMock(), "close": lambda self: None})()
        stdout = type("Output", (), {"readline": AsyncMock(side_effect=[json.dumps({"event": "result", "result": {"status": "ERROR", "error": "authentication required"}}).encode() + b"\n", b""])})()
        stderr = type("Errors", (), {"read": AsyncMock(return_value=b"")})()
        wait = AsyncMock(return_value=1)

    async def start(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    with pytest.raises(RuntimeError, match="authentication required"):
        async for _ in AgyCLI().run_completion("ping"):
            pass


def test_agy_request_skips_claude_auth(monkeypatch, caplog):
    import main

    caplog.set_level(logging.INFO, logger="main")
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        yield {"content": [{"type": "text", "text": "pong"}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_cli, "parse_claude_message", lambda chunks: "pong")
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_cli, "list_models", AsyncMock(return_value=[
        {"id": "gemini-3.8-flash-high", "object": "model", "name": "Gemini 3.8 Flash (High)", "owned_by": "antigravity"}
    ]), raising=False)
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (False, {"errors": ["missing"]}))
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={"model": "agy-default", "messages": [{"role": "user", "content": "ping"}], "effort": "high"})
        streamed = client.post("/v1/chat/completions", json={"model": "agy-default", "messages": [{"role": "user", "content": "ping"}], "stream": True, "reasoning_effort": "medium"})
        models = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "pong"
    assert streamed.status_code == 200
    assert '"content":"pong"' in streamed.text
    assert "data: [DONE]" in streamed.text
    assert models.json()["data"][0]["id"] == "gemini-3.8-flash-high"
    assert [call["effort"] for call in calls] == ["high", "medium"]
    assert "may not be supported by Claude Code SDK" not in caplog.text
    assert "Antigravity plan mode selected" in caplog.text


def test_agy_stream_preserves_incremental_text_fragments(monkeypatch):
    import main

    async def completion(**kwargs):
        for part in ("Hello", " ", "world", "\n"):
            yield {"content": [{"type": "text", "text": part}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "agy-default", "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        })

    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: {")]
    fragments = [event["choices"][0]["delta"]["content"] for event in events
                 if "content" in event["choices"][0]["delta"]]
    assert "".join(fragments) == "Hello world\n"
    assert response.text.endswith("data: [DONE]\n\n")
