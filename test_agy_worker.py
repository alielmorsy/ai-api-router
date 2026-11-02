import asyncio
import json
import logging

import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from agy_cli import AgyCLI


class FakeProcess:
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
        self.events.put_nowait(json.dumps({
            "event": "result", "result": {"status": "SUCCESS", "response": f"turn {len(self.turns)}"}
        }).encode() + b"\n")

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
async def test_session_worker_reuses_process_and_sends_only_new_turn(monkeypatch):
    processes = []

    async def start(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = AgyCLI(timeout=30000)
    try:
        first = [x async for x in cli.run_completion("Human: one", session_id="s", incremental_prompt="Human: one")]
        second = [x async for x in cli.run_completion("Human: one\n\nAssistant: turn 1\n\nHuman: two", session_id="s", incremental_prompt="Human: two")]
        assert len(processes) == 1
        assert [x["message"]["content"] for x in processes[0].turns] == ["Human: one", "Human: two"]
        assert cli.parse_claude_message(first) == "turn 1"
        assert cli.parse_claude_message(second) == "turn 2"
    finally:
        await cli.shutdown()


@pytest.mark.asyncio
async def test_worker_isolated_by_session_and_restarted_on_settings_change(monkeypatch):
    processes = []

    async def start(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = AgyCLI(timeout=30000)
    try:
        for session, model in (("a", "m1"), ("b", "m1"), ("a", "m2")):
            _ = [x async for x in cli.run_completion("full", session_id=session, incremental_prompt="new", model=model)]
        assert len(processes) == 3
        assert processes[0].returncode == -9
        assert processes[2].turns[0]["message"]["content"] == "full"
    finally:
        await cli.shutdown()


@pytest.mark.asyncio
async def test_different_sessions_run_concurrently_and_idle_workers_close(monkeypatch):
    processes = []

    async def start(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = AgyCLI(timeout=30000)
    try:
        async def turn(session):
            return [x async for x in cli.run_completion(session, session_id=session)]

        await asyncio.gather(turn("a"), turn("b"))
        assert len(processes) == 2
        assert all(process.returncode is None for process in processes)
        await cli.reap_idle(0)
        assert all(process.returncode == -9 for process in processes)
    finally:
        await cli.shutdown()


@pytest.mark.asyncio
async def test_worker_logs_lifecycle_without_prompt(monkeypatch, caplog):
    async def start(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    caplog.set_level(logging.INFO, logger="agy_cli")
    cli = AgyCLI(timeout=30000)
    try:
        for prompt in ("private first message", "private second message"):
            _ = [x async for x in cli.run_completion(prompt, session_id="logged", request_id="req-1")]
    finally:
        await cli.shutdown()
    assert "action=start" in caplog.text
    assert "action=reuse" in caplog.text
    assert "action=result" in caplog.text
    assert "action=close" in caplog.text
    assert "private first message" not in caplog.text


@pytest.mark.asyncio
async def test_forbidden_builtin_tool_attempt_fails_before_timeout(monkeypatch):
    class ToolAttempt(FakeProcess):
        def write(self, data):
            self.turns.append(json.loads(data))
            self.events.put_nowait(json.dumps({"event": "step_update", "step_update": {
                "step_type": "tool", "tool_name": "read_file", "state": "ACTIVE"
            }}).encode() + b"\n")

    async def start(*args, **kwargs):
        return ToolAttempt()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    cli = AgyCLI(timeout=30000)
    with pytest.raises(RuntimeError, match="built-in tool read_file"):
        await asyncio.wait_for(
            anext(cli.run_completion("Choose get_number", external_tools=True,
                                     structured_schema={"type": "object"})), timeout=1
        )


@pytest.mark.asyncio
async def test_api_agent_finish_step_allows_structured_result(monkeypatch):
    class FinishProcess(FakeProcess):
        def write(self, data):
            self.turns.append(json.loads(data))
            self.events.put_nowait(json.dumps({"event": "step_update", "step_update": {
                "step_type": "tool", "tool_name": "finish", "state": "ACTIVE"
            }}).encode() + b"\n")
            self.events.put_nowait(json.dumps({"event": "result", "result": {
                "status": "SUCCESS", "structured_output": {"content": "ok", "tool_calls": []}
            }}).encode() + b"\n")

    async def start(*args, **kwargs):
        return FinishProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    chunks = [x async for x in AgyCLI().run_completion(
        "Return JSON", external_tools=True, structured_schema={"type": "object"}
    )]
    assert chunks == [{"type": "result", "structured_output": {"content": "ok", "tool_calls": []}}]


def test_api_passes_session_and_new_turn_to_worker(monkeypatch):
    import main

    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        yield {"content": [{"type": "text", "text": "answer"}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_cli, "shutdown", AsyncMock())
    with TestClient(main.app) as client:
        for message in ("one", "two"):
            response = client.post("/v1/chat/completions", json={
                "model": "agy-default", "session_id": "worker-test", "messages": [
                    {"role": "user", "content": message}
                ],
            })
            assert response.status_code == 200, response.text
    assert [call["session_id"] for call in calls] == ["worker-test", "worker-test"]
    assert calls[1]["incremental_prompt"] == "Human: two"
    assert "Human: one" in calls[1]["prompt"]


@pytest.mark.asyncio
async def test_same_session_requests_wait_for_prior_history_update(monkeypatch):
    import main

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        yield {"content": [{"type": "text", "text": "answer"}]}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        def body(text):
            return {"model": "agy-default", "session_id": "concurrent-worker-test",
                    "messages": [{"role": "user", "content": text}]}

        first = asyncio.create_task(client.post("/v1/chat/completions", json=body("one")))
        await entered.wait()
        second = asyncio.create_task(client.post("/v1/chat/completions", json=body("two")))
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        release.set()
        responses = await asyncio.gather(first, second)
    assert all(response.status_code == 200 for response in responses)
    assert "Assistant: answer" in calls[1]["prompt"]
