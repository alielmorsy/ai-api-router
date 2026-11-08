import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from models import ChatCompletionRequest, Message
from session_manager import SessionManager


def request(messages=None, **kwargs):
    return ChatCompletionRequest(
        model=kwargs.pop("model", "haiku"),
        messages=messages or [Message(role="user", content="one")],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_cache_claims_exact_history_once_and_closes_evicted_sessions():
    from auto_sessions import AutomaticSessionCache

    manager = SessionManager()
    close = AsyncMock()
    cache = AutomaticSessionCache(manager, close, capacity=2)
    first = await cache.reserve(request(effort="low"), {})
    manager.process_messages(request().messages, first)
    manager.add_assistant_response(first, Message(role="assistant", content="answer"))
    await cache.finish(first, True)
    history = [Message(role="user", content="one"), Message(role="assistant", content="answer")]
    followup = request(history + [Message(role="user", content="two")], reasoning_effort="low")
    assert await cache.reserve(followup, {}) == first
    # A concurrent branch cannot claim the same mutable process.
    branch = await cache.reserve(followup, {})
    assert branch != first
    # All slots are busy: fall back to a throwaway worker.
    assert await cache.reserve(followup, {}) is None
    await cache.finish(branch, False)
    close.assert_awaited_once_with(branch)
    assert manager.get_session(branch) is None
    await cache.finish(first, True)
    newer = await cache.reserve(request(), {})
    manager.process_messages(request().messages, newer)
    manager.add_assistant_response(newer, Message(role="assistant", content="new"))
    await cache.finish(newer, True)
    await cache.reserve(request(), {})
    assert first in [call.args[0] for call in close.await_args_list]
    assert manager.get_session(first) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"model": "sonnet"},
        {"effort": "high"},
        {"user": "different-user"},
        {"tools": []},
        {"tool_choice": "none"},
    ],
)
async def test_cache_isolates_settings_and_requires_exact_assistant_text(change):
    from auto_sessions import AutomaticSessionCache

    manager = SessionManager()
    cache = AutomaticSessionCache(manager, AsyncMock())
    sid = await cache.reserve(request(), {})
    manager.process_messages(request().messages, sid)
    manager.add_assistant_response(sid, Message(role="assistant", content="answer"))
    await cache.finish(sid, True)
    history = list(manager.get_session(sid).messages)
    assert (
        await cache.reserve(request(history + [Message(role="user", content="two")], **change), {})
        != sid
    )
    edited = [
        history[0],
        Message(role="assistant", content="edited"),
        Message(role="user", content="two"),
    ]
    assert await cache.reserve(request(edited), {}) != sid
    assert await cache.reserve(request(history + [Message(role="user", content="two")]), {}) == sid


@pytest.mark.asyncio
async def test_cache_expiry_and_failed_turn_remove_process_and_metadata():
    from auto_sessions import AutomaticSessionCache

    manager = SessionManager()
    close = AsyncMock()
    cache = AutomaticSessionCache(manager, close, ttl_seconds=0)
    sid = await cache.reserve(request(), {})
    manager.process_messages(request().messages, sid)
    manager.add_assistant_response(sid, Message(role="assistant", content="answer"))
    await cache.finish(sid, True)
    await cache.reap()
    assert manager.get_session(sid) is None
    close.assert_awaited_once_with(sid)
    assert sid not in manager._request_locks


@pytest.mark.asyncio
async def test_changed_system_instructions_or_headers_do_not_claim_old_worker():
    from auto_sessions import AutomaticSessionCache

    manager = SessionManager()
    cache = AutomaticSessionCache(manager, AsyncMock())
    sid = await cache.reserve(request(), {})
    manager.process_messages(request().messages, sid)
    manager.add_assistant_response(sid, Message(role="assistant", content="answer"))
    await cache.finish(sid, True)
    history = list(manager.get_session(sid).messages)
    assert (
        await cache.reserve(
            request(
                history
                + [
                    Message(role="system", content="new instructions"),
                    Message(role="user", content="two"),
                ]
            ),
            {},
        )
        != sid
    )
    assert (
        await cache.reserve(
            request(history + [Message(role="user", content="two")]), {"max_turns": 3}
        )
        != sid
    )
    assert await cache.reserve(request(history + [Message(role="user", content="two")]), {}) == sid


@pytest.fixture
def api(monkeypatch):
    import main
    from claude_cli import ClaudeCodeCLI
    from test_claude_worker import FakeClaudeProcess

    processes = []

    async def spawn(*args, **kwargs):
        process = FakeClaudeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(main, "session_manager", SessionManager())
    monkeypatch.setattr(main, "claude_worker_cli", ClaudeCodeCLI())
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main.claude_worker_cli, "verify_cli", AsyncMock(return_value=True))
    monkeypatch.setattr(main, "verify_api_key", AsyncMock())
    # A cache belongs to this server/session manager, not another test's server.
    from auto_sessions import AutomaticSessionCache

    monkeypatch.setattr(
        main,
        "auto_sessions",
        AutomaticSessionCache(main.session_manager, main.close_cached_session),
    )
    with TestClient(main.app) as client:
        yield client, processes, main


@pytest.mark.parametrize("stream", [False, True])
def test_standard_client_reuses_actual_process_with_only_new_messages(api, stream):
    client, processes, main = api
    history = [{"role": "user", "content": "one"}]
    for index in range(2):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "haiku",
                "effort": "low",
                "messages": history,
                "stream": stream,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            answer = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
        else:
            answer = response.json()["choices"][0]["message"]["content"]
        assert answer == f"turn {index + 1}"
        history += [{"role": "assistant", "content": answer}, {"role": "user", "content": "two"}]
    assert len(processes) == 1
    assert [turn["message"]["content"] for turn in processes[0].turns] == [
        "Human: one",
        "Human: two",
    ]


def test_edited_or_repeated_requests_get_independent_processes(api):
    client, processes, main = api
    for _ in range(2):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "haiku",
                "messages": [{"role": "user", "content": "one"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "turn 1"
    assert len(processes) == 2


def test_builtin_tools_keep_stateless_behavior(api):
    client, processes, main = api
    for _ in range(2):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "haiku",
                "enable_tools": True,
                "messages": [{"role": "user", "content": "one"}],
            },
        )
        assert response.status_code == 200
    assert len(processes) == 2
    assert all(process.returncode is not None for process in processes)
    assert not main.session_manager.sessions


@pytest.mark.parametrize("stream", [False, True])
def test_failed_turn_closes_worker_and_does_not_publish_history(api, monkeypatch, stream):
    client, processes, main = api
    from test_claude_worker import FakeClaudeProcess

    def fail(self, data):
        self.turns.append(json.loads(data))
        self.events.put_nowait(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "error",
                    "is_error": True,
                    "result": "test failure",
                }
            ).encode()
            + b"\n"
        )

    monkeypatch.setattr(FakeClaudeProcess, "write", fail)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "haiku",
            "messages": [{"role": "user", "content": "one"}],
            "stream": stream,
        },
    )
    assert "test failure" in response.text
    assert all(process.returncode is not None for process in processes)
    assert not main.session_manager.sessions
    assert not main.session_manager._request_locks


@pytest.mark.parametrize("stream", [False, True])
def test_program_tool_result_reuses_process_and_preserves_call_identity(api, monkeypatch, stream):
    client, processes, main = api
    from test_claude_worker import FakeClaudeProcess

    def structured(self, data):
        self.turns.append(json.loads(data))
        output = (
            {"content": "", "tool_calls": [{"name": "number", "arguments": {}}]}
            if len(self.turns) == 1
            else {"content": "42", "tool_calls": []}
        )
        self.events.put_nowait(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "structured_output": output,
                }
            ).encode()
            + b"\n"
        )

    monkeypatch.setattr(FakeClaudeProcess, "write", structured)
    tools = [{"type": "function", "function": {"name": "number", "parameters": {"type": "object"}}}]
    history = [{"role": "user", "content": "Get the number"}]
    response = client.post(
        "/v1/chat/completions", json={"model": "haiku", "tools": tools, "messages": history}
    )
    assert response.status_code == 200, response.text
    assistant = response.json()["choices"][0]["message"]
    history += [
        assistant,
        {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "42"},
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "haiku",
            "tools": tools,
            "messages": history,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    assert "42" in response.text
    assert len(processes) == 1
    assert len(processes[0].turns) == 2
    assert "Get the number" not in processes[0].turns[1]["message"]["content"]


def test_explicit_ids_take_precedence_and_deleted_auto_sessions_are_not_reused(api):
    client, processes, main = api
    history = [{"role": "user", "content": "one"}]
    first = client.post("/v1/chat/completions", json={"model": "haiku", "messages": history}).json()
    sid = next(iter(main.session_manager.sessions))
    history += [first["choices"][0]["message"], {"role": "user", "content": "two"}]
    explicit = client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "explicit"},
        json={
            "model": "haiku",
            "messages": history,
        },
    )
    assert explicit.status_code == 200
    assert len(processes) == 2
    assert client.delete(f"/v1/sessions/{sid}").status_code == 200
    assert processes[0].returncode is not None
    assert (
        client.post(
            "/v1/chat/completions", json={"model": "haiku", "messages": history}
        ).status_code
        == 200
    )
    assert len(processes) == 3


@pytest.mark.asyncio
async def test_stream_cancellation_releases_automatic_session(monkeypatch):
    import main
    from starlette.requests import Request
    from auto_sessions import AutomaticSessionCache
    from test_claude_worker import FakeClaudeProcess
    from claude_cli import ClaudeCodeCLI

    processes = []
    reached = asyncio.Event()

    class WaitingProcess(FakeClaudeProcess):
        def write(self, data):
            self.turns.append(json.loads(data))
            reached.set()

    async def spawn(*args, **kwargs):
        process = WaitingProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(main, "session_manager", SessionManager())
    monkeypatch.setattr(main, "claude_worker_cli", ClaudeCodeCLI())
    monkeypatch.setattr(main, "verify_api_key", AsyncMock())
    monkeypatch.setattr(
        main,
        "auto_sessions",
        AutomaticSessionCache(main.session_manager, main.close_cached_session),
    )
    http_request = Request({"type": "http", "headers": []})
    response = await main.chat_completions(request(stream=True), http_request, None)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        pass

    async def consume():
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    task = asyncio.create_task(consume())
    await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(processes) == 1 and processes[0].returncode is not None
    assert not main.session_manager.sessions
    assert not main.session_manager._request_locks


@pytest.mark.asyncio
async def test_cancellation_during_eviction_does_not_strand_busy_reservation():
    from auto_sessions import AutomaticSessionCache

    manager = SessionManager()
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = []

    async def close(sid):
        closing.set()
        await release.wait()
        closed.append(sid)

    cache = AutomaticSessionCache(manager, close, capacity=1)
    sid = await cache.reserve(request(), {})
    manager.process_messages(request().messages, sid)
    manager.add_assistant_response(sid, Message(role="assistant", content="answer"))
    await cache.finish(sid, True)
    task = asyncio.create_task(cache.reserve(request(), {}))
    await closing.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [sid]
    assert not manager.sessions
    # All capacity must be available after the cancelled reservation.
    assert await cache.reserve(request(), {}) is not None


@pytest.mark.asyncio
async def test_disconnect_before_stream_starts_does_not_strand_lease():
    from auto_sessions import AutomaticSessionCache, CachedStreamingResponse
    from starlette.responses import StreamingResponse

    manager = SessionManager()
    close = AsyncMock()
    cache = AutomaticSessionCache(manager, close, capacity=1)
    sid = await cache.reserve(request(stream=True), {})
    manager.process_messages(request().messages, sid)
    manager.add_assistant_response(
        sid, Message(role="assistant", content="already generated tool result")
    )

    async def chunks():
        yield "data: [DONE]\n\n"

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        raise RuntimeError("disconnect during response headers")

    response = CachedStreamingResponse(StreamingResponse(chunks()), cache, sid)
    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    close.assert_awaited_once_with(sid)
    assert not manager.sessions
    assert await cache.reserve(request(), {}) is not None


def test_branch_from_already_advanced_history_gets_fresh_process(api):
    client, processes, main = api
    history = [{"role": "user", "content": "one"}]
    first = client.post("/v1/chat/completions", json={"model": "haiku", "messages": history}).json()
    history += [first["choices"][0]["message"], {"role": "user", "content": "two"}]
    for expected in ("turn 2", "turn 1"):
        response = client.post("/v1/chat/completions", json={"model": "haiku", "messages": history})
        assert response.json()["choices"][0]["message"]["content"] == expected
    assert len(processes) == 2
    assert "Assistant: turn 1" in processes[1].turns[0]["message"]["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_agy_standard_client_also_reuses_actual_process(api, monkeypatch, stream):
    from agy_cli import AgyCLI
    from test_agy_worker import FakeProcess

    client, processes, main = api

    async def spawn(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(main, "claude_cli", AgyCLI())
    history = [{"role": "user", "content": "one"}]
    first = client.post("/v1/chat/completions", json={"model": "agy-default", "messages": history})
    assert first.status_code == 200
    history += [first.json()["choices"][0]["message"], {"role": "user", "content": "two"}]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "agy-default",
            "messages": history,
            "stream": stream,
        },
    )
    assert second.status_code == 200, second.text
    assert "turn 2" in second.text
    assert len(processes) == 1
    assert processes[0].turns[1]["message"]["content"] == "Human: two"
