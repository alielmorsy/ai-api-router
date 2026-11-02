import json
import logging
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


TOOL = {
    "type": "function",
    "function": {
        "name": "get_number",
        "description": "Return the test number",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_tool_messages_and_definitions_validate():
    from models import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate({
        "model": "agy-default", "tools": [TOOL], "tool_choice": "required",
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_number", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ],
    })
    assert request.messages[1].tool_call_id == "call_1"
    assert request.tools[0].function.name == "get_number"


def test_tool_history_reaches_prompt():
    from message_adapter import MessageAdapter
    from models import Message

    prompt, _ = MessageAdapter.messages_to_prompt([
        Message(role="assistant", content=None, tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "get_number", "arguments": "{}"}}
        ]),
        Message(role="tool", tool_call_id="call_1", content="42"),
    ])
    assert "get_number" in prompt
    assert "call_1" in prompt
    assert "42" in prompt


@pytest.mark.asyncio
async def test_agy_blocks_builtin_tools_and_requests_structured_output(monkeypatch):
    import asyncio
    from agy_cli import AgyCLI

    seen = {}

    class Input:
        def write(self, data):
            seen["prompt"] = data

        async def drain(self):
            pass

        def close(self):
            pass

    class Output:
        def __init__(self, lines):
            self.lines = iter(lines)

        async def readline(self):
            return next(self.lines, b"")

        async def read(self):
            return b""

    class Process:
        returncode = 0
        stdin = Input()
        stdout = Output([json.dumps({"event": "result", "result": {
            "status": "SUCCESS", "response": "", "structured_output": {
                "content": "", "tool_calls": [{"name": "get_number", "arguments": {}}]
            }
        }}).encode() + b"\n"])
        stderr = Output([])

        async def wait(self):
            return 0

    async def start(*args, **kwargs):
        seen["args"] = args
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    chunks = [chunk async for chunk in AgyCLI().run_completion(
        "call get_number", structured_schema={"type": "object"}, external_tools=True,
    )]
    assert chunks[-1]["structured_output"]["tool_calls"][0]["name"] == "get_number"
    settings = json.loads(seen["args"][seen["args"].index("--settings") + 1])
    assert "command(*)" in settings["permissions"]["deny"]
    assert seen["args"][seen["args"].index("--mode") + 1] == "plan"
    assert seen["args"][seen["args"].index("--agent") + 1] == "openai-api"
    assert "read_file(*)" in settings["permissions"]["deny"]
    assert "mcp(*)" in settings["permissions"]["deny"]
    assert "--json-schema" in seen["args"]


def test_openai_tool_round_trip_uses_program_tools_only(monkeypatch, caplog):
    import main

    caplog.set_level(logging.INFO, logger="main")
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if "Tool result" in kwargs["prompt"]:
            output = {"content": "The number is 42.", "tool_calls": []}
        else:
            output = {"content": "", "tool_calls": [{"name": "get_number", "arguments": {}}]}
        yield {"type": "result", "structured_output": output}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    monkeypatch.setattr(main.claude_cli, "verify_cli", AsyncMock(return_value=True))
    with TestClient(main.app) as client:
        first = client.post("/v1/chat/completions", json={
            "model": "agy-default", "tools": [TOOL], "tool_choice": "required",
            "messages": [{"role": "user", "content": "Test get_number."}],
        })
        assert first.status_code == 200, first.text
        choice = first.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["function"] == {"name": "get_number", "arguments": "{}"}

        second = client.post("/v1/chat/completions", json={
            "model": "agy-default", "tools": [TOOL], "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": "Test get_number."},
                choice["message"],
                {"role": "tool", "tool_call_id": tool_call["id"], "content": "42"},
            ],
        })
        assert second.status_code == 200, second.text
        assert second.json()["choices"][0]["message"]["content"] == "The number is 42."
        streamed = client.post("/v1/chat/completions", json={
            "model": "agy-default", "tools": [TOOL], "tool_choice": "required", "stream": True,
            "messages": [{"role": "user", "content": "Test get_number."}],
        })
    assert streamed.status_code == 200
    assert '"finish_reason":"tool_calls"' in streamed.text
    assert '"tool_calls"' in streamed.text
    assert all(call["external_tools"] for call in calls)
    assert "program tool requested" in caplog.text
    assert "program tool result" in caplog.text
    assert "get_number" in caplog.text


def test_rejects_unknown_returned_tool_name():
    from agy_tools import parse_tool_result
    from models import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate({
        "model": "agy-default", "messages": [{"role": "user", "content": "Test"}], "tools": [TOOL]
    })
    with pytest.raises(ValueError, match="not offered"):
        parse_tool_result({"content": "", "tool_calls": [{"name": "run_command", "arguments": {}}]}, request)


def test_nested_structured_answer_is_returned_as_plain_content():
    from agy_tools import parse_tool_result
    from models import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate({
        "model": "agy-default", "messages": [{"role": "user", "content": "What was my first message?"}],
        "tools": [TOOL], "tool_choice": "none",
    })
    inner = {"content": 'The first message you sent was: "Hello Mr gemini"', "tool_calls": []}
    content, calls = parse_tool_result({"content": json.dumps(inner), "tool_calls": []}, request)
    assert content == inner["content"]
    assert calls == []


def test_nested_structured_tool_call_is_returned_as_tool_call():
    from agy_tools import parse_tool_result
    from models import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate({
        "model": "agy-default", "messages": [{"role": "user", "content": "Test get_number"}],
        "tools": [TOOL], "tool_choice": "required",
    })
    inner = {"content": "", "tool_calls": [{"name": "get_number", "arguments": {}}]}
    content, calls = parse_tool_result({"content": json.dumps(inner), "tool_calls": []}, request)
    assert content is None
    assert calls[0]["function"] == {"name": "get_number", "arguments": "{}"}


def test_api_does_not_expose_nested_tool_envelope(monkeypatch):
    import main

    inner = {"content": 'The first message you sent was: "Hello Mr gemini"', "tool_calls": []}

    async def completion(**kwargs):
        yield {"type": "result", "structured_output": {
            "content": json.dumps(inner), "tool_calls": [],
        }}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "agy-default", "tools": [TOOL], "tool_choice": "none",
            "messages": [{"role": "user", "content": "What was my first message?"}],
        })
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == inner["content"]


def test_stream_rejects_unknown_tool_before_sse(monkeypatch):
    import main

    async def completion(**kwargs):
        yield {"type": "result", "structured_output": {
            "content": "", "tool_calls": [{"name": "run_command", "arguments": {}}]
        }}

    monkeypatch.setattr(main, "BACKEND", "agy")
    monkeypatch.setattr(main.claude_cli, "run_completion", completion)
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "agy-default", "tools": [TOOL], "stream": True,
            "messages": [{"role": "user", "content": "Test"}],
        })
    assert response.status_code == 502
