"""Regression coverage for large newline-delimited CLI events."""

import asyncio
import json

import pytest

from agy_cli import AgyCLI
from claude_cli import ClaudeCodeCLI


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["agy", "claude"])
async def test_worker_accepts_json_event_larger_than_default_stream_limit(monkeypatch, backend):
    answer = "x" * (128 * 1024)
    if backend == "agy":
        events = [{"event": "result", "result": {"status": "SUCCESS", "response": answer}}]
        cli = AgyCLI(timeout=30000)
    else:
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}},
            {"type": "result", "subtype": "success", "is_error": False},
        ]
        cli = ClaudeCodeCLI(timeout=30000)

    class Process:
        def __init__(self, limit):
            self.returncode = 0
            self.stdout = asyncio.StreamReader(limit=limit)
            self.stdin = self
            self.stderr = self

        def write(self, data):
            for event in events:
                self.stdout.feed_data((json.dumps(event) + "\n").encode())
            self.stdout.feed_eof()

        async def drain(self):
            pass

        def close(self):
            pass

        async def read(self):
            return b""

        async def wait(self):
            return self.returncode

    async def start(*args, **kwargs):
        return Process(kwargs.get("limit", 64 * 1024))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    chunks = [chunk async for chunk in cli.run_completion("hello")]
    assert cli.parse_claude_message(chunks) == answer
