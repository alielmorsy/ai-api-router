import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_hook_commands_resolve_from_agents_directory():
    agents_dir = Path(__file__).parent / ".agents"
    config = json.loads((agents_dir / "hooks.json").read_text(encoding="utf-8"))
    hook = config["agy-wrapper-tool-log"]
    commands = [entry["command"] for event in ("SessionStart", "PreInvocation") for entry in hook[event]]
    commands += [
        entry["command"]
        for event in ("PreToolUse", "PostToolUse")
        for group in hook[event]
        for entry in group["hooks"]
    ]
    assert all(command.startswith("python agy_hook_logger.py ") for command in commands)
    assert (agents_dir / "agy_hook_logger.py").is_file()


def test_hook_records_tool_name_without_arguments(tmp_path):
    hook_log = tmp_path / "hook-events.jsonl"
    env = {**os.environ, "AGY_WRAPPER_HOOK_LOG": str(hook_log)}
    hook_script = Path(__file__).parent / ".agents" / "agy_hook_logger.py"
    result = subprocess.run(
        [sys.executable, str(hook_script), "pre"],
        input=json.dumps({
            "conversationId": "conv-1", "modelName": "gemini-test", "stepIdx": 2,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "secret command"}},
        }),
        text=True, capture_output=True, env=env, check=True,
    )
    assert json.loads(result.stdout) == {"decision": "allow"}
    event = json.loads(hook_log.read_text(encoding="utf-8"))
    assert event["tool"] == "run_command"
    assert event["event"] == "pre"
    assert "secret command" not in hook_log.read_text(encoding="utf-8")


def test_hook_records_post_error(tmp_path):
    hook_log = tmp_path / "hook-events.jsonl"
    env = {**os.environ, "AGY_WRAPPER_HOOK_LOG": str(hook_log)}
    hook_script = Path(__file__).parent / ".agents" / "agy_hook_logger.py"
    result = subprocess.run(
        [sys.executable, str(hook_script), "post"],
        input=json.dumps({
            "conversationId": "conv-1", "stepIdx": 2,
            "toolCall": {"name": "run_command", "args": {}}, "error": "permission denied",
        }),
        text=True, capture_output=True, env=env, check=True,
    )
    assert json.loads(result.stdout) == {}
    event = json.loads(hook_log.read_text(encoding="utf-8"))
    assert event["outcome"] == "error"


def test_hook_records_session_start(tmp_path):
    hook_log = tmp_path / "hook-events.jsonl"
    env = {**os.environ, "AGY_WRAPPER_HOOK_LOG": str(hook_log)}
    hook_script = Path(__file__).parent / ".agents" / "agy_hook_logger.py"
    result = subprocess.run(
        [sys.executable, str(hook_script), "session"],
        input=json.dumps({"conversationId": "conv-1", "modelName": "gemini-test"}),
        text=True, capture_output=True, env=env, check=True,
    )
    assert json.loads(result.stdout) == {}
    event = json.loads(hook_log.read_text(encoding="utf-8"))
    assert event["outcome"] == "started"


@pytest.mark.asyncio
async def test_agy_replays_hook_events_into_server_log(monkeypatch, caplog):
    from agy_cli import AgyCLI

    class Output:
        def __init__(self, lines=()):
            self.lines = iter(lines)

        async def readline(self):
            return next(self.lines, b"")

        async def read(self):
            return b""

    class Process:
        returncode = 0
        stdin = type("Input", (), {
            "write": lambda self, data: None,
            "drain": lambda self: asyncio.sleep(0),
            "close": lambda self: None,
        })()
        stdout = Output([
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "pong"}}).encode() + b"\n"
        ])
        stderr = Output()

        async def wait(self):
            return 0

    async def start(*args, **kwargs):
        hook_log = Path(kwargs["env"]["AGY_WRAPPER_HOOK_LOG"])
        hook_log.write_text(json.dumps({
            "event": "pre", "tool": "view_file", "outcome": "attempted",
            "conversation_id": "conv-1", "step": 1,
        }) + "\n", encoding="utf-8")
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    caplog.set_level(logging.INFO, logger="agy_cli")
    chunks = [chunk async for chunk in AgyCLI().run_completion("ping")]
    assert chunks[0]["content"][0]["text"] == "pong"
    assert "tool=view_file" in caplog.text
    assert "event=pre" in caplog.text
