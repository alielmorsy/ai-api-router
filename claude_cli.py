"""Persistent Claude Code CLI worker for OpenAI-compatible chat requests."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional


logger = logging.getLogger(__name__)


class _ClaudeWorker:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.lock = asyncio.Lock()
        self.process = None
        self.command = None
        self.stderr_task = None
        self.last_used = time.monotonic()


async def _stop_worker(worker: _ClaudeWorker, request_id: str = "shutdown", reason: str = "stop") -> None:
    process = worker.process
    if process is None:
        return
    logger.info("claude worker request=%s session=%r pid=%s action=close reason=%s",
                request_id, worker.session_id, getattr(process, "pid", None), reason)
    try:
        if process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)
        if worker.stderr_task:
            try:
                await asyncio.wait_for(worker.stderr_task, timeout=5)
            except asyncio.TimeoutError:
                worker.stderr_task.cancel()
    finally:
        worker.process = None
        worker.command = None
        worker.stderr_task = None


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000
        self.cwd = Path(cwd) if cwd else Path.cwd()
        from auth import auth_manager, validate_claude_code_auth

        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning("Claude Code authentication issues detected: %s", auth_info["errors"])
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()
        self._workers: Dict[str, _ClaudeWorker] = {}
        self._workers_lock = asyncio.Lock()

    async def verify_cli(self) -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                "claude", "--version", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (OSError, asyncio.TimeoutError):
            logger.warning("Claude Code CLI is unavailable")
            return False

    async def close_session(self, session_id: str) -> None:
        async with self._workers_lock:
            worker = self._workers.pop(session_id, None)
        if worker:
            async with worker.lock:
                await _stop_worker(worker, reason="session_deleted")

    async def shutdown(self) -> None:
        async with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            async with worker.lock:
                await _stop_worker(worker)

    async def reap_idle(self, idle_seconds: float = 3600) -> None:
        now = time.monotonic()
        async with self._workers_lock:
            idle = [worker for worker in self._workers.values()
                    if worker.process is not None and now - worker.last_used >= idle_seconds]
        for worker in idle:
            async with worker.lock:
                if worker.process is not None and time.monotonic() - worker.last_used >= idle_seconds:
                    await _stop_worker(worker, reason="idle")

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        request_id: Optional[str] = None,
        incremental_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        command = [
            "claude", "--print", "--verbose", "--input-format", "stream-json",
            "--output-format", "stream-json", "--include-partial-messages",
            "--no-session-persistence", "--permission-prompts", "none",
            "--max-turns", str(max_turns),
        ]
        if model:
            command.extend(["--model", model])
        if effort:
            command.extend(["--effort", effort])
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        if disallowed_tools:
            command.extend(["--tools", ""])
        elif allowed_tools:
            command.extend(["--allowedTools", ",".join(allowed_tools)])

        if session_id is not None:
            async with self._workers_lock:
                worker = self._workers.setdefault(session_id, _ClaudeWorker(session_id))
        else:
            worker = _ClaudeWorker()
        run_id = request_id or "unknown"
        async with worker.lock:
            if worker.process is not None and (
                worker.process.returncode is not None or worker.command != command
            ):
                await _stop_worker(worker, run_id, reason="settings_changed_or_exited")
            fresh = worker.process is None
            if fresh:
                worker.process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
                    env={**os.environ, **self.claude_env_vars},
                )
                worker.command = command
                worker.stderr_task = asyncio.create_task(worker.process.stderr.read())
                logger.info("claude worker request=%s session=%r pid=%s action=start model=%s effort=%s",
                            run_id, session_id, getattr(worker.process, "pid", None), model, effort)
            else:
                logger.info("claude worker request=%s session=%r pid=%s action=reuse",
                            run_id, session_id, getattr(worker.process, "pid", None))

            process = worker.process
            completed = False
            saw_result = False
            streamed_current_assistant = False
            deadline = time.monotonic() + self.timeout + 5
            try:
                turn = prompt if fresh or session_id is None else incremental_prompt or prompt
                request = {"type": "user", "message": {"role": "user", "content": turn}}
                logger.info("claude worker request=%s session=%r action=turn_start", run_id, session_id)
                process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                await process.stdin.drain()
                if session_id is None:
                    process.stdin.close()

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                    if not line:
                        break
                    try:
                        event = json.loads(line.decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise RuntimeError("Claude Code CLI returned invalid JSON") from exc
                    event_type = event.get("type")
                    if event_type == "stream_event":
                        inner = event.get("event") or {}
                        delta = inner.get("delta") or {}
                        if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                            text = delta.get("text")
                            if text:
                                streamed_current_assistant = True
                                yield {"content": [{"type": "text", "text": text}]}
                    elif event_type == "assistant":
                        if not streamed_current_assistant:
                            blocks = event.get("message", {}).get("content", [])
                            text = "".join(block.get("text", "") for block in blocks
                                           if isinstance(block, dict) and block.get("type") == "text")
                            if text:
                                yield {"content": [{"type": "text", "text": text}]}
                        streamed_current_assistant = False
                    elif event_type == "result":
                        saw_result = True
                        logger.info("claude worker request=%s session=%r action=result subtype=%s cost=%s",
                                    run_id, session_id, event.get("subtype"), event.get("total_cost_usd"))
                        if event.get("is_error") or event.get("subtype") != "success":
                            raise RuntimeError(event.get("result") or event.get("error") or "Claude Code CLI failed")
                        completed = True
                        break
                if not saw_result:
                    if process.returncode is None:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    detail = await worker.stderr_task
                    raise RuntimeError("Claude Code CLI returned no result: " +
                                       detail.decode("utf-8", errors="replace").strip()[-500:])
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Claude Code CLI timed out") from exc
            finally:
                worker.last_used = time.monotonic()
                if session_id is None or not completed:
                    await _stop_worker(worker, run_id,
                                       reason="request_end" if completed else "incomplete")

    @staticmethod
    def parse_claude_message(messages: List[Dict[str, Any]]) -> Optional[str]:
        return "".join(
            block["text"] for message in messages for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ) or None
