"""Adapter from Antigravity's headless JSON output to the wrapper's message format."""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional


logger = logging.getLogger(__name__)
CLI_EVENT_LINE_LIMIT = 16 * 1024 * 1024
BUILTIN_TOOL_DENY_RULES = [
    "read_file(*)", "write_file(*)", "read_url(*)", "execute_url(*)",
    "command(*)", "mcp(*)",
]


class _AgyWorker:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.lock = asyncio.Lock()
        self.process = None
        self.command = None
        self.system_prompt = None
        self.hook_dir = None
        self.hook_log = None
        self.stderr_task = None
        self.last_used = time.monotonic()
        self.hook_lines = 0


async def _stop_worker(worker: _AgyWorker, request_id: str = "shutdown", reason: str = "stop") -> None:
    process = worker.process
    if process is None:
        return
    logger.info("agy worker request=%s session=%r pid=%s action=close reason=%s",
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
        worker.hook_lines = log_hook_events(worker.hook_log, request_id, worker.hook_lines)
    finally:
        worker.hook_dir.cleanup()
        worker.process = None
        worker.hook_dir = None
        worker.stderr_task = None


def log_hook_events(path: Path, request_id: str, start_line: int = 0) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return start_line
    except OSError as exc:
        logger.warning("agy hook request=%s could not read event log: %s", request_id, exc)
        return start_line
    for line in lines[start_line:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("agy hook request=%s invalid event record", request_id)
            continue
        if not isinstance(event, dict):
            logger.warning("agy hook request=%s invalid event record", request_id)
            continue
        logger.info(
            "agy hook request=%s conversation=%s event=%s tool=%s outcome=%s step=%s model=%s",
            request_id,
            event.get("conversation_id"),
            event.get("event"),
            event.get("tool"),
            event.get("outcome"),
            event.get("step"),
            event.get("model"),
        )
    return len(lines)


class AgyCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self._workers: Dict[str, _AgyWorker] = {}
        self._workers_lock = asyncio.Lock()

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

    async def verify_cli(self) -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                "agy", "--version", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (OSError, asyncio.TimeoutError):
            logger.warning("Antigravity CLI is unavailable")
            return False

    async def list_models(self) -> List[Dict[str, str]]:
        """Read the account's currently available model slugs from the CLI."""
        process = await asyncio.create_subprocess_exec(
            "agy", "models", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=min(self.timeout, 30)
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise RuntimeError("Antigravity model listing timed out") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Antigravity model listing failed: {detail[-500:]}")
        models = []
        for line in stdout.decode("utf-8-sig").splitlines():
            if "\t" not in line:
                continue
            slug, name = line.split("\t", 1)
            if slug.strip() and name.strip():
                models.append({
                    "id": slug.strip(), "object": "model", "name": name.strip(),
                    "owned_by": "antigravity",
                })
        if not models:
            raise RuntimeError("Antigravity returned no available models")
        return models

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
        structured_schema: Optional[Dict[str, Any]] = None,
        external_tools: bool = False,
        request_id: Optional[str] = None,
        incremental_prompt: Optional[str] = None,
        context_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        deny_builtin_tools = external_tools or bool(disallowed_tools)
        command = [
            "agy", "--mode", "plan" if deny_builtin_tools else "auto",
            "--sandbox", "--dangerously-skip-permissions",
            "--settings", json.dumps({
                "artifactReviewPolicy": "always-proceed",
                "permissions": {"deny": BUILTIN_TOOL_DENY_RULES} if deny_builtin_tools else {},
            }),
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--print-timeout", f"{int(self.timeout)}s",
        ]
        if deny_builtin_tools:
            command.extend(["--agent", os.getenv("AGY_API_AGENT", "openai-api")])
        if model and model != "agy-default":
            command.extend(["--model", model])
        if effort:
            command.extend(["--effort", effort])
        if structured_schema is not None:
            command.extend(["--json-schema", json.dumps(structured_schema)])

        run_id = request_id or f"agy-{uuid.uuid4().hex[:12]}"
        if session_id is not None:
            async with self._workers_lock:
                worker = self._workers.setdefault(session_id, _AgyWorker(session_id))
        else:
            worker = _AgyWorker()
        async with worker.lock:
            # A changed command or system instruction requires a fresh conversation.
            if worker.process is not None and (
                worker.process.returncode is not None
                or worker.command != command
                or worker.system_prompt != (context_key if context_key is not None else system_prompt)
            ):
                await _stop_worker(worker, run_id, reason="settings_changed_or_exited")
            fresh = worker.process is None
            if fresh:
                worker.hook_dir = tempfile.TemporaryDirectory(prefix="agy-wrapper-hooks-")
                worker.hook_log = Path(worker.hook_dir.name) / "events.jsonl"
                try:
                    worker.process = await asyncio.create_subprocess_exec(
                        *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
                        limit=CLI_EVENT_LINE_LIMIT,
                        env={**os.environ, "AGY_WRAPPER_HOOK_LOG": str(worker.hook_log)},
                    )
                except BaseException:
                    worker.hook_dir.cleanup()
                    worker.hook_dir = None
                    raise
                worker.command = command
                worker.system_prompt = context_key if context_key is not None else system_prompt
                worker.hook_lines = 0
                worker.stderr_task = asyncio.create_task(worker.process.stderr.read())
                logger.info("agy worker request=%s session=%r pid=%s action=start mode=%s model=%s",
                            run_id, session_id, getattr(worker.process, "pid", None),
                            command[2], model or "agy-default")
            else:
                logger.info("agy worker request=%s session=%r pid=%s action=reuse",
                            run_id, session_id, getattr(worker.process, "pid", None))
            process = worker.process
            deadline = time.monotonic() + self.timeout + 5
            result = None
            streamed = False
            completed = False
            try:
                turn = prompt if fresh or session_id is None else incremental_prompt or prompt
                if system_prompt and (fresh or session_id is None):
                    turn = f"System instructions:\n{system_prompt}\n\nUser conversation:\n{turn}"
                request = {"event": "user", "message": {"content": turn}}
                logger.info("agy worker request=%s session=%r action=turn_start", run_id, session_id)
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
                        raise RuntimeError("Antigravity CLI returned invalid JSON") from exc
                    if event.get("event") == "step_update":
                        step = event.get("step_update", {})
                        if step.get("step_type") == "tool":
                            tool_name = step.get("tool_name") or (step.get("tool_info") or {}).get("name")
                            logger.info("agy worker request=%s session=%r action=tool name=%s state=%s",
                                        run_id, session_id, tool_name, step.get("state"))
                            if deny_builtin_tools and tool_name != "finish" and step.get("state") == "ACTIVE":
                                raise RuntimeError(
                                    f"Antigravity attempted built-in tool {tool_name} while API tools are disabled"
                                )
                        delta = step.get("text_delta")
                        if structured_schema is None and step.get("step_type") == "agent_response" and delta:
                            streamed = True
                            yield {"content": [{"type": "text", "text": delta}]}
                    elif event.get("event") == "result":
                        result = event.get("result", {})
                        logger.info("agy worker request=%s session=%r action=result status=%s tokens=%s",
                                    run_id, session_id, result.get("status"),
                                    (result.get("usage") or {}).get("total_tokens"))
                        break
                if session_id is None:
                    await asyncio.wait_for(process.wait(), timeout=max(0.1, deadline - time.monotonic()))
                if result is None:
                    detail = b"" if process.returncode is None else await worker.stderr_task
                    raise RuntimeError(f"Antigravity CLI returned no result: {detail.decode('utf-8', errors='replace').strip()[-500:]}")
                if process.returncode not in (None, 0) or result.get("status") != "SUCCESS":
                    raise RuntimeError(result.get("error") or "Antigravity CLI failed")
                worker.hook_lines = log_hook_events(worker.hook_log, run_id, worker.hook_lines)
                if structured_schema is not None:
                    structured_output = result.get("structured_output")
                    if not isinstance(structured_output, dict):
                        raise RuntimeError("Antigravity CLI did not return structured output")
                    completed = True
                    yield {"type": "result", "structured_output": structured_output}
                else:
                    if not streamed and result.get("response"):
                        yield {"content": [{"type": "text", "text": result["response"]}]}
                    completed = True
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Antigravity CLI timed out") from exc
            finally:
                worker.last_used = time.monotonic()
                if session_id is None or not completed:
                    await _stop_worker(worker, run_id, reason="request_end" if completed else "incomplete")

    @staticmethod
    def parse_claude_message(messages: List[Dict[str, Any]]) -> Optional[str]:
        parts = [
            block["text"]
            for message in messages
            for block in message.get("content", [])
            if block.get("type") == "text"
        ]
        return "".join(parts) or None
