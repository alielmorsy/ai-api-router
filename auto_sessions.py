"""Lease persistent CLI sessions by hashes of completed OpenAI conversations."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

import anyio
from starlette.responses import StreamingResponse

from models import ChatCompletionRequest, Message
from session_manager import SessionManager

logger = logging.getLogger(__name__)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _history_hashes(messages: List[Message], settings: str) -> List[str]:
    digest = hashlib.sha256(settings.encode("ascii"))
    hashes = []
    for message in messages:
        encoded = _canonical(message.model_dump(exclude_none=True))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        hashes.append(digest.hexdigest())
    return hashes


@dataclass
class _Entry:
    settings: str
    last_used: float
    history_hash: Optional[str] = None
    busy: bool = True


class AutomaticSessionCache:
    def __init__(
        self,
        manager: SessionManager,
        close_session: Callable[[str], Awaitable[None]],
        capacity: int = 128,
        ttl_seconds: float = 3600,
    ):
        self.manager = manager
        self.close_session = close_session
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._index: Dict[str, str] = {}

    async def reserve(self, request: ChatCompletionRequest, options: dict) -> Optional[str]:
        """Atomically claim a matching idle worker, or allocate an independent session.

        Mutation before each await is intentional: two concurrent branches must
        never lease the same mutable worker. Call from the server's event loop.
        """
        if request.enable_tools:
            return None
        await self.reap()
        settings = hashlib.sha256(
            _canonical(
                {
                    "model": request.model,
                    "effort": request.agy_effort,
                    "user": request.user,
                    "system": [
                        message.model_dump(exclude_none=True)
                        for message in request.messages
                        if message.role == "system"
                    ],
                    "tools": (
                        None
                        if request.tools is None
                        else [tool.model_dump(exclude_none=True) for tool in request.tools]
                    ),
                    "tool_choice": request.tool_choice,
                    "parallel_tool_calls": request.parallel_tool_calls,
                    "options": options,
                }
            )
        ).hexdigest()
        # The complete input is deliberately excluded: a retry/regeneration
        # must generate again from its input, not continue after its old answer.
        for prefix in reversed(_history_hashes(request.messages, settings)[:-1]):
            sid = self._index.pop(prefix, None)
            if sid is not None:
                entry = self._entries[sid]
                entry.busy = True
                entry.history_hash = None
                entry.last_used = time.monotonic()
                self._entries.move_to_end(sid)
                if self.manager.get_session(sid) is not None:
                    logger.info("automatic session=%s action=reuse", sid)
                    return sid
                await self.discard(sid)
        victim = None
        if len(self._entries) >= self.capacity:
            victim = next((sid for sid, entry in self._entries.items() if not entry.busy), None)
            if victim is None:
                return None
            self._forget(victim)
        sid = "auto-" + uuid.uuid4().hex
        self._entries[sid] = _Entry(settings, time.monotonic())
        if victim is not None:
            try:
                await self._close_many([victim])
            except BaseException:
                self._forget(sid)
                raise
        logger.info("automatic session=%s action=create", sid)
        return sid

    async def finish(self, session_id: str, success: bool) -> None:
        entry = self._entries.get(session_id)
        if entry is None or not entry.busy:
            return
        session = self.manager.get_session(session_id) if success else None
        if session is None or not session.messages or session.messages[-1].role != "assistant":
            await self.discard(session_id)
            return
        entry.history_hash = _history_hashes(session.messages, entry.settings)[-1]
        entry.last_used = time.monotonic()
        entry.busy = False
        self._entries.move_to_end(session_id)
        self._index[entry.history_hash] = session_id

    def _forget(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry and entry.history_hash and self._index.get(entry.history_hash) == session_id:
            del self._index[entry.history_hash]

    async def _close(self, session_id: str) -> None:
        try:
            await self.close_session(session_id)
        finally:
            self.manager.delete_session(session_id)
            # Internal sessions have only one lease, so no waiting callers.
            self.manager._request_locks.pop(session_id, None)

    async def discard(self, session_id: str) -> None:
        if session_id in self._entries:
            self._forget(session_id)
            await self._close_many([session_id])

    async def _close_many(self, sessions: List[str]) -> None:
        """Finish every removed worker even when the calling request disconnects."""
        if not sessions:
            return
        cleanup = asyncio.gather(*(self._close(sid) for sid in sessions))
        with anyio.CancelScope(shield=True):
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def reap(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, entry in self._entries.items()
            if not entry.busy and now - entry.last_used >= self.ttl_seconds
        ]
        # Forget all victims before awaiting process cleanup.
        for sid in expired:
            self._forget(sid)
        await self._close_many(expired)

    async def shutdown(self) -> None:
        sessions = list(self._entries)
        self._entries.clear()
        self._index.clear()
        await self._close_many(sessions)


class CachedStreamingResponse(StreamingResponse):
    """Own the cache lease for the entire ASGI response, including headers."""

    def __init__(self, response: StreamingResponse, cache: AutomaticSessionCache, session_id: str):
        self._source = response.body_iterator
        self._cache = cache
        self._session_id = session_id
        self._complete = False
        super().__init__(
            self._stream(),
            status_code=response.status_code,
            media_type=response.media_type,
            background=response.background,
        )
        self.raw_headers = response.raw_headers

    async def _stream(self):
        async for chunk in self._source:
            marker = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            if marker.strip() == "data: [DONE]":
                self._complete = True
            yield chunk

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                    await self._source.aclose()
                finally:
                    await self._cache.finish(self._session_id, self._complete)
