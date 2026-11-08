# Automatic Worker Reuse Implementation Plan

> Use superpowers:executing-plans for inline execution. The user authorized completing implementation and verification without intermediate approval.

**Goal:** Reuse CLI workers automatically for standard clients that resend full message history.
**Architecture:** A bounded cache leases internal sessions by exact completed-history hashes and request settings. Existing session management and CLI worker pools retain their responsibilities.
**Tech Stack:** Python, FastAPI, asyncio, SHA-256, pytest.
**Spec:** ../specs/2026-11-08-automatic-worker-reuse-design.md

## Constraints and decisions

- Preserve explicit identifier precedence and built-in-tool behavior.
- Keep at most 128 automatically owned workers; expire idle entries after one hour.
- Publish only completed turns. Consume matches atomically before awaiting work.
- Work in the user's current checkout; retain their existing design artifact.
- No commit or push is needed for this local change.

## Review focus

- Concurrent branches must not append to an already advanced worker.
- Client disconnects and worker errors must close automatic sessions.
- Tool-call history and exact assistant text participate in matching.
- Settings changes must miss even when message history is identical.
- Eviction and deletion must remove cache, process, and session metadata.

## Tasks

- [x] Add failing tests in `test_auto_sessions.py` for exact-history continuation, branching, settings isolation, capacity/expiry, and invalidation.
- [x] Implement `AutomaticSessionCache` in `auto_sessions.py`, with reserve, publish, discard, and reap lifecycle operations.
- [x] Add API integration tests using real worker pools with fake CLI processes for streaming, non-streaming, explicit identifiers, tools, and failures.
- [x] Integrate automatic session reservation and completion in `main.py`; preserve streaming request-lock lifetime and close resources on disconnect.
- [x] Update README, run targeted tests, run full pytest, review the diff, and run Haiku/low through localhost without session identifiers.

## Execution record

Ruling: Implement inline and use the current checkout, per the user's direction to proceed A to Z without stopping. No unrelated changes or publication.

Review: Independent review found cancellation during eviction and disconnect
before first stream iteration could strand leases. Both received regression
tests and fixes; follow-up review found no remaining concrete blockers.

Validation: Focused backend suite passes. Bare pytest was attempted and stopped
when legacy live-server tests exercised the already-running service on port
8000; it is not claimed as a passing full-suite run.

Live validation: OpenAI AsyncOpenAI SDK, Haiku/low, no session identifiers,
full-history requests, temporary localhost server. Three turns used PID 95840;
times were 6.336s, 5.332s, and 1.694s (third turn streamed, first text at 1.407s).
The second turn correctly recalled cobalt-731. Temporary server and worker
were shut down afterward.
