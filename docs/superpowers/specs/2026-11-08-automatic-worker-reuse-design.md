# Automatic worker reuse from message history

## Intent

Let ordinary OpenAI Chat Completions clients reuse CLI workers without adding
`session_id` or a custom header. Avoid repeated CLI startup while preserving the
conversation represented by the client's messages.

## Approach

Use the requested message-hash cache. Hashing the entire new request would miss
follow-ups; hashing only the first user message could mix unrelated chats.
Instead, index each successfully completed conversation by a SHA-256 hash of
its canonical message history and effective worker settings.

On a subsequent request without an explicit identifier, look for an exact
completed-history prefix followed by new messages. Claim that cached worker
atomically and send only the new messages. A cache miss creates an internal
session with a unique ID and sends the complete history.

## Matching and lifecycle

- Existing explicit session IDs, headers, and prompt-cache keys take precedence.
- Include model, normalized effort, system messages, tool definitions, tool
  choice, and tool mode in matching. Preserve exact message content and tool
  call details; ignore serialization-only differences such as JSON key order.
- Cache completed histories including assistant messages and tool calls, for
  both streaming and non-streaming responses and both existing CLI backends.
- Consume a cache entry when claiming its worker. Concurrent branches or edits
  start independent workers rather than inheriting a worker's newer context.
- Publish a new cache entry only after successful completion. Failed or
  cancelled turns invalidate the automatic session and close its worker.
- Use an in-memory, bounded cache (128 automatic workers, one-hour idle expiry).
  Eviction, expiry, deletion, and shutdown release the process and its metadata.
- Built-in tools can introduce state outside the submitted transcript; keep
  those requests on the existing explicit-session/stateless behavior.
- This caches conversation-to-worker mappings, not generated answers. Every
  request still asks the model to generate a response.

## Validation

Verify full-history follow-ups reuse the actual worker and send only new turns;
cover streaming, program-tool exchanges, settings changes, edited histories,
concurrent branches, failure/cancellation, eviction/expiry, and explicit IDs.
Run the existing backend/session tests and a Haiku/low live comparison.

## Scope

Keep the existing Python/FastAPI architecture and public response shapes.
Document automatic reuse and its exact-history matching requirement in README.
No disk cache, response cache, or new API endpoint is needed.
