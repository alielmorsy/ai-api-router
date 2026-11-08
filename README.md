# OpenAI API Wrapper for Antigravity (`agy`) and Claude Code

Run a local OpenAI-compatible server on top of the CLIs you already have signed in. Point any OpenAI client at `http://127.0.0.1:8000/v1` and chat with Google Antigravity (`agy`) or Claude Code models. Sessions, streaming, reasoning effort, and caller-defined function calling all work through the standard `/v1/chat/completions` endpoint.

## How it works

```
OpenAI client ──► FastAPI (main.py) ──► model alias? ──► opus/sonnet/haiku ──► claude CLI worker
                                                    └──► anything else    ──► agy CLI worker
```

- **One server, two backends.** The request's `model` picks the CLI. `opus`, `sonnet` and `haiku` go to Claude Code. Every other slug, including `agy-default`, goes to `agy`. `GET /v1/models` lists your live `agy` model slugs plus the three Claude aliases.
- **Persistent workers.** A request with a `session_id` gets its own long-lived CLI process (`agy` or `claude`) in stream-JSON mode. Later requests in that session are sent as new turns over the open stdin, so the CLI keeps its own context and you skip process startup.
  - Different sessions run in parallel. Requests inside one session are serialized.
  - A change of model, effort, mode, schema or system prompt restarts that session's worker and replays the stored history.
  - Workers close when you delete the session, when the server shuts down, or after one idle hour.
- **Automatic worker reuse.** Clients that resend the full `messages` history need no custom session ID. The wrapper hashes completed histories and worker settings, reuses a matching idle process, and sends only the new messages. Model, effort, tool definitions, tool choice, and caller `user` participate in matching. Each request still generates a new answer.
  - Initial requests, edited histories, regenerations, and concurrent branches start independent processes. Matching requires the exact previous messages, including assistant text and tool-call IDs.
  - Up to 128 automatic workers are kept in memory and expire after one idle hour. Eviction, failed turns, disconnects, session deletion, and shutdown close their processes. If all cache slots are busy, new requests use throwaway processes.
  - Explicit `session_id`, `X-Session-ID`, or `prompt_cache_key` takes precedence. Requests with built-in tools enabled (`enable_tools: true`) need an explicit ID to keep a process; caller-defined program tools support automatic reuse.
- **No tools by default.** API requests run `agy` in `--mode plan` with the tool-less `openai-api` agent (`.agents/agents/openai-api.md`) and deny file, terminal, web and MCP tools. Claude runs with its tools disabled. Pass `enable_tools: true` to allow the CLI's built-in tools (`agy --mode auto`).
- **Program tools without execution.** When you send OpenAI `tools`, the wrapper asks `agy` for a structured JSON answer (`--json-schema`) naming the function and arguments, validates it against what you offered and `tool_choice`, and returns a normal `tool_calls` response. Your program runs the function and replies with a `role: "tool"` message. The wrapper never runs your functions, and `agy`'s built-in tools stay denied for these requests.
- **Observability.** Console logs show worker start, reuse and close, turn starts, tool states, result status and token totals, tagged with request and session IDs. Workspace hooks (`.agents/hooks.json`) add CLI-side session, model and tool events. Prompts and tool arguments are never logged.

## Quick start

Prerequisites: Python 3.10+, [uv](https://docs.astral.sh/uv/), and at least one signed-in CLI: `agy` ([headless docs](https://antigravity.google/docs/cli/headless/)) and/or `claude` (`npm install -g @anthropic-ai/claude-code`, then `claude auth login`).

```powershell
git clone https://github.com/RichardAtCT/claude-code-openai-wrapper
cd claude-code-openai-wrapper
uv sync

agy --version                      # sanity check
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Test it from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models

$body = @{ model = 'agy-default'; effort = 'high'; messages = @(@{ role = 'user'; content = 'Hello' }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

Or with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="no-auth-required")
r = client.chat.completions.create(
    model="haiku",                      # Claude; use an agy slug for Antigravity
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"session_id": "demo", "effort": "low"},
)
print(r.choices[0].message.content)
```

## Request options

| Field | Meaning |
|-------|---------|
| `model` | `opus` / `sonnet` / `haiku` → Claude; `agy-default` or any `agy` slug → Antigravity |
| `effort` / `reasoning_effort` | `low`, `medium`, `high`, `xhigh`, `max`. Passed as `--effort`; the CLI rejects levels the model lacks. If both are set they must match. |
| `session_id` | Keeps a persistent worker and conversation history |
| `enable_tools` | Allow the CLI's built-in tools instead of the locked-down API mode |
| `tools`, `tool_choice`, `parallel_tool_calls` | OpenAI-style program function calling (agy only) |
| `stream` | OpenAI SSE |

### Sessions

Standard clients can send the complete conversation on every request:

```python
messages = [{"role": "user", "content": "My name is Alice."}]
first = client.chat.completions.create(
    model="haiku", messages=messages, extra_body={"effort": "low"},
)
messages += [first.choices[0].message.model_dump(exclude_none=True),
             {"role": "user", "content": "What's my name?"}]
second = client.chat.completions.create(
    model="haiku", messages=messages, extra_body={"effort": "low"},
)
```

The second call automatically reuses the first call's worker. To send only new
messages instead, supply a stable explicit session ID:

```python
client.chat.completions.create(model="sonnet", extra_body={"session_id": "s1"},
    messages=[{"role": "user", "content": "My name is Alice."}])
client.chat.completions.create(model="sonnet", extra_body={"session_id": "s1"},
    messages=[{"role": "user", "content": "What's my name?"}])
```

Manage them with `GET /v1/sessions`, `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}` (also closes the worker) and `GET /v1/sessions/stats`.

### Program tools

```python
tools = [{"type": "function", "function": {
    "name": "get_number", "description": "Return the test number",
    "parameters": {"type": "object", "properties": {}}}}]
messages = [{"role": "user", "content": "Test get_number and tell me its result."}]

first = client.chat.completions.create(model="agy-default", messages=messages,
                                       tools=tools, tool_choice="required")
call = first.choices[0].message.tool_calls[0]
messages += [first.choices[0].message.model_dump(exclude_none=True),
             {"role": "tool", "tool_call_id": call.id, "content": "42"}]  # you run it
final = client.chat.completions.create(model="agy-default", messages=messages,
                                       tools=tools, tool_choice="none")
print(final.choices[0].message.content)
```

`tool_choice` supports `auto`, `none`, `required` and a named function. Arguments come back as a JSON string, so validate them before executing. Works with both workers (`opus`/`sonnet`/`haiku` go to Claude Code via `--json-schema`, everything else to `agy`). Streaming works, but chunks are sent after the worker finishes, because structured output only arrives in the final `result` event. Plain text requests stream `text_delta` fragments live.

## Configuration

Set in `.env` or the environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGY_CWD` | current dir | Working directory for `agy` |
| `CLAUDE_CWD` | current dir | Working directory for `claude` |
| `AGY_API_AGENT` | `openai-api` | Tool-limited agent used for API requests |
| `API_KEY` | none | Require `Authorization: Bearer <key>`. If unset, startup offers to generate one. |
| `MAX_TIMEOUT` | `600000` | Per-request timeout in ms |
| `PORT` | `8000` | Port for `python main.py` |
| `CORS_ORIGINS` | `["*"]` | Allowed origins |
| `DEBUG_MODE`, `VERBOSE` | `false` | Extra logging |

If `AGY_CWD` points at another workspace, copy `.agents/agents/openai-api.md` (or set `AGY_API_AGENT`) and `.agents/hooks.json` + `.agents/agy_hook_logger.py` into that workspace's `.agents/` directory so the tool lockdown and hook logging still apply. Every `agy` run uses `--sandbox`, `--dangerously-skip-permissions` and an `always-proceed` artifact review setting so headless runs never wait on prompts.

## Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/auth/status`, `GET /health`
- `GET|DELETE /v1/sessions[/{id}]`, `GET /v1/sessions/stats`

## Limitations

- Images in messages become text placeholders.
- `temperature`, `top_p`, `max_tokens`, penalties and `logit_bias` are not mapped. `n > 1` is unsupported.
- Program function calling requires the `agy` backend.
- Session history lives in memory and expires after one idle hour.

## Tests

```powershell
uv run pytest test_agy_backend.py test_agy_tools.py test_agy_worker.py test_agy_hook_logging.py test_claude_worker.py
```

`test_endpoints.py` and `test_basic.py` expect a running server.

## License

MIT
