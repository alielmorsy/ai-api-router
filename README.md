# Claude Code OpenAI API Wrapper

## Use Google Antigravity CLI (`agy`)

The locally installed Antigravity CLI is the default response backend. Install and sign in to `agy` first, then start the server. The server keeps the same OpenAI-compatible `/v1/chat/completions` endpoint. `GET /v1/models` returns the model slugs and display names currently available to your `agy` account. You can still use `agy-default` to select the CLI's default model. Set `AGENT_BACKEND=claude` to use the Claude Code SDK instead. See [Antigravity's headless CLI documentation](https://antigravity.google/docs/cli/headless/) for CLI authentication and model selection.

In PowerShell:

```powershell
agy --version
agy -p 'Reply with pong' --output-format json
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Then send a request:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models'
$body = @{ model = 'agy-default'; effort = 'high'; messages = @(@{ role = 'user'; content = 'Hello' }); stream = $false } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $body
```

Set `AGY_CWD` to choose the working directory for the CLI. For requests with a `session_id`, the wrapper keeps one headless `agy` process per active session and sends later messages as new turns over its open stream JSON stdin. Different sessions can run concurrently; requests for one session are serialized. A model, effort, mode, schema, or system instruction change restarts that session's worker with its full stored history. Stateless requests still use isolated CLI processes so unrelated chats cannot share context. Deleting a session, shutting down the server, or leaving a worker idle for an hour closes its process. By default, requests use `agy --mode plan` with the `openai-api` agent in `.agents/agents/openai-api.md`, which has no file, terminal, web, or MCP tools. Requests with `enable_tools: true` use `agy --mode auto` and its built-in tools. If `AGY_CWD` points to another workspace, copy the `openai-api.md` agent into that workspace's `.agents/agents/` directory or install it globally, or set `AGY_API_AGENT` to the name of an equivalent tool-limited agent. Every run passes `--sandbox` and `--dangerously-skip-permissions`, plus a session-only artifact review setting of `always-proceed`, so headless requests do not wait for tool or plan review prompts. The API's `API_KEY` protection still applies if configured. Streaming requests use Antigravity's `stream-json` output.

Requests can set `effort` or `reasoning_effort` to `low`, `medium`, `high`, `xhigh`, or `max`. The wrapper passes it as `agy --effort` for that request; the CLI checks whether the selected model supports the level. If both fields are provided, they must match.

### Test tools supplied by your program

Send OpenAI-style function definitions in `tools`. The wrapper asks `agy` which function to call and returns an OpenAI `tool_calls` response. Your program executes the function and sends its output back as a `role: "tool"` message. The wrapper never executes the function itself. Antigravity's permission-controlled file, terminal, web, and MCP tools are denied for these API runs, even though `--dangerously-skip-permissions` is present. You do not need `enable_tools` for program tools.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="no-auth-required")
tools = [{"type": "function", "function": {
    "name": "get_number",
    "description": "Return the test number",
    "parameters": {"type": "object", "properties": {}},
}}]
messages = [{"role": "user", "content": "Test get_number and tell me its result."}]

first = client.chat.completions.create(
    model="agy-default", messages=messages, tools=tools, tool_choice="required"
)
call = first.choices[0].message.tool_calls[0]
assert call.function.name == "get_number"
result = "42"  # Your program runs get_number here.

messages.extend([
    first.choices[0].message.model_dump(exclude_none=True),
    {"role": "tool", "tool_call_id": call.id, "content": result},
])
final = client.chat.completions.create(
    model="agy-default", messages=messages, tools=tools, tool_choice="none"
)
print(final.choices[0].message.content)
```

The bridge supports `tool_choice` values `auto`, `none`, `required`, or a named function, as well as `parallel_tool_calls` and streaming chat completions. Tool arguments are JSON strings in the response. Validate them in your program before executing a function. If you configure `API_KEY`, use that key in the client instead of `no-auth-required`.

### Streaming

Set `stream: true` for OpenAI-compatible SSE. Ordinary text responses forward `agy`'s `agent_response.text_delta` fragments as they arrive, preserving spaces and newlines. Requests with program `tools` also return SSE chunks, but their content or `tool_calls` is sent after `agy` finishes: its `--json-schema` structured output is available only in the final `result` event. See [Antigravity's stream JSON documentation](https://antigravity.google/docs/cli/headless/).

### Server tool logs

The server's normal console log shows each `agy` worker starting, being reused, and closing, plus turn starts, tool states, result status, and token totals when the CLI reports them. Lines include request and session IDs so simultaneous chats can be followed separately. For example, run `uv run uvicorn main:app --host 127.0.0.1 --port 8000` and watch that terminal. These activity lines omit prompt text and tool arguments.

The server logs each program function call it returns and each `role: "tool"` result it receives, using the request ID, call ID, and function name. Antigravity's workspace hooks in `.agents/hooks.json` also report session starts, model invocations, and built-in tool attempts/completions to the server log. Hook entries include the CLI conversation ID, tool name, outcome, step, and model. Arguments, tool output, and prompts are not logged. Hook events are collected after each turn and once more when a worker closes; program function events are logged when the corresponding API request is handled.

The hooks load when `agy` runs in this repository. If you set `AGY_CWD` to another workspace, copy `.agents/hooks.json` and `.agents/agy_hook_logger.py` into that workspace's `.agents` directory to capture its CLI hook events. The program function logs work regardless of `AGY_CWD`. See [Antigravity's hooks documentation](https://antigravity.google/docs/hooks?tab=cli) for the workspace hook format.

An OpenAI API-compatible wrapper for Claude Code, allowing you to use Claude Code with any OpenAI client library. **Now powered by the official Claude Code Python SDK** with enhanced authentication and features.

## Status

🎉 **Production Ready** - All core features working and tested:
- ✅ Chat completions endpoint with **official Claude Code Python SDK**
- ✅ Streaming and non-streaming responses  
- ✅ Full OpenAI SDK compatibility
- ✅ **Multi-provider authentication** (API key, Bedrock, Vertex AI, CLI auth)
- ✅ **System prompt support** via SDK options
- ✅ Model selection support with validation
- ✅ **Fast by default** - Tools disabled for OpenAI compatibility (5-10x faster)
- ✅ Optional tool usage (Read, Write, Bash, etc.) when explicitly enabled
- ✅ **Real-time cost and token tracking** from SDK
- ✅ **Session continuity** with conversation history across requests 🆕
- ✅ **Session management endpoints** for full session control 🆕
- ✅ Health, auth status, and models endpoints
- ✅ **Development mode** with auto-reload

## Features

### 🔥 **Core API Compatibility**
- OpenAI-compatible `/v1/chat/completions` endpoint
- Support for both streaming and non-streaming responses
- Compatible with OpenAI Python SDK and all OpenAI client libraries
- Automatic model validation and selection

### 🛠 **Claude Code SDK Integration**
- **Official Claude Code Python SDK** integration (v0.0.14)
- **Real-time cost tracking** - actual costs from SDK metadata
- **Accurate token counting** - input/output tokens from SDK
- **Session management** - proper session IDs and continuity
- **Enhanced error handling** with detailed authentication diagnostics

### 🔐 **Multi-Provider Authentication**
- **Automatic detection** of authentication method
- **Claude CLI auth** - works with existing `claude auth` setup
- **Direct API key** - `ANTHROPIC_API_KEY` environment variable
- **AWS Bedrock** - enterprise authentication with AWS credentials
- **Google Vertex AI** - GCP authentication support

### ⚡ **Advanced Features**
- **System prompt support** via SDK options
- **Optional tool usage** - Enable Claude Code tools (Read, Write, Bash, etc.) when needed
- **Fast default mode** - Tools disabled by default for OpenAI API compatibility
- **Development mode** with auto-reload (`uvicorn --reload`)
- **Interactive API key protection** - Optional security with auto-generated tokens
- **Comprehensive logging** and debugging capabilities

## Quick Start

Get started in under 2 minutes:

```bash
# 1. Install Claude Code CLI (if not already installed)
npm install -g @anthropic-ai/claude-code

# 2. Authenticate (choose one method)
claude auth login  # Recommended for development
# OR set: export ANTHROPIC_API_KEY=your-api-key

# 3. Clone and setup the wrapper
git clone https://github.com/RichardAtCT/claude-code-openai-wrapper
cd claude-code-openai-wrapper
poetry install

# 4. Start the server
poetry run uvicorn main:app --reload --port 8000

# 5. Test it works
poetry run python test_endpoints.py
```

🎉 **That's it!** Your OpenAI-compatible Claude Code API is running on `http://localhost:8000`

## Prerequisites

1. **Claude Code CLI**: Install Claude Code CLI
   ```bash
   # Install Claude Code (follow Anthropic's official guide)
   npm install -g @anthropic-ai/claude-code
   ```

2. **Authentication**: Choose one method:
   - **Option A**: Authenticate via CLI (Recommended for development)
     ```bash
     claude auth login
     ```
   - **Option B**: Set environment variable
     ```bash
     export ANTHROPIC_API_KEY=your-api-key
     ```
   - **Option C**: Use AWS Bedrock or Google Vertex AI (see Configuration section)

3. **Python 3.10+**: Required for the server

4. **Poetry**: For dependency management
   ```bash
   # Install Poetry (if not already installed)
   curl -sSL https://install.python-poetry.org | python3 -
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RichardAtCT/claude-code-openai-wrapper
   cd claude-code-openai-wrapper
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all dependencies.

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

## Configuration

Edit the `.env` file:

```env
# Claude CLI path (usually just "claude")
CLAUDE_CLI_PATH=claude

# Optional API key for client authentication
# If not set, server will prompt for interactive API key protection on startup
# API_KEY=your-optional-api-key

# Server port
PORT=8000

# Timeout in milliseconds
MAX_TIMEOUT=600000

# CORS origins
CORS_ORIGINS=["*"]
```

### 🔐 **API Security Configuration**

The server supports **interactive API key protection** for secure remote access:

1. **No API key set**: Server prompts "Enable API key protection? (y/N)" on startup
   - Choose **No** (default): Server runs without authentication
   - Choose **Yes**: Server generates and displays a secure API key

2. **Environment API key set**: Uses the configured `API_KEY` without prompting

```bash
# Example: Interactive protection enabled
poetry run python main.py

# Output:
# ============================================================
# 🔐 API Endpoint Security Configuration
# ============================================================
# Would you like to protect your API endpoint with an API key?
# This adds a security layer when accessing your server remotely.
# 
# Enable API key protection? (y/N): y
# 
# 🔑 API Key Generated!
# ============================================================
# API Key: Xf8k2mN9-vLp3qR5_zA7bW1cE4dY6sT0uI
# ============================================================
# 📋 IMPORTANT: Save this key - you'll need it for API calls!
#    Example usage:
#    curl -H "Authorization: Bearer Xf8k2mN9-vLp3qR5_zA7bW1cE4dY6sT0uI" \
#         http://localhost:8000/v1/models
# ============================================================
```

**Perfect for:**
- 🏠 **Local development** - No authentication needed
- 🌐 **Remote access** - Secure with generated tokens
- 🔒 **VPN/Tailscale** - Add security layer for remote endpoints

## Running the Server

1. Verify Claude Code is installed and working:
   ```bash
   claude --version
   claude --print --model claude-3-5-haiku-20241022 "Hello"  # Test with fastest model
   ```

2. Start the server:

   **Development mode (recommended - auto-reloads on changes):**
   ```bash
   poetry run uvicorn main:app --reload --port 8000
   ```

   **Production mode:**
   ```bash
   poetry run python main.py
   ```

   **Port Options for production mode:**
   - Default: Uses port 8000 (or PORT from .env)
   - If port is in use, automatically finds next available port
   - Specify custom port: `poetry run python main.py 9000`
   - Set in environment: `PORT=9000 poetry run python main.py`

## Usage Examples

### Using curl

```bash
# Basic chat completion (no auth)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'

# With API key protection (when enabled)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-generated-api-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Write a Python hello world script"}
    ],
    "stream": true
  }'
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

# Configure client (automatically detects auth requirements)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key-if-required"  # Only needed if protection enabled
)

# Alternative: Let examples auto-detect authentication
# The wrapper's example files automatically check server auth status

# Basic chat completion
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What files are in the current directory?"}
    ]
)

print(response.choices[0].message.content)
# Output: Fast response without tool usage (default behavior)

# Enable tools when you need them (e.g., to read files)
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "What files are in the current directory?"}
    ],
    extra_body={"enable_tools": True}  # Enable tools for file access
)
print(response.choices[0].message.content)
# Output: Claude will actually read your directory and list the files!

# Check real costs and tokens
print(f"Cost: ${response.usage.total_tokens * 0.000003:.6f}")  # Real cost tracking
print(f"Tokens: {response.usage.total_tokens} ({response.usage.prompt_tokens} + {response.usage.completion_tokens})")

# Streaming
stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Supported Models

- `claude-sonnet-4-20250514` (Recommended)
- `claude-opus-4-20250514`
- `claude-3-7-sonnet-20250219`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`

The model parameter is passed to Claude Code via the `--model` flag.

## Session Continuity 🆕

The wrapper now supports **session continuity**, allowing you to maintain conversation context across multiple requests. This is a powerful feature that goes beyond the standard OpenAI API.

### How It Works

- **Stateless Mode** (default): Each request is independent, just like the standard OpenAI API
- **Session Mode**: Include a `session_id` to maintain conversation history across requests

### Using Sessions with OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Start a conversation with session continuity
response1 = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello! My name is Alice and I'm learning Python."}
    ],
    extra_body={"session_id": "my-learning-session"}
)

# Continue the conversation - Claude remembers the context
response2 = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022", 
    messages=[
        {"role": "user", "content": "What's my name and what am I learning?"}
    ],
    extra_body={"session_id": "my-learning-session"}  # Same session ID
)
# Claude will remember: "Your name is Alice and you're learning Python."
```

### Using Sessions with curl

```bash
# First message (add -H "Authorization: Bearer your-key" if auth enabled)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "My favorite color is blue."}],
    "session_id": "my-session"
  }'

# Follow-up message - context is maintained
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022", 
    "messages": [{"role": "user", "content": "What's my favorite color?"}],
    "session_id": "my-session"
  }'
```

### Session Management

The wrapper provides endpoints to manage active sessions:

- `GET /v1/sessions` - List all active sessions
- `GET /v1/sessions/{session_id}` - Get session details
- `DELETE /v1/sessions/{session_id}` - Delete a session
- `GET /v1/sessions/stats` - Get session statistics

```bash
# List active sessions
curl http://localhost:8000/v1/sessions

# Get session details
curl http://localhost:8000/v1/sessions/my-session

# Delete a session
curl -X DELETE http://localhost:8000/v1/sessions/my-session
```

### Session Features

- **Automatic Expiration**: Sessions expire after 1 hour of inactivity
- **Streaming Support**: Session continuity works with both streaming and non-streaming requests
- **Memory Persistence**: Full conversation history is maintained within the session
- **Efficient Storage**: Only active sessions are kept in memory

### Examples

See `examples/session_continuity.py` for comprehensive Python examples and `examples/session_curl_example.sh` for curl examples.

## API Endpoints

### Core Endpoints
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (supports `session_id`)
- `GET /v1/models` - List available models
- `GET /v1/auth/status` - Check authentication status and configuration
- `GET /health` - Health check endpoint

### Session Management Endpoints 🆕
- `GET /v1/sessions` - List all active sessions
- `GET /v1/sessions/{session_id}` - Get detailed session information
- `DELETE /v1/sessions/{session_id}` - Delete a specific session
- `GET /v1/sessions/stats` - Get session manager statistics

## Limitations & Roadmap

### 🚫 **Current Limitations**
- **Images in messages** are converted to text placeholders
- **Function calling** not supported (tools work automatically based on prompts)
- **OpenAI parameters** not yet mapped: `temperature`, `top_p`, `max_tokens`, `logit_bias`, `presence_penalty`, `frequency_penalty`
- **Multiple responses** (`n > 1`) not supported

### 🛣 **Planned Enhancements** 
- [ ] **Tool configuration** - allowed/disallowed tools endpoints  
- [ ] **OpenAI parameter mapping** - temperature, top_p, max_tokens support
- [ ] **Enhanced streaming** - better chunk handling
- [ ] **MCP integration** - Model Context Protocol server support

### ✅ **Recent Improvements**
- **✅ SDK Integration**: Official Python SDK replaces subprocess calls
- **✅ Real Metadata**: Accurate costs and token counts from SDK
- **✅ Multi-auth**: Support for CLI, API key, Bedrock, and Vertex AI authentication  
- **✅ Session IDs**: Proper session tracking and management
- **✅ System Prompts**: Full support via SDK options
- **✅ Session Continuity**: Conversation history across requests with session management

## Troubleshooting

1. **Claude CLI not found**:
   ```bash
   # Check Claude is in PATH
   which claude
   # Update CLAUDE_CLI_PATH in .env if needed
   ```

2. **Authentication errors**:
   ```bash
   # Test authentication with fastest model
   claude --print --model claude-3-5-haiku-20241022 "Hello"
   # If this fails, re-authenticate if needed
   ```

3. **Timeout errors**:
   - Increase `MAX_TIMEOUT` in `.env`
   - Note: Claude Code can take time for complex requests

## Testing

### 🧪 **Quick Test Suite**
Test all endpoints with a simple script:
```bash
# Make sure server is running first
poetry run python test_endpoints.py
```

### 📝 **Basic Test Suite**
Run the comprehensive test suite:
```bash
# Make sure server is running first  
poetry run python test_basic.py

# With API key protection enabled, set TEST_API_KEY:
TEST_API_KEY=your-generated-key poetry run python test_basic.py
```

The test suite automatically detects whether API key protection is enabled and provides helpful guidance for providing the necessary authentication.

### 🔍 **Authentication Test**
Check authentication status:
```bash
curl http://localhost:8000/v1/auth/status | python -m json.tool
```

### ⚙️ **Development Tools**
```bash
# Install development dependencies
poetry install --with dev

# Format code
poetry run black .

# Run full tests (when implemented)
poetry run pytest tests/
```

### ✅ **Expected Results**
All tests should show:
- **4/4 endpoint tests passing**
- **4/4 basic tests passing** 
- **Authentication method detected** (claude_cli, anthropic, bedrock, or vertex)
- **Real cost tracking** (e.g., $0.001-0.005 per test call)
- **Accurate token counts** from SDK metadata

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
