# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Claude on OpenAI** is a proxy server that translates between Anthropic's Claude API format and OpenAI's API format. It
enables Claude Code and other Anthropic clients to use any OpenAI-compatible model (GPT-4o, OpenRouter, DeepSeek, Ollama,
etc.) by intercepting requests and converting them on the fly using the OpenAI Python SDK directly.

**Key capability**: Allows Claude Code CLI to run with any OpenAI-compatible model by setting
`ANTHROPIC_BASE_URL=http://localhost:8082`.

## Development Commands

### Running the Server

```bash
# Standard mode
python main.py

# Using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8082

# With auto-reload
uvicorn src.main:app --host 0.0.0.0 --port 8082 --reload
```

### Testing

```bash
# Run all tests
python -m tests.test_api

# Run only non-streaming tests
python -m tests.test_api --no-streaming

# Run only simple tests (no tools)
python -m tests.test_api --simple

# Run only tool tests
python -m tests.test_api --tools-only

# Run converter unit tests
python -m tests.test_converter
```

### Dependencies

```bash
# Install with uv (recommended)
uv venv && uv pip install -r requirements.txt

# Install with pip
pip install -r requirements.txt
```

## Architecture

### Request Flow

1. **Client Request** → Anthropic format (`/v1/messages`)
2. **Endpoint** (`src/api/endpoints.py`) → Validates API key, generates request ID
3. **Request Converter** (`src/conversion/request_converter.py`) → Claude → OpenAI format
4. **OpenAI Client** (`src/core/client.py`) → AsyncOpenAI SDK with cancellation support
5. **Response Converter** (`src/conversion/response_converter.py`) → OpenAI → Claude format
6. **Client Response** ← Anthropic format (SSE stream or JSON)

### Provider System

The proxy supports any OpenAI-compatible provider via `OPENAI_BASE_URL`:

- **OpenAI** (default): `https://api.openai.com/v1`
- **OpenRouter**: `https://openrouter.ai/api/v1` (100+ models)
- **DeepSeek**: `https://api.deepseek.com/v1`
- **Azure OpenAI**: Uses `AsyncAzureOpenAI` when `AZURE_API_VERSION` is set
- **Ollama/Local**: `http://localhost:11434/v1`

**Model mapping logic** (in `src/core/model_manager.py`):

- Models with provider prefixes (`meta/`, `google/`, `openrouter/`, `qwen/`, etc.) are passed through
- Claude models are mapped: `haiku` → `SMALL_MODEL`, `sonnet` → `MIDDLE_MODEL`, `opus` → `BIG_MODEL`
- Unknown models default to `BIG_MODEL`

**Reasoning model support**:

- Models starting with `o1`, `o3`, `o4` are detected as reasoning models
- Use `max_completion_tokens` instead of `max_tokens`
- `temperature` parameter is omitted (rejected by reasoning models)
- `reasoning_content` in responses is converted to Claude `thinking` blocks

### Core Components

**`src/api/endpoints.py`**

- `/v1/messages` — Main messages endpoint (Anthropic format, streaming + non-streaming)
- `/v1/messages/count_tokens` — Token counting with tiktoken (cl100k_base)
- `/health` — Health check
- `/test-connection` — API connectivity test
- `/` — Root info with config summary
- API key validation via `x-api-key` or `Authorization: Bearer` headers

**`src/conversion/request_converter.py`**

- `convert_claude_to_openai()` — Full request format conversion
- Input sanitization: strips `thinking`/`cache_control` from messages
- Adaptive max_tokens: `max_completion_tokens` for reasoning models
- Tool conversion: Claude tools → OpenAI function calling format
- Tool choice mapping: `auto`→`auto`, `any`→`required`, `tool`→specific function
- Image conversion: Claude base64 → OpenAI image_url format

**`src/conversion/response_converter.py`**

- `convert_openai_to_claude_response()` — Non-streaming response conversion
- `convert_openai_streaming_to_claude()` — Streaming without cancellation
- `convert_openai_streaming_to_claude_with_cancellation()` — Streaming with client disconnect detection
- Reasoning/thinking block conversion from `reasoning_content`
- Incremental tool call argument streaming (sends each chunk as delta)
- All content blocks guaranteed closed to prevent client hangs

**`src/core/client.py`**

- `OpenAIClient` — Async wrapper with cancellation support
- Thread-safe `active_requests` dict with `asyncio.Lock`
- Auto-detects Azure vs standard OpenAI based on `api_version`
- Error classification with user-friendly messages
- Custom header support via `CUSTOM_HEADER_*` env vars

**`src/core/config.py`**

- `Config` — Centralized configuration from environment
- `is_reasoning_model()` — Detects o1/o3/o4 series
- `validate_client_api_key()` — Optional client auth
- `get_custom_headers()` — Dynamic header injection

**`src/core/model_manager.py`**

- `ModelManager` — Claude model name → OpenAI model mapping
- `PASSTHROUGH_PREFIXES` — Provider prefixes that skip remapping

**`src/core/constants.py`**

- All string constants for roles, content types, events, deltas
- `CLAUDE_ONLY_FIELDS` — Fields to strip from non-Claude provider requests

**`src/models/claude.py`**

- Pydantic models for Claude API requests/responses
- Content block types: text, image, tool_use, tool_result
- `ClaudeThinkingConfig` for thinking parameter
- Token count request model

### Streaming Implementation

The streaming handler follows Anthropic's SSE event format:

1. `message_start` — Initial message metadata
2. `content_block_start` — Start of text/thinking/tool block
3. `ping` — Keepalive event
4. `content_block_delta` — Incremental content (`text_delta`, `thinking_delta`, or `input_json_delta`)
5. `content_block_stop` — End of content block
6. `message_delta` — Final metadata (stop_reason, usage)
7. `message_stop` — Stream completion

**Critical details**:

- Text blocks must be closed before tool blocks start
- Thinking blocks (from reasoning models) close before text blocks start
- Tool call arguments are sent incrementally (each chunk as a delta)
- All started blocks are guaranteed closed, even on error
- Client disconnection triggers upstream request cancellation
- `prompt_tokens_details` is null-safe (OpenAI sometimes returns `null`)

### Environment Configuration

Required:

- `OPENAI_API_KEY` — API key for your provider

Optional:

- `ANTHROPIC_API_KEY` — If set, clients must provide this key to authenticate
- `OPENAI_BASE_URL` — Provider base URL (default: `https://api.openai.com/v1`)
- `AZURE_API_VERSION` — Enables Azure OpenAI mode
- `BIG_MODEL` — Maps Claude opus (default: `gpt-4o`)
- `MIDDLE_MODEL` — Maps Claude sonnet (default: same as BIG_MODEL)
- `SMALL_MODEL` — Maps Claude haiku (default: `gpt-4o-mini`)
- `HOST` — Server host (default: `0.0.0.0`)
- `PORT` — Server port (default: `8082`)
- `LOG_LEVEL` — `debug`, `info`, `warning`, `error`, `critical`
- `MAX_TOKENS_LIMIT` — Max output tokens (default: `16384`)
- `MIN_TOKENS_LIMIT` — Min output tokens (default: `100`)
- `REQUEST_TIMEOUT` — Request timeout in seconds (default: `120`)
- `CUSTOM_HEADER_*` — Custom headers (underscores become hyphens)

## Common Patterns

### Adding Support for New Providers

1. If the provider is OpenAI-compatible, just set `OPENAI_BASE_URL` — no code changes needed
2. If models need prefix passthrough, add to `PASSTHROUGH_PREFIXES` in `model_manager.py`
3. If using Azure, set `AZURE_API_VERSION` to enable `AsyncAzureOpenAI`
4. Test with both streaming and non-streaming requests

### Modifying Tool Call Handling

Tool conversion happens in two places:

1. **Request**: `request_converter.py` — tools, tool_choice conversion
2. **Response**: `response_converter.py` — tool_calls → tool_use blocks (non-streaming: lines 44-48, streaming: tool
   call delta handling)

### Debugging Streaming Issues

Set `LOG_LEVEL=debug` in `.env`. Check that:

- Text blocks are closed before tool blocks
- `finish_reason` triggers completion events
- Token usage is extracted from stream chunks
- All content blocks receive `content_block_stop` events

## Important Notes

- **No LiteLLM dependency** — Uses OpenAI Python SDK directly for fewer moving parts
- **Input sanitization** — Claude-only fields (`thinking`, `cache_control`) are stripped before forwarding
- **Reasoning models** — o1/o3/o4 automatically get `max_completion_tokens` and thinking block conversion
- **Token counting** — Uses tiktoken (cl100k_base) for accurate counts, falls back to estimation
- **Tool call IDs** — Generated using UUID if not provided by the LLM
- **Error handling** — All errors are converted to Anthropic format with proper `error` objects
- **Thread safety** — `active_requests` dict protected by `asyncio.Lock` to prevent race conditions
- **Content block guarantee** — All started blocks are closed in the finally path, preventing client hangs
