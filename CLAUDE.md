# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Claude on OpenAI** is a proxy server that translates between Anthropic's Claude API format and OpenAI's API format. It
enables Claude Code and other Anthropic clients to use OpenAI models (like GPT-4o) by intercepting requests and
converting them on the fly. The proxy uses LiteLLM as the translation layer.

**Key capability**: Allows Claude Code CLI to run with OpenAI models by setting
`ANTHROPIC_BASE_URL=http://localhost:8082`.

## Development Commands

### Running the Server

```bash
# Standard mode
python main.py

# Debug mode (with auto-reload and debug logging)
python main.py --debug

# Using uvicorn directly
uvicorn src.app.main:app --host 0.0.0.0 --port 8082

# With uvicorn and reload
uvicorn src.app.main:app --host 0.0.0.0 --port 8082 --reload
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
uv pip install -r requirements.txt

# Install with pip
pip install -r requirements.txt
```

## Architecture

### Request Flow

1. **Client Request** → Anthropic format (`/v1/messages`)
2. **Route Handler** (`src/app/api/routes.py`) → Validates request, determines provider
3. **Converter** (`src/app/services/converter.py`) → Anthropic → LiteLLM format
4. **LiteLLM** → Routes to appropriate provider (OpenAI/Anthropic/NVIDIA)
5. **Response Converter** → LiteLLM → Anthropic format
6. **Client Response** ← Anthropic format

### Provider System

The proxy supports multiple providers configured via `PREFERRED_PROVIDER` env var:

- `openai` (default): Routes to OpenAI models
- `anthropic`: Routes to native Anthropic models
- `nvidia`: Routes to NVIDIA NIM models

**Model mapping logic** (in `src/app/models/schema.py`):

- `_validate_model()` automatically prefixes models with provider (e.g., `openai/gpt-4o`)
- Claude models are mapped: `haiku` → `SMALL_MODEL`, `sonnet` → `BIG_MODEL`
- Original model name is preserved in `original_model` field

**Function calling support**:

- The proxy automatically checks if a model supports function calling using `litellm.supports_function_calling()`
- For models that don't support function calling (like some NVIDIA NIM models), `tools` and `tool_choice` parameters are
  automatically removed from requests
- Configured with `litellm.drop_params = True` to automatically drop unsupported parameters
- See routes.py:224-234 and routes.py:376-386 for implementation

### Core Components

**`src/app/api/routes.py`**

- `/v1/messages` - Main messages endpoint (Anthropic format)
- `/v1/messages/count_tokens` - Token counting endpoint
- `/v1/chat/completions` - OpenAI-compatible endpoint
- `/metrics` - HTML metrics dashboard
- Handles both streaming and non-streaming responses
- Special handling for OpenAI models: flattens complex content structures (tool results, images) to plain text

**`src/app/services/converter.py`**

- `convert_anthropic_to_litellm()` - Request format conversion
- `convert_litellm_to_anthropic()` - Response format conversion
- `handle_streaming()` - Streaming response handler with SSE events
- Tool call conversion: OpenAI function calls ↔ Anthropic tool_use blocks
- Complex streaming state machine manages content blocks and tool calls

**`src/app/models/schema.py`**

- Pydantic models for all request/response types
- Model validation and provider prefixing
- Content block types: text, image, tool_use, tool_result
- Token count request/response models

**`src/app/main.py`**

- FastAPI app initialization
- Exception handlers (HTTPException, generic Exception)
- CORS middleware
- Request logging middleware

### Streaming Implementation

The streaming handler (`handle_streaming()`) follows Anthropic's SSE event format:

1. `message_start` - Initial message metadata
2. `content_block_start` - Start of text/tool block
3. `ping` - Keepalive event
4. `content_block_delta` - Incremental content (text_delta or input_json_delta)
5. `content_block_stop` - End of content block
6. `message_delta` - Final metadata (stop_reason, usage)
7. `message_stop` - Stream completion
8. `[DONE]` - Terminator

**Critical detail**: Text blocks must be closed before tool blocks start. The handler manages state to ensure proper
event ordering.

### OpenAI Model Compatibility

For OpenAI models, the proxy performs special content flattening:

- **Tool results**: Converted from structured blocks to plain text with `Tool Result:` prefix
- **Complex content arrays**: Flattened to text strings
- **Image blocks**: Replaced with placeholder text
- **Empty content**: Replaced with `"..."` (OpenAI doesn't accept null/empty)

This happens in routes.py lines 98-216.

### Environment Configuration

Required keys (at least one):

- `ANTHROPIC_API_KEY` - For Anthropic models
- `OPENAI_API_KEY` - For OpenAI models
- `NVIDIA_NIM_API_KEY` - For NVIDIA models (optional)

Model mapping:

- `BIG_MODEL` - Maps Sonnet models (default: `gpt-4o`)
- `SMALL_MODEL` - Maps Haiku models (default: `gpt-4o-mini`)

Provider selection:

- `PREFERRED_PROVIDER` - `openai`, `anthropic`, or `nvidia` (default: `openai`)

Server config:

- `SERVER_HOST` - Default: `0.0.0.0`
- `SERVER_PORT` - Default: `8082`
- `LOG_LEVEL` - `debug`, `info`, `warning`, `error`, `critical`
- `OPENAI_BASE_URL` - Optional custom OpenAI base URL

## Common Patterns

### Adding Support for New Providers

1. Add API key environment variable to `.env`
2. Update `routes.py` provider selection logic (lines 71-96)
3. Add model mapping logic in `schema.py` `_validate_model()` (lines 16-58)
4. Test with both streaming and non-streaming requests

### Modifying Tool Call Handling

Tool conversion happens in two places:

1. **Request**: `converter.py` `_convert_tools()` and `_convert_tool_choice()`
2. **Response**: `converter.py` lines 262-298 (non-streaming) and lines 476-553 (streaming)

Claude models receive native `tool_use` blocks. Non-Claude models get tool calls flattened to text.

### Debugging Streaming Issues

Enable debug logging with `--debug` flag. The streaming handler logs:

- Chunk processing at line 442
- Tool call details at lines 263, 268, 291
- Content block events throughout `handle_streaming()`

Check that:

- Text blocks are closed before tool blocks
- `finish_reason` triggers completion events
- Token usage is extracted from chunks

### Testing API Compatibility

The test suite (`tests/test_api.py`) covers:

- Simple text completions
- Tool calls (function calling)
- Streaming responses
- Token counting

Run specific test categories with flags (`--simple`, `--tools-only`, `--no-streaming`).

## NVIDIA NIM Integration

The proxy fully supports NVIDIA NIM models. Key considerations:

**Configuration**:

- Set `PREFERRED_PROVIDER=nvidia` in `.env`
- Provide `NVIDIA_NIM_API_KEY`
- Models are automatically prefixed with `nvidia_nim/` (e.g., `nvidia_nim/meta/llama3-70b-instruct`)

**Function calling limitations**:

- Most NVIDIA NIM models don't support native function calling
- The proxy automatically detects this using `litellm.supports_function_calling()`
- When unsupported, `tools` parameters are stripped from requests
- No errors are thrown - the proxy gracefully handles this

**Recommended NVIDIA models**:

- For large tasks: Set `BIG_MODEL=meta/llama3-70b-instruct` or `mistralai/mixtral-8x22b-instruct`
- For small tasks: Set `SMALL_MODEL=meta/llama3-8b` or `microsoft/phi-3-mini-4k-instruct`

See [LiteLLM NVIDIA NIM docs](https://docs.litellm.ai/docs/providers/nvidia_nim) for full model list.

## Important Notes

- **Message content validation**: OpenAI models require string content, not arrays. The proxy automatically flattens
  complex content.
- **Token limits**: OpenAI models are capped at 16384 max_tokens (line 182 in converter.py)
- **API key routing**: The proxy selects the API key based on `PREFERRED_PROVIDER`, not the model name
- **Tool call IDs**: Generated using UUID if not provided by the LLM
- **Error handling**: All errors are converted to Anthropic format with proper `error` objects
- **Function calling**: Automatically disabled for models that don't support it (no manual configuration needed)
