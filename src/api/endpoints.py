import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

import httpx
import tiktoken
from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.client import OpenAIClient
from src.core.config import config
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest

# Pre-load tokenizer (cl100k_base covers gpt-4o, gpt-4, gpt-3.5-turbo)
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)


async def validate_api_key(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None

    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return

    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning("Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key.",
        )


def _get_passthrough_api_key(http_request: Request) -> str:
    """Extract the API key from the request headers, falling back to config."""
    api_key = http_request.headers.get("x-api-key")
    if not api_key:
        auth = http_request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            api_key = auth.removeprefix("Bearer ")
    # Fall back to configured key (may be the same if client sends the env key)
    return api_key or config.anthropic_api_key


async def _handle_passthrough(
    request: ClaudeMessagesRequest,
    http_request: Request,
) -> StreamingResponse | JSONResponse:
    """Forward a Claude request directly to Anthropic's API without conversion."""
    api_key = _get_passthrough_api_key(http_request)
    url = f"{config.anthropic_base_url}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = request.model_dump(exclude_none=True)

    logger.info(
        f"Passthrough → Anthropic: model={request.model}, stream={request.stream}"
    )

    try:
        if request.stream:
            # Streaming: keep the connection open and pipe SSE events back
            client = httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout))
            upstream = await client.send(
                client.build_request("POST", url, json=body, headers=headers),
                stream=True,
            )

            if upstream.status_code != 200:
                resp_body = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                logger.error(
                    f"Passthrough upstream error {upstream.status_code}: {resp_body.decode()}"
                )
                return JSONResponse(
                    status_code=upstream.status_code,
                    content={
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": resp_body.decode(),
                        },
                    },
                )

            async def _streaming_generator() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                except httpx.RemoteProtocolError:
                    pass
                finally:
                    await upstream.aclose()
                    await client.aclose()

            return StreamingResponse(
                _streaming_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            # Non-streaming: simple request/response
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(config.request_timeout)
            ) as client:
                upstream = await client.post(url, json=body, headers=headers)

            if upstream.status_code != 200:
                logger.error(
                    f"Passthrough upstream error {upstream.status_code}: {upstream.text}"
                )
                return JSONResponse(
                    status_code=upstream.status_code,
                    content=upstream.json(),
                )

            return JSONResponse(content=upstream.json())

    except httpx.TimeoutException:
        logger.error("Passthrough request timed out")
        raise HTTPException(
            status_code=504, detail="Upstream Anthropic request timed out"
        )
    except httpx.HTTPError as exc:
        logger.error(f"Passthrough HTTP error: {exc}")
        raise HTTPException(status_code=502, detail=f"Upstream Anthropic error: {exc}")


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest,
    http_request: Request,
    _: None = Depends(validate_api_key),
):
    try:
        start_time = time.monotonic()

        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Anthropic passthrough: forward Claude model requests directly to Anthropic API
        if (
            config.anthropic_api_key
            and config.enable_passthrough
            and "claude" in request.model.lower()
        ):
            return await _handle_passthrough(request, http_request)

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, model_manager)

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                        start_time=start_time,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )

            # Log throughput
            elapsed = time.monotonic() - start_time
            output_tokens = claude_response.get("usage", {}).get("output_tokens", 0)
            tok_s = output_tokens / elapsed if elapsed > 0 else 0
            model = request.model
            logger.info(
                f"Request completed: model={model}, {output_tokens} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)"
            )

            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)
):
    try:
        total_text = []

        # Collect system message text
        if request.system:
            if isinstance(request.system, str):
                total_text.append(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_text.append(block.text)

        # Collect message text
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_text.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_text.append(block.text)

        combined = "\n".join(total_text)

        # Use tiktoken for accurate counting, fallback to estimation
        if _tokenizer:
            estimated_tokens = len(_tokenizer.encode(combined))
        else:
            estimated_tokens = max(1, len(combined) // 4)

        return {"input_tokens": max(1, estimated_tokens)}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v2.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
