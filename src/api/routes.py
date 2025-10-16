"""
API routes for Anthropic-OpenAI proxy.

This module defines all API endpoints for the proxy server using native OpenAI client.
"""
import logging
import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import HTMLResponse

from src.core.config import config
from src.core.constants import Constants
from src.core.client import OpenAIClient
from src.conversion.request_converter import convert_anthropic_to_openai
from src.conversion.response_converter import (
    convert_openai_to_anthropic,
    handle_streaming
)
from src.app.models.schema import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
    ChatCompletionRequest
)
from src.app.utils.helpers import log_request_beautifully

# Create router
router = APIRouter()

# Get logger
logger = logging.getLogger(__name__)

# Get custom headers from config
custom_headers = config.get_custom_headers()

# Initialize OpenAI client
openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url or "https://api.openai.com/v1",
    config.request_timeout,
    custom_headers=custom_headers,
)

logger.info(f"OpenAI client initialized with base URL: {config.openai_base_url or 'https://api.openai.com/v1'}")


async def validate_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """
    Validate the client's API key from either x-api-key header or Authorization header.

    Args:
        x_api_key: API key from x-api-key header
        authorization: API key from Authorization header

    Raises:
        HTTPException: If API key is invalid
    """
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
    if not client_api_key or client_api_key != config.anthropic_api_key:
        logger.warning("Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )


@router.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    http_request: Request,
    _: None = Depends(validate_api_key)
):
    """
    Main endpoint for handling Anthropic Messages API requests.

    Converts Anthropic API format to OpenAI format, sends to OpenAI API,
    and converts response back to Anthropic format.
    """
    try:
        # Extract model information for logging
        display_model = request.model.split("/")[-1] if "/" in request.model else request.model

        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Convert Anthropic request to OpenAI format
        openai_request = convert_anthropic_to_openai(request)

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        # Count tools for logging
        num_tools = len(request.tools) if request.tools else 0

        # Log request summary
        log_request_beautifully(
            "POST",
            http_request.url.path,
            display_model,
            openai_request.get('model'),
            len(openai_request['messages']),
            num_tools,
            200
        )

        if request.stream:
            # Streaming response
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    handle_streaming(
                        openai_stream,
                        request
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
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": e.detail},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            anthropic_response = convert_openai_to_anthropic(
                openai_response, request
            )
            return anthropic_response

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
    request: TokenCountRequest,
    _: None = Depends(validate_api_key)
):
    """
    Endpoint to count tokens for a request.

    Uses simple character-based estimation (4 characters per token).
    """
    try:
        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return TokenCountResponse(input_tokens=estimated_tokens)

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """
    Root endpoint to verify the server is running.
    """
    return {
        "message": "Anthropic-OpenAI Proxy v2.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url or "https://api.openai.com/v1",
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


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
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


async def handle_openai_streaming(response_generator):
    """
    Handle streaming responses from OpenAI API in OpenAI format.

    Args:
        response_generator: Async generator from OpenAI

    Yields:
        OpenAI-formatted SSE chunks
    """
    async for chunk in response_generator:
        yield f"{chunk}\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _: None = Depends(validate_api_key)
):
    """
    Endpoint for handling OpenAI-compatible chat completions.

    This endpoint accepts OpenAI format directly and passes it through
    to the OpenAI API without conversion.
    """
    try:
        openai_request = request.dict(exclude_none=True)

        # Add extra_body parameters if present
        if request.extra_body:
            openai_request.update(request.extra_body)

        logger.debug(
            f"OpenAI chat completion: Model={openai_request.get('model')}, "
            f"Tools={len(openai_request.get('tools', []))}"
        )

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Handle streaming
        if request.stream:
            openai_stream = openai_client.create_chat_completion_stream(
                openai_request, request_id
            )
            return StreamingResponse(
                handle_openai_streaming(openai_stream),
                media_type="text/event-stream"
            )

        # Handle non-streaming
        else:
            response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard():
    """
    Render a simple metrics dashboard for monitoring proxy performance.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anthropic-OpenAI Proxy Metrics</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .metric-card {
                @apply bg-white rounded-lg shadow p-4 hover:shadow-lg transition-shadow;
            }
            .metric-value {
                @apply text-3xl font-bold text-blue-600;
            }
            .metric-label {
                @apply text-gray-500 text-sm uppercase;
            }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-2xl font-bold text-gray-800 mb-6">Anthropic-OpenAI Proxy Metrics</h1>

            <div id="metrics-content" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <div class="metric-card">
                    <p class="metric-value" id="total-requests">0</p>
                    <p class="metric-label">Total Requests</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="success-rate">100%</p>
                    <p class="metric-label">Success Rate</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="avg-latency">-</p>
                    <p class="metric-label">Avg. Latency (ms)</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="provider">OpenAI</p>
                    <p class="metric-label">Provider</p>
                </div>
            </div>

            <div class="mt-8 bg-white rounded-lg shadow p-4">
                <h2 class="text-lg font-semibold mb-4">Configuration</h2>
                <dl class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <dt class="text-sm text-gray-500">Base URL</dt>
                        <dd class="text-sm font-medium">{{base_url}}</dd>
                    </div>
                    <div>
                        <dt class="text-sm text-gray-500">Big Model</dt>
                        <dd class="text-sm font-medium">{{big_model}}</dd>
                    </div>
                    <div>
                        <dt class="text-sm text-gray-500">Small Model</dt>
                        <dd class="text-sm font-medium">{{small_model}}</dd>
                    </div>
                    <div>
                        <dt class="text-sm text-gray-500">Max Tokens Limit</dt>
                        <dd class="text-sm font-medium">{{max_tokens}}</dd>
                    </div>
                </dl>
            </div>
        </div>
    </body>
    </html>
    """.replace("{{base_url}}", config.openai_base_url or "https://api.openai.com/v1") \
       .replace("{{big_model}}", config.big_model) \
       .replace("{{small_model}}", config.small_model) \
       .replace("{{max_tokens}}", str(config.max_tokens_limit))

    return HTMLResponse(content=html_content)
