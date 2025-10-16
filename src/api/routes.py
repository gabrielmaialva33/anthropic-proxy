"""
API routes for Anthropic-OpenAI proxy.

This module defines all API endpoints for the proxy server.
"""
import json
import logging
import time

import litellm
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import HTMLResponse

from src.core.config import config
from src.core.constants import Constants
from src.conversion.request_converter import convert_anthropic_to_litellm
from src.conversion.response_converter import (
    convert_litellm_to_anthropic,
    handle_streaming
)
from src.app.models.schema import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
    ChatCompletionRequest
)
from src.app.utils.helpers import log_request_beautifully

# Configure LiteLLM to automatically drop unsupported parameters
litellm.drop_params = True

# Create router
router = APIRouter()

# Get logger
logger = logging.getLogger(__name__)


def get_provider_api_key(provider: str) -> str:
    """
    Get the API key for the specified provider.

    Args:
        provider: Provider name (openai, anthropic, nvidia)

    Returns:
        API key for the provider

    Raises:
        HTTPException: If API key is not configured
    """
    api_key = config.get_api_key_for_provider(provider)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"{provider.upper()} API key is required for {provider} models but was not provided"
        )
    return api_key


@router.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    """
    Main endpoint for handling Anthropic Messages API requests.

    Converts Anthropic API format to LiteLLM format, sends to provider,
    and converts response back to Anthropic format.
    """
    # Extract model information for logging
    original_model = request.model
    display_model = original_model.split("/")[-1] if "/" in original_model else original_model

    # Log request details
    logger.debug(f"Processing request: Model={request.model}, Stream={request.stream}")

    # Convert Anthropic request to LiteLLM format
    litellm_request = convert_anthropic_to_litellm(request)

    # Get API key for the preferred provider
    api_key = get_provider_api_key(config.preferred_provider)
    litellm_request["api_key"] = api_key

    # Add OpenAI base URL if configured
    if config.preferred_provider == "openai" and config.openai_base_url:
        litellm_request["api_base"] = config.openai_base_url
        logger.debug(f"Using OpenAI base URL: {config.openai_base_url}")

    # Log request summary
    num_tools = len(request.tools) if request.tools else 0
    logger.debug(
        f"Request converted: Model={litellm_request.get('model')}, "
        f"Messages={len(litellm_request['messages'])}, "
        f"Tools={num_tools}, "
        f"Stream={litellm_request.get('stream', False)}"
    )

    # IMPORTANT: Tools are ALWAYS included if present - no filtering
    # The provider will handle tool support appropriately

    # Count messages and tools for logging
    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        litellm_request.get('model'),
        len(litellm_request['messages']),
        num_tools,
        200
    )

    # Handle streaming responses
    if request.stream:
        headers = {"Authorization": f"Bearer {api_key}"}
        api_key_param = litellm_request.pop("api_key", None)

        # Call LiteLLM with allowed OpenAI parameters
        response_generator = await litellm.acompletion(
            **litellm_request,
            api_key=api_key_param,
            headers=headers,
            allowed_openai_params=["tools"]
        )

        return StreamingResponse(
            handle_streaming(response_generator, request),
            media_type="text/event-stream"
        )

    # Handle non-streaming responses
    else:
        start_time = time.time()
        headers = {"Authorization": f"Bearer {api_key}"}
        api_key_param = litellm_request.pop("api_key", None)

        # Call LiteLLM with allowed OpenAI parameters
        litellm_response = litellm.completion(
            **litellm_request,
            api_key=api_key_param,
            headers=headers,
            allowed_openai_params=["tools"]
        )

        logger.debug(
            f"Response received: Model={litellm_request.get('model')}, "
            f"Time={time.time() - start_time:.2f}s"
        )

        # Convert LiteLLM response to Anthropic format
        anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

        return anthropic_response


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    """
    Endpoint to count tokens for a request.

    Uses LiteLLM's token counting functionality.
    """
    # Extract model information
    original_model = request.original_model or request.model
    display_model = original_model.split("/")[-1] if "/" in original_model else original_model

    # Convert to LiteLLM request format for token counting
    converted_request = convert_anthropic_to_litellm(
        MessagesRequest(
            model=request.model,
            max_tokens=100,  # Arbitrary value not used for token counting
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking
        )
    )

    from litellm import token_counter

    # Count messages and tools for logging
    num_tools = len(request.tools) if request.tools else 0

    # Log the request
    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        converted_request.get('model'),
        len(converted_request['messages']),
        num_tools,
        200
    )

    # Count tokens
    token_count = token_counter(
        model=converted_request["model"],
        messages=converted_request["messages"],
    )

    return TokenCountResponse(input_tokens=token_count)


@router.get("/")
async def root():
    """
    Root endpoint to verify the server is running.
    """
    return {
        "message": "Anthropic-OpenAI Proxy",
        "version": "2.0.0",
        "preferred_provider": config.preferred_provider
    }


async def handle_openai_streaming(response_generator):
    """
    Handle streaming responses from LiteLLM in OpenAI format.

    Args:
        response_generator: Async generator from LiteLLM

    Yields:
        OpenAI-formatted SSE chunks
    """
    async for chunk in response_generator:
        yield f"data: {chunk.json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request
):
    """
    Endpoint for handling OpenAI-compatible chat completions.

    This endpoint accepts OpenAI format directly and passes it through
    to LiteLLM without conversion.
    """
    litellm_request = request.dict(exclude_none=True)

    # Add extra_body parameters if present
    if request.extra_body:
        litellm_request.update(request.extra_body)

    # Get API key for the preferred provider
    api_key = get_provider_api_key(config.preferred_provider)
    litellm_request["api_key"] = api_key

    # Add OpenAI base URL if configured
    if config.openai_base_url:
        litellm_request["api_base"] = config.openai_base_url

    # IMPORTANT: Tools are ALWAYS included if present - no filtering
    # The provider will handle tool support appropriately

    logger.debug(
        f"OpenAI chat completion: Model={litellm_request.get('model')}, "
        f"Tools={len(litellm_request.get('tools', []))}"
    )

    headers = {"Authorization": f"Bearer {api_key}"}
    api_key_param = litellm_request.pop("api_key", None)

    # Handle streaming
    if request.stream:
        response_generator = await litellm.acompletion(
            **litellm_request,
            api_key=api_key_param,
            headers=headers,
            allowed_openai_params=["tools"]
        )
        return StreamingResponse(
            handle_openai_streaming(response_generator),
            media_type="text/event-stream"
        )

    # Handle non-streaming
    else:
        response = litellm.completion(
            **litellm_request,
            api_key=api_key_param,
            headers=headers,
            allowed_openai_params=["tools"]
        )
        return response


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
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                    <p class="metric-value" id="total-requests">...</p>
                    <p class="metric-label">Total Requests</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="success-rate">...</p>
                    <p class="metric-label">Success Rate</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="avg-latency">...</p>
                    <p class="metric-label">Avg. Latency (ms)</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="cache-hit-rate">...</p>
                    <p class="metric-label">Cache Hit Rate</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-white rounded-lg shadow p-4">
                    <h2 class="text-lg font-semibold mb-4">Requests per Minute</h2>
                    <canvas id="requests-chart"></canvas>
                </div>
                <div class="bg-white rounded-lg shadow p-4">
                    <h2 class="text-lg font-semibold mb-4">Response Time (ms)</h2>
                    <canvas id="latency-chart"></canvas>
                </div>
            </div>

            <div class="mt-8 bg-white rounded-lg shadow p-4">
                <h2 class="text-lg font-semibold mb-4">Recent Requests</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full table-auto">
                        <thead>
                            <tr class="bg-gray-100">
                                <th class="px-4 py-2 text-left">Time</th>
                                <th class="px-4 py-2 text-left">Source Model</th>
                                <th class="px-4 py-2 text-left">Target Model</th>
                                <th class="px-4 py-2 text-left">Status</th>
                                <th class="px-4 py-2 text-left">Duration</th>
                            </tr>
                        </thead>
                        <tbody id="requests-table">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function fetchMetrics() {
                fetch('/api/metrics-data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-requests').textContent = data.total_requests;
                        document.getElementById('success-rate').textContent = data.success_rate + '%';
                        document.getElementById('avg-latency').textContent = data.avg_latency + 'ms';
                        document.getElementById('cache-hit-rate').textContent = data.cache_hit_rate + '%';
                        updateRequestTable(data.recent_requests);
                    })
                    .catch(error => console.error('Error fetching metrics:', error));
            }

            fetchMetrics();
            setInterval(fetchMetrics, 5000);

            function updateRequestTable(requests) {
                const tableBody = document.getElementById('requests-table');
                tableBody.innerHTML = '';

                requests.forEach(req => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="border px-4 py-2">${req.timestamp}</td>
                        <td class="border px-4 py-2">${req.source_model}</td>
                        <td class="border px-4 py-2">${req.target_model}</td>
                        <td class="border px-4 py-2">
                            <span class="px-2 py-1 rounded ${req.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                ${req.status}
                            </span>
                        </td>
                        <td class="border px-4 py-2">${req.duration}ms</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/api/metrics-data")
async def get_metrics_data():
    """
    Return metrics data for the dashboard.

    This would be implemented to return real metrics from a monitoring system.
    For now, returns sample data.
    """
    return {
        "total_requests": 1258,
        "success_rate": 99.2,
        "avg_latency": 245,
        "cache_hit_rate": 42.5,
        "requests_per_minute": [12, 15, 10, 8, 14, 18, 22, 25, 20, 18],
        "latency_over_time": [230, 245, 260, 220, 210, 240, 250, 270, 230, 220],
        "recent_requests": [
            {
                "timestamp": "2023-06-14 15:42:33",
                "source_model": "claude-3-sonnet",
                "target_model": config.big_model,
                "status": "success",
                "duration": 245
            },
            {
                "timestamp": "2023-06-14 15:41:58",
                "source_model": "claude-3-haiku",
                "target_model": config.small_model,
                "status": "success",
                "duration": 134
            },
            {
                "timestamp": "2023-06-14 15:41:22",
                "source_model": "claude-3-sonnet",
                "target_model": config.big_model,
                "status": "error",
                "duration": 320
            }
        ]
    }
